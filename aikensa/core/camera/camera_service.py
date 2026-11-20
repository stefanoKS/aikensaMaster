"""Single ownership of the physical cameras.

Previously ``calibration_thread`` and ``inspection_thread`` each opened their
own cameras from hardcoded serial numbers, so both could stream the same device
and a mode switch could release a handle while a frame read was in progress.
This module replaces that with:

* one logical owner per camera, enforced by a role-based lease;
* a per-camera re-entrant lock so a handle can never be released while another
  thread is inside ``read_frame``;
* an explicit lifecycle state machine;
* refusal to treat a placeholder image as a production image source unless the
  process was explicitly started in offline development mode.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from aikensa.core.settings.schema import CameraSettings
from aikensa.core.settings.service import SettingsService

LOGGER = logging.getLogger(__name__)

OFFLINE_MODE_ENV = "AIKENSA_ALLOW_PLACEHOLDER_CAMERAS"


class CameraRole(str, Enum):
    IDLE = "idle"
    CAMERA_PREVIEW = "camera_preview"
    CALIBRATION = "calibration"
    FINE_ADJUSTMENT = "fine_adjustment"
    INSPECTION = "inspection"
    ERROR = "error"

    @property
    def label_ja(self) -> str:
        return _ROLE_LABELS_JA[self]


_ROLE_LABELS_JA = {
    CameraRole.IDLE: "待機",
    CameraRole.CAMERA_PREVIEW: "プレビュー",
    CameraRole.CALIBRATION: "校正",
    CameraRole.FINE_ADJUSTMENT: "微調整",
    CameraRole.INSPECTION: "検査",
    CameraRole.ERROR: "異常",
}


class CameraUnavailableError(RuntimeError):
    """Raised when a camera cannot be opened (missing, unassigned, busy)."""

    def __init__(self, camera_index: int, reason_ja: str):
        self.camera_index = camera_index
        self.reason_ja = reason_ja
        super().__init__(f"camera {camera_index}: {reason_ja}")


class CameraBusyError(CameraUnavailableError):
    """Raised when another role already owns the camera."""


@dataclass
class CameraHandle:
    index: int
    capture: object
    role: CameraRole
    serial: Optional[str]
    is_placeholder: bool = False
    readback: Dict[str, object] = field(default_factory=dict)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.capture.read()


class CameraService:
    """Open / close / read cameras with deterministic ownership."""

    def __init__(
        self,
        settings_service: Optional[SettingsService] = None,
        *,
        capture_factory: Optional[Callable[..., object]] = None,
        allow_placeholders: Optional[bool] = None,
    ):
        self.settings = settings_service or SettingsService()
        self._capture_factory = capture_factory
        self._allow_placeholders = (
            bool(os.environ.get(OFFLINE_MODE_ENV, ""))
            if allow_placeholders is None
            else allow_placeholders
        )
        self._locks: Dict[int, threading.RLock] = {0: threading.RLock(), 1: threading.RLock()}
        self._handles: Dict[int, Optional[CameraHandle]] = {0: None, 1: None}
        self._owners: Dict[int, Optional[CameraRole]] = {0: None, 1: None}
        self._states: Dict[int, CameraRole] = {0: CameraRole.IDLE, 1: CameraRole.IDLE}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def state(self, index: int) -> CameraRole:
        return self._states[index]

    def owner(self, index: int) -> Optional[CameraRole]:
        return self._owners[index]

    def is_open(self, index: int) -> bool:
        handle = self._handles[index]
        return handle is not None

    def handle(self, index: int) -> Optional[CameraHandle]:
        return self._handles[index]

    def is_production_source(self, index: int) -> bool:
        """False when the open capture is a placeholder (never a valid source)."""

        handle = self._handles[index]
        return handle is not None and not handle.is_placeholder

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def acquire(self, index: int, role: CameraRole) -> CameraHandle:
        """Open *index* for *role*, or raise a Japanese-reasoned error."""

        if index not in self._handles:
            raise CameraUnavailableError(index, f"カメラ番号 {index} は未対応です。")

        lock = self._locks[index]
        with lock:
            current_owner = self._owners[index]
            if current_owner is not None and current_owner is not role:
                raise CameraBusyError(
                    index,
                    f"カメラ{index + 1} は{current_owner.label_ja}で使用中です。"
                    "先に現在の処理を終了してください。",
                )

            existing = self._handles[index]
            if existing is not None and existing.role is role:
                self._states[index] = role if role is not CameraRole.ERROR else CameraRole.ERROR
                return existing

            settings = self._load_camera_settings(index)
            identifier = settings.assignment(index).resolved_identifier()
            if identifier is None:
                self._states[index] = CameraRole.ERROR
                raise CameraUnavailableError(
                    index,
                    f"カメラ{index + 1} のシリアル番号が未設定です。"
                    "[設備設定] > [カメラ設定] で設定してください。",
                )

            hardware = settings.hardware_for(index)
            capture = self._open_capture(index, identifier, hardware, settings)

            placeholder = bool(getattr(capture, "is_placeholder", False)) or _is_dummy(capture)
            if placeholder and not self._allow_placeholders:
                _safe_release(capture)
                self._handles[index] = None
                self._owners[index] = None
                self._states[index] = CameraRole.ERROR
                raise CameraUnavailableError(
                    index,
                    f"カメラ{index + 1} (シリアル {identifier}) を開けませんでした。"
                    "接続・電源・シリアル設定を確認してください。",
                )

            handle = CameraHandle(
                index=index,
                capture=capture,
                role=role,
                serial=str(identifier),
                is_placeholder=placeholder,
                readback=self._readback(capture),
            )
            self._handles[index] = handle
            self._owners[index] = role
            self._states[index] = role
            return handle

    def release(self, index: int, role: Optional[CameraRole] = None) -> None:
        """Release the camera; a different role cannot release someone else's lease."""

        if index not in self._handles:
            return
        lock = self._locks[index]
        with lock:
            owner = self._owners[index]
            if owner is None:
                return
            if role is not None and owner is not role:
                raise CameraBusyError(
                    index,
                    f"カメラ{index + 1} は{owner.label_ja}が使用中のため解放できません。",
                )
            handle = self._handles[index]
            if handle is not None:
                _safe_release(handle.capture)
            self._handles[index] = None
            self._owners[index] = None
            self._states[index] = CameraRole.IDLE

    def release_all(self, role: Optional[CameraRole] = None) -> None:
        for index in list(self._handles):
            try:
                self.release(index, role)
            except CameraBusyError:
                LOGGER.warning("camera %s still owned by %s", index, self._owners[index])

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------
    def read_frame(self, index: int, role: Optional[CameraRole] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one frame while holding the camera lock.

        Holding the lock for the duration of the read is what prevents another
        thread from releasing the handle mid-read.
        """

        lock = self._locks[index]
        with lock:
            handle = self._handles[index]
            if handle is None:
                self._states[index] = CameraRole.ERROR
                return False, None
            if role is not None and self._owners[index] is not role:
                return False, None
            ok, frame = handle.read()
            return ok, frame

    def mark_error(self, index: int, reason_ja: str) -> None:
        LOGGER.error("camera %s error: %s", index, reason_ja)
        self._states[index] = CameraRole.ERROR

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_camera_settings(self, index: int) -> CameraSettings:
        self.settings.bootstrap()
        cameras, _issues = self.settings.repository.load_camera_settings()
        return cameras

    def _open_capture(self, index, identifier, hardware, settings):
        if self._capture_factory is not None:
            return self._capture_factory(
                identifier,
                width=hardware.width_px,
                height=hardware.height_px,
                fps=hardware.fps,
                exposure_us=hardware.exposure_us,
                gain_db=hardware.gain_db,
                wb_temperature=hardware.wb_temperature_k,
                wb_ratio_r=hardware.wb_ratio_r,
                wb_ratio_g=hardware.wb_ratio_g,
                wb_ratio_b=hardware.wb_ratio_b,
                rotate_180=hardware.rotate_180,
            )
        # Imported lazily: the IC4 SDK is not installed on engineering machines
        # that only need the settings UI.
        from aikensa.core.scripts.cam_init.cam_init_ic4 import open_ic4_camera_or_placeholder

        return open_ic4_camera_or_placeholder(
            identifier,
            width=hardware.width_px,
            height=hardware.height_px,
            fps=hardware.fps,
            wb_temperature=hardware.wb_temperature_k,
            wb_ratio_r=hardware.wb_ratio_r,
            wb_ratio_g=hardware.wb_ratio_g,
            wb_ratio_b=hardware.wb_ratio_b,
            rotate_180=hardware.rotate_180,
        )

    @staticmethod
    def _readback(capture) -> Dict[str, object]:
        readback: Dict[str, object] = {}
        for name, prop in (
            ("width", getattr(capture, "_W", 3)),
            ("height", getattr(capture, "_H", 4)),
            ("fps", getattr(capture, "_FPS", 5)),
            ("exposure_us", getattr(capture, "_EXP", 15)),
            ("gain_db", getattr(capture, "_GAIN", 14)),
            ("wb_temperature_k", getattr(capture, "_WB", 45)),
        ):
            try:
                readback[name] = capture.get(prop)
            except Exception:  # pragma: no cover - hardware dependent
                readback[name] = None
        return readback


def _is_dummy(capture) -> bool:
    """True for the real ``DummyCapture`` placeholder used when no camera exists."""

    return capture.__class__.__name__ == "DummyCapture"


def _safe_release(capture) -> None:
    try:
        capture.release()
    except Exception:  # pragma: no cover - hardware dependent
        LOGGER.exception("failed to release capture %r", capture)
