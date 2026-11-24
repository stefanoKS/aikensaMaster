"""Single ownership of the physical cameras.

Previously ``calibration_thread`` and ``inspection_thread`` each opened their
own cameras from hardcoded serial numbers, so both could stream the same device
and a mode switch could release a handle while a frame read was in progress.
This module replaces that with:

* one logical owner per camera, enforced by a role-based lease;
* a per-camera re-entrant lock so a handle can never be released while another
  thread is inside ``read_frame``;
* an explicit lifecycle state machine;
* refusal to treat a placeholder image as an inspection or calibration source,
  even in offline development mode.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from cv2 import error as OpenCVError

from aikensa.core.settings.schema import CameraSettings
from aikensa.core.settings.service import SettingsService

LOGGER = logging.getLogger(__name__)

OFFLINE_MODE_ENV = "AIKENSA_ALLOW_PLACEHOLDER_CAMERAS"
_RECONNECT_INTERVAL = 2.0
# IC4 SnapSink waits up to 1 second per capture. Reject an older frame
# rather than pairing it with a much later snapshot from the other camera.
_MAX_PAIR_AGE = 1.25


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


@dataclass(frozen=True)
class RealCameraFrame:
    index: int
    image: np.ndarray
    received_at: float


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
            if identifier == settings.assignment(1 - index).resolved_identifier():
                self._states[index] = CameraRole.ERROR
                raise CameraUnavailableError(index, "カメラ①とカメラ②の割り当てが重複しています。")

            hardware = settings.hardware_for(index)
            try:
                capture = self._open_capture(index, identifier, hardware)
            except (ImportError, OSError, RuntimeError) as exc:
                self._states[index] = CameraRole.ERROR
                LOGGER.debug("camera %s could not be opened", index, exc_info=True)
                raise CameraUnavailableError(
                    index, f"カメラ{index + 1} を開けません。接続・電源・IC4 を確認してください。"
                ) from exc

            placeholder = bool(getattr(capture, "is_placeholder", False)) or _is_dummy(capture)
            if placeholder and not (
                self._allow_placeholders and role is CameraRole.CAMERA_PREVIEW
            ):
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
            with self._locks[index]:
                if role is None or self._owners[index] is role:
                    self.release(index, role)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------
    def read_frame(self, index: int, role: Optional[CameraRole] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """Compatibility API: read only real frames."""
        frame = self.read_real_frame(index, role)
        return (True, frame.image) if frame is not None else (False, None)

    def read_real_frame(self, index: int, role: Optional[CameraRole]) -> Optional[RealCameraFrame]:
        """Read a new BGR8 snapshot without a display or previous-frame fallback."""
        lock = self._locks[index]
        with lock:
            handle = self._handles[index]
            if handle is None or handle.index != index or handle.is_placeholder:
                return None
            if role is not None and (handle.role is not role or self._owners[index] is not role):
                return None
            try:
                ok, frame = handle.read()
            except (OSError, RuntimeError, OpenCVError):
                LOGGER.debug("camera %s read failed", index, exc_info=True)
                return None
            received_at = time.monotonic()
            if not (
                ok and isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                and frame.ndim == 3 and frame.shape[2] == 3
                and frame.shape[0] > 0 and frame.shape[1] > 0
            ):
                return None
            return RealCameraFrame(index, frame, received_at)

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

    def _open_capture(self, index, identifier, hardware):
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
            camera_index=index,
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


class CameraStream:
    """Acquire real frames for a worker and back off failed opens/reads."""

    def __init__(self, service: CameraService, role: CameraRole):
        self.service = service
        self.role = role
        self.errors: Dict[int, str] = {}
        self._retry_at = {0: 0.0, 1: 0.0}

    def read(self, index: int) -> Optional[RealCameraFrame]:
        if time.monotonic() < self._retry_at[index]:
            return None
        if self.service.owner(index) is not self.role:
            try:
                self.service.acquire(index, self.role)
            except CameraUnavailableError as exc:
                self._fail(index, exc.reason_ja)
                return None
        frame = self.service.read_real_frame(index, self.role)
        if frame is None:
            self._fail(index, f"カメラ{index + 1} の画像取得エラー。接続・電源を確認してください。")
        else:
            self.errors.pop(index, None)
        return frame

    def read_pair(self) -> Tuple[Optional[RealCameraFrame], Optional[RealCameraFrame]]:
        first, second = self.read(0), self.read(1)
        if first is not None and second is not None and (
            second.received_at - first.received_at > _MAX_PAIR_AGE
        ):
            self.errors[0] = "カメラ1 の画像が古すぎます。接続・FPS 設定を確認してください。"
            first = None
        return first, second

    def close(self) -> None:
        self.service.release_all(self.role)
        self.errors.clear()
        self._retry_at = {0: 0.0, 1: 0.0}

    def _fail(self, index: int, reason: str) -> None:
        self.errors[index] = reason
        self._retry_at[index] = time.monotonic() + _RECONNECT_INTERVAL
        with self.service._locks[index]:
            if self.service.owner(index) is self.role:
                self.service.release(index, self.role)


def _is_dummy(capture) -> bool:
    """True for the real ``DummyCapture`` placeholder used when no camera exists."""

    return capture.__class__.__name__ == "DummyCapture"


def _safe_release(capture) -> None:
    try:
        capture.release()
    except Exception:  # pragma: no cover - hardware dependent
        LOGGER.exception("failed to release capture %r", capture)
