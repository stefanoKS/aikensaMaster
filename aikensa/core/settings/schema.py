"""Typed settings objects.

Every field name carries its unit so that millimetres, pixels, microseconds,
decibels and degrees can never be confused:

``*_px``      full-resolution image pixels
``*_us``      microseconds
``*_db``      decibels
``*_deg``     degrees
``*_mm``      millimetres
``*_fps``     frames per second

Values are stored in memory as ``numpy`` arrays for matrices but are always
persisted as plain YAML lists (see :mod:`aikensa.core.settings.yaml_store`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

#: Logical camera indices used across the application.  0 is labelled
#: "カメラ① / Camera 1" in the UI and owns ``Calibration_camera_0.yaml``.
CAMERA_INDICES: Tuple[int, int] = (0, 1)

CAMERA_UI_LABELS_JA: Dict[int, str] = {0: "カメラ①", 1: "カメラ②"}


class CalibrationState(str, Enum):
    """Lifecycle state of a calibration artifact or of a whole machine profile."""

    MISSING = "missing"
    """The file / value does not exist at all."""

    UNCONFIGURED = "unconfigured"
    """The file exists but still contains ``null`` placeholders."""

    INVALID = "invalid"
    """The file exists but its content fails validation."""

    INCOMPATIBLE = "incompatible"
    """Valid data that does not match the currently configured camera/resolution."""

    READY = "ready"
    """Usable for production inspection."""

    @property
    def is_usable(self) -> bool:
        return self is CalibrationState.READY

    @property
    def label_ja(self) -> str:
        return _STATE_LABELS_JA[self]


_STATE_LABELS_JA = {
    CalibrationState.MISSING: "未作成",
    CalibrationState.UNCONFIGURED: "未設定",
    CalibrationState.INVALID: "不正",
    CalibrationState.INCOMPATIBLE: "不適合",
    CalibrationState.READY: "準備完了",
}


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding, always phrased for an operator."""

    severity: Severity
    field: str
    message_ja: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.severity.value}] {self.field}: {self.message_ja}"


@dataclass
class CameraAssignment:
    """Mapping between a logical camera index and a physical IC4 serial.

    ``legacy_index`` marks data inherited from the historical
    ``cam_id.yaml`` (``0: 0``), where the value was an enumeration index rather
    than a serial number.  Such values keep working but are reported as a
    warning so an engineer can migrate them to real serials.
    """

    index: int
    serial: Optional[str] = None
    legacy_index: bool = False

    @property
    def label_ja(self) -> str:
        return CAMERA_UI_LABELS_JA.get(self.index, f"カメラ{self.index}")

    @property
    def is_assigned(self) -> bool:
        return bool(self.serial and str(self.serial).strip())

    def normalized_serial(self) -> Optional[str]:
        if not self.is_assigned:
            return None
        return str(self.serial).strip()

    def resolved_identifier(self) -> Optional[object]:
        """Identifier handed to the acquisition layer.

        An ``int`` means "IC4 enumeration index" (legacy data), a ``str`` means
        "serial number".
        """

        if not self.is_assigned:
            return None
        text = str(self.serial).strip()
        if self.legacy_index:
            try:
                return int(text)
            except ValueError:
                return text
        return text


@dataclass
class CameraHardwareSettings:
    """Operating parameters for one physical camera.

    All values are applied at stream setup.  Auto exposure / gain / white
    balance are *deliberately unsupported* for production colour stability;
    the UI must therefore show them as fixed, not as editable options.
    """

    index: int
    width_px: int = 3072
    height_px: int = 2048
    fps: float = 15.0
    rotate_180: bool = True
    exposure_us: float = 9000.0
    gain_db: float = 15.0
    wb_temperature_k: int = 4500
    wb_ratio_r: Optional[float] = 2.2
    wb_ratio_g: Optional[float] = 1.0
    wb_ratio_b: Optional[float] = 2.2

    #: Auto modes are not applied by the acquisition layer; these flags exist
    #: so that the UI can explain the restriction instead of silently ignoring
    #: operator input.
    auto_exposure_supported: bool = False
    auto_gain_supported: bool = False
    auto_wb_supported: bool = False

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width_px, self.height_px)

    @property
    def label_ja(self) -> str:
        return CAMERA_UI_LABELS_JA.get(self.index, f"カメラ{self.index}")

    def same_acquisition_geometry(self, other: "CameraHardwareSettings") -> bool:
        return (self.width_px, self.height_px, self.rotate_180) == (
            other.width_px,
            other.height_px,
            other.rotate_180,
        )


@dataclass
class CameraSettings:
    """Camera assignments plus hardware parameters for the whole machine."""

    assignments: Dict[int, CameraAssignment] = field(default_factory=dict)
    hardware: Dict[int, CameraHardwareSettings] = field(default_factory=dict)
    rotate_180_global: bool = True

    @classmethod
    def unconfigured(cls) -> "CameraSettings":
        return cls(
            assignments={i: CameraAssignment(index=i) for i in CAMERA_INDICES},
            hardware={i: CameraHardwareSettings(index=i) for i in CAMERA_INDICES},
            rotate_180_global=True,
        )

    def assignment(self, index: int) -> CameraAssignment:
        return self.assignments.setdefault(index, CameraAssignment(index=index))

    def hardware_for(self, index: int) -> CameraHardwareSettings:
        return self.hardware.setdefault(index, CameraHardwareSettings(index=index))

    def assigned_serials(self) -> List[Tuple[int, str]]:
        return [
            (idx, a.normalized_serial())
            for idx, a in sorted(self.assignments.items())
            if a.normalized_serial()
        ]


@dataclass
class IntrinsicCalibration:
    """Intrinsic (ChArUco) calibration for a single camera."""

    camera_index: int
    camera_matrix: Optional[np.ndarray] = None
    distortion_coefficients: Optional[np.ndarray] = None
    rotation_vectors: Optional[np.ndarray] = None
    translation_vectors: Optional[np.ndarray] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    state: CalibrationState = CalibrationState.MISSING
    source_path: Optional[str] = None

    @property
    def is_null(self) -> bool:
        return self.camera_matrix is None or self.distortion_coefficients is None


@dataclass
class HomographyCalibration:
    """Homography (nominal template plane) for a single camera."""

    camera_index: int
    matrix: Optional[np.ndarray] = None
    scaled_matrix: Optional[np.ndarray] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    state: CalibrationState = CalibrationState.MISSING
    source_path: Optional[str] = None


@dataclass
class GlobalAlignment:
    """Machine-wide fine adjustment of camera 2 relative to camera 1.

    Applied in *destination* (merged template) space, in full-resolution
    pixels, after the base homography::

        H_effective = T_global @ H_base
    """

    cam2_x_px: float = 0.0
    cam2_y_px: float = 0.0
    cam2_rotation_deg: float = 0.0
    pivot_px: Optional[Tuple[float, float]] = None
    max_abs_shift_px: float = 500.0
    max_abs_rotation_deg: float = 2.0
    revision_id: Optional[str] = None

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.cam2_x_px) < 1e-9
            and abs(self.cam2_y_px) < 1e-9
            and abs(self.cam2_rotation_deg) < 1e-9
        )


@dataclass
class CropRect:
    """Axis-aligned ROI in full-resolution merged-image pixels."""

    x_px: int
    y_px: int
    width_px: int
    height_px: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x_px, self.y_px, self.width_px, self.height_px)

    def is_valid(self) -> bool:
        return self.width_px > 0 and self.height_px > 0

    def clamp_to(self, bounds_w: int, bounds_h: int) -> "CropRect":
        x = max(0, min(self.x_px, max(0, bounds_w - 1)))
        y = max(0, min(self.y_px, max(0, bounds_h - 1)))
        w = max(1, min(self.width_px, bounds_w - x))
        h = max(1, min(self.height_px, bounds_h - y))
        return CropRect(x_px=x, y_px=y, width_px=w, height_px=h)


@dataclass
class WidgetAlignment:
    """Per-part / per-widget correction layered on top of the global alignment.

    ``source`` records whether the effective value comes from the shared
    registry defaults or from a machine-local override, so the UI can offer a
    truthful "restore shared default" action.
    """

    widget_id: int
    part_number: str = ""
    cam2_offset_x_px: float = 0.0
    cam2_offset_y_px: float = 0.0
    cam2_rotation_deg: float = 0.0
    crop: Optional[CropRect] = None
    katabu_crop: Optional[CropRect] = None
    source: str = "shared_default"
    max_abs_offset_px: float = 300.0

    def is_overridden(self) -> bool:
        return self.source == "machine_override"

    def same_as(self, other: "WidgetAlignment") -> bool:
        return (
            abs(self.cam2_offset_x_px - other.cam2_offset_x_px) < 1e-9
            and abs(self.cam2_offset_y_px - other.cam2_offset_y_px) < 1e-9
            and abs(self.cam2_rotation_deg - other.cam2_rotation_deg) < 1e-9
            and self.crop == other.crop
            and self.katabu_crop == other.katabu_crop
        )


@dataclass(frozen=True)
class SettingsRevision:
    """One recorded revision of machine-local adjustment settings."""

    revision_id: str
    created_at: datetime
    operator: str = ""
    machine_id: str = ""
    changed_files: Tuple[str, ...] = ()
    changes: Tuple[str, ...] = ()
    activated: bool = True
    note: str = ""


@dataclass
class MachineSettingsSnapshot:
    """Internally consistent view of everything the inspection needs."""

    cameras: CameraSettings
    intrinsics: Dict[int, IntrinsicCalibration] = field(default_factory=dict)
    homographies: Dict[int, HomographyCalibration] = field(default_factory=dict)
    global_alignment: GlobalAlignment = field(default_factory=GlobalAlignment)
    widget_alignments: Dict[int, WidgetAlignment] = field(default_factory=dict)
    revision_id: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)

    def issues_for_camera(self, index: int, error_only: bool = True) -> List[ValidationIssue]:
        prefix = f"camera{index}."
        return [
            issue
            for issue in self.issues
            if issue.field.startswith(prefix)
            and (not error_only or issue.severity is Severity.ERROR)
        ]
