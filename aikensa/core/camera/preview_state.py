"""Equipment state shared by the inspection worker and the operator display."""

from dataclasses import dataclass
from enum import Enum, auto


class PreviewReadiness(Enum):
    CONNECTING = auto()
    CAMERA_UNAVAILABLE = auto()
    CALIBRATION_REQUIRED = auto()
    GEOMETRY_ERROR = auto()
    READY = auto()


class CameraPreviewState(Enum):
    UNKNOWN = auto()
    AVAILABLE = auto()
    DISCONNECTED = auto()
    READ_FAILED = auto()
    STALE_FRAME = auto()


@dataclass(frozen=True)
class EquipmentPreview:
    readiness: PreviewReadiness = PreviewReadiness.CONNECTING
    missing_cameras: tuple[int, ...] = ()
    message: str = ""
    cameras: tuple[CameraPreviewState, ...] = ()
