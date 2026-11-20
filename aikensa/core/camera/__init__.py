"""Shared camera acquisition and ownership."""

from aikensa.core.camera.camera_service import (
    CameraBusyError,
    CameraHandle,
    CameraRole,
    CameraService,
    CameraUnavailableError,
)

__all__ = [
    "CameraBusyError",
    "CameraHandle",
    "CameraRole",
    "CameraService",
    "CameraUnavailableError",
]
