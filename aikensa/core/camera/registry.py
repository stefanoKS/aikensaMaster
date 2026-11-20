"""Process-wide access to the single :class:`CameraService` instance.

Both worker threads must share one service: that is what makes "the same camera
cannot be opened by two owners at once" enforceable across threads.
"""

from __future__ import annotations

import threading
from typing import Optional

from aikensa.core.camera.camera_service import CameraService
from aikensa.core.settings.service import SettingsService

_service: Optional[CameraService] = None
_lock = threading.Lock()


def get_camera_service(settings_service: Optional[SettingsService] = None) -> CameraService:
    """Return the shared camera service, creating it on first use."""

    global _service
    with _lock:
        if _service is None:
            _service = CameraService(settings_service or SettingsService())
        return _service


def reset_camera_service(service: Optional[CameraService] = None) -> None:
    """Replace the shared service (used by tests)."""

    global _service
    with _lock:
        _service = service
