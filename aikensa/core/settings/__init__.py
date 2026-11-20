"""Centralised settings subsystem.

Responsibilities are split so that GUI code never touches YAML directly:

* :mod:`paths`      - resolves the application root and every runtime path.
* :mod:`yaml_store` - null-aware, atomic YAML persistence.
* :mod:`schema`     - typed settings objects with explicit units.
* :mod:`validator`  - rejects unsafe / inconsistent values.
* :mod:`repository` - reads and writes the machine-local files.
* :mod:`revisions`  - versioned revisions plus rollback metadata.
* :mod:`migration`  - backup / restore of machine-specific configuration.
* :mod:`service`    - the single facade used by the GUI and the workers.
"""

from aikensa.core.settings.paths import AppPaths, find_app_root
from aikensa.core.settings.schema import (
    CalibrationState,
    CameraAssignment,
    CameraHardwareSettings,
    CameraSettings,
    GlobalAlignment,
    HomographyCalibration,
    IntrinsicCalibration,
    MachineSettingsSnapshot,
    SettingsRevision,
    ValidationIssue,
    WidgetAlignment,
)
from aikensa.core.settings.service import SettingsService

__all__ = [
    "AppPaths",
    "find_app_root",
    "CalibrationState",
    "CameraAssignment",
    "CameraHardwareSettings",
    "CameraSettings",
    "GlobalAlignment",
    "HomographyCalibration",
    "IntrinsicCalibration",
    "MachineSettingsSnapshot",
    "SettingsRevision",
    "ValidationIssue",
    "WidgetAlignment",
    "SettingsService",
]
