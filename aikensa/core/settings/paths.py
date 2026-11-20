"""Path resolution for shared defaults and machine-local runtime data.

The application historically used a mix of repository-relative relative paths
(``"aikensa/core/param/camera/homography_param_cam1.yaml"``) and
``os.path.dirname(__file__)`` based paths.  Both break as soon as the process
working directory changes, so the settings subsystem resolves everything from
one place.

Two locations are distinguished:

``shared_dir``
    Git-tracked application defaults (templates, schemas).  Safe to overwrite
    on ``git pull``.

``runtime_dir``
    Machine-specific configuration.  Must never be silently replaced by a
    ``git pull`` once the two-stage migration has been executed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../aikensa
REPO_ROOT = PACKAGE_ROOT.parent
APP_ROOT_ENV = "AIKENSA_ROOT"

#: Names that must never change: downstream tooling and deployed machines
#: depend on them.
CAM_ID_FILENAME = "cam_id.yaml"
CAM_CONFIG_FILENAME = "cam_config.yaml"
GLOBAL_ALIGNMENT_FILENAME = "global_alignment.yaml"
WIDGET_ADJUSTMENTS_FILENAME = "widget_adjustments.yaml"

INTRINSIC_FILENAMES = (
    "Calibration_camera_0.yaml",
    "Calibration_camera_1.yaml",
    "Calibration_camera_scaled_0.yaml",
    "Calibration_camera_scaled_1.yaml",
)

HOMOGRAPHY_FILENAMES = (
    "homography_param_cam1.yaml",
    "homography_param_cam2.yaml",
    "homography_param_cam1_scaled.yaml",
    "homography_param_cam2_scaled.yaml",
    # Optional legacy high-resolution variants kept for compatibility.
    "homography_param_cam1_high.yaml",
    "homography_param_cam2_high.yaml",
    "homography_param_cam1_high_scaled.yaml",
    "homography_param_cam2_high_scaled.yaml",
)

PLANARIZE_FILENAMES = (
    "planarizeTransform_wide.yaml",
    "planarizeTransform_wide_scaled.yaml",
    # Optional legacy variants kept for compatibility.
    "planarizeTransform_narrow.yaml",
    "planarizeTransform_narrow_scaled.yaml",
    "planarizeTransform_high_wide.yaml",
    "planarizeTransform_high_wide_scaled.yaml",
    "planarizeTransform_high_narrow.yaml",
    "planarizeTransform_high_narrow_scaled.yaml",
)

#: Every machine-specific runtime file, grouped by owning directory.
RUNTIME_CAM_FILES = (CAM_ID_FILENAME, CAM_CONFIG_FILENAME)
RUNTIME_CAMERA_FILES = INTRINSIC_FILENAMES + HOMOGRAPHY_FILENAMES + PLANARIZE_FILENAMES
RUNTIME_ADJUSTMENT_FILES = (GLOBAL_ALIGNMENT_FILENAME, WIDGET_ADJUSTMENTS_FILENAME)


def find_app_root(start: Optional[Union[os.PathLike, str]] = None) -> Path:
    """Return the repository root that contains the ``aikensa`` package.

    Resolution order:

    1. ``AIKENSA_ROOT`` environment variable, when set.
    2. Walk upwards from *start* (default: this file) looking for a directory
       that contains ``aikensa/AIKensa.py``.
    3. Fall back to the repository root derived from this module's location.
    """

    env_root = os.environ.get(APP_ROOT_ENV)
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "aikensa").is_dir():
            return candidate

    anchor = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (anchor, *anchor.parents):
        if (candidate / "aikensa" / "AIKensa.py").is_file():
            return candidate

    return REPO_ROOT


@dataclass(frozen=True)
class AppPaths:
    """Immutable bundle of every directory the settings subsystem needs."""

    root: Path

    @classmethod
    def discover(cls, start: Optional[Union[os.PathLike, str]] = None) -> "AppPaths":
        return cls(root=find_app_root(start))

    # ---- shared (git tracked) -------------------------------------------------
    @property
    def shared_dir(self) -> Path:
        return self.root / "aikensa" / "core" / "config" / "defaults"

    @property
    def shared_cam_dir(self) -> Path:
        return self.shared_dir / "cam"

    @property
    def shared_camera_dir(self) -> Path:
        return self.shared_dir / "camera"

    @property
    def shared_adjustments_dir(self) -> Path:
        return self.shared_dir / "adjustments"

    # ---- machine specific (never committed) ----------------------------------
    @property
    def runtime_cam_dir(self) -> Path:
        return self.root / "aikensa" / "core" / "config" / "cam"

    @property
    def runtime_camera_dir(self) -> Path:
        return self.root / "aikensa" / "core" / "param" / "camera"

    @property
    def runtime_adjustments_dir(self) -> Path:
        return self.root / "aikensa" / "core" / "param" / "adjustments"

    @property
    def runtime_history_dir(self) -> Path:
        return self.runtime_adjustments_dir / "revisions"

    @property
    def backup_dir(self) -> Path:
        """Machine-local backup root, deliberately outside the work tree."""
        override = os.environ.get("AIKENSA_BACKUP_DIR")
        if override:
            return Path(override).expanduser().resolve()
        return self.root.parent / "aikensa_backups"

    # ---- fully qualified helpers ---------------------------------------------
    def cam_id_file(self) -> Path:
        return self.runtime_cam_dir / CAM_ID_FILENAME

    def cam_config_file(self) -> Path:
        return self.runtime_cam_dir / CAM_CONFIG_FILENAME

    def global_alignment_file(self) -> Path:
        return self.runtime_adjustments_dir / GLOBAL_ALIGNMENT_FILENAME

    def widget_adjustments_file(self) -> Path:
        return self.runtime_adjustments_dir / WIDGET_ADJUSTMENTS_FILENAME

    def camera_file(self, filename: str) -> Path:
        return self.runtime_camera_dir / filename

    def shared_cam_template(self, filename: str) -> Path:
        return self.shared_cam_dir / filename

    def shared_camera_template(self, filename: str) -> Path:
        return self.shared_camera_dir / filename

    def shared_adjustment_template(self, filename: str) -> Path:
        return self.shared_adjustments_dir / filename

    def calibration_file(self, camera_index: int) -> Path:
        return self.camera_file(f"Calibration_camera_{camera_index}.yaml")

    def calibration_scaled_file(self, camera_index: int) -> Path:
        return self.camera_file(f"Calibration_camera_scaled_{camera_index}.yaml")

    def homography_file(self, camera_index: int, *, scaled: bool = False) -> Path:
        # Runtime names are 1-based ("cam1"/"cam2") for compatibility.
        suffix = "_scaled" if scaled else ""
        return self.camera_file(f"homography_param_cam{camera_index + 1}{suffix}.yaml")
