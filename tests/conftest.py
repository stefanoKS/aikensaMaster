"""Shared fixtures for the AIKensaMaster software-only test suite.

No physical camera, GPU or PLC is required.  Every test builds an isolated
machine root under ``tmp_path`` so a test can never touch real calibration.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from aikensa.core.settings.paths import AppPaths  # noqa: E402
from aikensa.core.settings.service import SettingsService  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def machine_root(tmp_path, monkeypatch) -> Path:
    """An unconfigured machine root with the shared defaults copied in."""

    root = tmp_path / "machine"
    shared = root / "aikensa" / "core" / "config" / "defaults"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "aikensa" / "core" / "config" / "defaults", shared)

    # find_app_root() looks for aikensa/AIKensa.py; AppPaths is used directly in
    # tests, but keeping the marker makes the fixtures behave like a real clone.
    _write(root / "aikensa" / "AIKensa.py", "")

    _write(
        root / "aikensa" / "core" / "config" / "part_registry.yaml",
        """
parts:
  - widget_id: 17
    part_number: "808387UA1A"
    part_name: "Seal FR Door Parting"
  - widget_id: 36
    part_number: "P731957YA0A"
    part_name: "Seal Roof"
  - widget_id: 43
    part_number: "P658107YA0A"
    part_name: "Seal Assy Rad Core"
widget_h2_offsets_cam2:
  43: [1.5, -2.5]
widget_crop_points:
  43: [10, 20, 210, 20, 210, 120, 10, 120]
  44: [0, 0, 100, 0, 100, 100, 0, 100]
""",
    )
    _write(
        root / "aikensa" / "core" / "config" / "parts_specifications.yaml",
        """
parts:
  "808387UA1A":
    pitchSpec: [15, 123]
    tolerance_pitch: [3.0, 1.7]
    idSpec: [0, 0, 0]
    pixelMultiplier: 0.161358
""",
    )
    _write(
        root / "aikensa" / "core" / "config" / "models.yaml",
        """
device: cpu
models:
  Dummy_Model:
    loader: yolo
    path: ./aikensa/core/checkpoints/does_not_exist.pt
""",
    )

    monkeypatch.setenv("AIKENSA_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.delenv("AIKENSA_ROOT", raising=False)
    return root


@pytest.fixture()
def paths(machine_root: Path) -> AppPaths:
    return AppPaths(root=machine_root)


@pytest.fixture()
def settings(paths: AppPaths) -> SettingsService:
    service = SettingsService(paths)
    service.bootstrap()
    return service


@pytest.fixture()
def qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def valid_camera_matrix() -> np.ndarray:
    return np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])


@pytest.fixture()
def valid_distortion() -> np.ndarray:
    return np.array([[-0.2, 0.05, 0.001, 0.002, 0.0]])


@pytest.fixture()
def valid_homography() -> np.ndarray:
    return np.array([[1.1, 0.01, -20.0], [-0.005, 1.05, 15.0], [1e-7, 2e-6, 1.0]])


@pytest.fixture()
def identity_homography() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def write_serial(settings: SettingsService, serials: dict) -> None:
    """Persist camera serials through the public API."""

    cameras, _issues = settings.repository.load_camera_settings()
    for index, serial in serials.items():
        cameras.assignments[index].serial = serial
        cameras.assignments[index].legacy_index = False
    settings.save_camera_settings(cameras)


@pytest.fixture(autouse=True)
def _silence_modal_dialogs(monkeypatch):
    """Headless tests must never block on a modal QMessageBox."""

    try:
        from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox
    except Exception:  # pragma: no cover - PyQt5 always present in this project
        yield
        return

    QApplication.instance() or QApplication([])
    messages = []

    def _information(parent, title, text, *args, **kwargs):
        messages.append(("info", title, text))
        return QMessageBox.Ok

    def _critical(parent, title, text, *args, **kwargs):
        messages.append(("error", title, text))
        return QMessageBox.Ok

    def _warning(parent, title, text, *args, **kwargs):
        messages.append(("warning", title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "information", staticmethod(_information))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_critical))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_warning))
    monkeypatch.setattr(
        QInputDialog, "getItem", staticmethod(lambda *a, **k: ("", False))
    )
    yield messages
