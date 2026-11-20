"""Group A: machine configuration behaviour."""

from __future__ import annotations

import pytest

from aikensa.core.settings.paths import AppPaths
from aikensa.core.settings.schema import CalibrationState, Severity
from aikensa.core.settings.service import SettingsService
from aikensa.core.settings.yaml_store import SettingsValidationError, read_yaml, write_yaml_atomic
from tests.conftest import REPO_ROOT, write_serial


def test_bootstrap_creates_null_templates(settings):
    data = read_yaml(settings.paths.cam_id_file())
    assert data == {0: None, 1: None}

    snapshot = settings.load_snapshot()
    assert snapshot.cameras.assignment(0).normalized_serial() is None
    assert settings.camera_state(snapshot, 0) is CalibrationState.UNCONFIGURED
    assert settings.production_ready(snapshot) is False


def test_bootstrap_never_overwrites_existing_values(settings):
    write_serial(settings, {0: "SN-A", 1: "SN-B"})
    before = settings.paths.cam_id_file().read_text(encoding="utf-8")

    created = settings.repository.ensure_runtime_layout()

    assert created == []
    assert settings.paths.cam_id_file().read_text(encoding="utf-8") == before
    cameras, _issues = settings.repository.load_camera_settings()
    assert cameras.assignment(0).normalized_serial() == "SN-A"


def test_null_camera_id_is_unconfigured_not_error(settings):
    snapshot = settings.load_snapshot()
    errors = [i for i in snapshot.issues if i.severity is Severity.ERROR]
    assert any("シリアル番号が未設定" in i.message_ja for i in errors)


def test_valid_serial_round_trip_after_restart(settings, paths):
    write_serial(settings, {0: "40720154", 1: "40720155"})

    restarted = SettingsService(AppPaths(root=paths.root))
    cameras, issues = restarted.repository.load_camera_settings()

    assert cameras.assignment(0).normalized_serial() == "40720154"
    assert cameras.assignment(1).normalized_serial() == "40720155"
    assert not [i for i in issues if i.severity is Severity.ERROR]


def test_duplicate_serial_is_rejected(settings):
    cameras, _ = settings.repository.load_camera_settings()
    cameras.assignments[0].serial = "SAME"
    cameras.assignments[1].serial = "SAME"

    with pytest.raises(SettingsValidationError) as excinfo:
        settings.save_camera_settings(cameras)

    assert any("重複" in i.message_ja for i in excinfo.value.issues)


def test_invalid_resolution_is_rejected(settings):
    cameras, _ = settings.repository.load_camera_settings()
    cameras.assignments[0].serial = "SN-A"
    cameras.assignments[1].serial = "SN-B"
    cameras.hardware_for(0).width_px = 999999

    with pytest.raises(SettingsValidationError):
        settings.save_camera_settings(cameras)


def test_invalid_yaml_is_reported_not_silently_defaulted(settings):
    settings.paths.cam_id_file().write_text("0: [unclosed\n", encoding="utf-8")

    snapshot = settings.load_snapshot()

    assert any("cam_id.yaml" in i.message_ja for i in snapshot.issues)
    assert settings.camera_state(snapshot, 0) is CalibrationState.UNCONFIGURED


def test_legacy_index_assignment_is_preserved(settings):
    write_yaml_atomic(settings.paths.cam_id_file(), {0: 0, 1: 1})

    cameras, issues = settings.repository.load_camera_settings()

    assert cameras.assignment(0).legacy_index is True
    assert cameras.assignment(0).resolved_identifier() == 0
    assert any(i.severity is Severity.WARNING for i in issues)


def test_two_machines_are_isolated(machine_root, tmp_path):
    machine_b = tmp_path / "machine_b"
    machine_b.mkdir()
    (machine_b / "aikensa").mkdir()
    (machine_b / "aikensa" / "AIKensa.py").write_text("", encoding="utf-8")

    service_a = SettingsService(AppPaths(root=machine_root))
    service_b = SettingsService(AppPaths(root=machine_b))
    service_a.bootstrap()
    service_b.bootstrap()

    write_serial(service_a, {0: "MACHINE-A-0", 1: "MACHINE-A-1"})

    cameras_b, _ = service_b.repository.load_camera_settings()
    assert cameras_b.assignment(0).normalized_serial() is None


def test_repository_defaults_are_null_not_developer_serials():
    """A new clone must not inherit this developer machine's camera IDs."""

    template = read_yaml(REPO_ROOT / "aikensa" / "core" / "config" / "defaults" / "cam" / "cam_id.yaml")
    assert template == {0: None, 1: None}


def test_runtime_serials_are_not_in_shared_defaults(settings):
    write_serial(settings, {0: "ONLY-IN-RUNTIME", 1: "ONLY-IN-RUNTIME-B"})

    shared = read_yaml(settings.paths.shared_cam_template("cam_id.yaml"))
    assert shared == {0: None, 1: None}


def test_camera_geometry_change_marks_calibration_incompatible(
    settings, valid_camera_matrix, valid_distortion, valid_homography
):
    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    settings.save_intrinsic(0, valid_camera_matrix, valid_distortion)
    settings.save_homography(0, valid_homography)
    assert settings.camera_state(settings.load_snapshot(), 0) is CalibrationState.READY

    cameras, _ = settings.repository.load_camera_settings()
    cameras.hardware_for(0).width_px = 2592
    cameras.hardware_for(0).height_px = 1944
    settings.save_camera_settings(cameras)

    snapshot = settings.load_snapshot()
    assert settings.camera_state(snapshot, 0) is CalibrationState.INCOMPATIBLE
    assert any(i.severity is Severity.ERROR for i in snapshot.issues)
