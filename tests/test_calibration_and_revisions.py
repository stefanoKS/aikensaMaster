"""Group B: calibration files and group G: revision / backup behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from aikensa.core.settings.migration import (
    backup_machine_files,
    restore_machine_files,
    verify_backup,
)
from aikensa.core.settings.schema import CalibrationState, GlobalAlignment
from aikensa.core.settings.service import SettingsService
from aikensa.core.settings.yaml_store import SettingsValidationError, read_yaml
from tests.conftest import REPO_ROOT, write_serial


# ---------------------------------------------------------------------------
# Intrinsic calibration
# ---------------------------------------------------------------------------
def test_null_intrinsic_is_unconfigured(settings):
    intrinsic = settings.repository.load_intrinsic(0)
    assert intrinsic.state is CalibrationState.UNCONFIGURED
    assert intrinsic.camera_matrix is None
    assert intrinsic.is_null


def test_load_intrinsic_never_invents_a_matrix(settings):
    snapshot = settings.load_snapshot()
    intrinsic = snapshot.intrinsics[0]
    assert intrinsic.camera_matrix is None


def test_valid_intrinsic_round_trip(settings, valid_camera_matrix, valid_distortion):
    settings.save_intrinsic(0, valid_camera_matrix, valid_distortion)

    loaded = settings.repository.load_intrinsic(0)

    assert loaded.state is CalibrationState.READY
    np.testing.assert_allclose(loaded.camera_matrix, valid_camera_matrix)
    stored = read_yaml(settings.paths.calibration_file(0))
    assert "camera_matrix" in stored
    assert "distortion_coefficients" in stored


def test_invalid_intrinsic_is_refused_and_previous_file_survives(
    settings, valid_camera_matrix, valid_distortion
):
    settings.save_intrinsic(0, valid_camera_matrix, valid_distortion)
    before = settings.paths.calibration_file(0).read_text(encoding="utf-8")

    with pytest.raises(SettingsValidationError):
        settings.save_intrinsic(0, None, valid_distortion)

    assert settings.paths.calibration_file(0).read_text(encoding="utf-8") == before
    assert settings.repository.load_intrinsic(0).state is CalibrationState.READY


def test_broken_intrinsic_file_is_invalid(settings):
    settings.paths.calibration_file(0).write_text("camera_matrix: [[1, 2]]\n", encoding="utf-8")
    assert settings.repository.load_intrinsic(0).state is CalibrationState.INVALID


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------
def test_missing_homography_state(settings):
    settings.paths.homography_file(0).unlink()
    assert settings.repository.load_homography(0).state is CalibrationState.MISSING


def test_null_homography_is_unconfigured(settings):
    assert settings.repository.load_homography(0).state is CalibrationState.UNCONFIGURED


def test_degenerate_homography_is_invalid(settings):
    settings.paths.homography_file(0).write_text("[[0, 0, 0], [0, 0, 0], [0, 0, 1]]\n", encoding="utf-8")
    assert settings.repository.load_homography(0).state is CalibrationState.INVALID


def test_valid_homography_round_trip(settings, valid_homography):
    settings.save_homography(0, valid_homography)

    loaded = settings.repository.load_homography(0)

    assert loaded.state is CalibrationState.READY
    np.testing.assert_allclose(loaded.matrix, valid_homography)


def test_homography_save_rejects_none(settings, valid_homography):
    settings.save_homography(0, valid_homography)
    before = settings.paths.homography_file(0).read_text(encoding="utf-8")

    with pytest.raises(SettingsValidationError):
        settings.save_homography(0, None)

    assert settings.paths.homography_file(0).read_text(encoding="utf-8") == before


def test_existing_repository_calibration_files_still_load():
    """The real machine files committed in the repository must stay readable."""

    from aikensa.core.settings.paths import AppPaths

    service = SettingsService(AppPaths(root=REPO_ROOT))
    for index in (0, 1):
        intrinsic = service.repository.load_intrinsic(index)
        homography = service.repository.load_homography(index)
        assert intrinsic.state is CalibrationState.READY
        assert intrinsic.camera_matrix.shape == (3, 3)
        assert homography.state is CalibrationState.READY
        assert homography.matrix.shape == (3, 3)


def test_scaled_calibration_filenames_are_preserved(settings, valid_camera_matrix, valid_distortion):
    settings.save_intrinsic(0, valid_camera_matrix, valid_distortion, scaled=True)

    assert settings.paths.calibration_scaled_file(0).is_file()
    stored = read_yaml(settings.paths.calibration_scaled_file(0))
    assert stored["camera_matrix"] is not None


# ---------------------------------------------------------------------------
# Group G: revisions and backups
# ---------------------------------------------------------------------------
def test_global_alignment_save_and_restart(settings, paths):
    settings.save_global_alignment(
        GlobalAlignment(cam2_x_px=12.5, cam2_y_px=-3.25, cam2_rotation_deg=0.25)
    )

    restarted = SettingsService(paths)
    alignment = restarted.active_global_alignment()

    assert alignment.cam2_x_px == pytest.approx(12.5)
    assert alignment.cam2_y_px == pytest.approx(-3.25)
    assert alignment.cam2_rotation_deg == pytest.approx(0.25)


def test_out_of_range_alignment_is_rejected(settings):
    with pytest.raises(SettingsValidationError):
        settings.save_global_alignment(GlobalAlignment(cam2_x_px=10_000.0))

    assert settings.active_global_alignment().cam2_x_px == 0.0


def test_rollback_restores_previous_values(settings):
    settings.save_global_alignment(GlobalAlignment(cam2_x_px=5.0))
    first_revision = settings.revisions.active_revision_id()

    settings.save_global_alignment(GlobalAlignment(cam2_x_px=25.0))
    assert settings.active_global_alignment().cam2_x_px == pytest.approx(25.0)

    settings.rollback(first_revision)
    assert settings.active_global_alignment().cam2_x_px == pytest.approx(5.0)


def test_corrupted_staged_revision_is_rejected(settings):
    import json

    revision = settings.save_global_alignment(GlobalAlignment(cam2_x_px=1.0))
    revision_dir = settings.paths.runtime_history_dir / revision.revision_id
    manifest = json.loads((revision_dir / "manifest.json").read_text(encoding="utf-8"))
    stored = revision_dir / "files" / manifest["files"][0]["stored"]
    stored.write_text("cam2: {x_px: 999.0}\n", encoding="utf-8")

    with pytest.raises(Exception):
        settings.rollback(revision.revision_id)

    assert settings.active_global_alignment().cam2_x_px == pytest.approx(1.0)


def test_backup_is_verified_and_restorable(settings, paths):
    write_serial(settings, {0: "BACKUP-0", 1: "BACKUP-1"})

    result = backup_machine_files(paths, reason="test")

    assert result.ok, result.problems
    assert verify_backup(result.backup_dir) == []
    assert any("cam_id.yaml" in name for name in result.copied)

    # Destroy the runtime value, then restore it.
    paths.cam_id_file().write_text("0: null\n1: null\n", encoding="utf-8")
    restore_machine_files(paths, result.backup_dir)

    restored, _ = settings.repository.load_camera_settings()
    assert restored.assignment(0).normalized_serial() == "BACKUP-0"


def test_tampered_backup_fails_verification(settings, paths):
    settings.save_global_alignment(GlobalAlignment(cam2_x_px=2.0))
    result = backup_machine_files(paths, reason="tamper")
    assert result.ok

    target = result.backup_dir / "adjustments" / "global_alignment.yaml"
    target.write_text("cam2: {x_px: 900.0}\n", encoding="utf-8")

    problems = verify_backup(result.backup_dir)
    assert problems
    with pytest.raises(RuntimeError):
        restore_machine_files(paths, result.backup_dir)


def test_backup_is_restorable_on_a_different_machine_root(settings, paths, tmp_path):
    """A backup made on machine A must restore into machine B's own tree."""

    write_serial(settings, {0: "MACHINE-A-0", 1: "MACHINE-A-1"})
    result = backup_machine_files(paths, reason="portable")
    assert result.ok

    replacement = tmp_path / "replacement"
    (replacement / "aikensa").mkdir(parents=True)
    (replacement / "aikensa" / "AIKensa.py").write_text("", encoding="utf-8")

    from aikensa.core.settings.paths import AppPaths

    new_paths = AppPaths(root=replacement)
    restored = restore_machine_files(new_paths, result.backup_dir)

    assert all(str(replacement) in name for name in restored)
    restored_service = SettingsService(new_paths)
    restored_service.bootstrap()
    cameras, _ = restored_service.repository.load_camera_settings()
    assert cameras.assignment(0).normalized_serial() == "MACHINE-A-0"
    # The source machine's own file is untouched by a cross-machine restore.
    source_cameras, _ = settings.repository.load_camera_settings()
    assert source_cameras.assignment(0).normalized_serial() == "MACHINE-A-0"
