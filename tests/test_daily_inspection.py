"""Daily measurement behavior with synthetic points and no device dependencies."""

from types import SimpleNamespace
from dataclasses import FrozenInstanceError
import shutil
import sqlite3

import numpy as np
import pytest

from aikensa.core.inspection.daily import (
    DailyFailure,
    DailyInferenceSpec,
    DailyInspectionError,
    DailyInspectionStrategy,
    DailyROI,
    DailyReferenceSpec,
)
from aikensa.core.settings.revisions import RevisionError
from aikensa.core.settings.service import SettingsService
from tests.conftest import REPO_ROOT


def pose(*points):
    return [SimpleNamespace(keypoints=SimpleNamespace(xy=np.array([points], dtype=float)))]


@pytest.mark.parametrize(
    ("pixels", "actual", "status"),
    [(618, 99.2, "OK"), (617, 99.0, "OK"), (616, 98.8, "NG"),
     (623, 100.0, "OK"), (627, 100.6, "OK"), (629.1, 100.9, "OK"), (630, 101.1, "NG"),
     (629.9, 101.1, "NG")],
)
def test_horizontal_rounded_daily_judgment_retains_actual_value(pixels, actual, status):
    image = np.zeros((40, 800, 3), np.uint8)
    result = DailyInspectionStrategy().inspect(
        image, DailyReferenceSpec(100.0, 1.0, 0.1604477), DailyROI(10, 2, 700, 38),
        DailyInferenceSpec(0.4, 1980), lambda **_: pose((pixels, 3), (0, 28)),
    )
    assert result.status == status
    assert result.actual_mm == actual
    assert result.endpoints[0] == (10, 30)


@pytest.mark.parametrize("x", [0, 2100, 4200])
def test_daily_position_uses_approved_bounded_crop(x):
    image = np.zeros((431, 6528, 3), np.uint8)
    image[4, x] = [10, 20, 30]

    def infer(*, source, conf, imgsz, verbose):
        assert source.shape == (431, 1890, 3)
        assert tuple(source[4, 0]) == (10, 20, 30)
        assert (conf, imgsz, verbose) == (0.4, 1980, False)
        source[4, 0] = 0
        return pose((633, 6), (0, 4), (12, 11))

    result = DailyInspectionStrategy().inspect(
        image, DailyReferenceSpec(100, 2, 0.1604477),
        DailyROI(x, 0, 1890, 1342, allow_bottom_clip=True), DailyInferenceSpec(0.4, 1980), infer,
    )
    assert result.endpoints == ((x, 4), (x + 633, 6))
    assert result.status == "OK" and result.actual_mm == 101.6
    assert tuple(image[4, x]) == (10, 20, 30)


@pytest.mark.parametrize("output", [[SimpleNamespace(keypoints=None)], pose((3, 4), (float("nan"), 2))])
def test_successful_inference_without_two_finite_points_has_no_measurement(output):
    result = DailyInspectionStrategy().inspect(
        np.zeros((10, 10, 3), np.uint8), DailyReferenceSpec(100, 1, 0.16),
        DailyROI(0, 0, 10, 10), DailyInferenceSpec(0.4, 1980), lambda **_: output,
    )
    assert (result.status, result.reason, result.actual_mm) == ("NG", "DETECTION NG", None)


def broken_model(**_):
    raise RuntimeError("device failure")


@pytest.mark.parametrize(
    ("model", "roi", "failure"),
    [(None, DailyROI(0, 0, 10, 10), DailyFailure.MODEL_ERROR),
     (broken_model, DailyROI(0, 0, 10, 10), DailyFailure.INFERENCE_ERROR),
     (lambda **_: [object()], DailyROI(0, 0, 10, 10), DailyFailure.INFERENCE_ERROR),
     (lambda **_: pose((0, 0), (500, 0)), DailyROI(0, 0, 10, 10), DailyFailure.INFERENCE_ERROR),
     (None, DailyROI(12, 0, 10, 10, True), DailyFailure.CALIBRATION_ERROR),
     (None, DailyROI(0, 9, 10, 10, True), DailyFailure.CALIBRATION_ERROR)],
)
def test_daily_equipment_errors_never_return_ng_or_ok(model, roi, failure):
    with pytest.raises(DailyInspectionError) as caught:
        DailyInspectionStrategy().inspect(
            np.zeros((10, 10, 3), np.uint8), DailyReferenceSpec(100, 1, 0.16),
            roi, DailyInferenceSpec(0.4, 1980), model,
        )
    assert caught.value.category is failure


@pytest.mark.parametrize("values", [(True, 1, 0.16), (100, -1, 0.16), (100, 1, 0), (100, np.nan, 0.16)])
def test_invalid_reference_spec_rejected(values):
    with pytest.raises(DailyInspectionError) as caught:
        DailyReferenceSpec(*values)
    assert caught.value.category is DailyFailure.CONFIGURATION_ERROR


@pytest.fixture()
def daily_settings(settings):
    target = settings.paths.root / "aikensa/core/config"
    source = REPO_ROOT / "aikensa/core/config"
    for name in ("part_registry.yaml", "parts_specifications.yaml", "models.yaml"):
        shutil.copyfile(source / name, target / name)
    return settings


def test_daily_approval_persists_and_rolls_back_without_changing_other_positions(daily_settings):
    settings = daily_settings
    original = settings.load_daily_configuration()
    original_adjustments = settings.revisions.active_revision_id()
    draft = original.editable_values()
    draft[original.profile_for_widget(22).profile_id] = {"tolerance_mm": 2.0}
    assert settings.load_daily_configuration() == original
    approved = settings.save_daily_configuration(draft, expected_revision=original.revision_id)
    restarted = SettingsService(settings.paths)
    updated = restarted.load_daily_configuration()
    assert updated.revision_id == approved.revision_id
    assert updated.profile_for_widget(22).reference.tolerance_mm == 2.0
    for widget in (21, 23):
        assert updated.profile_for_widget(widget) == original.profile_for_widget(widget)
    assert original.profile_for_widget(22).reference.tolerance_mm == 1.0
    with pytest.raises(FrozenInstanceError):
        updated.profile_for_widget(22).reference.tolerance_mm = 9.0
    assert restarted.revisions.active_revision_id() == original_adjustments
    restarted.rollback_daily_configuration(original.revision_id)
    assert restarted.load_daily_configuration().profiles == original.profiles
    rollback = restarted.rollback_daily_configuration(approved.revision_id, operator="engineer", reason="confirm")
    restored = restarted.load_daily_configuration()
    assert restored.revision_id == rollback.revision_id != approved.revision_id
    assert restored.profiles == updated.profiles
    assert rollback.operator == "engineer" and "confirm" in rollback.note


def test_daily_approved_effective_configuration_survives_shared_yaml_change(daily_settings):
    import yaml

    service = daily_settings
    initial = service.load_daily_configuration()
    approved = service.save_daily_configuration(
        {"dailyTenken_02": {"tolerance_mm": 2}}, expected_revision=initial.revision_id,
    )
    effective = service.load_daily_configuration()
    config_dir = service.paths.root / "aikensa/core/config"
    path = config_dir / "parts_specifications.yaml"
    specs = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs["parts"]["NICHIJOU_TENKEN"]["pitchSpec"] = 102.0
    path.write_text(yaml.safe_dump(specs), encoding="utf-8")
    restarted = SettingsService(service.paths)
    assert restarted.load_daily_configuration() == effective
    reverted = restarted.rollback_daily_configuration(initial.revision_id)
    defaults = restarted.load_daily_configuration()
    assert defaults.revision_id == reverted.revision_id
    assert defaults.profile_for_widget(22).reference.nominal_mm == 102.0
    assert defaults.profile_for_widget(22).configuration_digest != effective.profile_for_widget(22).configuration_digest
    restarted.rollback_daily_configuration(approved.revision_id)
    assert restarted.load_daily_configuration().profiles == effective.profiles

    path = config_dir / "models.yaml"
    models = yaml.safe_load(path.read_text(encoding="utf-8"))
    models["models"]["NICHIJOU_TENKEN_Model"]["path"] = "./different.pt"
    path.write_text(yaml.safe_dump(models), encoding="utf-8")
    active = restarted.load_daily_configuration()
    with pytest.raises(RevisionError, match="AIモデル"):
        restarted.rollback_daily_configuration(approved.revision_id)
    assert restarted.load_daily_configuration() == active


def test_legacy_daily_revision_is_preserved_and_upgraded_on_approval(daily_settings):
    from pathlib import Path

    service = daily_settings
    legacy = service.daily_revisions.stage({
        Path("daily_inspection.yaml"): {
            "schema_version": 1, "positions": {"dailyTenken_02": {"tolerance_mm": 1.7}},
        },
    })
    service.daily_revisions.activate(legacy)
    old = service.load_daily_configuration()
    assert old.profile_for_widget(22).reference.tolerance_mm == 1.7
    overrides = old.editable_values()
    overrides["dailyTenken_03"] = {"tolerance_mm": 1.8}
    approved = service.save_daily_configuration(overrides, expected_revision=legacy)
    new = SettingsService(service.paths).load_daily_configuration()
    assert new.revision_id == approved.revision_id
    assert new.profile_for_widget(22) == old.profile_for_widget(22)
    assert new.profile_for_widget(23).reference.tolerance_mm == 1.8
    assert service.daily_revisions.staged_data(legacy)["daily_inspection.yaml"]["schema_version"] == 1


def test_v2_daily_revision_and_common_inheritance_preserve_position_overrides(daily_settings):
    import json
    from pathlib import Path
    from aikensa.core.settings.daily_config import DAILY_COMMON_SCOPE

    service = daily_settings
    current = service.load_daily_configuration()
    legacy = service.daily_revisions.stage({
        Path("daily_inspection.yaml"): {
            "schema_version": 2,
            "shared": {k: v for k, v in json.loads(current.shared_definition).items() if k != "save_diagnostic_images"},
            "positions": {"dailyTenken_02": {"tolerance_mm": 1.7}},
        },
    })
    service.daily_revisions.activate(legacy)
    old = service.load_daily_configuration()
    overrides = old.editable_values()
    overrides[DAILY_COMMON_SCOPE] = {"tolerance_mm": 2.0, "confidence": 0.6, "save_diagnostic_images": False}
    overrides["dailyTenken_02"]["width"] = 1000
    candidate = service.preview_daily_configuration(old, overrides)
    assert service.load_daily_configuration() == old
    assert [p.reference.tolerance_mm for p in candidate.profiles] == [2.0, 1.7, 2.0]
    assert [p.roi.x for p in candidate.profiles] == [0, 2100, 4200]
    assert [p.roi.width for p in candidate.profiles] == [1890, 1000, 1890]
    assert all(p.inference.confidence == 0.6 and not p.save_diagnostic_images for p in candidate.profiles)
    with pytest.raises(DailyInspectionError):
        service.preview_daily_configuration(old, {DAILY_COMMON_SCOPE: {"x": 25}})
    with pytest.raises(DailyInspectionError):
        service.preview_daily_configuration(old, {"dailyTenken_02": {"confidence": 1.1}})
    approved = service.save_daily_configuration(overrides, expected_revision=legacy)
    assert SettingsService(service.paths).load_daily_configuration().profiles == candidate.profiles
    assert service.daily_revisions.staged_data(approved.revision_id)["daily_inspection.yaml"]["schema_version"] == 3


def test_daily_draft_overlay_marks_same_region_that_reaches_inference(daily_settings):
    from dataclasses import replace
    from aikensa.core.inspection.daily_draft import DailyDraftRequest, run_daily_draft

    settings = daily_settings.load_daily_configuration()
    profile = settings.profile_for_widget(22)
    profile = replace(profile, roi=replace(profile.roi, x=20, y=10, width=80, height=100))
    raw = np.zeros((60, 120, 3), np.uint8)
    raw[20, 30] = (50, 100, 125)

    def model(source, **_):
        assert source.shape == (50, 80, 3)
        assert tuple(source[10, 10]) == (50, 100, 125)
        np.testing.assert_array_equal(source, raw[10:60, 20:100])
        return pose((5, 5), (75, 45))

    result = run_daily_draft(DailyDraftRequest.create(profile, True), raw, model)
    assert result.measurement.endpoints == ((25, 15), (95, 55))
    assert "実効 (20, 10, 80, 50)" in result.message
    assert tuple(result.full_image[35, 20]) == (40, 225, 60)
    assert tuple(raw[35, 20]) == (0, 0, 0)


def test_two_settings_services_cannot_both_approve_the_same_starting_revision(daily_settings):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    paths = daily_settings.paths
    original = daily_settings.load_daily_configuration()
    ready = Barrier(2)

    def approve(tolerance):
        service = SettingsService(paths)
        ready.wait(timeout=5)
        try:
            return service.save_daily_configuration(
                {"dailyTenken_02": {"tolerance_mm": tolerance}}, expected_revision=original.revision_id,
            ).revision_id
        except RevisionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(approve, (2, 3))
    assert (first is None) != (second is None)
    assert daily_settings.load_daily_configuration().revision_id == (first or second)


def test_daily_backup_restores_active_revision_and_history_to_another_machine(daily_settings, tmp_path):
    from aikensa.core.settings.migration import backup_machine_files, restore_machine_files
    from aikensa.core.settings.paths import AppPaths

    original = daily_settings.load_daily_configuration()
    earlier = daily_settings.save_daily_configuration(
        {"dailyTenken_02": {"tolerance_mm": 2.0}}, expected_revision=original.revision_id,
    )
    active = daily_settings.save_daily_configuration(
        {"dailyTenken_02": {"tolerance_mm": 3.0}}, expected_revision=earlier.revision_id,
    )
    backup = backup_machine_files(daily_settings.paths, tmp_path / "backup")
    assert backup.ok
    replacement = AppPaths(root=tmp_path / "replacement")
    shutil.copytree(
        daily_settings.paths.root / "aikensa/core/config",
        replacement.root / "aikensa/core/config", ignore=shutil.ignore_patterns("cam"),
    )
    restore_machine_files(replacement, backup.backup_dir)
    restarted = SettingsService(replacement)
    assert restarted.load_daily_configuration().revision_id == active.revision_id
    assert restarted.load_daily_configuration().profile_for_widget(22).reference.tolerance_mm == 3.0
    restarted.rollback_daily_configuration(earlier.revision_id)
    assert restarted.load_daily_configuration().profile_for_widget(22).reference.tolerance_mm == 2.0


def test_invalid_or_failed_activation_leaves_previous_revision(daily_settings, monkeypatch):
    settings = daily_settings
    previous = settings.load_daily_configuration()
    with pytest.raises(DailyInspectionError):
        settings.save_daily_configuration(
            {"dailyTenken_02": {"tolerance_mm": -1}}, expected_revision=previous.revision_id,
        )
    with pytest.raises(RevisionError):
        settings.save_daily_configuration(
            {"dailyTenken_02": {"tolerance_mm": 2}}, expected_revision="stale",
        )

    def fail_pointer(_revision_id):
        raise OSError("write failure")

    monkeypatch.setattr(settings.daily_revisions, "_mark_active", fail_pointer)
    with pytest.raises(OSError):
        settings.save_daily_configuration(
            {"dailyTenken_02": {"tolerance_mm": 2}}, expected_revision=previous.revision_id,
        )
    assert SettingsService(settings.paths).load_daily_configuration() == previous
    assert len(settings.daily_revisions.list_revisions()) == 1


def test_daily_recording_is_idempotent_and_failed_images_do_not_create_a_result(
    daily_settings, tmp_path, monkeypatch,
):
    from aikensa.core.inspection import daily_recording as recording
    from aikensa.core.inspection.daily import measure_reference

    config = daily_settings.load_daily_configuration()
    profile = config.profile_for_widget(22)

    def cycle():
        return recording.DailyCycle.begin(profile, config.revision_id, "session", "", (0, 0), (0, 0))

    root = tmp_path / "results"
    image = np.zeros((40, 700, 3), np.uint8)
    sample = measure_reference(((0, 0), (627, 2)), profile.reference)
    con = sqlite3.connect(":memory:")
    try:
        con.execute(f"CREATE TABLE inspection_results (id INTEGER PRIMARY KEY, {recording._RESULT_FIELDS})")
        first = cycle()
        assert recording.write_daily_result(con, root, first, sample, image, image)
        assert not recording.write_daily_result(con, root, first, sample, image, image)
        failed = cycle()
        write = recording.cv2.imwrite
        # Write the first image for real so cleanup is exercised when the second fails.
        monkeypatch.setattr(recording.cv2, "imwrite", lambda path, array: write(path, array) if "nama" in path else False)
        with pytest.raises(OSError):
            recording.write_daily_result(con, root, failed, sample, image, image)
        assert con.execute("SELECT count(*) FROM inspection_results").fetchone() == (1,)
        assert con.execute("SELECT cycle_id FROM daily_inspection_metadata").fetchall() == [(first.cycle_id,)]
        assert len(list(root.rglob("*.png"))) == 2
    finally:
        con.close()


def test_current_daily_session_rejects_replaced_attempts_and_old_configurations(daily_settings):
    from dataclasses import replace
    from aikensa.core.inspection.daily_session import DailyPositionState as State
    from aikensa.core.inspection.daily_session import DailyPositionUpdate, DailySession, RecordingState

    settings = daily_settings
    config = settings.load_daily_configuration()
    session = DailySession("current")
    assert "未完了" in session.summary
    completed = {}

    for widget in (21, 22, 23):
        attempt = DailyPositionUpdate(widget, "current", f"cycle-{widget}", State.INSPECTING)
        assert session.apply(attempt)
        completed[widget] = replace(
            attempt, state=State.OK, profile=config.profile_for_widget(widget),
            judgment="OK", local=RecordingState.RECORDED,
        )
        assert session.apply(completed[widget])
        assert ("全位置OK" in session.summary) == (widget == 23)

    retry = replace(completed[22], cycle_id="retry", state=State.INSPECTING, profile=None)
    assert session.apply(retry)
    assert "全位置OK" not in session.summary and "点検中" in session.summary
    assert not session.apply(completed[22])
    assert not session.apply(replace(retry, session_id="old"))
    assert session.apply(replace(retry, state=State.NG, judgment="NG"))
    assert "全位置実施・NGあり" in session.summary

    assert session.apply(replace(retry, cycle_id="retry-2"))
    session.interrupt("camera disconnected")
    assert session.positions[22].state is State.EQUIPMENT_ERROR
    assert not session.apply(replace(completed[22], cycle_id="retry-2"))

    assert session.apply(replace(retry, cycle_id="retry-3"))
    assert session.apply(replace(retry, cycle_id="retry-3", state=State.RECORDING_FAILED, local=RecordingState.RECORDED,
                                 remote=RecordingState.FAILED))
    assert "記録失敗" in session.summary and "全位置OK" not in session.summary

    assert session.apply(replace(retry, cycle_id="retry-4"))
    settings.save_daily_configuration(
        {"dailyTenken_02": {"tolerance_mm": 2}}, expected_revision=config.revision_id,
    )
    updated = settings.load_daily_configuration()
    session.update_configuration({profile.widget_id: profile for profile in updated.profiles})
    assert session.apply(replace(completed[22], cycle_id="retry-4"))
    assert session.positions[22].state is State.REINSPECTION_REQUIRED
    assert session.positions[21].state is session.positions[23].state is State.OK
