"""Daily measurement behavior with synthetic points and no device dependencies."""

from types import SimpleNamespace
from dataclasses import FrozenInstanceError
import shutil

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
    restarted.rollback_daily_configuration(approved.revision_id)
    assert restarted.load_daily_configuration() == updated


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
