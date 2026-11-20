from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aikensa.core.inspection import (
    InspectionContext, InspectionCoordinator, InspectionError,
    InspectionProfile, InspectionResult, P808387UA1AStrategy, load_profiles,
)


def profile(part_id="P808387UA1A", strategy="p808387ua1a"):
    return InspectionProfile(part_id, strategy, part_id, "Test part", 17)


def make_context(partcheck=None):
    image = np.zeros((64, 1200, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(1200, dtype=np.uint16) % 256
    inference = image[:, :, ::-1].copy()
    calls = []

    def keypoint(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(keypoints=SimpleNamespace(xy=np.array([[[10.0, 20.0]]])))]

    def sliced(image, model, **kwargs):
        calls.append((image, model, kwargs))
        return SimpleNamespace(object_prediction_list=[])

    return InspectionContext(image, inference, object(), keypoint, sliced, partcheck=partcheck), calls


def coordinator(*profiles):
    return InspectionCoordinator({p.part_id: p for p in profiles}, {"p808387ua1a": P808387UA1AStrategy()})


def test_matches_original_partcheck_and_inference_contract(monkeypatch):
    from aikensa.core.config.parts.NISSAN.M_JC2D import P808387UA1A as legacy

    monkeypatch.setattr(legacy, "draw_status_text_PIL", lambda image, *a, **k: image)
    original, _ = make_context(legacy.partcheck)
    clip = original.sliced_prediction(original.inference_image, original.clip_model)
    left = original.keypoint_model(source=original.image[:, :512, :], conf=0.6, imgsz=512, verbose=False)
    right = original.keypoint_model(source=original.image[:, -512:, :], conf=0.6, imgsz=512, verbose=False)
    expected = legacy.partcheck(original.image, clip.object_prediction_list, left, right, None)

    context, calls = make_context(legacy.partcheck)
    result = coordinator(profile()).inspect("P808387UA1A", context)
    assert np.array_equal(result.image, expected[0])
    assert (result.measurements, result.judgments, result.id_judgments, result.status, result.ng_reason) == expected[1:]
    assert result.status == "NG" and result.ng_reason == "NUMBER OF CLIP MISMATCH"
    assert calls[0][0] is context.inference_image
    assert calls[0][2] == dict(slice_height=512, slice_width=1980, overlap_height_ratio=0.0,
                               overlap_width_ratio=0.2, postprocess_match_metric="IOS",
                               postprocess_match_threshold=0.2, postprocess_class_agnostic=True,
                               postprocess_type="GREEDYNMM", verbose=0, perform_standard_pred=False)
    for call, expected_crop in zip(calls[1:], (context.image[:, :512, :], context.image[:, -512:, :])):
        assert np.array_equal(call["source"], expected_crop)
        assert (call["conf"], call["imgsz"], call["verbose"]) == (0.6, 512, False)


@pytest.mark.parametrize("offset, expected_status", [(0, "OK"), (20, "NG")])
def test_real_partcheck_measurement_order_matches_original(monkeypatch, offset, expected_status):
    from aikensa.core.config.parts.NISSAN.M_JC2D import P808387UA1A as legacy

    monkeypatch.setattr(legacy, "draw_status_text_PIL", lambda image, *a, **k: image)
    pitches = [15, 123, 122, 81, 81, 122, 123, 15]
    points = [10.0]
    for pitch in pitches:
        points.append(points[-1] + pitch / legacy.pixelMultiplier)
    centers = points[1:-1]
    centers[3] += offset / legacy.pixelMultiplier
    detections = [
        SimpleNamespace(bbox=SimpleNamespace(minx=x - 1, maxx=x + 1, miny=19, maxy=21),
                        category=SimpleNamespace(id=0))
        for x in reversed(centers)
    ]

    def context():
        image = np.zeros((64, 4512, 3), dtype=np.uint8)
        keypoints = iter([points[0], points[-1] - 4000])
        def predict_keypoint(**kwargs):
            return [SimpleNamespace(keypoints=SimpleNamespace(
                xy=np.array([[[next(keypoints), 20.0]]])))]
        return InspectionContext(image, image.copy(), object(), predict_keypoint,
                                 lambda *args, **kwargs: SimpleNamespace(object_prediction_list=detections))

    original = context()
    left = original.keypoint_model(source=original.image[:, :512, :])
    right = original.keypoint_model(source=original.image[:, -512:, :])
    expected = legacy.partcheck(original.image, detections, left, right, None)
    result = coordinator(profile()).inspect("P808387UA1A", context())
    assert (result.measurements, result.judgments, result.id_judgments, result.status, result.ng_reason) == expected[1:]
    assert result.status == expected_status
    if expected_status == "OK":
        assert result.measurements == pytest.approx([15, 123, 122, 81, 81, 122, 123, 15, 682], abs=0.2)
        assert result.judgments == [1] * 9
        assert result.id_judgments == [1] * 7
    else:
        assert result.ng_reason == "CLIP PITCH NG"


def test_synthetic_profile_runs_real_partcheck_and_resolves_persistence(tmp_path, monkeypatch):
    from aikensa.core.config.parts.NISSAN.M_JC2D import P808387UA1A as legacy

    monkeypatch.setattr(legacy, "draw_status_text_PIL", lambda image, *a, **k: image)
    path = tmp_path / "profiles.yaml"
    path.write_text("""profiles:
  - part_id: TEST_NEW_PART_001
    strategy: p808387ua1a
    directory: TEST_NEW_PART_001
    name: Synthetic
    widget_id: 50
""")
    engine = InspectionCoordinator(load_profiles(path), {"p808387ua1a": P808387UA1AStrategy()})
    context, _ = make_context()
    result = engine.inspect("TEST_NEW_PART_001", context)
    resolved, _ = engine.resolve("TEST_NEW_PART_001")
    assert isinstance(result, InspectionResult)
    assert result.status == "NG"
    assert result.ng_reason == "NUMBER OF CLIP MISMATCH"
    assert (resolved.name, resolved.directory) == ("Synthetic", "TEST_NEW_PART_001")


def test_unknown_profile_and_strategy_are_errors():
    engine = coordinator(profile(strategy="missing"))
    with pytest.raises(InspectionError, match="unknown inspection profile"):
        engine.resolve("missing")
    with pytest.raises(InspectionError, match="unknown inspection strategy"):
        engine.resolve("P808387UA1A")


def test_invalid_profile_registration_is_rejected(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text("""profiles:
  - {part_id: part, strategy: p808387ua1a, directory: part, name: Part, widget_id: 50}
  - {part_id: other, strategy: p808387ua1a, directory: other, name: Other, widget_id: 50}
""")
    with pytest.raises(InspectionError, match="duplicate profile widget"):
        load_profiles(path)


@pytest.mark.parametrize("failure", ["model", "image", "keypoints", "partcheck", "result"])
def test_equipment_failures_do_not_yield_product_result(failure):
    context, _ = make_context()
    if failure == "model":
        context.clip_model = None
    elif failure == "image":
        context.image = np.zeros((4, 4, 3), dtype=np.uint8)
    elif failure == "keypoints":
        context.keypoint_model = lambda **kwargs: []
    elif failure == "partcheck":
        def fail(*args):
            raise ValueError("model error")
        context.partcheck = fail
    else:
        context.partcheck = lambda *args: (context.image, [], [], [], "OK", "")
    with pytest.raises(InspectionError):
        coordinator(profile()).inspect("P808387UA1A", context)


def test_finalize_only_once():
    context, _ = make_context(lambda image, *args: (image, [0] * 9, [0] * 9, [], "NG", "test"))
    engine = coordinator(profile())
    result = engine.inspect("P808387UA1A", context)
    seen = []
    engine.finalize(result, seen.append)
    with pytest.raises(InspectionError, match="already finalized"):
        engine.finalize(result, seen.append)
    assert seen == [result]


def test_optional_anomaly_model_passed_to_partcheck():
    anomaly = object()
    seen = []
    def partcheck(image, detections, left, right, model):
        seen.append(model)
        return image, [0] * 9, [0] * 9, [], "NG", "test"
    context, _ = make_context(partcheck)
    context.anomaly_model = anomaly
    coordinator(profile()).inspect("P808387UA1A", context)
    assert seen == [anomaly]


def test_production_profile_is_resolvable():
    path = Path(__file__).resolve().parents[1] / "aikensa/core/config/inspection_profiles.yaml"
    engine = InspectionCoordinator(load_profiles(path), {"p808387ua1a": P808387UA1AStrategy()})
    assert engine.resolve("P808387UA1A")[0].directory == "808387UA1A"
