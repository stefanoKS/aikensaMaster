"""Group C: geometry correctness."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aikensa.core.geometry.pipeline import GeometryPipeline, UncalibratedError
from aikensa.core.geometry.transforms import (
    compose_into_base,
    homography_for_scale,
    is_valid_homography,
    merge_warped,
    rotation_about_pivot,
    translation_matrix,
    warp_with_mask,
)
from aikensa.core.settings.schema import (
    CalibrationState,
    CropRect,
    GlobalAlignment,
    HomographyCalibration,
    IntrinsicCalibration,
    WidgetAlignment,
)
from aikensa.core.settings.validator import crop_from_legacy_points, legacy_points_from_crop


# ---------------------------------------------------------------------------
# Scaled preview consistency
# ---------------------------------------------------------------------------
def test_preview_homography_matches_full_resolution_geometry(valid_homography):
    full_source = (3072, 2048)
    full_destination = (5672, 1342)
    source = (full_source[0] // 5, full_source[1] // 5)
    destination = (full_destination[0] // 5, full_destination[1] // 5)

    preview = homography_for_scale(
        valid_homography,
        source_full_size=full_source,
        destination_full_size=full_destination,
        source_size=source,
        destination_size=destination,
    )

    for point in (np.array([0.0, 0.0]), np.array([1000.0, 500.0]), np.array([3000.0, 2000.0])):
        full_out = cv2.perspectiveTransform(point.reshape(1, 1, 2), valid_homography).reshape(2)
        scaled_in = np.array([point[0] * source[0] / full_source[0], point[1] * source[1] / full_source[1]])
        preview_out = cv2.perspectiveTransform(scaled_in.reshape(1, 1, 2), preview).reshape(2)
        expected = np.array(
            [
                full_out[0] * destination[0] / full_destination[0],
                full_out[1] * destination[1] / full_destination[1],
            ]
        )
        np.testing.assert_allclose(preview_out, expected, atol=1e-6)


def test_preview_homography_uses_actual_rounded_sizes(valid_homography):
    """Integer division makes the scale slightly non-uniform; it must be honoured."""

    preview = homography_for_scale(
        valid_homography,
        source_full_size=(3072, 2048),
        destination_full_size=(5672, 1342),
        source_size=(614, 409),  # 3072//5, 2048//5
        destination_size=(1134, 268),  # 5672//5, 1342//5
    )
    expected = np.diag([1134 / 5672, 268 / 1342, 1.0]) @ valid_homography @ np.diag(
        [1 / (614 / 3072), 1 / (409 / 2048), 1.0]
    )
    np.testing.assert_allclose(preview, expected, atol=1e-9)


def test_preview_homography_rejects_null(identity_homography):
    with pytest.raises(ValueError):
        homography_for_scale(
            None,
            source_full_size=(10, 10),
            destination_full_size=(10, 10),
            source_size=(2, 2),
            destination_size=(2, 2),
        )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def test_translation_composition_is_exact_even_for_projective_transform():
    # A homography with a non-zero perspective row: editing H[0,2] would not be
    # a pure image-space translation.
    base = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [1e-3, 2e-3, 1.0]])

    composed = compose_into_base(base, [translation_matrix(37.0, -11.0)])
    naive = base.copy()
    naive[0, 2] += 37.0
    naive[1, 2] -= 11.0

    point = np.array([500.0, 300.0, 1.0])
    before = base @ point
    before /= before[2]
    after = composed @ point
    after /= after[2]

    assert after[0] == pytest.approx(before[0] + 37.0, abs=1e-9)
    assert after[1] == pytest.approx(before[1] - 11.0, abs=1e-9)
    # The naive approach is measurably different for this matrix.
    naive_point = naive @ point
    naive_point /= naive_point[2]
    assert abs(naive_point[0] - after[0]) > 1.0


def test_rotation_about_pivot_keeps_pivot_fixed():
    pivot = (700.0, 400.0)
    matrix = rotation_about_pivot(0.75, pivot)

    result = matrix @ np.array([pivot[0], pivot[1], 1.0])
    result /= result[2]

    assert result[0] == pytest.approx(pivot[0], abs=1e-9)
    assert result[1] == pytest.approx(pivot[1], abs=1e-9)


def test_rotation_moves_point_by_expected_angle():
    pivot = (0.0, 0.0)
    matrix = rotation_about_pivot(90.0, pivot)
    result = matrix @ np.array([10.0, 0.0, 1.0])
    assert result[0] == pytest.approx(0.0, abs=1e-9)
    assert result[1] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Mask / merge
# ---------------------------------------------------------------------------
def test_identity_merge_covers_whole_canvas():
    source = np.full((40, 60, 3), 200, dtype=np.uint8)
    canvas = np.zeros((40, 60, 3), dtype=np.uint8)

    warped, mask = warp_with_mask(source, np.eye(3), (60, 40))
    merged = merge_warped(canvas, warped, mask)

    assert mask.min() == 255
    assert (merged == 200).all()


def test_mask_is_created_in_source_coordinates():
    """A shifted warp must leave the uncovered border at the canvas value.

    The previous implementation warped the already-warped mask a second time,
    which eroded the composited region by the border of the shift.
    """

    source = np.full((40, 60, 3), 255, dtype=np.uint8)
    canvas = np.zeros((40, 60, 3), dtype=np.uint8)

    warped, mask = warp_with_mask(source, translation_matrix(10.0, 0.0), (60, 40))
    merged = merge_warped(canvas, warped, mask)

    assert (merged[:, :10] == 0).all()          # uncovered strip stays black
    assert (merged[:, 10:] == 255).all()        # covered area comes from the source


def test_merge_rejects_size_mismatch():
    with pytest.raises(ValueError):
        merge_warped(
            np.zeros((10, 10, 3), np.uint8), np.zeros((20, 20, 3), np.uint8), np.ones((20, 20), np.uint8)
        )


# ---------------------------------------------------------------------------
# Pipeline: null awareness, per-widget isolation, crop
# ---------------------------------------------------------------------------
def _pipeline(widget_alignments=None, global_alignment=None) -> GeometryPipeline:
    intrinsics = {
        i: IntrinsicCalibration(
            camera_index=i,
            camera_matrix=np.array([[1000.0, 0, 50.0], [0, 1000.0, 50.0], [0, 0, 1]]),
            distortion_coefficients=np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            width_px=100,
            height_px=100,
            state=CalibrationState.READY,
        )
        for i in (0, 1)
    }
    homographies = {
        i: HomographyCalibration(
            camera_index=i,
            matrix=np.eye(3),
            scaled_matrix=np.eye(3),
            width_px=100,
            height_px=100,
            state=CalibrationState.READY,
        )
        for i in (0, 1)
    }
    return GeometryPipeline(
        canvas_size=(100, 100),
        intrinsics=intrinsics,
        homographies=homographies,
        global_alignment=global_alignment or GlobalAlignment(),
        widget_alignments=widget_alignments or {},
    )


def test_pipeline_raises_for_uncalibrated_camera():
    pipeline = GeometryPipeline(
        canvas_size=(100, 100),
        intrinsics={0: IntrinsicCalibration(camera_index=0)},
        homographies={0: HomographyCalibration(camera_index=0)},
    )
    with pytest.raises(UncalibratedError):
        pipeline.undistort(np.zeros((100, 100, 3), np.uint8), 0)
    with pytest.raises(UncalibratedError):
        pipeline.effective_homography(0)


def test_widget_offset_is_isolated_per_widget():
    pipeline = _pipeline(
        widget_alignments={
            17: WidgetAlignment(widget_id=17, cam2_offset_x_px=12.0),
            36: WidgetAlignment(widget_id=36),
        }
    )

    widget17 = pipeline.effective_homography(1, widget_id=17)
    widget36 = pipeline.effective_homography(1, widget_id=36)

    assert widget17[0, 2] == pytest.approx(12.0)
    assert widget36[0, 2] == pytest.approx(0.0)


def test_global_alignment_applies_only_to_camera_two():
    pipeline = _pipeline(global_alignment=GlobalAlignment(cam2_x_px=7.0, cam2_y_px=-3.0))

    cam1 = pipeline.effective_homography(0)
    cam2 = pipeline.effective_homography(1)

    np.testing.assert_allclose(cam1, np.eye(3))
    assert cam2[0, 2] == pytest.approx(7.0)
    assert cam2[1, 2] == pytest.approx(-3.0)


def test_global_and_widget_translations_add_up():
    pipeline = _pipeline(
        widget_alignments={43: WidgetAlignment(widget_id=43, cam2_offset_x_px=2.0, cam2_offset_y_px=5.0)},
        global_alignment=GlobalAlignment(cam2_x_px=10.0),
    )

    matrix = pipeline.effective_homography(1, widget_id=43)

    assert matrix[0, 2] == pytest.approx(12.0)
    assert matrix[1, 2] == pytest.approx(5.0)


def test_crop_is_clamped_to_image_bounds():
    image = np.zeros((100, 120, 3), np.uint8)
    cropped = GeometryPipeline.crop(image, CropRect(x_px=100, y_px=90, width_px=200, height_px=200))
    assert cropped.shape == (10, 20, 3)


def test_eight_number_crop_is_interpreted_as_bounding_box():
    crop = crop_from_legacy_points([10, 20, 210, 20, 210, 120, 10, 120])
    assert crop.as_tuple() == (10, 20, 200, 100)
    assert legacy_points_from_crop(crop) == [10, 20, 210, 20, 210, 120, 10, 120]


def test_invalid_homography_detection():
    assert is_valid_homography(None) is False
    assert is_valid_homography(np.zeros((3, 3))) is False
    assert is_valid_homography(np.eye(3)) is True
