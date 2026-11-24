"""Prove fixed-canvas clipping and read-only full-coverage rendering."""

import numpy as np
import pytest

from aikensa.core.geometry.coverage import analyze_merge_coverage, render_merge_coverage
from aikensa.core.geometry.transforms import translation_matrix


def overflow_fixture():
    frames = (np.zeros((60, 100, 3), np.uint8), np.zeros((60, 100, 3), np.uint8))
    frames[0][8:14, 4:10] = (255, 255, 255)
    frames[1][45:51, 90:96] = (255, 255, 255)
    matrices = (translation_matrix(-20, -5), translation_matrix(100, 15))
    return frames, matrices, (180, 65)


def test_full_coverage_retains_native_markers_outside_fixed_canvas():
    from aikensa.core.scripts.img_processing.cameracalibrate import warpTwoImages_template

    frames, matrices, canvas = overflow_fixture()
    original = [h.copy() for h in matrices]
    fixed = np.zeros((canvas[1], canvas[0], 3), np.uint8)
    for frame, matrix in zip(frames, matrices):
        fixed = warpTwoImages_template(fixed, frame, matrix)
    assert not fixed.any()
    report = analyze_merge_coverage(((100, 60), (100, 60)), matrices, canvas)
    assert report.origin == (-20, -5)
    assert report.union_size == (220, 80)
    assert report.footprints[0].overflow == (20, 0, 5, 0)
    assert report.footprints[1].overflow == (0, 20, 0, 10)
    assert report.overlap_px2 == 0
    image = render_merge_coverage(frames, matrices, report)
    assert tuple(image[12, 8]) == (255, 255, 255)
    assert tuple(image[70, 214]) == (255, 255, 255)
    for matrix, wanted in zip(matrices, original):
        np.testing.assert_array_equal(matrix, wanted)


@pytest.mark.parametrize("matrix", [
    np.zeros((3, 3)), np.diag([1., 1., float("nan")]),
    np.array([[1., 0., 0.], [0., 1., 0.], [-.03, 0., 1.]]),
    translation_matrix(200_000, 0),
])
def test_coverage_rejects_unusable_projective_geometry(matrix):
    with pytest.raises(ValueError):
        analyze_merge_coverage(((100, 60), (100, 60)), (matrix, np.eye(3)), (180, 65))


def test_distant_projection_is_bounded_before_allocation():
    frames, _, canvas = overflow_fixture()
    matrices = translation_matrix(-9000, -2500), translation_matrix(9000, 2500)
    report = analyze_merge_coverage(((100, 60), (100, 60)), matrices, canvas)
    image = render_merge_coverage(frames, matrices, report)
    assert image.shape[1] <= 1805 and image.shape[0] <= 1005


def test_overlap_is_geometric_even_when_both_camera_images_are_black():
    sizes = ((101, 61), (101, 61))
    matrices = (np.eye(3), translation_matrix(75, 10))
    coverage = analyze_merge_coverage(sizes, matrices, (180, 75))
    assert coverage.overlap_px2 == pytest.approx(25 * 50)
