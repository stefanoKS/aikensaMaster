"""Pure projective-geometry helpers.

Coordinate conventions
=======================

* All matrices are ``3x3`` and operate on homogeneous *pixel* coordinates with
  ``(0, 0)`` at the top-left of the image, ``x`` to the right and ``y`` down -
  the OpenCV convention.
* A "full-resolution" homography maps a camera image to the merged template
  canvas at native camera resolution.
* Fine adjustments are expressed in *destination* (template) pixels and applied
  on the left of the base transform::

      H_effective = T_adjustment @ H_base

  Rotating the base matrix's ``H[0,2]`` / ``H[1,2]`` entries would only be a
  translation when the transform is affine; composing a translation matrix is
  correct for any projective transform.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

Matrix = np.ndarray

_DEGENERATE_DET_EPS = 1e-12


def is_valid_homography(matrix: Optional[Matrix], eps: float = _DEGENERATE_DET_EPS) -> bool:
    """True when *matrix* can safely be passed to ``cv2.warpPerspective``."""

    if matrix is None:
        return False
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        return False
    if not np.all(np.isfinite(array)):
        return False
    if abs(float(np.linalg.det(array))) < eps:
        return False
    return abs(float(array[2, 2])) >= 1e-12


def scale_matrix(sx: float, sy: float) -> Matrix:
    """Anisotropic scaling about the origin."""

    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def homography_for_scale(
    full_matrix: Matrix,
    *,
    source_full_size: Tuple[int, int],
    destination_full_size: Tuple[int, int],
    source_size: Tuple[int, int],
    destination_size: Tuple[int, int],
) -> Matrix:
    """Derive a down-scaled homography from the authoritative full one.

    ``H_preview = S_destination @ H_full @ inverse(S_source)``

    Scale factors are computed from the *actual* integer image sizes rather
    than assuming an exact factor of 5, because the existing pipeline uses
    integer division (``size // 5``) and therefore slightly non-uniform scales.
    """

    if not is_valid_homography(full_matrix):
        raise ValueError("full-resolution homography is not usable")

    if min(*source_full_size, *destination_full_size, *source_size, *destination_size) <= 0:
        raise ValueError("image sizes must be positive")

    sx_src = source_size[0] / float(source_full_size[0])
    sy_src = source_size[1] / float(source_full_size[1])
    sx_dst = destination_size[0] / float(destination_full_size[0])
    sy_dst = destination_size[1] / float(destination_full_size[1])

    s_src_inv = np.linalg.inv(scale_matrix(sx_src, sy_src))
    s_dst = scale_matrix(sx_dst, sy_dst)
    return s_dst @ np.asarray(full_matrix, dtype=np.float64) @ s_src_inv


def translation_matrix(dx_px: float, dy_px: float) -> Matrix:
    """Pure destination-space translation in pixels."""

    return np.array(
        [[1.0, 0.0, float(dx_px)], [0.0, 1.0, float(dy_px)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_about_pivot(rotation_deg: float, pivot: Sequence[float]) -> Matrix:
    """Rotation in destination space about an explicit pivot (pixels)."""

    px, py = float(pivot[0]), float(pivot[1])
    theta = math.radians(float(rotation_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    translate_to_origin = translation_matrix(-px, -py)
    rotate = np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    translate_back = translation_matrix(px, py)
    return translate_back @ rotate @ translate_to_origin


def compose_into_base(base: Matrix, transforms: Iterable[Matrix]) -> Matrix:
    """Apply destination-space transforms to *base* (``T_n ... T_1 @ base``)."""

    result = np.asarray(base, dtype=np.float64)
    for transform in transforms:
        result = np.asarray(transform, dtype=np.float64) @ result
    return result


def warp_with_mask(
    source: np.ndarray, matrix: Matrix, canvas_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp *source* and produce a matching validity mask.

    The mask is built in *source image* coordinates and warped exactly once,
    which is what the previous implementation got wrong: it warped the already
    warped image's mask a second time, eroding the valid region.

    ``canvas_size`` is ``(width, height)`` as expected by OpenCV.
    """

    warped = cv2.warpPerspective(source, matrix, canvas_size)
    source_mask = np.ones(source.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        source_mask, matrix, canvas_size, flags=cv2.INTER_NEAREST, borderValue=0
    )
    return warped, warped_mask


def merge_warped(
    canvas: np.ndarray, warped: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Composite ``warped`` into ``canvas`` where ``mask`` is set."""

    result = canvas.copy()
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if warped.ndim == 2:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    if result.shape[:2] != warped.shape[:2]:
        raise ValueError(
            f"canvas {result.shape[:2]} and warped {warped.shape[:2]} sizes differ"
        )
    bool_mask = mask.astype(bool)
    if bool_mask.ndim == 2:
        bool_mask = np.repeat(bool_mask[:, :, None], result.shape[2], axis=2)
    np.copyto(result, warped, where=bool_mask)
    return result
