"""Shared image-geometry implementation used by calibration and inspection."""

from aikensa.core.geometry.pipeline import GeometryPipeline, UncalibratedError
from aikensa.core.geometry.transforms import (
    compose_into_base,
    homography_for_scale,
    is_valid_homography,
    merge_warped,
    rotation_about_pivot,
    scale_matrix,
    translation_matrix,
    warp_with_mask,
)

__all__ = [
    "GeometryPipeline",
    "UncalibratedError",
    "compose_into_base",
    "homography_for_scale",
    "is_valid_homography",
    "merge_warped",
    "rotation_about_pivot",
    "scale_matrix",
    "translation_matrix",
    "warp_with_mask",
]
