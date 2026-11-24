"""Read-only footprint and full-coverage rendering in native merge coordinates."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from aikensa.core.geometry.transforms import scale_matrix, translation_matrix


@dataclass(frozen=True)
class CameraFootprint:
    corners: tuple[tuple[float, float], ...]
    bounds: tuple[float, float, float, float]
    overflow: tuple[float, float, float, float]  # left, right, top, bottom


@dataclass(frozen=True)
class MergeCoverage:
    footprints: tuple[CameraFootprint, CameraFootprint]
    production_size: tuple[int, int]
    origin: tuple[int, int]
    union_size: tuple[int, int]
    overlap_px2: float


def _footprint(size: tuple[int, int], matrix: np.ndarray, canvas: tuple[int, int]) -> CameraFootprint:
    width, height = size
    matrix = np.asarray(matrix, dtype=np.float64)
    if min(width, height) < 2 or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("画像サイズまたはホモグラフィが不正です。")
    scale = np.max(np.abs(matrix))
    if scale == 0 or np.linalg.matrix_rank(matrix / scale) != 3:
        raise ValueError("ホモグラフィが特異です。")
    matrix = matrix / scale
    source = np.asarray(((0, 0, 1), (width - 1, 0, 1), (width - 1, height - 1, 1), (0, height - 1, 1)))
    homogeneous = source @ matrix.T
    denominator = homogeneous[:, 2]
    if (np.min(np.abs(denominator)) < 1e-10
            or np.min(denominator) < 0 < np.max(denominator)):
        raise ValueError("カメラ範囲が射影の無限遠を横切ります。校正を確認してください。")
    points = homogeneous[:, :2] / denominator[:, None]
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 100_000:
        raise ValueError("射影範囲が大きすぎます。校正を確認してください。")
    left, top = map(float, points.min(axis=0))
    right, bottom = map(float, points.max(axis=0))
    overflow = (
        max(0.0, -left), max(0.0, right - (canvas[0] - 1)),
        max(0.0, -top), max(0.0, bottom - (canvas[1] - 1)),
    )
    return CameraFootprint(tuple(map(tuple, points.tolist())), (left, top, right, bottom), overflow)


def analyze_merge_coverage(
    source_sizes: tuple[tuple[int, int], tuple[int, int]],
    matrices: tuple[np.ndarray, np.ndarray],
    canvas: tuple[int, int],
) -> MergeCoverage:
    if min(canvas) < 2:
        raise ValueError("マージ範囲のサイズが不正です。")
    first, second = (_footprint(size, matrix, canvas) for size, matrix in zip(source_sizes, matrices))
    l1, t1, r1, b1 = first.bounds
    l2, t2, r2, b2 = second.bounds
    x0, y0 = math.floor(min(0, l1, l2)), math.floor(min(0, t1, t2))
    x1, y1 = math.ceil(max(canvas[0] - 1, r1, r2)), math.ceil(max(canvas[1] - 1, b1, b2))
    overlap, _ = cv2.intersectConvexConvex(
        np.asarray(first.corners, dtype=np.float32), np.asarray(second.corners, dtype=np.float32),
    )
    return MergeCoverage((first, second), canvas, (x0, y0), (x1 - x0 + 1, y1 - y0 + 1), overlap)


def render_merge_coverage(
    frames: tuple[np.ndarray, np.ndarray],
    matrices: tuple[np.ndarray, np.ndarray],
    coverage: MergeCoverage,
    crop: tuple[int, ...] | None = None,
) -> np.ndarray:
    width, height = coverage.union_size
    # Bound allocation even when a valid but distant projection is submitted.
    scale = min(1.0, 1800 / (width + 4), 1000 / (height + 4))
    margin = 2
    size = (math.ceil((width - 1) * scale) + 2 * margin + 1,
            math.ceil((height - 1) * scale) + 2 * margin + 1)
    x0, y0 = coverage.origin
    transform = translation_matrix(margin, margin) @ scale_matrix(scale, scale) @ translation_matrix(-x0, -y0)
    display = np.zeros((size[1], size[0], 3), np.uint8)
    for frame, matrix in zip(frames, matrices):
        mapping = transform @ matrix
        warped = cv2.warpPerspective(frame, mapping, size)
        mask = cv2.warpPerspective(
            np.full(frame.shape[:2], 255, np.uint8), mapping, size, flags=cv2.INTER_NEAREST,
        )
        cv2.copyTo(warped, mask, display)

    def outline(points, color):
        vertices = np.rint(cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float64).reshape(1, -1, 2), transform,
        )[0]).astype(np.int32)
        cv2.polylines(display, [vertices], True, color, 1, cv2.LINE_8)

    for footprint, color in zip(coverage.footprints, ((255, 190, 0), (120, 80, 255))):
        outline(footprint.corners, color)
    w, h = coverage.production_size
    outline(((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)), (0, 225, 255))
    if crop is not None:
        left, right = min(crop[::2]), max(crop[::2]) - 1
        top, bottom = min(crop[1::2]), max(crop[1::2]) - 1
        outline(((left, top), (right, top), (right, bottom), (left, bottom)), (80, 255, 80))
    return display
