"""Validation rules for settings.

Every rule returns :class:`ValidationIssue` objects with Japanese messages so
the operator-facing UI can display them verbatim.  Nothing here writes files;
callers decide whether a problem blocks activation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from aikensa.core.settings.schema import (
    CameraHardwareSettings,
    CameraSettings,
    CropRect,
    GlobalAlignment,
    ValidationIssue,
    Severity,
    WidgetAlignment,
)

# Hardware envelope.  These are *guard rails*, not calibration tolerances:
# they stop typographical errors (e.g. width 30720) from being written to a
# camera.  They deliberately do not encode any quality judgement.
MIN_WIDTH_PX = 320
MAX_WIDTH_PX = 5472
MIN_HEIGHT_PX = 240
MAX_HEIGHT_PX = 3648
MIN_FPS = 1.0
MAX_FPS = 120.0
MIN_EXPOSURE_US = 10.0
MAX_EXPOSURE_US = 5_000_000.0
MIN_GAIN_DB = 0.0
MAX_GAIN_DB = 48.0
MIN_WB_TEMPERATURE_K = 1000
MAX_WB_TEMPERATURE_K = 12000
MIN_WB_RATIO = 0.05
MAX_WB_RATIO = 20.0

HOMOGRAPHY_DEGENERATE_DET_EPS = 1e-12


def _error(field: str, message_ja: str) -> ValidationIssue:
    return ValidationIssue(Severity.ERROR, field, message_ja)


def _warning(field: str, message_ja: str) -> ValidationIssue:
    return ValidationIssue(Severity.WARNING, field, message_ja)


# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------
def validate_camera_settings(camera: CameraSettings) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    seen: Dict[str, int] = {}
    for index in sorted(camera.assignments):
        assignment = camera.assignments[index]
        serial = assignment.normalized_serial()
        if serial is None:
            issues.append(
                _error(
                    f"camera{index}.serial",
                    f"{assignment.label_ja} のシリアル番号が未設定です。",
                )
            )
            continue
        if assignment.legacy_index:
            issues.append(
                _warning(
                    f"camera{index}.serial",
                    f"{assignment.label_ja} は旧形式のカメラ番号で設定されています。"
                    " シリアル番号を設定してください。",
                )
            )
        if serial in seen:
            issues.append(
                _error(
                    f"camera{index}.serial",
                    f"シリアル番号 {serial} がカメラ{seen[serial] + 1}と重複しています。",
                )
            )
        else:
            seen[serial] = index

    for index in sorted(camera.hardware):
        issues.extend(validate_camera_hardware(index, camera.hardware[index]))

    return issues


def validate_camera_hardware(
    index: int, hardware: CameraHardwareSettings
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    prefix = f"camera{index}"

    if not (MIN_WIDTH_PX <= hardware.width_px <= MAX_WIDTH_PX):
        issues.append(
            _error(
                f"{prefix}.width_px",
                f"解像度の幅は {MIN_WIDTH_PX}〜{MAX_WIDTH_PX} px の範囲で指定してください。",
            )
        )
    if not (MIN_HEIGHT_PX <= hardware.height_px <= MAX_HEIGHT_PX):
        issues.append(
            _error(
                f"{prefix}.height_px",
                f"解像度の高さは {MIN_HEIGHT_PX}〜{MAX_HEIGHT_PX} px の範囲で指定してください。",
            )
        )
    if hardware.width_px % 4 != 0 or hardware.height_px % 4 != 0:
        issues.append(
            _warning(
                f"{prefix}.resolution",
                "IC4 カメラは通常 4 の倍数の解像度を要求します。",
            )
        )
    if not (MIN_FPS <= hardware.fps <= MAX_FPS):
        issues.append(
            _error(f"{prefix}.fps", f"FPS は {MIN_FPS}〜{MAX_FPS} の範囲で指定してください。")
        )
    if not (MIN_EXPOSURE_US <= hardware.exposure_us <= MAX_EXPOSURE_US):
        issues.append(
            _error(
                f"{prefix}.exposure_us",
                f"露光時間は {MIN_EXPOSURE_US:.0f}〜{MAX_EXPOSURE_US:.0f} μs の範囲で指定してください。",
            )
        )
    if not (MIN_GAIN_DB <= hardware.gain_db <= MAX_GAIN_DB):
        issues.append(
            _error(
                f"{prefix}.gain_db",
                f"ゲインは {MIN_GAIN_DB:.0f}〜{MAX_GAIN_DB:.0f} dB の範囲で指定してください。",
            )
        )
    if not (MIN_WB_TEMPERATURE_K <= hardware.wb_temperature_k <= MAX_WB_TEMPERATURE_K):
        issues.append(
            _error(
                f"{prefix}.wb_temperature_k",
                f"ホワイトバランス色温度は {MIN_WB_TEMPERATURE_K}〜{MAX_WB_TEMPERATURE_K} K で指定してください。",
            )
        )
    for channel, ratio in (
        ("r", hardware.wb_ratio_r),
        ("g", hardware.wb_ratio_g),
        ("b", hardware.wb_ratio_b),
    ):
        if ratio is None:
            continue
        if not (MIN_WB_RATIO <= ratio <= MAX_WB_RATIO):
            issues.append(
                _error(
                    f"{prefix}.wb_ratio_{channel}",
                    f"ホワイトバランス比({channel.upper()}) は {MIN_WB_RATIO}〜{MAX_WB_RATIO} で指定してください。",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Calibration matrices
# ---------------------------------------------------------------------------
def validate_camera_matrix(matrix: Optional[np.ndarray], field: str) -> List[ValidationIssue]:
    if matrix is None:
        return [_error(field, "内部パラメータが未設定(null)です。")]
    if matrix.shape != (3, 3):
        return [_error(field, f"内部パラメータは 3x3 である必要があります (実際: {matrix.shape})。")]
    if not np.all(np.isfinite(matrix)):
        return [_error(field, "内部パラメータに NaN / 無限大が含まれています。")]
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        return [_error(field, "焦点距離 (fx, fy) は正の値である必要があります。")]
    if abs(float(matrix[2, 2]) - 1.0) > 1e-6:
        return [_error(field, "内部パラメータの (3,3) 成分は 1.0 である必要があります。")]
    return []


def validate_distortion_coefficients(
    coefficients: Optional[np.ndarray], field: str
) -> List[ValidationIssue]:
    if coefficients is None:
        return [_error(field, "歪み係数が未設定(null)です。")]
    flat = np.asarray(coefficients).reshape(-1)
    if flat.size < 4:
        return [_error(field, f"歪み係数は 4 個以上必要です (実際: {flat.size})。")]
    if not np.all(np.isfinite(flat)):
        return [_error(field, "歪み係数に NaN / 無限大が含まれています。")]
    return []


def validate_homography(matrix: Optional[np.ndarray], field: str) -> List[ValidationIssue]:
    """Validate a 3x3 homography used with ``cv2.warpPerspective``."""

    if matrix is None:
        return [_error(field, "ホモグラフィが未計算(null)です。")]
    if matrix.shape != (3, 3):
        return [_error(field, f"ホモグラフィは 3x3 である必要があります (実際: {matrix.shape})。")]
    if not np.all(np.isfinite(matrix)):
        return [_error(field, "ホモグラフィに NaN / 無限大が含まれています。")]
    determinant = abs(float(np.linalg.det(matrix)))
    if determinant < HOMOGRAPHY_DEGENERATE_DET_EPS:
        return [_error(field, "ホモグラフィが退化しています (行列式がゼロ)。")]
    if abs(float(matrix[2, 2])) < 1e-12:
        return [_error(field, "ホモグラフィの (3,3) 成分がゼロのため変換できません。")]
    return []


def validate_correspondence_points(
    source: Sequence[Sequence[float]],
    destination: Sequence[Sequence[float]],
    field: str,
    min_count: int = 4,
    min_area_px: float = 1.0,
) -> List[ValidationIssue]:
    """Reject degenerate correspondences before calling ``cv2.findHomography``."""

    src = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(destination, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape:
        return [_error(field, "対応点の数が一致しません。")]
    if src.shape[0] < min_count:
        return [_error(field, f"対応点は最低 {min_count} 点必要です (実際: {src.shape[0]})。")]
    for name, points in (("src", src), ("dst", dst)):
        if not np.all(np.isfinite(points)):
            return [_error(field, f"{name} の対応点に NaN / 無限大が含まれています。")]
        if _polygon_area(points) < min_area_px:
            return [_error(field, f"{name} の対応点が一直線上にあり、変換を計算できません。")]
    return []


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def validate_global_alignment(alignment: GlobalAlignment) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if abs(alignment.cam2_x_px) > alignment.max_abs_shift_px:
        issues.append(
            _error(
                "global_alignment.cam2.x_px",
                f"X 方向の補正量が許容範囲(±{alignment.max_abs_shift_px:.0f} px)を超えています。",
            )
        )
    if abs(alignment.cam2_y_px) > alignment.max_abs_shift_px:
        issues.append(
            _error(
                "global_alignment.cam2.y_px",
                f"Y 方向の補正量が許容範囲(±{alignment.max_abs_shift_px:.0f} px)を超えています。",
            )
        )
    if abs(alignment.cam2_rotation_deg) > alignment.max_abs_rotation_deg:
        issues.append(
            _error(
                "global_alignment.cam2.rotation_deg",
                f"回転補正量が許容範囲(±{alignment.max_abs_rotation_deg:.2f}°)を超えています。",
            )
        )
    return issues


def validate_widget_alignment(alignment: WidgetAlignment) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    field = f"widget_adjustments.{alignment.widget_id}"
    if abs(alignment.cam2_offset_x_px) > alignment.max_abs_offset_px:
        issues.append(
            _error(
                f"{field}.cam2_offset_x_px",
                f"ウィジェット {alignment.widget_id} のX補正が許容範囲を超えています。",
            )
        )
    if abs(alignment.cam2_offset_y_px) > alignment.max_abs_offset_px:
        issues.append(
            _error(
                f"{field}.cam2_offset_y_px",
                f"ウィジェット {alignment.widget_id} のY補正が許容範囲を超えています。",
            )
        )
    if abs(alignment.cam2_rotation_deg) > 2.0:
        issues.append(
            _error(
                f"{field}.cam2_rotation_deg",
                f"ウィジェット {alignment.widget_id} の回転補正が許容範囲(±2°)を超えています。",
            )
        )
    for name, crop in (("crop", alignment.crop), ("katabu_crop", alignment.katabu_crop)):
        if crop is not None and not crop.is_valid():
            issues.append(_error(f"{field}.{name}", "切り出し範囲の幅・高さは正の値が必要です。"))
    return issues


# ---------------------------------------------------------------------------
# Legacy crop compatibility
# ---------------------------------------------------------------------------
def crop_from_legacy_points(points: Iterable[float]) -> Optional[CropRect]:
    """Convert the legacy eight-number crop representation into a rectangle.

    The historical format is ``[x1, y1, x2, y2, x3, y3, x4, y4]`` describing a
    quadrilateral.  The inspection code has always used only its bounding
    rectangle, so the adapter keeps exactly that behaviour.
    """

    values = [float(v) for v in points]
    if len(values) != 8:
        return None
    xs = values[0::2]
    ys = values[1::2]
    x0, x1 = int(round(min(xs))), int(round(max(xs)))
    y0, y1 = int(round(min(ys))), int(round(max(ys)))
    return CropRect(x_px=x0, y_px=y0, width_px=x1 - x0, height_px=y1 - y0)


def legacy_points_from_crop(crop: CropRect) -> List[int]:
    """Inverse of :func:`crop_from_legacy_points` (axis-aligned rectangle)."""

    x0, y0 = crop.x_px, crop.y_px
    x1, y1 = crop.x_px + crop.width_px, crop.y_px + crop.height_px
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def clamp_crop(crop: CropRect, width_px: int, height_px: int) -> Tuple[CropRect, List[ValidationIssue]]:
    clamped = crop.clamp_to(width_px, height_px)
    issues: List[ValidationIssue] = []
    if clamped != crop:
        issues.append(
            _warning(
                "crop",
                f"切り出し範囲を画像サイズ({width_px}x{height_px})に収まるよう補正しました。",
            )
        )
    return clamped, issues
