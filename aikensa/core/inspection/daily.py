"""Daily reference measurement without camera, audio, Qt or storage ownership."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Mapping

import numpy as np


class DailyFailure(str, Enum):
    CALIBRATION_ERROR = "CALIBRATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"


class DailyInspectionError(RuntimeError):
    def __init__(self, category: DailyFailure, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class DailyReferenceSpec:
    nominal_mm: float
    tolerance_mm: float
    mm_per_px: float

    def __post_init__(self):
        for name in ("nominal_mm", "tolerance_mm", "mm_per_px"):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and (
                    value >= 0 if name == "tolerance_mm" else value > 0
                )
            except OverflowError:
                valid = False
            if not valid:
                raise DailyInspectionError(DailyFailure.CONFIGURATION_ERROR, f"{name}: 無効な測定設定")

    @classmethod
    def from_legacy(cls, spec: Mapping) -> "DailyReferenceSpec":
        try:
            return cls(spec["pitchSpec"], spec["pitchTolerance"], spec["pixelMultiplier"])
        except KeyError as exc:
            raise DailyInspectionError(
                DailyFailure.CONFIGURATION_ERROR, f"日常点検の必須項目がありません: {exc}"
            ) from exc


@dataclass(frozen=True)
class DailyROI:
    x: int
    y: int
    width: int
    height: int
    allow_bottom_clip: bool = False

    def __post_init__(self):
        if (any(type(value) is not int for value in (self.x, self.y, self.width, self.height))
                or min(self.x, self.y) < 0 or min(self.width, self.height) < 2):
            raise DailyInspectionError(DailyFailure.CONFIGURATION_ERROR, "日常点検の検出範囲が不正です。")

    def crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        right, bottom = self.x + self.width, self.y + self.height
        if self.allow_bottom_clip:
            bottom = min(bottom, height)
        if right > width or bottom > height or bottom - self.y < 2:
            raise DailyInspectionError(
                DailyFailure.CALIBRATION_ERROR, "日常点検の検出範囲が結合画像に収まりません。"
            )
        return np.ascontiguousarray(image[self.y:bottom, self.x:right]).copy()


@dataclass(frozen=True)
class DailyInferenceSpec:
    confidence: float
    image_size: int

    def __post_init__(self):
        if (type(self.confidence) not in (int, float) or not 0 <= self.confidence <= 1
                or type(self.image_size) is not int or not 32 <= self.image_size <= 4096):
            raise DailyInspectionError(DailyFailure.CONFIGURATION_ERROR, "日常点検の推論設定が不正です。")


@dataclass(frozen=True)
class DailyMeasurement:
    actual_mm: float | None
    reference: DailyReferenceSpec
    endpoints: tuple[tuple[int, int], ...]
    status: str
    reason: str

    @property
    def difference_mm(self) -> float | None:
        return None if self.actual_mm is None else self.actual_mm - self.reference.nominal_mm


def find_reference_points(output, offset: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    if not isinstance(output, (list, tuple)) or not output:
        raise DailyInspectionError(DailyFailure.INFERENCE_ERROR, "日常点検の推論結果が不正です。")
    points: list[tuple[int, int]] = []
    for result in output:
        if not hasattr(result, "keypoints"):
            raise DailyInspectionError(DailyFailure.INFERENCE_ERROR, "推論結果にキーポイント項目がありません。")
        keypoints = result.keypoints
        if keypoints is None:
            continue
        try:
            xy = keypoints.xy
            if hasattr(xy, "cpu"):
                xy = xy.cpu().numpy()
            xy = np.asarray(xy, dtype=np.float64)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise DailyInspectionError(DailyFailure.INFERENCE_ERROR, "キーポイントを読み取れません。") from exc
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise DailyInspectionError(DailyFailure.INFERENCE_ERROR, "キーポイントの座標形式が不正です。")
        for x, y in xy.reshape(-1, 2):
            if math.isfinite(x) and math.isfinite(y):
                points.append((round(offset[0] + float(x)), round(offset[1] + float(y))))
    points.sort(key=lambda point: point[0])
    return (points[0], points[-1]) if len(points) >= 2 else tuple(points)


def measure_reference(
    endpoints: tuple[tuple[int, int], ...], reference: DailyReferenceSpec
) -> DailyMeasurement:
    if len(endpoints) < 2:
        return DailyMeasurement(None, reference, endpoints, "NG", "DETECTION NG")
    actual = round(abs(endpoints[-1][0] - endpoints[0][0]) * reference.mm_per_px, 1)
    if not math.isfinite(actual):
        raise DailyInspectionError(DailyFailure.INFERENCE_ERROR, "日常点検の実測値が無効です。")
    if abs(reference.nominal_mm - actual) <= reference.tolerance_mm:
        return DailyMeasurement(actual, reference, endpoints, "OK", "")
    return DailyMeasurement(actual, reference, endpoints, "NG", "CALIBRATION NG")


class DailyInspectionStrategy:
    def inspect(
        self,
        image: np.ndarray,
        reference: DailyReferenceSpec,
        roi: DailyROI,
        inference: DailyInferenceSpec,
        model: Callable | None,
    ) -> DailyMeasurement:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
            raise DailyInspectionError(DailyFailure.CALIBRATION_ERROR, "日常点検の結合画像が不正です。")
        crop = roi.crop(image)
        if not callable(model):
            raise DailyInspectionError(DailyFailure.MODEL_ERROR, "日常点検のAIモデルが読み込まれていません。")
        try:
            output = model(source=crop, conf=inference.confidence, imgsz=inference.image_size, verbose=False)
        except Exception as exc:
            raise DailyInspectionError(DailyFailure.INFERENCE_ERROR, "日常点検のAI推論に失敗しました。") from exc
        return measure_reference(find_reference_points(output, (roi.x, roi.y)), reference)
