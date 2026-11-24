"""One-shot daily Settings measurement; no result lifecycle side effects."""

from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np

from aikensa.core.inspection.daily import DailyInspectionError, DailyInspectionStrategy, DailyMeasurement
from aikensa.core.settings.daily_config import DailyProfileSpec


@dataclass(frozen=True)
class DailyDraftRequest:
    profile: DailyProfileSpec
    measure: bool
    request_id: str

    @classmethod
    def create(cls, profile: DailyProfileSpec, measure: bool) -> "DailyDraftRequest":
        return cls(profile, measure, uuid4().hex)


@dataclass(frozen=True)
class DailyDraftResult:
    request: DailyDraftRequest
    message: str
    full_image: np.ndarray | None = None
    cropped_image: np.ndarray | None = None
    measurement: DailyMeasurement | None = None
    geometry_digest: str | None = None
    geometry_files_digest: str | None = None
    effective_bounds: tuple[int, int, int, int] | None = None


def run_daily_draft(
    request: DailyDraftRequest, raw: np.ndarray, model: Callable | None, *,
    geometry_digest: str | None = None, geometry_files_digest: str | None = None,
) -> DailyDraftResult:
    roi = request.profile.roi
    height, width = raw.shape[:2]
    x0, y0, x1, y1 = roi.effective_bounds(width, height)
    crop = roi.crop(raw)
    full = raw.copy()
    # Draw in native prepared coordinates before fitting the image to a label.
    cv2.rectangle(full, (x0, y0), (x1 - 1, y1 - 1), (40, 225, 60), 5)
    requested_bottom = y0 + roi.height
    measurement = None
    message = (
        f"全体 {width}×{height} px / 要求 ({roi.x}, {roi.y}, {roi.width}, {roi.height}) / "
        f"実効 ({x0}, {y0}, {x1 - x0}, {y1 - y0})"
    )
    if y1 != requested_bottom:
        message += "（下端で切り詰め）"
    if request.measure:
        profile = request.profile
        try:
            measurement = DailyInspectionStrategy().inspect(raw, profile.reference, roi, profile.inference, model)
        except DailyInspectionError as exc:
            message += f"\n{profile.model_key}: {exc.category.value} / {exc}"
        else:
            for x, y in measurement.endpoints:
                cv2.circle(full, (x, y), 12, (255, 100, 20), 4)
                cv2.circle(crop, (x - x0, y - y0), 10, (255, 100, 20), 3)
            actual = "未検出" if measurement.actual_mm is None else f"{measurement.actual_mm:.1f} mm"
            difference = "" if measurement.difference_mm is None else f" / 差 {measurement.difference_mm:+.1f} mm"
            message += (
                f"\n{measurement.status} / 実測 {actual} / "
                f"基準 {profile.reference.nominal_mm:g} ± {profile.reference.tolerance_mm:g} mm{difference}"
            )
    return DailyDraftResult(
        request, message, full, crop, measurement,
        geometry_digest, geometry_files_digest, (x0, y0, x1, y1),
    )
