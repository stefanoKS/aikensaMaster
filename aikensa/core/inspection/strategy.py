"""Widget-17 inspection contract, independent of camera and UI ownership."""

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np
import yaml


class InspectionError(RuntimeError):
    """Equipment, configuration or inspection execution failure, not product NG."""


@dataclass(frozen=True)
class InspectionProfile:
    part_id: str
    strategy: str
    directory: str
    name: str
    widget_id: int


@dataclass
class InspectionContext:
    image: np.ndarray
    inference_image: np.ndarray
    clip_model: object
    keypoint_model: Callable
    sliced_prediction: Callable
    anomaly_model: object = None
    partcheck: Callable = None


@dataclass
class InspectionResult:
    image: np.ndarray
    measurements: list
    judgments: list
    id_judgments: list
    status: str
    ng_reason: str
    _finalized: bool = False


class InspectionStrategy(Protocol):
    def inspect(self, context: InspectionContext) -> InspectionResult:
        ...


def load_profiles(path: Path) -> dict[str, InspectionProfile]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
        raise InspectionError("profiles must be a list")
    profiles = {}
    for item in data["profiles"]:
        if not isinstance(item, dict) or set(item) != {"part_id", "strategy", "directory", "name", "widget_id"}:
            raise InspectionError("invalid profile fields")
        if any(not isinstance(item[key], str) or not item[key].strip()
               for key in ("part_id", "strategy", "directory", "name")):
            raise InspectionError("profile fields must be nonempty strings")
        if type(item["widget_id"]) is not int or item["widget_id"] < 0:
            raise InspectionError("invalid profile widget")
        profile = InspectionProfile(**item)
        if profile.part_id in profiles:
            raise InspectionError(f"duplicate profile: {profile.part_id}")
        if any(existing.widget_id == profile.widget_id for existing in profiles.values()):
            raise InspectionError(f"duplicate profile widget: {profile.widget_id}")
        if (profile.directory in (".", "..") or "/" in profile.directory
                or "\\" in profile.directory or Path(profile.directory).name != profile.directory):
            raise InspectionError("invalid persistence directory")
        profiles[profile.part_id] = profile
    return profiles


class P808387UA1AStrategy:
    """Adapter for the five-argument P808387UA1A partcheck."""

    def inspect(self, context: InspectionContext) -> InspectionResult:
        for image in (context.image, context.inference_image):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3
                    or image.shape[0] == 0 or image.shape[1] < 512):
                raise InspectionError("invalid inspection image")
        if context.image.shape != context.inference_image.shape:
            raise InspectionError("inspection image shapes differ")
        if context.clip_model is None or not callable(context.keypoint_model) or not callable(context.sliced_prediction):
            raise InspectionError("required inference model or detector missing")
        try:
            detections = context.sliced_prediction(
                context.inference_image, context.clip_model,
                slice_height=512, slice_width=1980,
                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                postprocess_match_metric="IOS", postprocess_match_threshold=0.2,
                postprocess_class_agnostic=True, postprocess_type="GREEDYNMM",
                verbose=0, perform_standard_pred=False,
            )
            left = context.keypoint_model(source=context.image[:, :512, :], conf=0.6, imgsz=512, verbose=False)
            right = context.keypoint_model(source=context.image[:, -512:, :], conf=0.6, imgsz=512, verbose=False)
            if not isinstance(left, (list, tuple)) or not left or not isinstance(right, (list, tuple)) or not right:
                raise InspectionError("invalid keypoint inference output")
            partcheck = context.partcheck
            if partcheck is None:
                from aikensa.core.config.parts.NISSAN.M_JC2D.P808387UA1A import partcheck
            values = partcheck(context.image, detections.object_prediction_list, left, right, context.anomaly_model)
        except InspectionError:
            raise
        except Exception as exc:
            raise InspectionError("P808387UA1A inspection failed") from exc
        if not isinstance(values, tuple) or len(values) != 6:
            raise InspectionError("partcheck must return six values")
        image, measured, judgments, ids, status, reason = values
        if (not isinstance(image, np.ndarray) or image.shape != context.image.shape
                or not isinstance(measured, list) or len(measured) != 9
                or any(not isinstance(value, Real) or not np.isfinite(value) for value in measured)
                or not isinstance(judgments, list) or len(judgments) != 9
                or any(value not in (0, 1) for value in judgments)
                or not isinstance(ids, list) or status not in ("OK", "NG")
                or not isinstance(reason, str)
                or (status == "OK" and (reason or any(value != 1 for value in judgments)))
                or (status == "NG" and not reason)):
            raise InspectionError("invalid partcheck result")
        return InspectionResult(image, measured, judgments, ids, status, reason)


class InspectionCoordinator:
    def __init__(self, profiles: Mapping[str, InspectionProfile], strategies: Mapping[str, InspectionStrategy]):
        self.profiles = dict(profiles)
        self.strategies = dict(strategies)

    def resolve(self, part_id: str) -> tuple[InspectionProfile, InspectionStrategy]:
        if part_id not in self.profiles:
            raise InspectionError(f"unknown inspection profile: {part_id}")
        profile = self.profiles[part_id]
        strategy = self.strategies.get(profile.strategy)
        if strategy is None or not callable(getattr(strategy, "inspect", None)):
            raise InspectionError(f"unknown inspection strategy: {profile.strategy}")
        return profile, strategy

    def inspect(self, part_id: str, context: InspectionContext) -> InspectionResult:
        _, strategy = self.resolve(part_id)
        result = strategy.inspect(context)
        if (not isinstance(result, InspectionResult) or result.status not in ("OK", "NG")
                or not isinstance(result.ng_reason, str)
                or (result.status == "OK" and result.ng_reason)
                or (result.status == "NG" and not result.ng_reason)):
            raise InspectionError("invalid strategy result")
        return result

    def finalize(self, result: InspectionResult, callback: Callable[[InspectionResult], None]) -> None:
        if result._finalized:
            raise InspectionError("inspection already finalized")
        result._finalized = True
        callback(result)
