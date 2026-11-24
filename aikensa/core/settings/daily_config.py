"""Resolve the daily family from the existing registry and shared specifications."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from aikensa.core.inspection.daily import (
    DailyFailure,
    DailyInferenceSpec,
    DailyInspectionError,
    DailyReferenceSpec,
    DailyROI,
)
from aikensa.core.settings.yaml_store import read_yaml_result

DAILY_MODEL = "NICHIJOU_TENKEN_Model"
DAILY_DEFAULT_REVISION = "shared-defaults"
_REFERENCE_FIELDS = frozenset(("nominal_mm", "tolerance_mm", "mm_per_px"))


@dataclass(frozen=True)
class DailyProfileSpec:
    profile_id: str
    widget_id: int
    directory: str
    model_key: str
    inference: DailyInferenceSpec
    roi: DailyROI
    reference: DailyReferenceSpec


@dataclass(frozen=True)
class DailyConfiguration:
    revision_id: str
    profiles: tuple[DailyProfileSpec, ...]
    overrides: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]

    def profile_for_widget(self, widget_id: int) -> DailyProfileSpec:
        for profile in self.profiles:
            if profile.widget_id == widget_id:
                return profile
        raise _invalid(f"点検ページ {widget_id} の設定がありません。")

    def editable_values(self) -> dict[str, dict[str, float]]:
        return {profile: dict(values) for profile, values in self.overrides}


def _invalid(message: str) -> DailyInspectionError:
    return DailyInspectionError(DailyFailure.CONFIGURATION_ERROR, message)


def _read_mapping(path: Path) -> Mapping:
    data, error = read_yaml_result(path)
    if error or not isinstance(data, Mapping):
        raise _invalid(f"{path.name}: 日常点検の共通設定を読み込めません。")
    return data


def resolve_daily_config(
    config_dir: Path, revision_id: str, payload: object,
) -> DailyConfiguration:
    specs = _read_mapping(config_dir / "parts_specifications.yaml").get("parts")
    models = _read_mapping(config_dir / "models.yaml").get("models")
    entries = _read_mapping(config_dir / "part_registry.yaml").get("parts")
    if not isinstance(specs, Mapping) or not isinstance(specs.get("NICHIJOU_TENKEN"), Mapping):
        raise _invalid("日常点検の基準仕様がありません。")
    if not isinstance(models, Mapping) or not isinstance(models.get(DAILY_MODEL), Mapping):
        raise _invalid("日常点検のAIモデル定義がありません。")
    model = models[DAILY_MODEL]
    if model.get("loader") != "yolo" or not isinstance(model.get("path"), str) or not model["path"].strip():
        raise _invalid("日常点検のAIモデル定義が不正です。")
    if not isinstance(entries, list):
        raise _invalid("品番レジストリが不正です。")
    shared = DailyReferenceSpec.from_legacy(specs["NICHIJOU_TENKEN"])
    profiles = []
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("inspection_type") != "daily_inspection":
            continue
        profile_id = entry.get("part_number")
        widget_id = entry.get("widget_id")
        directory = entry.get("directory_name")
        roi = entry.get("inference_crop")
        if (
            not isinstance(profile_id, str) or not profile_id.strip()
            or type(widget_id) is not int
            or not isinstance(directory, str) or not directory.strip()
            or directory in (".", "..") or "/" in directory or "\\" in directory
            or not isinstance(roi, list) or len(roi) != 4
        ):
            raise _invalid("日常点検のプロファイルが不正です。")
        profiles.append(DailyProfileSpec(
            profile_id, widget_id, directory, DAILY_MODEL,
            DailyInferenceSpec(0.4, entry.get("keypoint_model_imgsz")),
            DailyROI(*roi, allow_bottom_clip=True), shared,
        ))
    if (
        {profile.widget_id for profile in profiles} != {21, 22, 23}
        or len(profiles) != 3 or len({profile.profile_id for profile in profiles}) != 3
        or len({profile.directory for profile in profiles}) != 3
    ):
        raise _invalid("日常点検の3つの点検位置を重複せずに設定してください。")
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "positions"}:
        raise _invalid("日常点検の設定ファイル形式が不正です。")
    positions = payload["positions"]
    if (type(payload["schema_version"]) is not int or payload["schema_version"] != 1
            or not isinstance(positions, Mapping)):
        raise _invalid("日常点検の設定ファイルの版または点検位置が不正です。")
    if set(positions) - {profile.profile_id for profile in profiles}:
        raise _invalid("登録されていない日常点検の位置が含まれています。")
    normalized = []
    effective = []
    for profile in sorted(profiles, key=lambda profile: profile.widget_id):
        override = positions.get(profile.profile_id, {})
        if not isinstance(override, Mapping) or set(override) - _REFERENCE_FIELDS:
            raise _invalid(f"{profile.profile_id}: 日常点検の上書き項目が不正です。")
        reference = replace(shared, **override)
        effective.append(replace(profile, reference=reference))
        if override:
            normalized.append((profile.profile_id, tuple(sorted(override.items()))))
    return DailyConfiguration(revision_id, tuple(effective), tuple(normalized))
