"""Resolve the daily family from the existing registry and shared specifications."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
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
DAILY_COMMON_SCOPE = "__shared__"
_REFERENCE_FIELDS = frozenset(("nominal_mm", "tolerance_mm", "mm_per_px"))
_INFERENCE_FIELDS = frozenset(("confidence", "image_size"))
_ROI_FIELDS = frozenset(("x", "y", "width", "height"))
DAILY_COMMON_FIELDS = _REFERENCE_FIELDS | _INFERENCE_FIELDS | {"save_diagnostic_images"}
DAILY_POSITION_FIELDS = DAILY_COMMON_FIELDS | _ROI_FIELDS


@dataclass(frozen=True)
class DailyProfileSpec:
    profile_id: str
    widget_id: int
    directory: str
    model_key: str
    inference: DailyInferenceSpec
    roi: DailyROI
    reference: DailyReferenceSpec
    model_path: str
    save_diagnostic_images: bool = True

    def parameter_values(self) -> dict:
        return {
            **asdict(self.reference), **asdict(self.inference),
            **{key: getattr(self.roi, key) for key in _ROI_FIELDS},
            "save_diagnostic_images": self.save_diagnostic_images,
        }

    def with_parameters(self, overrides: Mapping) -> "DailyProfileSpec":
        save_diagnostics = overrides.get("save_diagnostic_images", self.save_diagnostic_images)
        if type(save_diagnostics) is not bool:
            raise _invalid("診断画像保存には真偽値を指定してください。")
        return replace(
            self,
            reference=replace(self.reference, **{k: v for k, v in overrides.items() if k in _REFERENCE_FIELDS}),
            inference=replace(self.inference, **{k: v for k, v in overrides.items() if k in _INFERENCE_FIELDS}),
            roi=replace(self.roi, **{k: v for k, v in overrides.items() if k in _ROI_FIELDS}),
            save_diagnostic_images=save_diagnostics,
        )

    @property
    def configuration_digest(self) -> str:
        profile = asdict(self)
        profile["reference"] = {key: float(value) for key, value in profile["reference"].items()}
        return sha256(_canonical({"strategy": "daily-horizontal-v1", "profile": profile}).encode()).hexdigest()


@dataclass(frozen=True)
class DailyConfiguration:
    revision_id: str
    profiles: tuple[DailyProfileSpec, ...]
    overrides: tuple[tuple[str, tuple[tuple[str, float | int | bool], ...]], ...]
    shared_definition: str

    def profile_for_widget(self, widget_id: int) -> DailyProfileSpec:
        for profile in self.profiles:
            if profile.widget_id == widget_id:
                return profile
        raise _invalid(f"点検ページ {widget_id} の設定がありません。")

    def editable_values(self) -> dict[str, dict[str, float | int | bool]]:
        return {profile: dict(values) for profile, values in self.overrides}

    def revision_payload(self, overrides: Mapping) -> dict:
        return {
            "schema_version": 3,
            "positions": {key: value for key, value in overrides.items() if key != DAILY_COMMON_SCOPE},
            "common": overrides.get(DAILY_COMMON_SCOPE, {}),
            "shared": json.loads(self.shared_definition),
        }


def _canonical(value: Mapping) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _invalid(message: str) -> DailyInspectionError:
    return DailyInspectionError(DailyFailure.CONFIGURATION_ERROR, message)


def _read_mapping(path: Path) -> Mapping:
    data, error = read_yaml_result(path)
    if error or not isinstance(data, Mapping):
        raise _invalid(f"{path.name}: 日常点検の共通設定を読み込めません。")
    return data


def load_daily_shared_definition(config_dir: Path) -> dict:
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
    spec = specs["NICHIJOU_TENKEN"]
    reference = {name: spec[name] for name in ("pitchSpec", "pitchTolerance", "pixelMultiplier") if name in spec}
    return {
        "reference": reference,
        "model": {"loader": model["loader"], "path": model["path"]},
        "save_diagnostic_images": spec.get("debug_save_images", False),
        "profiles": [
            {name: entry.get(name) for name in (
                "part_number", "widget_id", "directory_name", "inference_crop", "keypoint_model_imgsz",
            )}
            for entry in entries if isinstance(entry, Mapping) and entry.get("inspection_type") == "daily_inspection"
        ],
    }


def resolve_daily_config(
    config_dir: Path, revision_id: str, payload: object,
) -> DailyConfiguration:
    if not isinstance(payload, Mapping) or type(payload.get("schema_version")) is not int:
        raise _invalid("日常点検の設定ファイル形式が不正です。")
    version = payload["schema_version"]
    if version == 1 and set(payload) == {"schema_version", "positions"}:
        definition = load_daily_shared_definition(config_dir)
    elif version == 2 and set(payload) == {"schema_version", "positions", "shared"}:
        definition = payload["shared"]
    elif version == 3 and set(payload) == {"schema_version", "positions", "common", "shared"}:
        definition = payload["shared"]
    else:
        raise _invalid("日常点検の設定ファイルの版が不正です。")

    if (not isinstance(definition, Mapping)
            or set(definition) - {"save_diagnostic_images"} != {"reference", "model", "profiles"}):
        raise _invalid("日常点検の保存済み定義が不正です。")
    model, entries = definition["model"], definition["profiles"]
    if (not isinstance(model, Mapping) or model.get("loader") != "yolo"
            or not isinstance(model.get("path"), str) or not model["path"].strip()
            or not isinstance(entries, list) or not isinstance(definition["reference"], Mapping)):
        raise _invalid("日常点検のモデルまたはプロファイル定義が不正です。")
    shared = DailyReferenceSpec.from_legacy(definition["reference"])
    diagnostics = definition.get("save_diagnostic_images", True)
    if type(diagnostics) is not bool:
        raise _invalid("診断画像保存には真偽値を指定してください。")
    profiles = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise _invalid("日常点検のプロファイルが不正です。")
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
            DailyROI(*roi, allow_bottom_clip=True), shared, model["path"], diagnostics,
        ))
    if (
        {profile.widget_id for profile in profiles} != {21, 22, 23}
        or len(profiles) != 3 or len({profile.profile_id for profile in profiles}) != 3
        or len({profile.directory for profile in profiles}) != 3
    ):
        raise _invalid("日常点検の3つの点検位置を重複せずに設定してください。")
    positions = payload["positions"]
    if not isinstance(positions, Mapping):
        raise _invalid("日常点検の点検位置が不正です。")
    if set(positions) - {profile.profile_id for profile in profiles}:
        raise _invalid("登録されていない日常点検の位置が含まれています。")
    common = payload.get("common", {})
    if not isinstance(common, Mapping) or set(common) - DAILY_COMMON_FIELDS:
        raise _invalid("共通設定の上書き項目が不正です。検出範囲は位置ごとに設定してください。")
    normalized = [(DAILY_COMMON_SCOPE, tuple(sorted(common.items())))] if common else []
    effective = []
    for profile in sorted(profiles, key=lambda profile: profile.widget_id):
        override = positions.get(profile.profile_id, {})
        if not isinstance(override, Mapping) or set(override) - (DAILY_POSITION_FIELDS if version == 3 else _REFERENCE_FIELDS):
            raise _invalid(f"{profile.profile_id}: 日常点検の上書き項目が不正です。")
        inherited = profile.with_parameters(common)
        effective.append(inherited.with_parameters(override))
        if override:
            normalized.append((profile.profile_id, tuple(sorted(override.items()))))
    return DailyConfiguration(revision_id, tuple(effective), tuple(normalized), _canonical(definition))
