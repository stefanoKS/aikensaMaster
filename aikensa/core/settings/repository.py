"""Read / write the machine-local YAML files.

This is the only module that knows the on-disk shapes.  Everything above it
(GUI, workers, geometry) works with typed objects from
:mod:`aikensa.core.settings.schema`.

Backwards compatibility that must be preserved:

* ``cam_id.yaml`` historically held ``0: 0`` / ``1: 1`` (enumeration indices).
* ``cam_config.yaml`` holds ``ic4_rotate_180`` plus per-index ``name``,
  ``resolution: [w, h]`` and ``fps``.
* Calibration files hold ``camera_matrix``, ``distortion_coefficients``,
  ``rotation_vectors`` and ``translation_vectors`` as YAML lists.
* Homography / planarisation files are bare YAML list-of-lists, or ``null``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from aikensa.core.settings.paths import (
    CAM_CONFIG_FILENAME,
    CAM_ID_FILENAME,
    GLOBAL_ALIGNMENT_FILENAME,
    HOMOGRAPHY_FILENAMES,
    INTRINSIC_FILENAMES,
    PLANARIZE_FILENAMES,
    WIDGET_ADJUSTMENTS_FILENAME,
    AppPaths,
)
from aikensa.core.settings.schema import (
    CalibrationState,
    CameraAssignment,
    CameraHardwareSettings,
    CameraSettings,
    CropRect,
    GlobalAlignment,
    HomographyCalibration,
    IntrinsicCalibration,
    ValidationIssue,
    WidgetAlignment,
    CAMERA_INDICES,
)
from aikensa.core.settings.validator import (
    crop_from_legacy_points,
    validate_camera_matrix,
    validate_camera_settings,
    validate_distortion_coefficients,
    validate_homography,
)
from aikensa.core.settings.yaml_store import (
    SettingsFileError,
    SettingsValidationError,
    ensure_null_template,
    is_null_value,
    matrix_or_none,
    read_yaml,
    read_yaml_result,
    write_yaml_atomic,
)

#: Template payloads used when a runtime file is missing.  Keeping them in code
#: as well as in ``config/defaults`` means a partially cloned repository still
#: initialises to a *null* state rather than inheriting someone else's data.
_TEMPLATE_CAM_ID = {0: None, 1: None}

_TEMPLATE_CAM_CONFIG = {
    "ic4_rotate_180": True,
    0: {
        "name": "Camera_0",
        "resolution": [3072, 2048],
        "fps": 15,
        "exposure_us": 9000,
        "gain_db": 15,
        "wb_temperature_k": 4500,
        "wb_ratio_r": 2.2,
        "wb_ratio_g": 1.0,
        "wb_ratio_b": 2.2,
    },
    1: {
        "name": "Camera_1",
        "resolution": [3072, 2048],
        "fps": 15,
        "exposure_us": 9000,
        "gain_db": 15,
        "wb_temperature_k": 4500,
        "wb_ratio_r": 2.2,
        "wb_ratio_g": 1.0,
        "wb_ratio_b": 2.2,
    },
}

_TEMPLATE_INTRINSIC = {
    "camera_matrix": None,
    "distortion_coefficients": None,
    "rotation_vectors": None,
    "translation_vectors": None,
}

_TEMPLATE_GLOBAL_ALIGNMENT = {
    "cam2": {"x_px": 0.0, "y_px": 0.0, "rotation_deg": 0.0},
    "pivot_px": None,
    "limits": {"max_abs_shift_px": 500.0, "max_abs_rotation_deg": 2.0},
}

_TEMPLATE_WIDGET_ADJUSTMENTS: dict = {"widgets": {}}


class SettingsRepository:
    """File-level access to machine-specific configuration."""

    def __init__(self, paths: Optional[AppPaths] = None):
        self.paths = paths or AppPaths.discover()

    # ------------------------------------------------------------------
    # Layout initialisation
    # ------------------------------------------------------------------
    def ensure_runtime_layout(self) -> List[str]:
        """Create missing runtime files from templates.

        Existing files are *never* replaced, so a valid calibration or a real
        camera serial can never be overwritten by a default.  Returns the list
        of files that were created.
        """

        created: List[str] = []
        for directory in (
            self.paths.runtime_cam_dir,
            self.paths.runtime_camera_dir,
            self.paths.runtime_adjustments_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        # Create from the git-tracked template when present, otherwise from the
        # in-code template.  Either way the runtime file is only created when
        # absent.
        if ensure_null_template(
            self.paths.cam_id_file(),
            self._template_payload("cam", CAM_ID_FILENAME, _TEMPLATE_CAM_ID),
        ):
            created.append(str(self.paths.cam_id_file()))

        if ensure_null_template(
            self.paths.cam_config_file(),
            self._template_payload("cam", CAM_CONFIG_FILENAME, _TEMPLATE_CAM_CONFIG),
        ):
            created.append(str(self.paths.cam_config_file()))

        for name in INTRINSIC_FILENAMES:
            if ensure_null_template(
                self.paths.camera_file(name),
                self._template_payload("camera", name, _TEMPLATE_INTRINSIC),
            ):
                created.append(str(self.paths.camera_file(name)))

        for name in HOMOGRAPHY_FILENAMES + PLANARIZE_FILENAMES:
            if ensure_null_template(
                self.paths.camera_file(name), self._template_payload("camera", name, None)
            ):
                created.append(str(self.paths.camera_file(name)))

        if ensure_null_template(
            self.paths.global_alignment_file(),
            self._template_payload(
                "adjustments", GLOBAL_ALIGNMENT_FILENAME, _TEMPLATE_GLOBAL_ALIGNMENT
            ),
        ):
            created.append(str(self.paths.global_alignment_file()))

        if ensure_null_template(
            self.paths.widget_adjustments_file(),
            self._template_payload(
                "adjustments", WIDGET_ADJUSTMENTS_FILENAME, _TEMPLATE_WIDGET_ADJUSTMENTS
            ),
        ):
            created.append(str(self.paths.widget_adjustments_file()))

        return created

    def _template_payload(self, group: str, filename: str, fallback: Any) -> Any:
        if group == "cam":
            template_path = self.paths.shared_cam_template(filename)
        elif group == "camera":
            template_path = self.paths.shared_camera_template(filename)
        else:
            template_path = self.paths.shared_adjustment_template(filename)
        if template_path.is_file():
            try:
                return read_yaml(template_path)
            except SettingsFileError:
                return fallback
        return fallback

    # ------------------------------------------------------------------
    # Camera settings
    # ------------------------------------------------------------------
    def load_camera_settings(self) -> Tuple[CameraSettings, List[ValidationIssue]]:
        settings = CameraSettings.unconfigured()
        issues: List[ValidationIssue] = []

        assignments, assign_issues = self._load_cam_id()
        hardware, hardware_issues, rotate_global = self._load_cam_config()
        settings.assignments = assignments
        settings.hardware = hardware
        settings.rotate_180_global = rotate_global
        issues.extend(assign_issues)
        issues.extend(hardware_issues)
        issues.extend(validate_camera_settings(settings))
        return settings, issues

    def _load_cam_id(self) -> Tuple[Dict[int, CameraAssignment], List[ValidationIssue]]:
        result: Dict[int, CameraAssignment] = {i: CameraAssignment(index=i) for i in CAMERA_INDICES}
        issues: List[ValidationIssue] = []
        path = self.paths.cam_id_file()
        data, error = read_yaml_result(path)
        if error:
            issues.append(
                ValidationIssue(
                    "error", "cam_id", f"cam_id.yaml を読み込めません: {error}"
                )
            )
            return result, issues
        if data is None:
            return result, issues
        if not isinstance(data, Mapping):
            issues.append(
                ValidationIssue("error", "cam_id", "cam_id.yaml の形式が不正です (連想配列が必要)。")
            )
            return result, issues

        for key, value in data.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if index not in CAMERA_INDICES:
                continue
            if value is None or (isinstance(value, str) and not value.strip()):
                result[index] = CameraAssignment(index=index)
            elif isinstance(value, bool):
                issues.append(
                    ValidationIssue(
                        "error", f"camera{index}.serial", "シリアル番号の形式が不正です (真偽値)。"
                    )
                )
            elif isinstance(value, int):
                result[index] = CameraAssignment(index=index, serial=str(value), legacy_index=True)
            else:
                result[index] = CameraAssignment(index=index, serial=str(value).strip())
        return result, issues

    def _load_cam_config(
        self,
    ) -> Tuple[Dict[int, CameraHardwareSettings], List[ValidationIssue], bool]:
        hardware = {i: CameraHardwareSettings(index=i) for i in CAMERA_INDICES}
        issues: List[ValidationIssue] = []
        path = self.paths.cam_config_file()
        data, error = read_yaml_result(path)
        if error:
            issues.append(
                ValidationIssue("error", "cam_config", f"cam_config.yaml を読み込めません: {error}")
            )
            return hardware, issues, True
        if data is None:
            return hardware, issues, True
        if not isinstance(data, Mapping):
            issues.append(
                ValidationIssue(
                    "error", "cam_config", "cam_config.yaml の形式が不正です (連想配列が必要)。"
                )
            )
            return hardware, issues, True

        rotate_global = bool(data.get("ic4_rotate_180", True))
        for index in CAMERA_INDICES:
            raw = data.get(index, data.get(str(index)))
            settings = hardware[index]
            settings.rotate_180 = rotate_global
            if not isinstance(raw, Mapping):
                continue
            resolution = raw.get("resolution")
            if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
                try:
                    settings.width_px = int(resolution[0])
                    settings.height_px = int(resolution[1])
                except (TypeError, ValueError):
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"camera{index}.resolution",
                            f"カメラ{index + 1} の解像度の形式が不正です。",
                        )
                    )
            settings.fps = _as_float(raw.get("fps"), settings.fps)
            settings.exposure_us = _as_float(raw.get("exposure_us"), settings.exposure_us)
            settings.gain_db = _as_float(raw.get("gain_db"), settings.gain_db)
            settings.wb_temperature_k = int(
                _as_float(raw.get("wb_temperature_k", raw.get("wb_temperature")),
                         float(settings.wb_temperature_k))
            )
            if "rotate_180" in raw:
                settings.rotate_180 = bool(raw.get("rotate_180"))
            settings.wb_ratio_r = _as_optional_float(raw.get("wb_ratio_r"), settings.wb_ratio_r)
            settings.wb_ratio_g = _as_optional_float(raw.get("wb_ratio_g"), settings.wb_ratio_g)
            settings.wb_ratio_b = _as_optional_float(raw.get("wb_ratio_b"), settings.wb_ratio_b)

        return hardware, issues, rotate_global

    def save_camera_settings(self, settings: CameraSettings) -> None:
        """Validate and persist camera assignments and hardware parameters."""

        issues = validate_camera_settings(settings)
        errors = [i for i in issues if i.severity.value == "error"]
        if errors:
            raise SettingsValidationError(errors)

        cam_id_payload: Dict[int, Optional[str]] = {}
        for index in CAMERA_INDICES:
            cam_id_payload[index] = settings.assignment(index).normalized_serial()
        write_yaml_atomic(self.paths.cam_id_file(), cam_id_payload)

        config_payload: Dict[str, object] = {"ic4_rotate_180": settings.rotate_180_global}
        for index in CAMERA_INDICES:
            hw = settings.hardware_for(index)
            config_payload[index] = {
                "name": f"Camera_{index}",
                "resolution": [hw.width_px, hw.height_px],
                "fps": hw.fps,
                "exposure_us": hw.exposure_us,
                "gain_db": hw.gain_db,
                "wb_temperature_k": hw.wb_temperature_k,
                "wb_ratio_r": hw.wb_ratio_r,
                "wb_ratio_g": hw.wb_ratio_g,
                "wb_ratio_b": hw.wb_ratio_b,
            }
        write_yaml_atomic(self.paths.cam_config_file(), config_payload)

    # ------------------------------------------------------------------
    # Intrinsic calibration
    # ------------------------------------------------------------------
    def load_intrinsic(self, camera_index: int) -> IntrinsicCalibration:
        path = self.paths.calibration_file(camera_index)
        calibration = IntrinsicCalibration(camera_index=camera_index, source_path=str(path))
        data, error = read_yaml_result(path)

        if error is not None:
            calibration.state = CalibrationState.INVALID
            return calibration
        if data is None:
            calibration.state = CalibrationState.MISSING
            return calibration
        if not isinstance(data, Mapping):
            calibration.state = CalibrationState.INVALID
            return calibration

        matrix_raw = data.get("camera_matrix")
        dist_raw = data.get("distortion_coefficients")
        if is_null_value(matrix_raw) and is_null_value(dist_raw):
            calibration.state = CalibrationState.UNCONFIGURED
            return calibration

        try:
            calibration.camera_matrix = matrix_or_none(matrix_raw, (3, 3))
            calibration.distortion_coefficients = matrix_or_none(dist_raw)
            calibration.rotation_vectors = matrix_or_none(data.get("rotation_vectors"))
            calibration.translation_vectors = matrix_or_none(data.get("translation_vectors"))
        except ValueError:
            calibration.state = CalibrationState.INVALID
            return calibration

        problems = validate_camera_matrix(calibration.camera_matrix, "camera_matrix")
        problems += validate_distortion_coefficients(
            calibration.distortion_coefficients, "distortion_coefficients"
        )
        calibration.state = CalibrationState.INVALID if problems else CalibrationState.READY
        return calibration

    def save_intrinsic(
        self,
        camera_index: int,
        camera_matrix: Optional[np.ndarray],
        distortion_coefficients: Optional[np.ndarray],
        rotation_vectors: Optional[np.ndarray] = None,
        translation_vectors: Optional[np.ndarray] = None,
        *,
        scaled: bool = False,
    ) -> None:
        """Persist an intrinsic calibration, refusing to write invalid data.

        A failed calculation must never overwrite the previous valid file, so
        validation happens before anything touches the disk.
        """

        problems = validate_camera_matrix(camera_matrix, "camera_matrix")
        problems += validate_distortion_coefficients(
            distortion_coefficients, "distortion_coefficients"
        )
        if problems:
            raise SettingsValidationError(problems)

        payload = {
            "camera_matrix": np.asarray(camera_matrix, dtype=np.float64).tolist(),
            "distortion_coefficients": np.asarray(
                distortion_coefficients, dtype=np.float64
            ).tolist(),
            "rotation_vectors": None
            if rotation_vectors is None
            else np.asarray(rotation_vectors, dtype=np.float64).tolist(),
            "translation_vectors": None
            if translation_vectors is None
            else np.asarray(translation_vectors, dtype=np.float64).tolist(),
        }
        target = (
            self.paths.calibration_scaled_file(camera_index)
            if scaled
            else self.paths.calibration_file(camera_index)
        )
        write_yaml_atomic(target, payload)

    # ------------------------------------------------------------------
    # Homography
    # ------------------------------------------------------------------
    def load_homography(self, camera_index: int, *, scaled: bool = False) -> HomographyCalibration:
        path = self.paths.homography_file(camera_index, scaled=scaled)
        calibration = HomographyCalibration(
            camera_index=camera_index, source_path=str(path)
        )
        data, error = read_yaml_result(path)
        if error is not None:
            calibration.state = CalibrationState.INVALID
            return calibration
        if data is None:
            # Distinguish "file absent" from "file present but null".
            calibration.state = (
                CalibrationState.MISSING if not path.exists() else CalibrationState.UNCONFIGURED
            )
            return calibration
        try:
            calibration.matrix = matrix_or_none(data, (3, 3))
        except ValueError:
            calibration.state = CalibrationState.INVALID
            return calibration
        if calibration.matrix is None:
            calibration.state = CalibrationState.UNCONFIGURED
            return calibration

        problems = validate_homography(calibration.matrix, "homography")
        calibration.state = CalibrationState.INVALID if problems else CalibrationState.READY
        return calibration

    def save_homography(
        self, camera_index: int, matrix: Optional[np.ndarray], *, scaled: bool = False
    ) -> None:
        problems = validate_homography(matrix, "homography")
        if problems:
            raise SettingsValidationError(problems)
        target = self.paths.homography_file(camera_index, scaled=scaled)
        write_yaml_atomic(target, np.asarray(matrix, dtype=np.float64).tolist())

    def load_planarize_transform(self, name: str) -> Tuple[Optional[np.ndarray], CalibrationState]:
        """Load ``planarizeTransform_<name>.yaml`` without ever raising."""

        path = self.paths.camera_file(f"planarizeTransform_{name}.yaml")
        if not path.exists():
            return None, CalibrationState.MISSING
        data, error = read_yaml_result(path)
        if error is not None:
            return None, CalibrationState.INVALID
        try:
            matrix = matrix_or_none(data, (3, 3))
        except ValueError:
            return None, CalibrationState.INVALID
        if matrix is None:
            return None, CalibrationState.UNCONFIGURED
        return matrix, CalibrationState.READY

    # ------------------------------------------------------------------
    # Adjustments
    # ------------------------------------------------------------------
    def load_global_alignment(self) -> GlobalAlignment:
        data, error = read_yaml_result(self.paths.global_alignment_file())
        alignment = GlobalAlignment()
        if error is not None or not isinstance(data, Mapping):
            return alignment

        cam2 = data.get("cam2") or {}
        if isinstance(cam2, Mapping):
            alignment.cam2_x_px = _as_float(cam2.get("x_px"), 0.0)
            alignment.cam2_y_px = _as_float(cam2.get("y_px"), 0.0)
            alignment.cam2_rotation_deg = _as_float(cam2.get("rotation_deg"), 0.0)
        pivot = data.get("pivot_px")
        if isinstance(pivot, (list, tuple)) and len(pivot) == 2:
            alignment.pivot_px = (float(pivot[0]), float(pivot[1]))
        limits = data.get("limits") or {}
        if isinstance(limits, Mapping):
            alignment.max_abs_shift_px = _as_float(
                limits.get("max_abs_shift_px"), alignment.max_abs_shift_px
            )
            alignment.max_abs_rotation_deg = _as_float(
                limits.get("max_abs_rotation_deg"), alignment.max_abs_rotation_deg
            )
        return alignment

    def global_alignment_payload(self, alignment: GlobalAlignment) -> Dict[str, object]:
        return {
            "cam2": {
                "x_px": float(alignment.cam2_x_px),
                "y_px": float(alignment.cam2_y_px),
                "rotation_deg": float(alignment.cam2_rotation_deg),
            },
            "pivot_px": None if alignment.pivot_px is None else list(alignment.pivot_px),
            "limits": {
                "max_abs_shift_px": float(alignment.max_abs_shift_px),
                "max_abs_rotation_deg": float(alignment.max_abs_rotation_deg),
            },
        }

    def load_widget_adjustments(self) -> Dict[int, WidgetAlignment]:
        data, error = read_yaml_result(self.paths.widget_adjustments_file())
        result: Dict[int, WidgetAlignment] = {}
        if error is not None or not isinstance(data, Mapping):
            return result
        raw_widgets = data.get("widgets") or {}
        if not isinstance(raw_widgets, Mapping):
            return result
        for key, raw in raw_widgets.items():
            try:
                widget_id = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(raw, Mapping):
                continue
            alignment = WidgetAlignment(
                widget_id=widget_id,
                part_number=str(raw.get("part_number", "") or ""),
                cam2_offset_x_px=_as_float(raw.get("cam2_offset_x_px"), 0.0),
                cam2_offset_y_px=_as_float(raw.get("cam2_offset_y_px"), 0.0),
                cam2_rotation_deg=_as_float(raw.get("cam2_rotation_deg"), 0.0),
                crop=_crop_from_mapping(raw.get("crop")),
                katabu_crop=_crop_from_mapping(raw.get("katabu_crop")),
                source="machine_override",
            )
            result[widget_id] = alignment
        return result

    def widget_adjustments_payload(
        self, adjustments: Mapping[int, WidgetAlignment]
    ) -> Dict[str, object]:
        widgets: Dict[str, object] = {}
        for widget_id in sorted(adjustments):
            alignment = adjustments[widget_id]
            widgets[str(widget_id)] = {
                "part_number": alignment.part_number,
                "cam2_offset_x_px": float(alignment.cam2_offset_x_px),
                "cam2_offset_y_px": float(alignment.cam2_offset_y_px),
                "cam2_rotation_deg": float(alignment.cam2_rotation_deg),
                "crop": None if alignment.crop is None else list(alignment.crop.as_tuple()),
                "katabu_crop": None
                if alignment.katabu_crop is None
                else list(alignment.katabu_crop.as_tuple()),
            }
        return {"widgets": widgets}

    def shared_widget_defaults(self) -> Dict[int, WidgetAlignment]:
        """Read the shared (Git-tracked) per-widget defaults from the registry.

        These values are *defaults*: a machine-local override in
        ``widget_adjustments.yaml`` always wins, and "restore shared default"
        simply deletes the override.
        """

        result: Dict[int, WidgetAlignment] = {}
        registry_path = self.paths.root / "aikensa" / "core" / "config" / "part_registry.yaml"
        if not registry_path.is_file():
            return result
        try:
            data = read_yaml(registry_path)
        except SettingsFileError:
            return result
        if not isinstance(data, Mapping):
            return result

        raw_offsets = data.get("widget_h2_offsets_cam2") or {}
        if isinstance(raw_offsets, Mapping):
            for key, value in raw_offsets.items():
                try:
                    widget_id = int(key)
                except (TypeError, ValueError):
                    continue
                dx = dy = 0.0
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    dx, dy = float(value[0]), float(value[1])
                elif isinstance(value, Mapping):
                    dx = float(value.get("x", 0.0))
                    dy = float(value.get("y", 0.0))
                else:
                    continue
                result[widget_id] = WidgetAlignment(
                    widget_id=widget_id, cam2_offset_x_px=dx, cam2_offset_y_px=dy
                )

        raw_crops = data.get("widget_crop_points") or {}
        if isinstance(raw_crops, Mapping):
            for key, value in raw_crops.items():
                try:
                    widget_id = int(key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(value, (list, tuple)):
                    continue
                crop = crop_from_legacy_points(value)
                if crop is None:
                    continue
                existing = result.get(widget_id) or WidgetAlignment(widget_id=widget_id)
                existing.crop = crop
                result[widget_id] = existing

        return result

    def effective_widget_alignment(
        self, widget_id: int, overrides: Mapping[int, WidgetAlignment], defaults: Mapping[int, WidgetAlignment]
    ) -> Optional[WidgetAlignment]:
        if widget_id in overrides:
            return overrides[widget_id]
        default = defaults.get(widget_id)
        return default

    # ------------------------------------------------------------------
    # Shared templates (git tracked)
    # ------------------------------------------------------------------
    def load_shared_template_camera_settings(self) -> CameraSettings:
        """Load the git-tracked default camera settings (never the machine file)."""

        settings = CameraSettings.unconfigured()
        cam_id = self._template_payload("cam", CAM_ID_FILENAME, _TEMPLATE_CAM_ID)
        if isinstance(cam_id, Mapping):
            for index in CAMERA_INDICES:
                value = cam_id.get(index, cam_id.get(str(index)))
                if value is None:
                    settings.assignments[index] = CameraAssignment(index=index)
                elif isinstance(value, int) and not isinstance(value, bool):
                    settings.assignments[index] = CameraAssignment(
                        index=index, serial=str(value), legacy_index=True
                    )
                else:
                    settings.assignments[index] = CameraAssignment(
                        index=index, serial=str(value).strip()
                    )

        config = self._template_payload("cam", CAM_CONFIG_FILENAME, _TEMPLATE_CAM_CONFIG)
        if isinstance(config, Mapping):
            rotate = bool(config.get("ic4_rotate_180", True))
            settings.rotate_180_global = rotate
            for index in CAMERA_INDICES:
                raw = config.get(index, config.get(str(index)))
                hardware = CameraHardwareSettings(index=index, rotate_180=rotate)
                if isinstance(raw, Mapping):
                    resolution = raw.get("resolution")
                    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
                        hardware.width_px = int(resolution[0])
                        hardware.height_px = int(resolution[1])
                    hardware.fps = _as_float(raw.get("fps"), hardware.fps)
                    hardware.exposure_us = _as_float(raw.get("exposure_us"), hardware.exposure_us)
                    hardware.gain_db = _as_float(raw.get("gain_db"), hardware.gain_db)
                    hardware.wb_temperature_k = int(
                        _as_float(raw.get("wb_temperature_k"), float(hardware.wb_temperature_k))
                    )
                    hardware.wb_ratio_r = _as_optional_float(raw.get("wb_ratio_r"), hardware.wb_ratio_r)
                    hardware.wb_ratio_g = _as_optional_float(raw.get("wb_ratio_g"), hardware.wb_ratio_g)
                    hardware.wb_ratio_b = _as_optional_float(raw.get("wb_ratio_b"), hardware.wb_ratio_b)
                settings.hardware[index] = hardware
        return settings



# ---------------------------------------------------------------------------
# Small conversion helpers
# ---------------------------------------------------------------------------
def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _crop_from_mapping(value: Any) -> Optional[CropRect]:
    if isinstance(value, Mapping):
        try:
            crop = CropRect(
                x_px=int(value["x_px"]),
                y_px=int(value["y_px"]),
                width_px=int(value["width_px"]),
                height_px=int(value["height_px"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        return crop if crop.is_valid() else None
    if isinstance(value, (list, tuple)):
        if len(value) == 4:
            try:
                crop = CropRect(
                    x_px=int(value[0]),
                    y_px=int(value[1]),
                    width_px=int(value[2]),
                    height_px=int(value[3]),
                )
            except (TypeError, ValueError):
                return None
            return crop if crop.is_valid() else None
        if len(value) == 8:
            return crop_from_legacy_points(value)
    return None

    # ------------------------------------------------------------------