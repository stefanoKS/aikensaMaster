"""The single façade the GUI and the workers talk to.

Resolution precedence (lowest to highest)::

    shared defaults  ->  machine-specific configuration
                     ->  part/widget-specific machine overrides
                     ->  unsaved UI preview (never used in production)

Only values that passed validation can become active.  The service also keeps
a machine-local metadata record so that replacing a camera or changing its
resolution marks the dependent calibration as ``INCOMPATIBLE`` instead of
silently using geometry that no longer matches the optical setup.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from aikensa.core.settings.daily_config import (
    DAILY_COMMON_SCOPE,
    DAILY_DEFAULT_REVISION,
    DailyConfiguration,
    resolve_daily_config,
)
from aikensa.core.settings.paths import AppPaths
from aikensa.core.settings.repository import SettingsRepository
from aikensa.core.settings.revisions import DAILY_REVISION_LOCK, RevisionError, RevisionManager
from aikensa.core.settings.schema import (
    CalibrationState,
    CameraSettings,
    GlobalAlignment,
    HomographyCalibration,
    IntrinsicCalibration,
    MachineSettingsSnapshot,
    Severity,
    SettingsRevision,
    ValidationIssue,
    WidgetAlignment,
)
from aikensa.core.settings.validator import (
    validate_global_alignment,
    validate_widget_alignment,
)
from aikensa.core.settings.yaml_store import (
    SettingsValidationError,
    read_yaml_result,
    write_yaml_atomic,
)

LOGGER = logging.getLogger(__name__)

CALIBRATION_METADATA_FILENAME = "calibration_metadata.yaml"

Listener = Callable[[str], None]


class SettingsService:
    """High-level settings access used by the application."""

    def __init__(
        self,
        paths: Optional[AppPaths] = None,
        repository: Optional[SettingsRepository] = None,
        revisions: Optional[RevisionManager] = None,
    ):
        self.paths = paths or AppPaths.discover()
        self.repository = repository or SettingsRepository(self.paths)
        self.revisions = revisions or RevisionManager(self.paths)
        self.daily_revisions = RevisionManager(self.paths, snapshot_scope="daily_inspection")
        self._listeners: List[Listener] = []
        self._bootstrapped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def bootstrap(self) -> List[str]:
        """Create missing runtime files and load the active snapshot."""

        created = [] if self._bootstrapped else self.repository.ensure_runtime_layout()
        self._bootstrapped = True
        if created:
            LOGGER.info("Initialised %d machine configuration file(s)", len(created))
        return created

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def _notify(self, what: str) -> None:
        for listener in list(self._listeners):
            try:
                listener(what)
            except Exception:  # pragma: no cover - listener bugs must not break saving
                LOGGER.exception("settings listener failed for %s", what)

    # Judgment-only edits use an immutable snapshot and a single active pointer.
    # They do not reopen cameras or reload models and take effect on the next cycle.
    def load_daily_configuration(self) -> DailyConfiguration:
        revision_id = self.daily_revisions.active_revision_id(strict=True)
        return self._load_daily_revision(revision_id or DAILY_DEFAULT_REVISION)

    def load_daily_defaults(self) -> DailyConfiguration:
        return self._load_daily_revision(DAILY_DEFAULT_REVISION)

    def daily_inheritance_defaults(
        self, configuration: DailyConfiguration, *, include_common: bool = True,
    ) -> DailyConfiguration:
        overrides = configuration.editable_values()
        inherited = {DAILY_COMMON_SCOPE: overrides.get(DAILY_COMMON_SCOPE, {})} if include_common else {}
        return self.preview_daily_configuration(configuration, inherited)

    def preview_daily_configuration(self, configuration: DailyConfiguration, overrides: Mapping) -> DailyConfiguration:
        return resolve_daily_config(
            self.paths.root / "aikensa/core/config",
            configuration.revision_id, configuration.revision_payload(overrides),
        )

    def _load_daily_revision(self, revision_id: str) -> DailyConfiguration:
        if revision_id == DAILY_DEFAULT_REVISION:
            payload = {"schema_version": 1, "positions": {}}
        else:
            if not revision_id.split("-")[-1].isalnum() or "/" in revision_id or "\\" in revision_id:
                raise RevisionError("日常点検の設定版が不正です。")
            staged = self.daily_revisions.staged_data(revision_id)
            if set(staged) != {str(self._daily_payload_path)}:
                raise RevisionError("日常点検の設定スナップショットが不正です。")
            payload = staged[str(self._daily_payload_path)]
        return resolve_daily_config(self.paths.root / "aikensa/core/config", revision_id, payload)

    @property
    def _daily_payload_path(self) -> Path:
        return Path("daily_inspection.yaml")

    def save_daily_configuration(
        self, overrides: Mapping, *, expected_revision: str, operator: str = "", reason: str = "",
    ) -> SettingsRevision:
        with DAILY_REVISION_LOCK:
            current = self.load_daily_configuration()
            if current.revision_id != expected_revision:
                raise RevisionError("日常点検の設定が更新されました。読み込み直してください。")
            payload = current.revision_payload(overrides)
            candidate = resolve_daily_config(self.paths.root / "aikensa/core/config", "", payload)
            if candidate.overrides == current.overrides:
                raise RevisionError("日常点検の設定に変更がありません。")
            changed = [
                f"{new.profile_id}: " + ", ".join(
                    f"{key} {value} → {new.parameter_values()[key]}"
                    for key, value in old.parameter_values().items() if new.parameter_values()[key] != value
                )
                for old, new in zip(current.profiles, candidate.profiles)
                if old != new
            ]
            if not changed:
                changed = ["上書きの継承設定を更新（実効値は同じ）"]
            revision_id = self.daily_revisions.stage(
                {self._daily_payload_path: payload}, operator=operator, changes=changed,
                note=f"{reason} (previous: {current.revision_id})",
            )
            revision = self.daily_revisions.activate(revision_id)
        self._notify("daily_inspection")
        return revision

    def rollback_daily_configuration(
        self, revision_id: str, *, operator: str = "", reason: str = "", expected_revision: Optional[str] = None,
    ) -> SettingsRevision:
        with DAILY_REVISION_LOCK:
            active_id = self.daily_revisions.active_revision_id(strict=True) or DAILY_DEFAULT_REVISION
            if expected_revision is not None and active_id != expected_revision:
                raise RevisionError("日常点検の設定が更新されました。読み込み直してください。")
            restored = self._load_daily_revision(revision_id)
            deployed = self.load_daily_defaults()
            if any(
                profile.model_path != deployed.profile_for_widget(profile.widget_id).model_path
                for profile in restored.profiles
            ):
                raise RevisionError("復元先のAIモデルが現在の配備と異なります。現在の設定は変更されません。")
            payload = restored.revision_payload(restored.editable_values())
            new_id = self.daily_revisions.stage(
                {self._daily_payload_path: payload}, operator=operator,
                changes=(f"{revision_id} に復元",),
                note=f"{reason} (previous: {active_id}; restored: {revision_id})",
            )
            revision = self.daily_revisions.activate(new_id)
        self._notify("daily_inspection")
        return revision

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def load_snapshot(self) -> MachineSettingsSnapshot:
        self.bootstrap()
        cameras, issues = self.repository.load_camera_settings()
        metadata = self.load_calibration_metadata()

        intrinsics: Dict[int, IntrinsicCalibration] = {}
        homographies: Dict[int, HomographyCalibration] = {}
        for index in (0, 1):
            intrinsic = self.repository.load_intrinsic(index)
            homography = self.repository.load_homography(index)
            intrinsic, extra = self._check_compatibility(cameras, metadata, index, intrinsic, homography)
            homographies[index] = homography
            intrinsics[index] = intrinsic
            issues.extend(extra)

        return MachineSettingsSnapshot(
            cameras=cameras,
            intrinsics=intrinsics,
            homographies=homographies,
            global_alignment=self.repository.load_global_alignment(),
            widget_alignments=self.repository.load_widget_adjustments(),
            revision_id=self.revisions.active_revision_id(),
            issues=issues,
        )

    def camera_state(self, snapshot: MachineSettingsSnapshot, index: int) -> CalibrationState:
        """State usable by the Settings / Calibration pages for one camera."""

        assignment = snapshot.cameras.assignment(index)
        if not assignment.is_assigned:
            return CalibrationState.UNCONFIGURED
        intrinsic = snapshot.intrinsics.get(index)
        homography = snapshot.homographies.get(index)
        states = [s.state for s in (intrinsic, homography) if s is not None]
        if not states:
            return CalibrationState.MISSING
        for state in (
            CalibrationState.INVALID,
            CalibrationState.INCOMPATIBLE,
            CalibrationState.MISSING,
            CalibrationState.UNCONFIGURED,
        ):
            if state in states:
                return state
        return CalibrationState.READY

    def production_ready(self, snapshot: Optional[MachineSettingsSnapshot] = None) -> bool:
        """Production inspection needs both cameras fully calibrated."""

        snapshot = snapshot or self.load_snapshot()
        return all(
            self.camera_state(snapshot, index) is CalibrationState.READY for index in (0, 1)
        )

    def readiness_report(self, snapshot: Optional[MachineSettingsSnapshot] = None) -> List[str]:
        """Operator-facing Japanese summary of what still has to be done."""

        snapshot = snapshot or self.load_snapshot()
        lines: List[str] = []
        for index in (0, 1):
            assignment = snapshot.cameras.assignment(index)
            state = self.camera_state(snapshot, index)
            label = assignment.label_ja
            if not assignment.is_assigned:
                lines.append(f"{label}: シリアル番号が未設定です。[設定] > [カメラ設定] で設定してください。")
                continue
            if state is CalibrationState.READY:
                lines.append(f"{label}: 校正済み (準備完了)")
            elif state is CalibrationState.UNCONFIGURED:
                lines.append(f"{label}: 校正データが未設定です。校正を実行してください。")
            elif state is CalibrationState.MISSING:
                lines.append(f"{label}: 校正ファイルがありません。校正を実行してください。")
            elif state is CalibrationState.INVALID:
                lines.append(f"{label}: 校正データが不正です。再校正してください。")
            elif state is CalibrationState.INCOMPATIBLE:
                lines.append(f"{label}: カメラ設定が変更されたため再校正が必要です。")
        return lines

    # ------------------------------------------------------------------
    # Camera settings
    # ------------------------------------------------------------------
    def save_camera_settings(self, cameras: CameraSettings) -> None:
        self.repository.save_camera_settings(cameras)
        self._update_calibration_metadata_after_camera_save(cameras)
        self._notify("camera")

    def _update_calibration_metadata_after_camera_save(self, cameras: CameraSettings) -> None:
        """Record acquisition geometry only for cameras that were never calibrated.

        An existing record is deliberately left untouched: it describes the
        geometry the current calibration belongs to.  Changing width, height,
        rotation or the camera serial therefore makes the next snapshot report
        the calibration as ``INCOMPATIBLE`` instead of silently reusing it.
        """

        metadata = self.load_calibration_metadata()
        changed = False
        for index in (0, 1):
            key = str(index)
            if key in metadata:
                continue
            hw = cameras.hardware_for(index)
            assignment = cameras.assignment(index)
            metadata[key] = {
                "serial": assignment.normalized_serial(),
                "width_px": hw.width_px,
                "height_px": hw.height_px,
                "rotate_180": bool(hw.rotate_180),
                "recorded_at": None,
                "source": "camera_settings",
            }
            changed = True
        if changed:
            write_yaml_atomic(
                self.paths.runtime_adjustments_dir / CALIBRATION_METADATA_FILENAME, metadata
            )

    def _check_compatibility(
        self,
        cameras: CameraSettings,
        metadata: Mapping[str, Mapping],
        index: int,
        intrinsic: IntrinsicCalibration,
        homography: HomographyCalibration,
    ) -> Tuple[IntrinsicCalibration, List[ValidationIssue]]:
        """Downgrade a READY calibration to INCOMPATIBLE when geometry changed."""

        issues: List[ValidationIssue] = []
        record = metadata.get(str(index))
        if not record or intrinsic.state is not CalibrationState.READY:
            return intrinsic, issues

        hw = cameras.hardware_for(index)
        recorded = (
            record.get("width_px"),
            record.get("height_px"),
            record.get("rotate_180"),
        )
        width, height, _rotation = recorded
        if type(width) is int and type(height) is int and width > 0 and height > 0:
            intrinsic.width_px, intrinsic.height_px = width, height
        current = (hw.width_px, hw.height_px, bool(hw.rotate_180))
        serial = record.get("serial")
        if (
            (all(value is not None for value in recorded) and recorded != current)
            or (serial and serial != cameras.assignment(index).normalized_serial())
        ):
            intrinsic.state = CalibrationState.INCOMPATIBLE
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"camera{index}.calibration",
                    intrinsic.state.label_ja
                    + f": カメラ{index + 1} の割当・解像度・向きを確認し、再校正してください。",
                )
            )
        return intrinsic, issues

    def record_calibration_metadata(self, camera_index: int) -> None:
        """Record the geometry that a freshly saved calibration belongs to."""

        cameras, _ = self.repository.load_camera_settings()
        hw = cameras.hardware_for(camera_index)
        assignment = cameras.assignment(camera_index)
        metadata = self.load_calibration_metadata()
        metadata[str(camera_index)] = {
            "serial": assignment.normalized_serial(),
            "width_px": hw.width_px,
            "height_px": hw.height_px,
            "rotate_180": bool(hw.rotate_180),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_yaml_atomic(
            self.paths.runtime_adjustments_dir / CALIBRATION_METADATA_FILENAME, metadata
        )

    def load_calibration_metadata(self) -> Dict[str, dict]:
        data, error = read_yaml_result(
            self.paths.runtime_adjustments_dir / CALIBRATION_METADATA_FILENAME
        )
        if error is not None or not isinstance(data, Mapping):
            return {}
        return {str(k): dict(v) for k, v in data.items() if isinstance(v, Mapping)}

    # ------------------------------------------------------------------
    # Calibration persistence
    # ------------------------------------------------------------------
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
        self.repository.save_intrinsic(
            camera_index,
            camera_matrix,
            distortion_coefficients,
            rotation_vectors,
            translation_vectors,
            scaled=scaled,
        )
        if not scaled:
            self.record_calibration_metadata(camera_index)
        self._notify("intrinsic")

    def save_homography(
        self, camera_index: int, matrix: Optional[np.ndarray], *, scaled: bool = False
    ) -> None:
        self.repository.save_homography(camera_index, matrix, scaled=scaled)
        if not scaled:
            self.record_calibration_metadata(camera_index)
        self._notify("homography")

    # ------------------------------------------------------------------
    # Adjustments
    # ------------------------------------------------------------------
    def active_global_alignment(self) -> GlobalAlignment:
        return self.repository.load_global_alignment()

    def active_widget_overrides(self) -> Dict[int, WidgetAlignment]:
        return self.repository.load_widget_adjustments()

    def effective_widget_alignment(
        self,
        widget_id: int,
        overrides: Optional[Mapping[int, WidgetAlignment]] = None,
        defaults: Optional[Mapping[int, WidgetAlignment]] = None,
    ) -> Optional[WidgetAlignment]:
        overrides = overrides if overrides is not None else self.active_widget_overrides()
        defaults = defaults if defaults is not None else self.repository.shared_widget_defaults()
        return self.repository.effective_widget_alignment(widget_id, overrides, defaults)

    def save_global_alignment(
        self, alignment: GlobalAlignment, *, operator: str = "", note: str = ""
    ) -> SettingsRevision:
        issues = validate_global_alignment(alignment)
        errors = [i for i in issues if i.severity is Severity.ERROR]
        if errors:
            raise SettingsValidationError(errors)

        payload = self.repository.global_alignment_payload(alignment)
        revision_id = self.revisions.stage(
            {self.paths.global_alignment_file(): payload},
            operator=operator,
            changes=[
                f"global_alignment.cam2=({alignment.cam2_x_px:.3f}, {alignment.cam2_y_px:.3f}, {alignment.cam2_rotation_deg:.3f}deg)"
            ],
            note=note,
        )
        revision = self.revisions.activate(revision_id)
        self._notify("global_alignment")
        return revision

    def save_widget_adjustments(
        self,
        overrides: Mapping[int, WidgetAlignment],
        *,
        operator: str = "",
        note: str = "",
    ) -> SettingsRevision:
        issues: List[ValidationIssue] = []
        for alignment in overrides.values():
            issues.extend(validate_widget_alignment(alignment))
        errors = [i for i in issues if i.severity is Severity.ERROR]
        if errors:
            raise SettingsValidationError(errors)

        payload = self.repository.widget_adjustments_payload(overrides)
        revision_id = self.revisions.stage(
            {self.paths.widget_adjustments_file(): payload},
            operator=operator,
            changes=[f"widget_adjustments: {len(overrides)} override(s)"],
            note=note,
        )
        revision = self.revisions.activate(revision_id)
        self._notify("widget_adjustments")
        return revision

    def save_adjustments(
        self,
        global_alignment: Optional[GlobalAlignment] = None,
        widget_overrides: Optional[Mapping[int, WidgetAlignment]] = None,
        *,
        operator: str = "",
        note: str = "",
    ) -> SettingsRevision:
        """Stage global and widget changes together so activation is all-or-nothing."""

        files: Dict[Path, object] = {}
        changes: List[str] = []

        if global_alignment is not None:
            errors = [
                i
                for i in validate_global_alignment(global_alignment)
                if i.severity is Severity.ERROR
            ]
            if errors:
                raise SettingsValidationError(errors)
            files[self.paths.global_alignment_file()] = self.repository.global_alignment_payload(
                global_alignment
            )
            changes.append(
                "global_alignment.cam2="
                f"({global_alignment.cam2_x_px:.3f}, {global_alignment.cam2_y_px:.3f}, "
                f"{global_alignment.cam2_rotation_deg:.3f}deg)"
            )

        if widget_overrides is not None:
            errors = [
                i
                for a in widget_overrides.values()
                for i in validate_widget_alignment(a)
                if i.severity is Severity.ERROR
            ]
            if errors:
                raise SettingsValidationError(errors)
            files[self.paths.widget_adjustments_file()] = (
                self.repository.widget_adjustments_payload(widget_overrides)
            )
            changes.append(f"widget_adjustments: {len(widget_overrides)} override(s)")

        if not files:
            raise SettingsValidationError([ValidationIssue(Severity.ERROR, "adjustments", "変更がありません。")])

        revision_id = self.revisions.stage(
            files, operator=operator, changes=changes, note=note
        )
        revision = self.revisions.activate(revision_id)
        self._notify("adjustments")
        return revision

    def rollback(self, revision_id: str) -> SettingsRevision:
        revision = self.revisions.rollback(revision_id)
        self._notify("rollback")
        return revision

    def history(self) -> List[SettingsRevision]:
        return self.revisions.list_revisions()

    # ------------------------------------------------------------------
    # Import / export helpers used by the "copy from another part" UI action
    # ------------------------------------------------------------------
    def import_widget_alignment(
        self, source: WidgetAlignment, target_widget_id: int, part_number: str
    ) -> WidgetAlignment:
        return WidgetAlignment(
            widget_id=target_widget_id,
            part_number=part_number,
            cam2_offset_x_px=source.cam2_offset_x_px,
            cam2_offset_y_px=source.cam2_offset_y_px,
            cam2_rotation_deg=source.cam2_rotation_deg,
            crop=source.crop,
            katabu_crop=source.katabu_crop,
            source="machine_override",
            max_abs_offset_px=source.max_abs_offset_px,
        )
