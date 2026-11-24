"""Edit approved daily measurements using the existing Settings service."""

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QWidget
from yaml import YAMLError

from aikensa.core.inspection.daily import DailyInspectionError, DailyReferenceSpec
from aikensa.core.settings.daily_config import DAILY_DEFAULT_REVISION, DailyConfiguration
from aikensa.core.settings.revisions import RevisionError
from aikensa.gui.settings_pages.common import confirm_change, load_settings_ui


_CONFIG_ERRORS = (DailyInspectionError, RevisionError, OSError, ValueError, YAMLError)
_LABELS = ("基準寸法", "許容差", "画素換算係数")
_FIELDS = ("nominal_mm", "tolerance_mm", "mm_per_px")


class DailyInspectionSettingsPage(QWidget):
    def __init__(self, settings, on_saved, parent=None):
        super().__init__(parent)
        load_settings_ui("daily_inspection_settings.ui", self)
        self.settings = settings
        self.on_saved = on_saved
        self.active: DailyConfiguration | None = None
        self.defaults: DailyConfiguration | None = None
        self._edits = (self.dailyNominalEdit, self.dailyToleranceEdit, self.dailyScaleEdit)
        self._position = 0
        self._inherit = False
        self._dirty = False
        self.dailyPositionCombo.currentIndexChanged.connect(self._select_position)
        self.dailyValidateButton.clicked.connect(self._validate)
        self.dailySaveButton.clicked.connect(self._save)
        self.dailyDiscardButton.clicked.connect(self._reload)
        self.dailyInheritButton.clicked.connect(self._use_shared)
        self.dailyRollbackButton.clicked.connect(self._rollback)
        for edit in self._edits:
            edit.textEdited.connect(self._mark_draft)

    def refresh(self):
        if not self._dirty:
            self._reload()

    def _reload(self):
        self._dirty = self._inherit = False
        try:
            self.active = self.settings.load_daily_configuration()
            self.defaults = self.settings.load_daily_defaults()
        except _CONFIG_ERRORS as exc:
            self.active = None
            self.dailyRevisionLabel.setText("日常点検の有効な設定がありません。")
            self.dailyDraftLabel.setText(str(exc))
        enabled = self.active is not None
        for widget in (
            *self._edits, self.dailyValidateButton, self.dailySaveButton,
            self.dailyInheritButton, self.dailyPositionCombo,
        ):
            widget.setEnabled(enabled)
        with QSignalBlocker(self.dailyHistoryCombo):
            self.dailyHistoryCombo.clear()
            self.dailyHistoryCombo.addItem("共通の初期設定", DAILY_DEFAULT_REVISION)
            for revision in self.settings.daily_revisions.list_revisions():
                self.dailyHistoryCombo.addItem(
                    f"{revision.revision_id} / {revision.operator} / {', '.join(revision.changes)}",
                    revision.revision_id,
                )
        if enabled:
            with QSignalBlocker(self.dailyPositionCombo):
                self.dailyPositionCombo.clear()
                for profile in self.active.profiles:
                    self.dailyPositionCombo.addItem(f"点検位置{profile.widget_id - 20}", profile.profile_id)
                self.dailyPositionCombo.setCurrentIndex(self._position)
            self._display_active()

    def _display_active(self):
        profile = self.active.profiles[self._position]
        default = self.defaults.profiles[self._position].reference
        overrides = self.active.editable_values().get(profile.profile_id, {})
        for field, edit in zip(_FIELDS, self._edits):
            edit.setText(str(getattr(profile.reference, field)))
            source = "この位置の上書き" if field in overrides else "共通値"
            edit.setToolTip(f"{source} / 共通値: {getattr(default, field)}")
        self.dailyRevisionLabel.setText(f"適用中の版: {self.active.revision_id}（保存後、次回点検から適用）")
        roi = profile.roi
        self.dailyProfileDetailsLabel.setText(
            f"水平方向 / 共通値: {default.nominal_mm:g} ± {default.tolerance_mm:g} mm, {default.mm_per_px:g} mm/px\n"
            f"検出範囲 (結合切り出し画像): [{roi.x}, {roi.y}, {roi.width}, {roi.height}] px\n"
            f"AIモデル: {profile.model_key} / 信頼度: {profile.inference.confidence:g} / "
            f"推論サイズ: {profile.inference.image_size}"
        )
        self.dailyDraftLabel.setText("適用中の値")

    def _select_position(self, row):
        if row < 0:
            return
        if self._dirty and not confirm_change(self, "下書き", "現在の下書きを破棄して位置を変更しますか？"):
            with QSignalBlocker(self.dailyPositionCombo):
                self.dailyPositionCombo.setCurrentIndex(self._position)
            return
        self._position = row
        self._dirty = self._inherit = False
        self._display_active()

    def _mark_draft(self):
        self._dirty, self._inherit = True, False
        self.dailyDraftLabel.setText("未保存の下書き（実際の点検には未適用）")

    def _use_shared(self):
        self._mark_draft()
        self._inherit = True
        default = self.defaults.profiles[self._position].reference
        for field, edit in zip(_FIELDS, self._edits):
            edit.setText(str(getattr(default, field)))

    def _build_draft(self):
        profile = self.active.profiles[self._position]
        reference = DailyReferenceSpec(*(float(edit.text()) for edit in self._edits))
        overrides = self.active.editable_values()
        updated = {} if self._inherit else overrides.get(profile.profile_id, {}).copy()
        for field in _FIELDS:
            old, new = getattr(profile.reference, field), getattr(reference, field)
            if old != new and not self._inherit:
                updated[field] = new
        if updated:
            overrides[profile.profile_id] = updated
        else:
            overrides.pop(profile.profile_id, None)
        differences = "\n".join(
            f"{label}: {getattr(profile.reference, field):g} → {getattr(reference, field):g}"
            for label, field in zip(_LABELS, _FIELDS)
        )
        return overrides, f"{self.dailyPositionCombo.currentText()}\n{differences}"

    def _validate(self):
        try:
            _overrides, difference = self._build_draft()
            self.dailyDraftLabel.setText(f"下書きは有効です。未保存・未適用。\n{difference}")
        except _CONFIG_ERRORS as exc:
            self.dailyDraftLabel.setText(f"設定を確認してください: {exc}")

    def _save(self):
        try:
            overrides, difference = self._build_draft()
            if not confirm_change(self, "日常点検設定の承認", f"次回点検から適用されます。\n{difference}"):
                return
            self.settings.save_daily_configuration(
                overrides, expected_revision=self.active.revision_id, operator=self.dailyOperatorEdit.text().strip(),
            )
        except _CONFIG_ERRORS as exc:
            self.dailyDraftLabel.setText(f"保存できません: {exc}")
            return
        self._reload()
        self.on_saved("daily_inspection")

    def _rollback(self):
        revision = self.dailyHistoryCombo.currentData()
        if not revision or not confirm_change(self, "日常点検の設定復元", f"現在の下書きを破棄し、{revision} に戻しますか？"):
            return
        try:
            self.settings.rollback_daily_configuration(revision, operator=self.dailyOperatorEdit.text().strip())
        except _CONFIG_ERRORS as exc:
            self.dailyDraftLabel.setText(f"復元できません: {exc}")
            return
        self._reload()
        self.on_saved("daily_inspection")
