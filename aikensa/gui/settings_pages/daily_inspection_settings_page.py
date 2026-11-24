"""Edit approved daily measurements using the existing Settings service."""

from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMessageBox, QSizePolicy, QWidget
from yaml import YAMLError

from aikensa.core.inspection.daily import DailyInspectionError
from aikensa.core.inspection.daily_draft import DailyDraftRequest
from aikensa.core.settings.daily_config import DAILY_COMMON_FIELDS, DAILY_COMMON_SCOPE, DAILY_DEFAULT_REVISION, DailyConfiguration
from aikensa.gui.page_viewport import display_image
from aikensa.core.settings.revisions import RevisionError
from aikensa.gui.settings_pages.common import confirm_change, load_settings_ui


_CONFIG_ERRORS = (DailyInspectionError, RevisionError, OSError, ValueError, YAMLError)
_FIELDS = {
    "nominal_mm": ("基準寸法", float), "tolerance_mm": ("許容差", float),
    "mm_per_px": ("画素換算係数", float), "confidence": ("検出信頼度", float),
    "image_size": ("推論画像サイズ", int), "x": ("検出範囲 X", int), "y": ("検出範囲 Y", int),
    "width": ("検出範囲 幅", int), "height": ("検出範囲 高さ", int),
    "save_diagnostic_images": ("診断画像保存", bool),
}


class DailyInspectionSettingsPage(QWidget):
    def __init__(self, settings, on_saved, parent=None):
        super().__init__(parent)
        load_settings_ui("daily_inspection_settings.ui", self)
        self.settings = settings
        self.on_saved = on_saved
        self.active: DailyConfiguration | None = None
        self.defaults: DailyConfiguration | None = None
        self._edits = dict(zip(_FIELDS, (
            self.dailyNominalEdit, self.dailyToleranceEdit, self.dailyScaleEdit,
            self.dailyConfidenceEdit, self.dailyImageSizeEdit, self.dailyXEdit, self.dailyYEdit,
            self.dailyWidthEdit, self.dailyHeightEdit, self.dailyDiagnosticCheck,
        )))
        self._position = 0
        self._forms = {}
        self._inherit = set()
        self._generation = 0
        self._pending = None
        self._verified_rois = set()
        self.submit_draft = None
        self.dailyPositionCombo.currentIndexChanged.connect(self._select_position)
        self.dailyValidateButton.clicked.connect(self._validate)
        self.dailySaveButton.clicked.connect(self._save)
        self.dailyDiscardButton.clicked.connect(self._discard)
        self.dailyInheritButton.clicked.connect(self._use_shared)
        self.dailyRollbackButton.clicked.connect(self._rollback)
        self.dailyCaptureButton.clicked.connect(lambda: self._request_reference(False))
        self.dailyTestButton.clicked.connect(lambda: self._request_reference(True))
        self.dailyReferencePositionCombo.currentIndexChanged.connect(self._invalidate_reference)
        for field, edit in self._edits.items():
            signal = edit.toggled if field == "save_diagnostic_images" else edit.textChanged
            signal.connect(self._mark_draft)
        for label in (self.dailyFullPreviewLabel, self.dailyCropPreviewLabel):
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._update_test_actions()

    def refresh(self):
        if not self._forms:
            self._reload()

    def _reload(self):
        self._forms.clear()
        self._inherit.clear()
        self._verified_rois.clear()
        self._invalidate_reference()
        try:
            self.active = self.settings.load_daily_configuration()
            self.defaults = self.settings.daily_inheritance_defaults(self.active, include_common=False)
        except _CONFIG_ERRORS as exc:
            self.active = None
            self.dailyRevisionLabel.setText("日常点検の有効な設定がありません。")
            self.dailyDraftLabel.setText(str(exc))
        enabled = self.active is not None
        for widget in (
            *self._edits.values(), self.dailyValidateButton, self.dailySaveButton,
            self.dailyInheritButton, self.dailyPositionCombo,
        ):
            widget.setEnabled(enabled)
        with QSignalBlocker(self.dailyHistoryCombo):
            self.dailyHistoryCombo.clear()
            self.dailyHistoryCombo.addItem("現在配備されている共通設定", DAILY_DEFAULT_REVISION)
            for revision in self.settings.daily_revisions.list_revisions():
                self.dailyHistoryCombo.addItem(
                    f"{revision.revision_id} / {revision.operator} / {', '.join(revision.changes)}",
                    revision.revision_id,
                )
                self.dailyHistoryCombo.setItemData(self.dailyHistoryCombo.count() - 1, revision.note, Qt.ToolTipRole)
        if enabled:
            with QSignalBlocker(self.dailyPositionCombo):
                self.dailyPositionCombo.clear()
                for profile in self.active.profiles:
                    self.dailyPositionCombo.addItem(f"点検位置{profile.widget_id - 20}", profile.profile_id)
                self.dailyPositionCombo.addItem("共通設定", DAILY_COMMON_SCOPE)
                self.dailyPositionCombo.setCurrentIndex(self._position)
            self._display_active()
        self._update_test_actions()

    def _scope(self):
        return self.dailyPositionCombo.itemData(self._position)

    def _values(self):
        return {
            name: (edit.isChecked() if name == "save_diagnostic_images" else edit.text())
            for name, edit in self._edits.items() if self._scope() != DAILY_COMMON_SCOPE or name in DAILY_COMMON_FIELDS
        }

    def _base_values(self, scope, overrides):
        common = scope == DAILY_COMMON_SCOPE
        if common:
            profile = self.defaults.profiles[0]
        else:
            inherited = self.settings.preview_daily_configuration(
                self.active, {DAILY_COMMON_SCOPE: overrides.get(DAILY_COMMON_SCOPE, {})},
            )
            profile = next(p for p in inherited.profiles if p.profile_id == scope)
        values = profile.parameter_values()
        return {k: v for k, v in values.items() if not common or k in DAILY_COMMON_FIELDS}

    def _form_values(self, scope, overrides):
        values = {**self._base_values(scope, overrides), **overrides.get(scope, {})}
        return {name: (value if type(value) is bool else str(value)) for name, value in values.items()}

    def _display_active(self):
        scope = self._scope()
        common = scope == DAILY_COMMON_SCOPE
        profile = self.active.profiles[0 if common else self._position]
        overrides = self.active.editable_values()
        baseline = self._base_values(scope, overrides)
        original = self._form_values(scope, overrides)
        values = self._forms.get(scope, original)
        for field, edit in self._edits.items():
            edit.setEnabled(field in values)
            with QSignalBlocker(edit):
                if field == "save_diagnostic_images":
                    edit.setChecked(values[field])
                else:
                    edit.setText(values.get(field, ""))
            if field in values:
                source = "上書き" if field in overrides.get(scope, {}) else "継承値"
                edit.setToolTip(f"{source} / 継承元: {baseline[field]}")
        with QSignalBlocker(self.dailyReferencePositionCombo):
            self.dailyReferencePositionCombo.clear()
            for item in (self.active.profiles if common else (profile,)):
                self.dailyReferencePositionCombo.addItem(f"点検位置{item.widget_id - 20}", item.widget_id)
        self.dailyReferencePositionCombo.setEnabled(common)
        self.dailyRevisionLabel.setText(f"適用中の版: {self.active.revision_id}（保存後、次回点検から適用）")
        self.dailyProfileDetailsLabel.setText(
            f"使用AIモデル: {profile.model_key} (YOLO pose)\n"
            "配備済みの点検モデルのみ使用できます。設定テストで読込状態を確認します。"
        )
        self._show_draft_state()
        self._update_test_actions()

    def _select_position(self, row):
        if row < 0 or self.active is None:
            return
        self._position = row
        self._invalidate_reference()
        self._display_active()

    def _mark_draft(self):
        self._forms[self._scope()] = self._values()
        self._show_draft_state()
        self._invalidate_reference()

    def _show_draft_state(self):
        self.dailyDraftLabel.setText(
            f"未保存の下書き: {len(self._forms)} 件（実際の点検には未適用）" if self._forms else "適用中の値"
        )

    def _use_shared(self):
        scope = self._scope()
        base = self._base_values(scope, self.active.editable_values())
        self._inherit.add(scope)
        self._forms[scope] = {k: v if type(v) is bool else str(v) for k, v in base.items()}
        self._display_active()
        self._invalidate_reference()

    def _discard(self):
        scope = self._scope()
        self._forms.pop(scope, None)
        self._inherit.discard(scope)
        if not self._forms:
            self._reload()
        else:
            self._display_active()
            self._invalidate_reference()

    def _build_draft(self):
        overrides = self.active.editable_values()
        for scope in sorted(self._forms, key=lambda scope: scope != DAILY_COMMON_SCOPE):
            form = self._forms[scope]
            inherited = scope in self._inherit
            baseline = self._form_values(scope, self.active.editable_values())
            old_base = self._base_values(scope, self.active.editable_values())
            updated = {} if inherited else overrides.get(scope, {}).copy()
            for field, text in form.items():
                previous = (old_base[field] if type(old_base[field]) is bool else str(old_base[field])) if inherited else baseline[field]
                if text != previous:
                    name, convert = _FIELDS[field]
                    try:
                        updated[field] = convert(text)
                    except ValueError as exc:
                        raise ValueError(f"{name}: 数値を確認してください。") from exc
            if updated:
                overrides[scope] = updated
            else:
                overrides.pop(scope, None)
        candidate = self.settings.preview_daily_configuration(self.active, overrides)
        differences = []
        for old, new in zip(self.active.profiles, candidate.profiles):
            before, after = old.parameter_values(), new.parameter_values()
            changes = [f"{_FIELDS[key][0]}: {value} → {after[key]}" for key, value in before.items() if value != after[key]]
            differences.append(f"点検位置{new.widget_id - 20}: " + (" / ".join(changes) if changes else "変更なし"))
        return overrides, "\n".join(differences), candidate

    def _validate(self):
        try:
            _overrides, difference, _candidate = self._build_draft()
            self.dailyDraftLabel.setText(f"下書きは有効です。未保存・未適用。\n{difference}")
        except _CONFIG_ERRORS as exc:
            self.dailyDraftLabel.setText(f"設定を確認してください: {exc}")

    def _save(self):
        try:
            operator, reason = self._approval_details()
            overrides, difference, candidate = self._build_draft()
            for old, new in zip(self.active.profiles, candidate.profiles):
                if old.roi != new.roi and (new.widget_id, new.roi) not in self._verified_rois:
                    raise ValueError(f"点検位置{new.widget_id - 20}の検出範囲を基準画像で確認してください。")
            if not confirm_change(self, "日常点検設定の承認", f"次回点検から適用されます。\n{difference}"):
                return
            self.settings.save_daily_configuration(
                overrides, expected_revision=self.active.revision_id, operator=operator, reason=reason,
            )
        except _CONFIG_ERRORS as exc:
            self.dailyDraftLabel.setText(f"保存できません: {exc}")
            return
        self._reload()
        self.on_saved("daily_inspection")

    def can_leave(self):
        if not self._forms:
            return True
        dialog = QMessageBox(self)
        dialog.setWindowTitle("日常点検設定")
        dialog.setText("未保存の下書きがあります。")
        save = dialog.addButton("保存", QMessageBox.AcceptRole)
        discard = dialog.addButton("破棄", QMessageBox.DestructiveRole)
        cancel = dialog.addButton("キャンセル", QMessageBox.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.exec_()
        button = dialog.clickedButton()
        if button is save:
            self._save()
            return not self._forms
        if button is discard:
            self._reload()
            return True
        return False

    def _rollback(self):
        revision = self.dailyHistoryCombo.currentData()
        try:
            operator, reason = self._approval_details()
            if not revision or not confirm_change(self, "日常点検の設定復元", f"現在の下書きを破棄し、{revision} に戻しますか？"):
                return
            self.settings.rollback_daily_configuration(
                revision, operator=operator, reason=reason,
                expected_revision=self.active.revision_id if self.active else None,
            )
        except _CONFIG_ERRORS as exc:
            self.dailyDraftLabel.setText(f"復元できません: {exc}")
            return
        self._reload()
        self.on_saved("daily_inspection")

    def _approval_details(self):
        operator, reason = self.dailyOperatorEdit.text().strip(), self.dailyReasonEdit.text().strip()
        if not operator or not reason:
            raise ValueError("変更作業者と変更・復元理由を入力してください。")
        return operator, reason

    def set_draft_submitter(self, submit):
        self.submit_draft = submit
        self._update_test_actions()

    def _update_test_actions(self):
        enabled = self.active is not None and self.submit_draft is not None and self._pending is None
        self.dailyCaptureButton.setEnabled(enabled)
        self.dailyTestButton.setEnabled(enabled)

    def _invalidate_reference(self, *_args):
        self._generation += 1
        self.dailyFullPreviewLabel.setText("全体画像（未取得）")
        self.dailyCropPreviewLabel.setText("検出画像（未取得）")
        self.dailyTestResultLabel.setText(
            "設定テスト: 前の要求の完了を待っています…" if self._pending else "設定テスト結果: 未実施"
        )

    def hideEvent(self, event):
        self._invalidate_reference()
        super().hideEvent(event)

    def _request_reference(self, measure):
        if self._pending is not None or self.submit_draft is None:
            return
        try:
            _overrides, _description, candidate = self._build_draft()
            profile = candidate.profile_for_widget(self.dailyReferencePositionCombo.currentData())
            request = DailyDraftRequest.create(profile, measure)
            self._invalidate_reference()
            self._pending = (request.request_id, self._generation)
            self._update_test_actions()
            self.dailyTestResultLabel.setText("設定テスト: 実行中…" if measure else "基準画像を取得中…")
            if not self.submit_draft(request):
                raise ValueError("点検ワーカーが使用中です。完了後に再試行してください。")
        except _CONFIG_ERRORS as exc:
            self._pending = None
            self._update_test_actions()
            self.dailyTestResultLabel.setText(f"設定テスト結果: {exc}")

    def show_draft_result(self, result):
        if self._pending is None or self._pending[0] != result.request.request_id:
            return
        generation = self._pending[1]
        self._pending = None
        self._update_test_actions()
        if generation != self._generation:
            self.dailyTestResultLabel.setText("設定テスト結果: 下書き・画面が変更されたため再実行してください。")
            return
        if result.full_image is not None:
            profile = result.request.profile
            self._verified_rois.add((profile.widget_id, profile.roi))
            self._show_draft_state()
        scope = f"点検位置{result.request.profile.widget_id - 20}"
        title = "設定テスト結果（記録なし）" if result.request.measure else "基準画像（緑: 実効範囲）"
        self.dailyTestResultLabel.setText(f"{title} / {scope}\n{result.message}")
        for label, image in (
            (self.dailyFullPreviewLabel, result.full_image), (self.dailyCropPreviewLabel, result.cropped_image),
        ):
            if image is not None:
                height, width = image.shape[:2]
                qimage = QImage(image.data, width, height, image.strides[0], QImage.Format_BGR888)
                display_image(label, qimage)
