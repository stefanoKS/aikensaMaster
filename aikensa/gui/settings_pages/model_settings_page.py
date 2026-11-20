"""AI model settings page controller (AIモデル設定).

Model assignments and preprocessing live in the git-tracked ``models.yaml`` /
``parts_specifications.yaml``.  Runtime changes would conflict with an active
inference session, so this page reports state instead of editing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QTableWidgetItem, QWidget

from aikensa.core.settings.service import SettingsService
from aikensa.core.settings.yaml_store import SettingsFileError, read_yaml
from aikensa.gui.settings_pages.common import load_settings_ui, show_info


class ModelSettingsPage(QWidget):
    """Show model assignments, confidence and on-disk availability."""

    def __init__(self, settings: Optional[SettingsService] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        load_settings_ui("model_settings.ui", self)
        self.settings = settings or SettingsService()
        self._models_path = self.settings.paths.root / "aikensa" / "core" / "config" / "models.yaml"

        self.refreshModelStatusButton.clicked.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            data = read_yaml(self._models_path)
        except SettingsFileError as exc:
            self.modelStatusLabel.setText(f"models.yaml を読み込めません: {exc}")
            data = None

        if not isinstance(data, dict):
            self.modelTableWidget.setRowCount(0)
            return

        device = str(data.get("device", "-"))
        self.modelDeviceComboBox.blockSignals(True)
        self.modelDeviceComboBox.clear()
        self.modelDeviceComboBox.addItem(device)
        self.modelDeviceComboBox.blockSignals(False)

        models = data.get("models", {}) or {}
        table = self.modelTableWidget
        table.setRowCount(len(models))
        missing = 0
        first_conf = None

        for row, (name, spec) in enumerate(sorted(models.items())):
            spec = spec or {}
            path_text = str(spec.get("path", "") or "")
            resolved = Path(path_text)
            if not resolved.is_absolute():
                resolved = self.settings.paths.root / path_text
            exists = resolved.is_file()
            if not exists:
                missing += 1
            confidence = spec.get("conf")
            if confidence is not None and first_conf is None:
                first_conf = float(confidence)
            values = [
                name,
                str(spec.get("loader") or spec.get("model_type") or "-"),
                path_text,
                "-" if confidence is None else f"{float(confidence):.2f}",
                "あり" if exists else "なし",
            ]
            for column, text in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(text))

        if first_conf is not None:
            self.modelConfidenceSpinBox.setValue(first_conf)

        if missing:
            self.modelStatusLabel.setText(
                f"{len(models)} モデル中 {missing} 件が見つかりません。checkpoints を確認してください。"
            )
        else:
            self.modelStatusLabel.setText(f"{len(models)} モデルすべて配置済みです。")

    def show_models_path(self) -> None:
        show_info(self, "モデル設定ファイル", str(self._models_path))
