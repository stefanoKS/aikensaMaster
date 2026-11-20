"""Inspection conditions page controller (検査条件).

This page deliberately presents part specifications read-only.  Tolerance
changes affect the OK/NG judgement and must be reviewed against reference
samples, which cannot be done safely from an operator screen (see the STOP
conditions in the refactoring brief).
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QTableWidgetItem, QWidget

from aikensa.core.settings.part_registry import load_part_registry_entries
from aikensa.core.settings.service import SettingsService
from aikensa.core.settings.yaml_store import SettingsFileError, read_yaml
from aikensa.gui.settings_pages.common import load_settings_ui, show_info



class InspectionSettingsPage(QWidget):
    """Show nominal values, tolerances and reference measurements per part."""

    def __init__(self, settings: Optional[SettingsService] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        load_settings_ui("inspection_settings.ui", self)
        self.settings = settings or SettingsService()
        self._specs_path = self.settings.paths.root / "aikensa" / "core" / "config" / "parts_specifications.yaml"

        self._entries = load_part_registry_entries(self.settings.paths.root)
        self._specs = self._load_specs()

        for widget_id, part_number, label in self._entries:
            key = part_number
            if key in self._specs:
                self.inspectionPartComboBox.addItem(f"{part_number} ({widget_id})", key)

        self.inspectionPartComboBox.currentIndexChanged.connect(self.refresh)
        self.reloadInspectionSettingsButton.clicked.connect(self._reload)
        self.inspectionOpenSpecButton.clicked.connect(self.show_spec_path)
        self.refresh()

    # ------------------------------------------------------------------
    def _load_specs(self) -> dict:
        try:
            data = read_yaml(self._specs_path)
        except SettingsFileError as exc:
            self.inspectionInfoLabel.setText(f"仕様ファイルを読み込めません: {exc}")
            return {}
        if not isinstance(data, dict):
            return {}
        return data.get("parts", {}) or {}

    def _reload(self) -> None:
        self._specs = self._load_specs()
        current = self.inspectionPartComboBox.currentData()
        self.inspectionPartComboBox.blockSignals(True)
        self.inspectionPartComboBox.clear()
        for widget_id, part_number, _label in self._entries:
            if part_number in self._specs:
                self.inspectionPartComboBox.addItem(f"{part_number} ({widget_id})", part_number)
        self.inspectionPartComboBox.blockSignals(False)
        if current is not None:
            position = self.inspectionPartComboBox.findData(current)
            if position >= 0:
                self.inspectionPartComboBox.setCurrentIndex(position)
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self, *_args) -> None:
        key = self.inspectionPartComboBox.currentData()
        table = self.measurementTableWidget
        if key is None:
            table.setRowCount(0)
            self.inspectionInfoLabel.setText("この品番の検査仕様は parts_specifications.yaml にありません。")
            return

        spec = self._specs.get(key, {}) or {}
        pitch_spec = list(spec.get("pitchSpec", []) or [])
        tolerances = list(spec.get("tolerance_pitch", []) or [])
        id_spec = list(spec.get("idSpec", []) or [])
        multiplier = spec.get("pixelMultiplier")

        self.inspectionVariantLabel.setText(f"LH/RH: {spec.get('variant', '-')}")
        table.setRowCount(len(pitch_spec))
        for row, nominal in enumerate(pitch_spec):
            tolerance = tolerances[row] if row < len(tolerances) else None
            lower = f"{nominal - tolerance:.2f}" if tolerance is not None else "-"
            upper = f"{nominal + tolerance:.2f}" if tolerance is not None else "-"
            unit = "px" if multiplier in (None, 1, 1.0) else "mm"
            values = [
                f"P{row + 1}",
                f"{nominal}",
                lower,
                upper,
                "-",
                unit,
                "-",
                "ピッチ",
            ]
            for column, text in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(text))

        self.inspectionInfoLabel.setText(
            f"{key}: ピッチ {len(pitch_spec)} 点 / ID仕様 {len(id_spec)} 点 / "
            f"pixelMultiplier={multiplier} ({self._specs_path.name})"
        )

    def show_spec_path(self) -> None:
        show_info(self, "仕様ファイル", str(self._specs_path))
