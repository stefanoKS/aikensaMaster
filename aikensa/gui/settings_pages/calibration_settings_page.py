"""Calibration status page controller (校正・位置調整)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtWidgets import QTableWidgetItem, QWidget

from aikensa.core.settings.migration import (
    list_backups,
    restore_machine_files,
    verify_backup,
)
from aikensa.core.settings.schema import CalibrationState
from aikensa.core.settings.service import SettingsService
from aikensa.gui.settings_pages.common import load_settings_ui, show_error, show_info

_NEXT_ACTION_JA = {
    CalibrationState.MISSING: "校正を実行してください",
    CalibrationState.UNCONFIGURED: "校正を実行してください",
    CalibrationState.INVALID: "再校正してください",
    CalibrationState.INCOMPATIBLE: "カメラ設定が変わったため再校正が必要です",
    CalibrationState.READY: "対応不要",
}


class CalibrationSettingsPage(QWidget):
    """Show calibration readiness and launch recalibration."""

    def __init__(
        self,
        settings: Optional[SettingsService] = None,
        open_calibration: Optional[Callable[[int], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        load_settings_ui("calibration_settings.ui", self)
        self.settings = settings or SettingsService()
        self.open_calibration = open_calibration

        self.refreshCalibrationButton.clicked.connect(self.refresh)
        self.recalibrateCamera0Button.clicked.connect(lambda: self._recalibrate(0))
        self.recalibrateCamera1Button.clicked.connect(lambda: self._recalibrate(1))
        self.reviewCalibrationButton.clicked.connect(self.review)
        self.restoreCalibrationButton.clicked.connect(self.restore_backup)
        self.refresh()

    # ------------------------------------------------------------------
    def _recalibrate(self, camera_index: int) -> None:
        if self.open_calibration is None:
            show_info(
                self,
                "校正画面",
                "校正画面を開けませんでした。メイン画面から カメラ校正 を選択してください。",
            )
            return
        self.open_calibration(camera_index)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        snapshot = self.settings.load_snapshot()
        metadata = self.settings.load_calibration_metadata()
        table = self.calibrationTableWidget
        table.setRowCount(2)

        for index in (0, 1):
            assignment = snapshot.cameras.assignment(index)
            intrinsic = snapshot.intrinsics.get(index)
            homography = snapshot.homographies.get(index)
            state = self.settings.camera_state(snapshot, index)
            record = metadata.get(str(index)) or {}

            values = [
                assignment.label_ja,
                assignment.normalized_serial() or "未設定",
                intrinsic.state.label_ja if intrinsic else "なし",
                homography.state.label_ja if homography else "なし",
                "-",
                str(record.get("recorded_at") or "-"),
                _NEXT_ACTION_JA.get(state, "-"),
            ]
            for column, text in enumerate(values):
                table.setItem(index, column, QTableWidgetItem(text))

        report = self.settings.readiness_report(snapshot)
        self.calibrationNextActionLabel.setText("\n".join(report))

        self.calibrationCamera0FilesValueLabel.setText(self._file_summary(0))
        self.calibrationCamera1FilesValueLabel.setText(self._file_summary(1))

        issues = snapshot.issues
        if issues:
            self.calibrationIssuesTextEdit.setPlainText(
                "\n".join(f"[{i.severity.value}] {i.message_ja}" for i in issues)
            )
        else:
            self.calibrationIssuesTextEdit.setPlainText("問題は検出されていません。")

    def _file_summary(self, index: int) -> str:
        paths = self.settings.paths
        entries = [
            paths.calibration_file(index),
            paths.calibration_scaled_file(index),
            paths.homography_file(index),
            paths.homography_file(index, scaled=True),
        ]
        lines = []
        for path in entries:
            exists = "あり" if Path(path).is_file() else "なし"
            lines.append(f"{Path(path).name}: {exists}")
        return " / ".join(lines)

    # ------------------------------------------------------------------
    def review(self) -> None:
        snapshot = self.settings.load_snapshot()
        lines = []
        for index in (0, 1):
            intrinsic = snapshot.intrinsics.get(index)
            homography = snapshot.homographies.get(index)
            lines.append(f"--- カメラ{index + 1} ---")
            lines.append(
                "内部校正: "
                + (intrinsic.state.label_ja if intrinsic else "なし")
                + (
                    f" (焦点距離 fx={intrinsic.camera_matrix[0, 0]:.1f}, fy={intrinsic.camera_matrix[1, 1]:.1f})"
                    if intrinsic is not None and intrinsic.camera_matrix is not None
                    else ""
                )
            )
            lines.append(
                "ホモグラフィ: "
                + (homography.state.label_ja if homography else "なし")
                + (f" ({homography.source_path})" if homography else "")
            )
        show_info(self, "校正結果", "\n".join(lines))

    def restore_backup(self) -> None:
        backups = list_backups(None, self.settings.paths)
        if not backups:
            show_info(
                self,
                "バックアップなし",
                f"バックアップが見つかりません: {self.settings.paths.backup_dir}",
            )
            return
        latest = backups[0]
        problems = verify_backup(latest)
        if problems:
            show_error(self, "復元できません", "バックアップが検証できません:\n" + "\n".join(problems))
            return
        try:
            restored = restore_machine_files(self.settings.paths, latest)
        except Exception as exc:
            show_error(self, "復元できません", str(exc))
            return
        self.refresh()
        show_info(
            self,
            "復元しました",
            f"{len(restored)} 件のファイルを {latest.name} から復元しました。",
        )
