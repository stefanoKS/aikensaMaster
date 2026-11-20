"""Settings history page controller (設定履歴)."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtWidgets import QTableWidgetItem, QWidget

from aikensa.core.settings.migration import backup_machine_files
from aikensa.core.settings.service import SettingsService
from aikensa.gui.settings_pages.common import load_settings_ui, show_error, show_info


class SettingsHistoryPage(QWidget):
    """List machine-local revisions and roll back to a previous one."""

    def __init__(
        self,
        settings: Optional[SettingsService] = None,
        on_saved: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        load_settings_ui("settings_history.ui", self)
        self.settings = settings or SettingsService()
        self.on_saved = on_saved

        self.refreshHistoryButton.clicked.connect(self.refresh)
        self.rollbackHistoryButton.clicked.connect(self.rollback)
        self.createBackupButton.clicked.connect(self.create_backup)
        self.historyTableWidget.itemSelectionChanged.connect(self._show_detail)
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        revisions = self.settings.history()
        active = self.settings.revisions.active_revision_id()
        table = self.historyTableWidget
        table.setRowCount(len(revisions))
        for row, revision in enumerate(revisions):
            values = [
                revision.revision_id,
                revision.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                revision.operator or "-",
                revision.machine_id or "-",
                " / ".join(revision.changes) or "-",
            ]
            for column, text in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(text))
        self.historyActiveRevisionLabel.setText(f"適用中の版: {active or 'なし'}")
        if revisions:
            table.selectRow(0)
        else:
            self.historyDetailTextEdit.setPlainText("履歴はまだありません。")

    def _selected_revision_id(self) -> Optional[str]:
        row = self.historyTableWidget.currentRow()
        if row < 0:
            return None
        item = self.historyTableWidget.item(row, 0)
        return item.text() if item else None

    def _show_detail(self) -> None:
        revision_id = self._selected_revision_id()
        if revision_id is None:
            return
        revision = self.settings.revisions.get_revision(revision_id)
        if revision is None:
            return
        lines = [
            f"版ID: {revision.revision_id}",
            f"作成日時: {revision.created_at.isoformat()}",
            f"作業者: {revision.operator or '-'}",
            f"端末ID: {revision.machine_id or '-'}",
            f"メモ: {revision.note or '-'}",
            "変更ファイル:",
        ]
        lines.extend(f"  - {path}" for path in revision.changed_files)
        if revision.changes:
            lines.append("変更内容:")
            lines.extend(f"  - {change}" for change in revision.changes)
        self.historyDetailTextEdit.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    def rollback(self) -> None:
        revision_id = self._selected_revision_id()
        if revision_id is None:
            show_info(self, "戻せません", "戻す版を選択してください。")
            return
        try:
            self.settings.rollback(revision_id)
        except Exception as exc:
            show_error(self, "戻せません", str(exc))
            return
        self.refresh()
        if self.on_saved is not None:
            self.on_saved("rollback")
        show_info(self, "戻しました", f"版 {revision_id} を再適用しました。")

    def create_backup(self) -> None:
        try:
            result = backup_machine_files(self.settings.paths, reason="manual_ui")
        except Exception as exc:
            show_error(self, "バックアップ失敗", str(exc))
            return
        if result.ok:
            show_info(
                self,
                "バックアップ完了",
                f"{len(result.copied)} 件を {result.backup_dir} に保存し、checksum を検証しました。",
            )
        else:
            show_error(
                self,
                "バックアップ未検証",
                f"{result.backup_dir} に保存しましたが検証で問題が見つかりました:\n"
                + "\n".join(result.problems),
            )
