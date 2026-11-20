"""Shared helpers for the settings page controllers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.uic import loadUi

UI_DIR = Path(__file__).resolve().parents[1] / "pages" / "SETTINGS"


def ui_path(name: str) -> str:
    return str(UI_DIR / name)


def load_settings_ui(name: str, widget: QWidget) -> QWidget:
    """Load a settings ``.ui`` file into *widget* (a QWidget subclass)."""

    loadUi(ui_path(name), widget)
    return widget


def show_info(parent: Optional[QWidget], title_ja: str, message_ja: str) -> None:
    QMessageBox.information(parent, title_ja, message_ja)


def show_error(parent: Optional[QWidget], title_ja: str, message_ja: str) -> None:
    QMessageBox.critical(parent, title_ja, message_ja)


def format_issues(issues: Iterable) -> str:
    lines: List[str] = []
    for issue in issues:
        severity = getattr(issue, "severity", None)
        prefix = "エラー" if str(severity).endswith("ERROR") or "error" in str(severity) else "注意"
        lines.append(f"[{prefix}] {issue.message_ja}")
    return "\n".join(lines)
