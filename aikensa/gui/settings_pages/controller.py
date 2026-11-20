"""Settings hub controller.

``settings_main.ui`` is a real QStackedWidget-based nested interface: a
navigation list on the left and the individual settings pages on the right.
The controller loads each page ``.ui``, inserts it into the nested stack and
keeps the stack index in sync with the navigation list.

The hub itself is appended to the application's main ``QStackedWidget`` after
the existing inspection pages, so indices 0-49 keep their meaning.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from PyQt5.QtWidgets import QWidget

from aikensa.core.settings.migration import backup_machine_files
from aikensa.core.settings.service import SettingsService
from aikensa.gui.settings_pages.calibration_settings_page import CalibrationSettingsPage
from aikensa.gui.settings_pages.camera_settings_page import CameraSettingsPage
from aikensa.gui.settings_pages.common import load_settings_ui, show_error
from aikensa.gui.settings_pages.fine_adjustment_page import FineAdjustmentPage
from aikensa.gui.settings_pages.history_page import SettingsHistoryPage
from aikensa.gui.settings_pages.inspection_settings_page import InspectionSettingsPage
from aikensa.gui.settings_pages.model_settings_page import ModelSettingsPage
from aikensa.gui.settings_pages.widget_adjustment_page import WidgetAdjustmentPage


class SettingsController(QWidget):
    """The 設備設定 page plus its child pages."""

    def __init__(
        self,
        settings: Optional[SettingsService] = None,
        on_back: Optional[Callable[[], None]] = None,
        on_open_calibration: Optional[Callable[[int], None]] = None,
        on_settings_saved: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        load_settings_ui("settings_main.ui", self)
        self.settings = settings or SettingsService()
        self.on_back = on_back
        self.on_open_calibration = on_open_calibration
        self.on_settings_saved = on_settings_saved

        self._pages: List[QWidget] = []
        self._build_pages()

        self.settingsNavList.currentRowChanged.connect(self._select_page)
        self.backToMainButton.clicked.connect(self._go_back)
        self.reloadSettingsButton.clicked.connect(self.refresh_all)

        self.settingsNavList.setCurrentRow(0)
        self.refresh_all()

    # ------------------------------------------------------------------
    def _build_pages(self) -> None:
        saved = self._handle_saved

        self.camera_page = CameraSettingsPage(self.settings, self)
        self.calibration_page = CalibrationSettingsPage(
            self.settings, self.on_open_calibration, self
        )
        self.fine_page = FineAdjustmentPage(self.settings, saved, self)
        self.widget_page = WidgetAdjustmentPage(self.settings, saved, self)
        self.inspection_page = InspectionSettingsPage(self.settings, self)
        self.model_page = ModelSettingsPage(self.settings, self)
        self.history_page = SettingsHistoryPage(self.settings, saved, self)

        stack = self.settingsContentStack
        # Replace the design-time placeholder with the real pages.
        while stack.count() > 0:
            page = stack.widget(0)
            stack.removeWidget(page)
            page.deleteLater()

        for page in (
            self.camera_page,
            self.calibration_page,
            self.fine_page,
            self.widget_page,
            self.inspection_page,
            self.model_page,
            self.history_page,
        ):
            stack.addWidget(page)
            self._pages.append(page)

    # ------------------------------------------------------------------
    def _select_page(self, row: int) -> None:
        if 0 <= row < len(self._pages):
            self.settingsContentStack.setCurrentIndex(row)

    def _go_back(self) -> None:
        if self.on_back is not None:
            self.on_back()

    def _handle_saved(self, what: str) -> None:
        self.refresh_status()
        if self.on_settings_saved is not None:
            self.on_settings_saved(what)

    # ------------------------------------------------------------------
    def refresh_status(self) -> None:
        snapshot = self.settings.load_snapshot()
        lines = self.settings.readiness_report(snapshot)
        if self.settings.production_ready(snapshot):
            header = "検査準備: 完了"
        else:
            header = "検査準備: 未完了 (校正またはカメラ設定が必要です)"
        self.settingsStatusLabel.setText(header + "\n" + "\n".join(lines))

    def refresh_all(self) -> None:
        self.refresh_status()
        for page in self._pages:
            refresh = getattr(page, "refresh", None)
            if callable(refresh):
                try:
                    refresh()
                except TypeError:
                    pass
        self.camera_page.load()
        self.fine_page.load()
        if self.widget_page.current_widget_id() is not None:
            self.widget_page.load_current()

    # ------------------------------------------------------------------
    def ensure_backup(self) -> bool:
        """Create and verify a backup of machine-specific files."""

        result = backup_machine_files(self.settings.paths, reason="pre_settings_change")
        if result.ok:
            return True
        show_error(
            self,
            "バックアップ未検証",
            "バックアップを検証できませんでした:\n" + "\n".join(result.problems),
        )
        return False

    def set_preview_image(self, image) -> None:
        """Feed a numpy merged image to the preview widgets."""

        self.fine_page.set_reference_image(image)
        self.widget_page.set_reference_image(image)

    def set_preview_qimage(self, image) -> None:
        """Feed the merged frame from a thread signal (QImage) when visible.

        The guard keeps a 15 fps signal from repainting hidden pages.
        """

        if not self.isVisible():
            return
        if self.settingsContentStack.currentIndex() not in (2, 3):
            return
        self.fine_page.set_reference_qimage(image)
        self.widget_page.set_reference_qimage(image)
