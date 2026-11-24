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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QListWidgetItem, QPlainTextEdit,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from aikensa.core.settings.migration import backup_machine_files
from aikensa.core.settings.service import SettingsService
from aikensa.gui.settings_pages.calibration_settings_page import CalibrationSettingsPage
from aikensa.gui.settings_pages.camera_settings_page import CameraSettingsPage
from aikensa.gui.settings_pages.common import load_settings_ui, show_error
from aikensa.gui.settings_pages.daily_inspection_settings_page import DailyInspectionSettingsPage
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
        self._preview_qimage = None
        self._preview_full_size = None
        self._readiness_details = ""

        self._nav_pages = [self.settingsNavList.item(index) for index in range(len(self._pages))]
        for index, item in enumerate(self._nav_pages):
            item.setData(Qt.UserRole, index)
        for index, title in reversed(((0, "設備・カメラ"), (3, "検査条件"), (7, "管理"))):
            heading = QListWidgetItem(title)
            heading.setFlags(Qt.NoItemFlags)
            heading.setForeground(QColor("#aab3bc"))
            self.settingsNavList.insertItem(index, heading)
        self.settingsNavList.currentItemChanged.connect(self._select_page)
        self.backToMainButton.clicked.connect(self._go_back)
        self.reloadSettingsButton.clicked.connect(self.refresh_all)
        self.settingsStatusDetailsButton.clicked.connect(self._show_readiness_details)

        self.select_page(0)
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
        self.daily_page = DailyInspectionSettingsPage(self.settings, saved, self)
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
            self.daily_page,
            self.history_page,
        ):
            scroll = QScrollArea()
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            stack.addWidget(scroll)
            self._pages.append(page)

        for label in (
            self.camera_page.camera0PreviewLabel,
            self.camera_page.camera1PreviewLabel,
            self.fine_page.finePreviewLabel,
            self.widget_page.widgetFullPreviewLabel,
            self.widget_page.widgetCropPreviewLabel,
        ):
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    # ------------------------------------------------------------------
    def select_page(self, index: int) -> None:
        self.settingsNavList.setCurrentItem(self._nav_pages[index])

    def _select_page(self, item, _previous) -> None:
        index = None if item is None else item.data(Qt.UserRole)
        if index is not None:
            self.settingsContentStack.setCurrentIndex(index)
            if self._preview_qimage is not None:
                self.set_preview_qimage(self._preview_qimage, self._preview_full_size)

    def _go_back(self) -> None:
        if not self.daily_page.can_leave():
            return
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
            header = "検査準備: 要確認"
        self.settingsStatusLabel.setText(header)
        self._readiness_details = "\n".join(lines)

    def _show_readiness_details(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("準備状況の詳細")
        details = QPlainTextEdit(self._readiness_details, dialog)
        details.setReadOnly(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dialog)
        buttons.button(QDialogButtonBox.Close).setText("閉じる")
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(details)
        layout.addWidget(buttons)
        dialog.resize(660, 380)
        dialog.exec_()

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

    def set_preview_qimage(self, image, full_size=None) -> None:
        """Feed the merged frame from a thread signal (QImage) when visible.

        The guard keeps a 15 fps signal from repainting hidden pages.
        """

        if image is None or image.isNull():
            self._preview_qimage = None
            self._preview_full_size = None
            self.fine_page.set_reference_qimage(image)
            self.widget_page.set_reference_qimage(image)
            return
        if not self.isVisible():
            return
        index = self.settingsContentStack.currentIndex()
        if index == 2:
            self._preview_qimage = image
            self._preview_full_size = full_size
            self.fine_page.set_reference_qimage(image)
        elif index == 3:
            self._preview_qimage = image
            self._preview_full_size = full_size
            self.widget_page.set_reference_qimage(image, full_size)
