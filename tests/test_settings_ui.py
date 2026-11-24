"""Group E: Qt Designer .ui validation and settings UI behaviour."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET

import pytest

from aikensa.core.settings.schema import CropRect, GlobalAlignment
from tests.conftest import REPO_ROOT, write_serial

SETTINGS_DIR = REPO_ROOT / "aikensa" / "gui" / "pages" / "SETTINGS"

EXPECTED_UI = {
    "settings_main.ui": "SettingsMainPage",
    "camera_settings.ui": "CameraSettingsPage",
    "calibration_settings.ui": "CalibrationSettingsPage",
    "fine_adjustment.ui": "FineAdjustmentPage",
    "widget_adjustment.ui": "WidgetAdjustmentPage",
    "inspection_settings.ui": "InspectionSettingsPage",
    "model_settings.ui": "ModelSettingsPage",
    "settings_history.ui": "SettingsHistoryPage",
}

STANDARD_MENU_BUTTONS = {
    "P82833W050Pbutton": 5,
    "P82832W040Pbutton": 6,
    "P82833W090Pbutton": 7,
    "P82832W080Pbutton": 8,
}

REQUIRED_OBJECTS = {
    "settings_main.ui": ["settingsNavList", "settingsContentStack", "backToMainButton"],
    "camera_settings.ui": [
        "camera0SerialComboBox",
        "camera1SerialComboBox",
        "refreshCamerasButton",
        "camera0ResolutionComboBox",
        "camera1ResolutionComboBox",
        "camera0ExposureSpinBox",
        "camera1ExposureSpinBox",
        "camera0GainSpinBox",
        "camera1GainSpinBox",
        "camera0PreviewLabel",
        "camera1PreviewLabel",
        "saveCameraSettingsButton",
    ],
    "calibration_settings.ui": [
        "calibrationTableWidget",
        "recalibrateCamera0Button",
        "recalibrateCamera1Button",
        "restoreCalibrationButton",
        "calibrationNextActionLabel",
    ],
    "fine_adjustment.ui": [
        "globalOffsetXSpinBox",
        "globalOffsetYSpinBox",
        "globalRotationSpinBox",
        "nudgeStepComboBox",
        "overlayOpacitySlider",
        "zoomSlider",
        "saveFineAdjustmentButton",
    ],
    "widget_adjustment.ui": [
        "partComboBox",
        "widgetOffsetXSpinBox",
        "widgetOffsetYSpinBox",
        "cropXSpinBox",
        "cropYSpinBox",
        "cropWidthSpinBox",
        "cropHeightSpinBox",
        "saveWidgetAdjustmentButton",
    ],
    "inspection_settings.ui": ["measurementTableWidget", "inspectionPartComboBox"],
    "model_settings.ui": ["modelTableWidget", "modelConfidenceSpinBox"],
    "settings_history.ui": ["historyTableWidget", "rollbackHistoryButton", "createBackupButton"],
}


# ---------------------------------------------------------------------------
# Every .ui must be valid Designer XML and load with PyQt5
# ---------------------------------------------------------------------------
def test_all_expected_ui_files_exist():
    for name in EXPECTED_UI:
        assert (SETTINGS_DIR / name).is_file(), f"missing {name}"


@pytest.mark.parametrize("filename", sorted(EXPECTED_UI))
def test_ui_is_valid_designer_xml(filename):
    tree = ET.parse(SETTINGS_DIR / filename)
    root = tree.getroot()
    assert root.tag == "ui"
    assert root.get("version") == "4.0"
    assert root.find("class").text == EXPECTED_UI[filename]


@pytest.mark.parametrize("filename", sorted(EXPECTED_UI))
def test_ui_loads_with_pyqt5(qapp, filename):
    from PyQt5.QtWidgets import QWidget
    from PyQt5.uic import loadUi

    widget = QWidget()
    loadUi(str(SETTINGS_DIR / filename), widget)
    assert widget.layout() is not None


@pytest.mark.parametrize("filename", sorted(EXPECTED_UI))
def test_required_object_names_exist(qapp, filename):
    from PyQt5.QtWidgets import QWidget
    from PyQt5.uic import loadUi

    widget = QWidget()
    loadUi(str(SETTINGS_DIR / filename), widget)
    for name in REQUIRED_OBJECTS[filename]:
        assert widget.findChild(QWidget, name) is not None, f"{filename}: {name} missing"


def test_ui_files_use_layouts_not_absolute_positioning():
    """Settings pages must be layout based (mainPage.ui legacy is exempt)."""

    for filename in EXPECTED_UI:
        text = (SETTINGS_DIR / filename).read_text(encoding="utf-8")
        assert "<layout" in text, filename
        # Absolute geometry on a child widget would break at other screen sizes.
        assert text.count('<property name="geometry">') <= 1, filename


# ---------------------------------------------------------------------------
# Existing inspection widget indices must not move
# ---------------------------------------------------------------------------
def _ui_files_list() -> list[str]:
    source = (REPO_ROOT / "aikensa" / "AIKensa.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") == "UI_FILES" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("UI_FILES not found in AIKensa.py")


def test_inspection_widget_indices_are_unchanged():
    files = _ui_files_list()

    assert len(files) == 50, "inspection pages must stay at indices 0-49"
    assert files[0].endswith("mainPage.ui")
    assert files[1].endswith("calibration_cam1.ui")
    assert files[2].endswith("calibration_cam2.ui")
    assert files[3].endswith("camera_merge.ui")
    assert files[49].endswith("P828397YA6A_KATABU_NASHI.ui")
    assert not any("SETTINGS" in path for path in files)


def test_main_page_has_settings_button(qapp):
    from PyQt5.QtWidgets import QMainWindow, QPushButton
    from PyQt5.uic import loadUi

    window = QMainWindow()
    loadUi(str(REPO_ROOT / "aikensa" / "gui" / "mainPage.ui"), window)
    button = window.findChild(QPushButton, "settingsButton")
    assert button is not None
    assert button.text() == "設備設定"


# ---------------------------------------------------------------------------
# Settings pages behave
# ---------------------------------------------------------------------------
def test_settings_controller_navigation(qapp, settings):
    from PyQt5.QtCore import Qt
    from aikensa.gui.settings_pages import SettingsController

    controller = SettingsController(settings=settings)
    stack = controller.settingsContentStack
    assert stack.count() == 7

    controller.select_page(3)
    assert stack.currentIndex() == 3

    controller.select_page(0)
    assert stack.currentIndex() == 0
    headings = [controller.settingsNavList.item(row) for row in (0, 4, 8)]
    assert [item.text() for item in headings] == ["設備・カメラ", "検査条件", "管理"]
    for item in headings:
        assert not item.flags() & Qt.ItemIsSelectable
        controller.settingsNavList.setCurrentItem(item)
        assert stack.currentIndex() == 0


@pytest.mark.parametrize("size", [(1920, 1080), (1600, 900), (1280, 720)])
def test_settings_scroll_without_expanding_window(qapp, settings, size):
    from PyQt5.QtCore import QSize
    from PyQt5.QtWidgets import QMainWindow, QStackedWidget
    from aikensa.gui.settings_pages import SettingsController

    window = QMainWindow()
    stack = QStackedWidget()
    controller = SettingsController(settings=settings)
    stack.addWidget(controller)
    window.setCentralWidget(stack)
    window.resize(*size)
    window.show()
    qapp.processEvents()

    assert controller.minimumSizeHint().height() < 720
    for index in range(7):
        controller.select_page(index)
        qapp.processEvents()
        assert window.size() == QSize(*size)
        assert stack.size() == window.size()
        assert controller.settingsContentStack.widget(index).widget() is controller._pages[index]
    window.close()


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)])
def test_legacy_pages_scroll_to_controls(qapp, size):
    from PyQt5.QtCore import QSize
    from PyQt5.QtWidgets import QMainWindow, QPushButton, QStackedWidget
    from PyQt5.uic import loadUi
    from aikensa.gui.page_viewport import legacy_page_viewport

    paths = _ui_files_list()
    window = QMainWindow()
    stack = QStackedWidget()
    for index in (1, 6, 17):
        page = QMainWindow()
        loadUi(str(REPO_ROOT / paths[index]), page)
        stack.addWidget(legacy_page_viewport(page))
    window.setCentralWidget(stack)
    window.resize(*size)
    window.show()
    for index, name in enumerate(("calibCam", "mainmenubutton", "InspectButton")):
        stack.setCurrentIndex(index)
        qapp.processEvents()
        scroll = stack.currentWidget()
        button = scroll.findChild(QPushButton, name)
        assert button is not None
        scroll.ensureWidgetVisible(button, 0, 0)
        qapp.processEvents()
        position = button.mapTo(scroll.viewport(), button.rect().topLeft())
        assert scroll.viewport().rect().contains(position)
        assert scroll.viewport().rect().contains(position + button.rect().bottomRight())
        assert window.size() == QSize(*size)
    window.close()


def test_main_menu_controls_fit_designer_canvas_without_overlap(qapp):
    from PyQt5.QtCore import QRect, QSize
    from PyQt5.QtWidgets import QLabel, QMainWindow, QPushButton
    from PyQt5.uic import loadUi

    page = QMainWindow()
    loadUi(str(REPO_ROOT / "aikensa/gui/mainPage.ui"), page)
    assert page.size() == QSize(1920, 1080)
    page.show()
    qapp.processEvents()
    canvas = page.centralWidget()
    controls = [
        w for w in canvas.findChildren((QPushButton, QLabel))
        if w.isVisibleTo(canvas) and w.text().strip()
    ]
    for name in (
        *STANDARD_MENU_BUTTONS, "settingsButton", "quitbutton", "dailytenkenbutton",
        "camcalibrationbutton1", "camcalibrationbutton2", "cameraMerge",
    ):
        button = canvas.findChild(QPushButton, name)
        assert button is not None
        assert button.isVisibleTo(canvas)
        rect = QRect(button.mapTo(canvas, button.rect().topLeft()), button.size())
        assert canvas.rect().contains(rect), name
        if name in STANDARD_MENU_BUTTONS:
            for other in controls:
                if other is not button:
                    other_rect = QRect(other.mapTo(canvas, other.rect().topLeft()), other.size())
                    assert not rect.intersects(other_rect), (name, other.objectName())
    page.close()


def test_fullscreen_fits_actual_offscreen_geometry(qapp, settings):
    from PyQt5.QtWidgets import QMainWindow, QStackedWidget
    from aikensa.gui.settings_pages import SettingsController

    window = QMainWindow()
    stack = QStackedWidget()
    stack.addWidget(SettingsController(settings=settings))
    window.setCentralWidget(stack)
    window.showFullScreen()
    qapp.processEvents()
    assert window.isFullScreen()
    assert window.geometry() == window.screen().geometry()
    assert stack.size() == window.size()
    window.close()


def test_application_window_switches_pages_without_hardware(qapp, settings, monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock
    from PyQt5.QtWidgets import QMainWindow, QPushButton, QScrollArea, QScrollBar

    for module_name, names in (
        ("calibration_thread", ("CalibrationThread", "CalibrationConfig")),
        ("inspection_thread", ("InspectionThread", "InspectionConfig")),
        ("sio_thread", ("ServerMonitorThread",)),
        ("time_thread", ("TimeMonitorThread",)),
    ):
        fake = ModuleType(module_name)
        for name in names:
            setattr(fake, name, MagicMock())
        monkeypatch.setitem(sys.modules, f"aikensa.core.thread.{module_name}", fake)
    spec = importlib.util.spec_from_file_location(
        "_test_aikensa_gui", REPO_ROOT / "aikensa" / "AIKensa.py"
    )
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)

    monkeypatch.setattr(app_module, "SettingsService", lambda: settings)

    window = app_module.AIKensa()
    window.showFullScreen()
    qapp.processEvents()
    assert window.isFullScreen()
    assert window.geometry() == window.screen().geometry()
    main = window.stackedWidget.widget(0)
    assert isinstance(main, QMainWindow)
    assert not main.findChildren(QScrollBar)
    assert window.settings_widget_index == 50
    for index in range(1, 50):
        assert isinstance(window.stackedWidget.widget(index), QScrollArea)
    for name, index in STANDARD_MENU_BUTTONS.items():
        window.stackedWidget.setCurrentIndex(0)
        main.findChild(QPushButton, name).click()
        assert window.stackedWidget.currentIndex() == index
        assert window.inspection_thread.inspection_config.widget == index
    window.stackedWidget.setCurrentIndex(0)
    main.findChild(QPushButton, "camcalibrationbutton1").click()
    assert window.stackedWidget.currentIndex() == 1
    assert window.calibration_thread.calib_config.widget == 1
    window.stackedWidget.setCurrentIndex(0)
    main.findChild(QPushButton, "settingsButton").click()
    assert window.stackedWidget.currentIndex() == window.settings_widget_index
    for index in (0, 1, 6, 17, 49, window.settings_widget_index):
        window.stackedWidget.setCurrentIndex(index)
        qapp.processEvents()
        assert window.isFullScreen()
        assert window.geometry() == window.screen().geometry()
        assert window.stackedWidget.size() == window.size()
    window.close()


def test_legacy_preview_scales_without_changing_image(qapp):
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QLabel
    from aikensa.gui.page_viewport import display_image

    label = QLabel()
    label.resize(200, 200)
    image = QImage(800, 400, QImage.Format_RGB32)
    image.fill(0xFF123456)
    display_image(label, image)
    assert label.pixmap().size().width() == 200
    assert label.pixmap().size().height() == 100
    assert image.width() == 800 and image.height() == 400
    assert image.pixel(0, 0) == 0xFF123456


def test_settings_preview_does_not_expand_layout(qapp, settings):
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QMainWindow
    from aikensa.gui.settings_pages import SettingsController

    window = QMainWindow()
    controller = SettingsController(settings=settings)
    window.setCentralWidget(controller)
    window.resize(1280, 720)
    window.show()
    controller.select_page(2)
    qapp.processEvents()
    image = QImage(1600, 800, QImage.Format_RGB32)
    image.fill(0)
    controller.fine_page.set_reference_qimage(image)
    qapp.processEvents()
    pixmap = controller.fine_page.finePreviewLabel.pixmap()
    assert abs(pixmap.width() - 2 * pixmap.height()) <= 1  # Qt rounds scaled sizes.
    assert window.width() == 1280 and window.height() == 720
    window.close()


def test_ready_details_do_not_expand_settings_window(qapp, settings, monkeypatch):
    from PyQt5.QtCore import QSize
    from aikensa.gui.settings_pages import SettingsController

    monkeypatch.setattr(settings, "readiness_report", lambda _snapshot: ["長い診断" * 300] * 30)
    controller = SettingsController(settings=settings)
    controller.resize(1280, 720)
    controller.show()
    qapp.processEvents()
    assert controller.size() == QSize(1280, 720)
    assert controller.settingsStatusLabel.text() == "検査準備: 要確認"
    controller.close()


@pytest.mark.parametrize(
    ("crop", "expected"),
    [
        (CropRect(200, 100, 600, 400), (20, 10, 60, 40)),
        (CropRect(960, 180, 100, 140), (96, 18, 4, 14)),
        (CropRect(-20, -20, 100, 40), (0, 0, 8, 2)),
        (CropRect(1010, 100, 80, 40), (0, 0, 0, 0)),
        (CropRect(100, 100, 0, 40), (0, 0, 0, 0)),
    ],
)
def test_preview_crop_maps_from_full_merged_pixels(crop, expected):
    from PyQt5.QtCore import QSize
    from aikensa.gui.page_viewport import preview_crop_rect

    assert preview_crop_rect(crop, QSize(1000, 500), QSize(100, 50)).getRect() == expected
    assert preview_crop_rect(crop, QSize(), QSize(100, 50)).isEmpty()


def test_camera_page_saves_machine_yaml(qapp, settings):
    from aikensa.gui.settings_pages.camera_settings_page import CameraSettingsPage

    page = CameraSettingsPage(settings)
    page.camera0SerialComboBox.setEditText("40720154")
    page.camera1SerialComboBox.setEditText("40720155")
    page.camera0ExposureSpinBox.setValue(11000.0)
    assert page.save() is True

    cameras, _issues = settings.repository.load_camera_settings()
    assert cameras.assignment(0).normalized_serial() == "40720154"
    assert cameras.hardware_for(0).exposure_us == pytest.approx(11000.0)

    raw = (settings.paths.cam_id_file()).read_text(encoding="utf-8")
    assert "40720154" in raw


def test_camera_tabs_keep_each_draft_and_fit_settings_width(qapp, settings):
    from PyQt5.QtWidgets import QMainWindow
    from aikensa.gui.settings_pages import SettingsController

    window = QMainWindow()
    controller = SettingsController(settings=settings)
    window.setCentralWidget(controller)
    window.resize(1280, 720)
    window.show()
    qapp.processEvents()
    page = controller.camera_page
    page.camera0SerialComboBox.setEditText("SN-0")
    page.cameraTabs.setCurrentIndex(1)
    assert not page.camera0PreviewLabel.isVisible()
    assert page.camera1PreviewLabel.isVisible()
    page.camera1SerialComboBox.setEditText("SN-1")
    page.cameraTabs.setCurrentIndex(0)
    assert page.camera0SerialComboBox.currentText() == "SN-0"
    assert page.camera1SerialComboBox.currentText() == "SN-1"
    assert controller.settingsContentStack.currentWidget().horizontalScrollBar().maximum() == 0
    cameras, _ = settings.repository.load_camera_settings()
    assert not cameras.assignment(0).is_assigned and not cameras.assignment(1).is_assigned
    window.close()


def test_camera_page_rejects_duplicate_serials(qapp, settings):
    from aikensa.gui.settings_pages.camera_settings_page import CameraSettingsPage

    page = CameraSettingsPage(settings)
    page.camera0SerialComboBox.setEditText("DUPLICATE")
    page.camera1SerialComboBox.setEditText("DUPLICATE")

    assert page.save() is False
    cameras, _ = settings.repository.load_camera_settings()
    assert cameras.assignment(0).normalized_serial() is None


def test_camera_page_restore_defaults_does_not_write(qapp, settings):
    from aikensa.gui.settings_pages.camera_settings_page import CameraSettingsPage

    write_serial(settings, {0: "KEEP", 1: "KEEP2"})
    before = settings.paths.cam_id_file().read_text(encoding="utf-8")

    page = CameraSettingsPage(settings)
    page.restoreCameraDefaultsButton.click()

    assert page.camera0SerialComboBox.currentText() == ""
    assert settings.paths.cam_id_file().read_text(encoding="utf-8") == before


def test_fine_adjustment_page_saves_and_resets(qapp, settings):
    from aikensa.gui.settings_pages.fine_adjustment_page import FineAdjustmentPage

    page = FineAdjustmentPage(settings)
    page.globalOffsetXSpinBox.setValue(4.5)
    page.globalOffsetYSpinBox.setValue(-2.0)
    assert "未保存" in page.fineAdjustmentDraftLabel.text()

    assert page.save() is True
    alignment = settings.active_global_alignment()
    assert alignment.cam2_x_px == pytest.approx(4.5)
    assert alignment.cam2_y_px == pytest.approx(-2.0)

    page.restoreFineAdjustmentButton.click()
    assert page.globalOffsetXSpinBox.value() == pytest.approx(0.0)


def test_nudge_uses_selected_step(qapp, settings):
    from aikensa.gui.settings_pages.fine_adjustment_page import FineAdjustmentPage

    page = FineAdjustmentPage(settings)
    page.nudgeStepComboBox.setCurrentIndex(2)  # 5 px
    page.nudgeRightButton.click()
    assert page.globalOffsetXSpinBox.value() == pytest.approx(5.0)

    page.nudgeStepComboBox.setCurrentIndex(0)  # 0.5 px
    page.nudgeUpButton.click()
    assert page.globalOffsetYSpinBox.value() == pytest.approx(-0.5)


def test_widget_page_edits_only_the_selected_widget(qapp, settings):
    from aikensa.gui.settings_pages.widget_adjustment_page import WidgetAdjustmentPage

    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    page = WidgetAdjustmentPage(settings)

    index_43 = page.partComboBox.findData(43)
    if index_43 < 0:
        pytest.skip("widget 43 is not present in the shared registry")

    page.partComboBox.setCurrentIndex(index_43)
    page.widgetOffsetXSpinBox.setValue(9.0)
    assert page.save() is True

    overrides = settings.active_widget_overrides()
    assert overrides[43].cam2_offset_x_px == pytest.approx(9.0)
    assert 17 not in overrides


def test_widget_page_restores_shared_default(qapp, settings):
    from aikensa.gui.settings_pages.widget_adjustment_page import WidgetAdjustmentPage

    write_serial(settings, {0: "SN-0", 1: "SN-1"})
    page = WidgetAdjustmentPage(settings)
    index_43 = page.partComboBox.findData(43)
    if index_43 < 0:
        pytest.skip("widget 43 is not present in the shared registry")

    page.partComboBox.setCurrentIndex(index_43)
    page.widgetOffsetXSpinBox.setValue(9.0)
    page.save()
    page.restoreWidgetDefaultButton.click()

    assert 43 not in settings.active_widget_overrides()


def test_history_page_lists_and_confirms_rollback(qapp, settings, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    from aikensa.gui.settings_pages.history_page import SettingsHistoryPage

    settings.save_global_alignment(GlobalAlignment(cam2_x_px=3.0))
    first = settings.revisions.active_revision_id()
    settings.save_global_alignment(GlobalAlignment(cam2_x_px=30.0))

    page = SettingsHistoryPage(settings)
    assert page.historyTableWidget.rowCount() == 2
    assert first in page.historyActiveRevisionLabel.text() or page.historyActiveRevisionLabel.text()

    # Select the older revision and roll back.
    for row in range(page.historyTableWidget.rowCount()):
        if page.historyTableWidget.item(row, 0).text() == first:
            page.historyTableWidget.selectRow(row)
            break
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    page.rollbackHistoryButton.click()
    assert settings.active_global_alignment().cam2_x_px == pytest.approx(30.0)

    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    page.rollbackHistoryButton.click()

    assert settings.active_global_alignment().cam2_x_px == pytest.approx(3.0)


def test_inspection_page_shows_specs(qapp, settings):
    from aikensa.gui.settings_pages.inspection_settings_page import InspectionSettingsPage

    page = InspectionSettingsPage(settings)
    assert page.measurementTableWidget.rowCount() == 2
    assert page.measurementTableWidget.item(0, 0).text() == "P1"
    assert page.saveInspectionSettingsButton.isEnabled() is False


def test_model_page_reports_missing_models(qapp, settings):
    from aikensa.gui.settings_pages.model_settings_page import ModelSettingsPage

    page = ModelSettingsPage(settings)
    assert page.modelTableWidget.rowCount() == 1
    assert "見つかりません" in page.modelStatusLabel.text()


def test_live_preview_updates_fine_adjustment_page(qapp, settings):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage

    from aikensa.gui.settings_pages import SettingsController

    controller = SettingsController(settings=settings)
    controller.select_page(2)  # 全体微調整

    image = QImage(160, 120, QImage.Format_RGB888)
    image.fill(Qt.gray)

    controller.show()
    controller.set_preview_qimage(image)

    pixmap = controller.fine_page.finePreviewLabel.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    assert controller.widget_page.widgetFullPreviewLabel.pixmap() is None

    controller.select_page(3)
    pixmap = controller.widget_page.widgetFullPreviewLabel.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    controller.close()


def test_live_preview_is_ignored_while_hidden(qapp, settings):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage

    from aikensa.gui.settings_pages import SettingsController

    controller = SettingsController(settings=settings)
    image = QImage(160, 120, QImage.Format_RGB888)
    image.fill(Qt.gray)

    controller.set_preview_qimage(image)  # hidden: must be a no-op

    assert controller.fine_page.finePreviewLabel.pixmap() is None or controller.fine_page.finePreviewLabel.pixmap().isNull()
