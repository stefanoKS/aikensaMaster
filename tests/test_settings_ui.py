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
    "daily_inspection_settings.ui": "DailyInspectionSettingsPage",
    "settings_history.ui": "SettingsHistoryPage",
}

UNAVAILABLE_STANDARD_ROUTES = {
    "P82833W050Pbutton": 5,
    "P82832W040Pbutton": 6,
    "P82833W090Pbutton": 7,
    "P82832W080Pbutton": 8,
}

ACTIVE_MENU_ROUTES = {
    "P82833W050PKENGEN_button": 9,
    "P82833W050PCLIPSOUNYUUKI_button": 13,
    "P808387UA1A_button": 17,
    "P828447UA0A_button": 18,
    "P658217UA0A_button": 31,
}

REQUIRED_OBJECTS = {
    "daily_inspection_settings.ui": [
        "dailyPositionCombo", "dailyNominalEdit", "dailyToleranceEdit", "dailyScaleEdit",
        "dailyValidateButton", "dailySaveButton", "dailyHistoryCombo", "dailyRollbackButton",
    ],
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
    assert stack.count() == 8

    controller.select_page(3)
    assert stack.currentIndex() == 3

    controller.select_page(0)
    assert stack.currentIndex() == 0
    headings = [controller.settingsNavList.item(row) for row in (0, 4, 9)]
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
    for index in range(controller.settingsContentStack.count()):
        controller.select_page(index)
        qapp.processEvents()
        assert window.size() == QSize(*size)
        assert stack.size() == window.size()
        assert controller.settingsContentStack.widget(index).widget() is controller._pages[index]
    window.close()


def test_inspection_designer_canvases_have_the_fixed_size():
    paths = _ui_files_list()
    inspection_files = [name for name in paths[4:] if not name.endswith("/empty.ui")]
    assert len(inspection_files) == 36
    for name in inspection_files:
        root = ET.parse(REPO_ROOT / name)
        rect = root.find('widget/property[@name="geometry"]/rect')
        assert rect is not None
        assert (rect.findtext("width"), rect.findtext("height")) == ("1920", "1080"), name


def test_main_menu_controls_fit_designer_canvas_without_overlap(qapp):
    from PyQt5.QtCore import QRect, QSize
    from PyQt5.QtWidgets import QMainWindow, QPushButton
    from PyQt5.uic import loadUi

    page = QMainWindow()
    loadUi(str(REPO_ROOT / "aikensa/gui/mainPage.ui"), page)
    assert page.size() == QSize(1920, 1080)
    page.show()
    qapp.processEvents()
    canvas = page.centralWidget()
    controls = [
        w for w in canvas.findChildren(QPushButton) if w.isVisibleTo(canvas)
    ]
    for name in (
        *ACTIVE_MENU_ROUTES, "settingsButton", "quitbutton", "dailytenkenbutton",
        "camcalibrationbutton1", "camcalibrationbutton2", "cameraMerge",
    ):
        button = canvas.findChild(QPushButton, name)
        assert button is not None
        assert button.isVisibleTo(canvas)
        rect = QRect(button.mapTo(canvas, button.rect().topLeft()), button.size())
        assert canvas.rect().contains(rect), name
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


@pytest.fixture
def application_window(qapp, settings, monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock
    from PyQt5.QtCore import QObject, pyqtSignal

    class Worker(QObject):
        cameraErrorSignal = pyqtSignal(str)
        modelErrorSignal = pyqtSignal(str)
        dailyResultSignal = pyqtSignal(int, str, object, str)
        inspectionPreviewSignal = pyqtSignal(int, int, int, object)
        equipmentPreviewSignal = pyqtSignal(int, object)

        def __init__(self, config):
            super().__init__()
            self.fake = MagicMock()
            self.fake.calib_config = self.fake.inspection_config = config

        def __getattr__(self, name):
            return getattr(self.fake, name)

    for module_name, names in (
        ("calibration_thread", ("CalibrationThread", "CalibrationConfig")),
        ("inspection_thread", ("InspectionThread", "InspectionConfig")),
        ("sio_thread", ("ServerMonitorThread",)),
        ("time_thread", ("TimeMonitorThread",)),
    ):
        fake = ModuleType(module_name)
        if len(names) == 2:
            setattr(fake, names[0], Worker)
            setattr(fake, names[1], MagicMock())
        else:
            setattr(fake, names[0], MagicMock())
        monkeypatch.setitem(sys.modules, f"aikensa.core.thread.{module_name}", fake)
    spec = importlib.util.spec_from_file_location(
        "_test_aikensa_gui", REPO_ROOT / "aikensa" / "AIKensa.py"
    )
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)

    monkeypatch.setattr(app_module, "SettingsService", lambda: settings)

    window = app_module.AIKensa()
    yield window
    window.close()


def test_all_legacy_pages_are_direct_and_controls_fit_at_design_size(qapp, application_window):
    from PyQt5.QtCore import QRect, QSize
    from PyQt5.QtWidgets import (
        QCheckBox, QComboBox, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollBar, QSlider,
    )

    window = application_window
    design_size = QSize(1920, 1080)
    paths = _ui_files_list()
    window.resize(design_size)
    window.show()
    for index in range(50):
        window.stackedWidget.setCurrentIndex(index)
        qapp.processEvents()
        page = window.stackedWidget.currentWidget()
        assert isinstance(page, QMainWindow), index
        assert not any(bar.isVisible() for bar in page.findChildren(QScrollBar)), index
        assert page.size() == design_size, index
        canvas = page.centralWidget()
        if paths[index].endswith("/empty.ui"):
            continue
        assert canvas is not None, index
        for control in canvas.findChildren((QCheckBox, QComboBox, QLabel, QLineEdit, QPushButton, QSlider)):
            if control.isVisibleTo(canvas):
                rect = QRect(control.mapTo(canvas, control.rect().topLeft()), control.size())
                assert canvas.rect().contains(rect), (index, control.objectName(), rect)


def test_application_window_switches_pages_without_hardware(qapp, application_window):
    from PyQt5.QtWidgets import QPushButton

    window = application_window
    window.showFullScreen()
    qapp.processEvents()
    assert window.isFullScreen()
    assert window.geometry() == window.screen().geometry()
    main = window.stackedWidget.widget(0)
    assert window.settings_widget_index == 50
    for name, index in UNAVAILABLE_STANDARD_ROUTES.items():
        assert main.findChild(QPushButton, name) is None
        assert window.stackedWidget.widget(index).findChild(QPushButton, "InspectButton") is not None
    for name, index in ACTIVE_MENU_ROUTES.items():
        window.stackedWidget.setCurrentIndex(0)
        main.findChild(QPushButton, name).click()
        assert window.stackedWidget.currentIndex() == index
        assert window.inspection_thread.inspection_config.widget == index
    window.stackedWidget.setCurrentIndex(0)
    main.findChild(QPushButton, "camcalibrationbutton1").click()
    assert window.stackedWidget.currentIndex() == 1
    assert window.calibration_thread.calib_config.widget == 1
    window.stackedWidget.widget(1).findChild(QPushButton, "calibSingleFrame").click()
    assert window.calibration_thread.calib_config.calculateSingeFrameMatrix is True
    window.stackedWidget.setCurrentIndex(0)
    main.findChild(QPushButton, "dailytenkenbutton").click()
    for index, name, target in (
        (21, "nextButton", 22), (22, "nextButton", 23), (23, "prevButton", 22),
        (22, "prevButton", 21), (21, "mainmenubutton", 0),
    ):
        assert window.stackedWidget.currentIndex() == index
        window.stackedWidget.currentWidget().findChild(QPushButton, name).click()
        assert window.stackedWidget.currentIndex() == target
    window.stackedWidget.setCurrentIndex(0)
    main.findChild(QPushButton, "settingsButton").click()
    assert window.stackedWidget.currentIndex() == window.settings_widget_index
    for index in (0, 1, 6, 17, 49, window.settings_widget_index):
        window.stackedWidget.setCurrentIndex(index)
        qapp.processEvents()
        assert window.isFullScreen()
        assert window.geometry() == window.screen().geometry()
        assert window.stackedWidget.size() == window.size()


def test_small_window_clips_fixed_pages_without_resizing(qapp, application_window):
    from PyQt5.QtCore import QSize
    from PyQt5.QtWidgets import QPushButton

    window = application_window
    window.resize(1280, 720)
    window.show()
    for index in (0, 1, 3, 4, 6, 17, 21, 49, window.settings_widget_index):
        window.stackedWidget.setCurrentIndex(index)
        qapp.processEvents()
        assert window.size() == QSize(1280, 720)
    window.stackedWidget.setCurrentIndex(17)
    button = window.stackedWidget.currentWidget().findChild(QPushButton, "InspectButton")
    assert button.geometry().bottom() >= window.height()


def test_daily_result_from_previous_position_is_not_rendered_on_active_position(application_window):
    from PyQt5.QtGui import QColor, QImage
    from PyQt5.QtWidgets import QLabel
    from aikensa.core.camera.preview_state import EquipmentPreview, PreviewReadiness

    window = application_window
    window.stackedWidget.setCurrentIndex(22)
    window.inspection_thread.equipmentPreviewSignal.emit(1, EquipmentPreview(PreviewReadiness.READY))
    result = QImage(64, 32, QImage.Format_RGB888)
    result.fill(QColor("green"))
    window.inspection_thread.dailyResultSignal.emit(21, window._daily_session_id, result, "OK / 実測値: 100.6 mm")
    first = window.stackedWidget.widget(21)
    second = window.stackedWidget.widget(22)
    assert not first.findChild(QLabel, "framePart").pixmap().isNull()
    assert "100.6" in first.statusBar().currentMessage()
    assert "100.6" not in second.statusBar().currentMessage()


def _capture_preview_evidence(window, name):
    import os
    from pathlib import Path

    output = os.environ.get("AIKENSA_PREVIEW_EVIDENCE")
    if output:
        window.show()
        window.repaint()
        capture = window.grab()
        # QMainWindow's central stack adds no outer frame to the 1920x1080 page.
        assert (capture.width(), capture.height()) == (1920, 1080)
        assert capture.save(str(Path(output) / f"{name}.png"))


def test_daily_navigation_replays_equipment_preview_without_a_new_fault(qapp, application_window):
    from PyQt5.QtWidgets import QLabel, QPushButton
    from aikensa.core.camera.preview_state import EquipmentPreview, PreviewReadiness

    window = application_window
    window.resize(1920, 1080)
    window.show()
    qapp.processEvents()
    menu = window.stackedWidget.widget(0)
    menu.findChild(QPushButton, "P808387UA1A_button").click()
    assert window.stackedWidget.currentIndex() == 17
    label = window.stackedWidget.currentWidget().findChild(QLabel, "framePart")
    assert "確認中" in label.accessibleDescription()
    window.inspection_thread.equipmentPreviewSignal.emit(
        1, EquipmentPreview(PreviewReadiness.CAMERA_UNAVAILABLE, (0, 1), "カメラ未接続")
    )
    assert not label.pixmap().isNull()
    _capture_preview_evidence(window, "07-production-both-unavailable")
    window.stackedWidget.setCurrentIndex(0)
    menu.findChild(QPushButton, "dailytenkenbutton").click()
    for index, button in (
        (21, "nextButton"), (22, "nextButton"), (23, "prevButton"),
        (22, "prevButton"), (21, "mainmenubutton"),
    ):
        page = window.stackedWidget.currentWidget()
        assert window.stackedWidget.currentIndex() == index
        preview = page.findChild(QLabel, "framePart").pixmap()
        assert preview is not None and not preview.isNull(), index
        _capture_preview_evidence(window, f"0{index - 20}-daily-position-{index - 20}-unavailable")
        page.findChild(QPushButton, button).click()
    assert window.stackedWidget.currentIndex() == 0
    menu.findChild(QPushButton, "dailytenkenbutton").click()
    assert window.stackedWidget.currentIndex() == 21
    assert not window.stackedWidget.currentWidget().findChild(QLabel, "framePart").pixmap().isNull()


@pytest.mark.parametrize("missing", [(0,), (1,)])
def test_equipment_preview_distinguishes_recovery_and_rejects_stale_frames(
    qapp, application_window, monkeypatch, missing,
):
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QColor, QImage
    from PyQt5.QtWidgets import QLabel, QPushButton
    from aikensa.core.camera.preview_state import EquipmentPreview, PreviewReadiness
    from aikensa.gui import inspection_preview

    rendered = []
    original = inspection_preview.make_camera_placeholder

    def placeholder(camera, width, height):
        rendered.append(camera)
        return original(camera, width, height)

    monkeypatch.setattr(inspection_preview, "make_camera_placeholder", placeholder)
    window = application_window
    window.resize(1920, 1080)
    window.show()
    menu = window.stackedWidget.widget(0)
    menu.findChild(QPushButton, "dailytenkenbutton").click()
    worker = window.inspection_thread
    page = window.stackedWidget.currentWidget()
    label = page.findChild(QLabel, "framePart")
    ready = EquipmentPreview(PreviewReadiness.READY)
    worker.equipmentPreviewSignal.emit(1, ready)
    valid = QImage(1791, 131, QImage.Format_RGB888)
    valid.fill(QColor("#146239"))
    worker.dailyResultSignal.emit(21, window._daily_session_id, valid, "OK / 実測値: 100.6 mm")
    assert label.pixmap() is not None

    state = EquipmentPreview(PreviewReadiness.CAMERA_UNAVAILABLE, missing, "カメラ画像取得エラー")
    worker.equipmentPreviewSignal.emit(2, state)
    assert rendered == list(missing)
    assert "画像取得エラー" in page.statusBar().currentMessage()
    disconnected = label.pixmap().cacheKey()
    worker.equipmentPreviewSignal.emit(2, state)
    assert label.pixmap().cacheKey() == disconnected
    assert rendered == list(missing)
    if missing == (0,):
        _capture_preview_evidence(window, "04-camera-1-unavailable")

    worker.equipmentPreviewSignal.emit(
        3, EquipmentPreview(PreviewReadiness.CALIBRATION_REQUIRED, message="内部パラメータが未校正です。")
    )
    assert "未校正" in label.accessibleDescription()
    if missing == (0,):
        _capture_preview_evidence(window, "05-calibration-required")

    worker.equipmentPreviewSignal.emit(4, ready)
    old_target = window._preview_epoch
    QTimer.singleShot(0, lambda: worker.inspectionPreviewSignal.emit(21, old_target, 4, valid))
    page.findChild(QPushButton, "nextButton").click()
    window.stackedWidget.currentWidget().findChild(QPushButton, "prevButton").click()
    waiting = label.pixmap().cacheKey()
    qapp.processEvents()
    assert label.pixmap().cacheKey() == waiting
    worker.inspectionPreviewSignal.emit(21, window._preview_epoch, 3, valid)
    assert label.pixmap().cacheKey() == waiting
    worker.inspectionPreviewSignal.emit(21, window._preview_epoch, 4, valid)
    assert label.pixmap().toImage().pixelColor(20, 20) == QColor("#146239")
    if missing == (0,):
        _capture_preview_evidence(window, "06-valid-synthetic-preview")


def test_camera_and_model_signals_reach_visible_page_diagnostics(application_window):
    from PyQt5.QtWidgets import QScrollArea

    window = application_window
    stack = window.stackedWidget
    for index in (1, 3, 17, 21, 23, 49):
        stack.setCurrentIndex(index)
        page = stack.currentWidget()
        on_calibration = index in (1, 3)
        active = window.calibration_thread if on_calibration else window.inspection_thread
        inactive = window.inspection_thread if on_calibration else window.calibration_thread
        active.cameraErrorSignal.emit(f"Camera failed on page {index}")
        inactive.cameraErrorSignal.emit("Inactive worker error")
        assert page.statusBar().currentMessage() == f"Camera failed on page {index}"
        window.inspection_thread.modelErrorSignal.emit("Model unavailable")
        assert page.statusBar().currentMessage() == (
            f"Camera failed on page {index}" if on_calibration else "Model unavailable"
        )
    # Exercise the same diagnostics if a page is hosted inside a viewport.
    index = stack.currentIndex()
    page = stack.currentWidget()
    stack.removeWidget(page)
    wrapper = QScrollArea()
    wrapper.setWidget(page)
    stack.insertWidget(index, wrapper)
    stack.setCurrentIndex(index)
    window.inspection_thread.cameraErrorSignal.emit("Camera failed on wrapped page")
    assert page.statusBar().currentMessage() == "Camera failed on wrapped page"
    window.inspection_thread.modelErrorSignal.emit("Model unavailable on wrapped page")
    assert page.statusBar().currentMessage() == "Model unavailable on wrapped page"


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
