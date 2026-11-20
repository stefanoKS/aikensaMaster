"""Group E: Qt Designer .ui validation and settings UI behaviour."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET

import pytest

from aikensa.core.settings.schema import GlobalAlignment
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
    from aikensa.gui.settings_pages import SettingsController

    controller = SettingsController(settings=settings)
    stack = controller.settingsContentStack
    assert stack.count() == 7

    controller.settingsNavList.setCurrentRow(3)
    assert stack.currentIndex() == 3

    controller.settingsNavList.setCurrentRow(0)
    assert stack.currentIndex() == 0


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


def test_history_page_lists_and_rolls_back(qapp, settings):
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
