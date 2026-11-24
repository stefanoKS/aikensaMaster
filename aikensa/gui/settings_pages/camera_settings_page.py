"""Camera settings page controller (カメラ設定)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget

from aikensa.core.camera.camera_service import CameraRole, CameraUnavailableError
from aikensa.core.camera.placeholder import make_camera_placeholder
from aikensa.core.camera.registry import get_camera_service
from aikensa.core.settings.repository import SettingsRepository
from aikensa.core.settings.schema import (
    CameraAssignment,
    CameraHardwareSettings,
    CameraSettings,
    Severity,
)
from aikensa.core.settings.service import SettingsService
from aikensa.core.settings.validator import validate_camera_settings
from aikensa.core.settings.yaml_store import SettingsValidationError
from aikensa.gui.settings_pages.common import load_settings_ui, show_error, show_info

RESOLUTIONS = [
    (1280, 960),
    (1920, 1080),
    (2048, 1536),
    (2592, 1944),
    (3072, 2048),
    (3840, 2160),
]


class CameraSettingsPage(QWidget):
    """Assign physical cameras and edit their operating parameters."""

    def __init__(self, settings: Optional[SettingsService] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        load_settings_ui("camera_settings.ui", self)
        self.settings = settings or SettingsService()
        self._populate_resolutions()
        for index in (0, 1):
            self._ctrl(index, "TestButton").clicked.connect(
                lambda _checked=False, i=index: self.test_connection(i)
            )
        self.refreshCamerasButton.clicked.connect(self.refresh_cameras)
        self.saveCameraSettingsButton.clicked.connect(self.save)
        self.revertCameraSettingsButton.clicked.connect(self.load)
        self.restoreCameraDefaultsButton.clicked.connect(self.restore_defaults)
        self.load()

    # ------------------------------------------------------------------
    def _ctrl(self, index: int, suffix: str):
        name = f"camera{index}{suffix}"
        widget = self.findChild(QWidget, name)
        if widget is None:
            raise AttributeError(f"widget {name} missing from camera_settings.ui")
        return widget

    def _populate_resolutions(self) -> None:
        for index in (0, 1):
            combo = self._ctrl(index, "ResolutionComboBox")
            combo.clear()
            for width, height in RESOLUTIONS:
                combo.addItem(f"{width} x {height}", (width, height))

    # ------------------------------------------------------------------
    def load(self) -> None:
        cameras, issues = self.settings.repository.load_camera_settings()
        for index in (0, 1):
            assignment = cameras.assignment(index)
            hardware = cameras.hardware_for(index)

            serial = self._ctrl(index, "SerialComboBox")
            serial.setEditText(assignment.normalized_serial() or "")

            combo = self._ctrl(index, "ResolutionComboBox")
            position = combo.findData((hardware.width_px, hardware.height_px))
            if position < 0:
                combo.addItem(
                    f"{hardware.width_px} x {hardware.height_px}",
                    (hardware.width_px, hardware.height_px),
                )
                position = combo.count() - 1
            combo.setCurrentIndex(position)

            self._ctrl(index, "FpsSpinBox").setValue(int(round(hardware.fps)))
            self._ctrl(index, "ExposureSpinBox").setValue(float(hardware.exposure_us))
            self._ctrl(index, "GainSpinBox").setValue(float(hardware.gain_db))
            self._ctrl(index, "WbTemperatureSpinBox").setValue(int(hardware.wb_temperature_k))
            self._ctrl(index, "WbRatioRSpinBox").setValue(float(hardware.wb_ratio_r or 1.0))
            self._ctrl(index, "WbRatioGSpinBox").setValue(float(hardware.wb_ratio_g or 1.0))
            self._ctrl(index, "WbRatioBSpinBox").setValue(float(hardware.wb_ratio_b or 1.0))
            self._ctrl(index, "Rotate180CheckBox").setChecked(bool(hardware.rotate_180))

            status = self._ctrl(index, "StatusLabel")
            if assignment.is_assigned:
                status.setText("設定済み (未接続確認)")
            else:
                status.setText("未設定")
            self._ctrl(index, "ReadbackLabel").clear()
            self._show_placeholder(index)

        self._set_validation_text(issues)

    # ------------------------------------------------------------------
    def collect(self) -> CameraSettings:
        cameras = CameraSettings.unconfigured()
        for index in (0, 1):
            serial = self._ctrl(index, "SerialComboBox").currentText().strip()
            cameras.assignments[index] = CameraAssignment(index=index, serial=serial or None)

            width, height = self._ctrl(index, "ResolutionComboBox").currentData()
            hardware = CameraHardwareSettings(index=index)
            hardware.width_px = int(width)
            hardware.height_px = int(height)
            hardware.fps = float(self._ctrl(index, "FpsSpinBox").value())
            hardware.exposure_us = float(self._ctrl(index, "ExposureSpinBox").value())
            hardware.gain_db = float(self._ctrl(index, "GainSpinBox").value())
            hardware.wb_temperature_k = int(self._ctrl(index, "WbTemperatureSpinBox").value())
            hardware.wb_ratio_r = float(self._ctrl(index, "WbRatioRSpinBox").value())
            hardware.wb_ratio_g = float(self._ctrl(index, "WbRatioGSpinBox").value())
            hardware.wb_ratio_b = float(self._ctrl(index, "WbRatioBSpinBox").value())
            hardware.rotate_180 = bool(self._ctrl(index, "Rotate180CheckBox").isChecked())
            cameras.hardware[index] = hardware

        cameras.rotate_180_global = cameras.hardware_for(0).rotate_180
        return cameras

    def save(self) -> bool:
        cameras = self.collect()
        issues = validate_camera_settings(cameras)
        errors = [i for i in issues if i.severity is Severity.ERROR]
        if errors:
            self._set_validation_text(issues)
            show_error(self, "保存できません", "\n".join(i.message_ja for i in errors))
            return False
        try:
            self.settings.save_camera_settings(cameras)
        except SettingsValidationError as exc:
            show_error(self, "保存できません", "\n".join(i.message_ja for i in exc.issues))
            return False
        except OSError as exc:
            show_error(self, "保存できません", f"ファイルに書き込めません: {exc}")
            return False

        self._set_validation_text(issues)
        show_info(self, "保存しました", "カメラ設定を保存しました。次回の検査開始時から適用されます。")
        return True

    # ------------------------------------------------------------------
    def restore_defaults(self) -> None:
        """Load the git-tracked template into the form (nothing is saved yet)."""

        repository = SettingsRepository(self.settings.paths)
        template = repository.load_shared_template_camera_settings()
        for index in (0, 1):
            self._ctrl(index, "SerialComboBox").setEditText("")
            hardware = template.hardware_for(index)
            combo = self._ctrl(index, "ResolutionComboBox")
            position = combo.findData((hardware.width_px, hardware.height_px))
            if position >= 0:
                combo.setCurrentIndex(position)
            self._ctrl(index, "FpsSpinBox").setValue(int(round(hardware.fps)))
            self._ctrl(index, "ExposureSpinBox").setValue(float(hardware.exposure_us))
            self._ctrl(index, "GainSpinBox").setValue(float(hardware.gain_db))
            self._ctrl(index, "WbTemperatureSpinBox").setValue(int(hardware.wb_temperature_k))
            self._ctrl(index, "WbRatioRSpinBox").setValue(float(hardware.wb_ratio_r or 1.0))
            self._ctrl(index, "WbRatioGSpinBox").setValue(float(hardware.wb_ratio_g or 1.0))
            self._ctrl(index, "WbRatioBSpinBox").setValue(float(hardware.wb_ratio_b or 1.0))
            self._ctrl(index, "Rotate180CheckBox").setChecked(bool(hardware.rotate_180))
        self._set_validation_text(
            [i for i in validate_camera_settings(self.collect()) if i.severity is Severity.WARNING]
        )

    # ------------------------------------------------------------------
    def discover_serials(self) -> List[str]:
        """Enumerate IC4 serial numbers; returns an empty list when unavailable."""

        try:
            import imagingcontrol4 as ic4  # type: ignore
        except Exception:
            self.cameraDiscoveryStatusLabel.setText(
                "IC4 ライブラリが見つかりません (手入力してください)"
            )
            return []

        try:
            ic4.Library.init_context()
            devices = ic4.DeviceEnum.devices()
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.cameraDiscoveryStatusLabel.setText(f"検索失敗: {exc}")
            return []

        serials = [str(getattr(device, "serial", "") or "") for device in devices]
        serials = [s for s in serials if s]
        self.cameraDiscoveryStatusLabel.setText(
            f"{len(serials)} 台検出: " + (", ".join(serials) if serials else "なし")
        )
        return serials

    def refresh_cameras(self) -> None:
        serials = self.discover_serials()
        if not serials:
            return
        for index in (0, 1):
            combo = self._ctrl(index, "SerialComboBox")
            current = combo.currentText()
            combo.clear()
            combo.addItems(serials)
            combo.setEditText(current)

    # ------------------------------------------------------------------
    def test_connection(self, index: int) -> None:
        """Open the camera once through the shared service and read a frame."""

        service = get_camera_service(self.settings)
        self._ctrl(index, "ReadbackLabel").clear()
        try:
            handle = service.acquire(index, CameraRole.CAMERA_PREVIEW)
        except CameraUnavailableError as exc:
            self._ctrl(index, "StatusLabel").setText(exc.reason_ja)
            self._show_placeholder(index)
            return

        try:
            frame = service.read_real_frame(index, CameraRole.CAMERA_PREVIEW)
            if frame is None:
                status = "カメラ未接続" if handle.is_placeholder else "画像取得エラー"
                self._ctrl(index, "StatusLabel").setText(f"{status}。接続・電源を確認してください。")
                self._show_placeholder(index)
                return
            image = frame.image
            self._ctrl(index, "StatusLabel").setText(f"接続OK ({image.shape[1]}x{image.shape[0]})")
            self._show_preview(index, image)
            readback = handle.readback or {}
            self._ctrl(index, "ReadbackLabel").setText(
                "解像度 {width}x{height} / FPS {fps} / 露光 {exposure_us} μs / "
                "ゲイン {gain_db} dB / WB {wb_temperature_k} K".format(
                    width=_fmt(readback.get("width")),
                    height=_fmt(readback.get("height")),
                    fps=_fmt(readback.get("fps")),
                    exposure_us=_fmt(readback.get("exposure_us")),
                    gain_db=_fmt(readback.get("gain_db")),
                    wb_temperature_k=_fmt(readback.get("wb_temperature_k")),
                )
            )
        finally:
            service.release(index, CameraRole.CAMERA_PREVIEW)

    def _show_placeholder(self, index: int) -> None:
        width, height = self._ctrl(index, "ResolutionComboBox").currentData()
        scale = min(1.0, 640 / width, 480 / height)
        self._show_preview(index, make_camera_placeholder(index, round(width * scale), round(height * scale)))

    def _show_preview(self, index: int, frame: np.ndarray) -> None:
        label = self._ctrl(index, "PreviewLabel")
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def _set_validation_text(self, issues) -> None:
        if not issues:
            self.cameraValidationLabel.setText("入力内容に問題はありません。")
            return
        lines = []
        for issue in issues:
            prefix = "エラー" if issue.severity is Severity.ERROR else "注意"
            lines.append(f"[{prefix}] {issue.message_ja}")
        self.cameraValidationLabel.setText("\n".join(lines))


def _fmt(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)
