"""Global fine adjustment page controller (全体微調整)."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QWidget

import numpy as np

from aikensa.core.settings.schema import GlobalAlignment, Severity
from aikensa.core.settings.service import SettingsService
from aikensa.core.settings.validator import validate_global_alignment
from aikensa.core.settings.yaml_store import SettingsValidationError
from aikensa.gui.settings_pages.common import load_settings_ui, show_error, show_info

NUDGE_STEPS = (0.5, 1.0, 5.0)


class FineAdjustmentPage(QWidget):
    """Edit the machine-wide camera-2 offset without touching the base homography."""

    def __init__(
        self,
        settings: Optional[SettingsService] = None,
        on_saved: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        load_settings_ui("fine_adjustment.ui", self)
        self.settings = settings or SettingsService()
        self.on_saved = on_saved
        self._active = GlobalAlignment()
        self._reference: Optional[np.ndarray] = None
        self._reference_pixmap: Optional[QPixmap] = None

        for spin in (self.globalOffsetXSpinBox, self.globalOffsetYSpinBox, self.globalRotationSpinBox):
            spin.valueChanged.connect(self._on_value_changed)

        self.nudgeLeftButton.clicked.connect(lambda: self._nudge(-1.0, 0.0))
        self.nudgeRightButton.clicked.connect(lambda: self._nudge(1.0, 0.0))
        self.nudgeUpButton.clicked.connect(lambda: self._nudge(0.0, -1.0))
        self.nudgeDownButton.clicked.connect(lambda: self._nudge(0.0, 1.0))
        self.nudgeCenterButton.clicked.connect(self._reset_offsets)
        self.saveFineAdjustmentButton.clicked.connect(self.save)
        self.cancelFineAdjustmentButton.clicked.connect(self.load)
        self.restoreFineAdjustmentButton.clicked.connect(self.restore_defaults)
        self.overlayOpacitySlider.valueChanged.connect(self._render_preview)
        self.zoomSlider.valueChanged.connect(self._render_preview)
        self.beforeAfterCheckBox.toggled.connect(self._render_preview)
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        self._active = self.settings.active_global_alignment()
        self.globalOffsetXSpinBox.blockSignals(True)
        self.globalOffsetYSpinBox.blockSignals(True)
        self.globalRotationSpinBox.blockSignals(True)
        self.globalOffsetXSpinBox.setValue(float(self._active.cam2_x_px))
        self.globalOffsetYSpinBox.setValue(float(self._active.cam2_y_px))
        self.globalRotationSpinBox.setValue(float(self._active.cam2_rotation_deg))
        self.globalOffsetXSpinBox.blockSignals(False)
        self.globalOffsetYSpinBox.blockSignals(False)
        self.globalRotationSpinBox.blockSignals(False)
        self.globalOffsetXSpinBox.setMaximum(self._active.max_abs_shift_px)
        self.globalOffsetXSpinBox.setMinimum(-self._active.max_abs_shift_px)
        self.globalOffsetYSpinBox.setMaximum(self._active.max_abs_shift_px)
        self.globalOffsetYSpinBox.setMinimum(-self._active.max_abs_shift_px)
        self.globalRotationSpinBox.setMaximum(self._active.max_abs_rotation_deg)
        self.globalRotationSpinBox.setMinimum(-self._active.max_abs_rotation_deg)
        self.fineAdjustmentRangeLabel.setText(
            f"許容範囲: ±{self._active.max_abs_shift_px:.0f} px / "
            f"±{self._active.max_abs_rotation_deg:.2f} deg"
        )
        self._update_draft_label()
        self._render_preview()

    def collect(self) -> GlobalAlignment:
        return GlobalAlignment(
            cam2_x_px=float(self.globalOffsetXSpinBox.value()),
            cam2_y_px=float(self.globalOffsetYSpinBox.value()),
            cam2_rotation_deg=float(self.globalRotationSpinBox.value()),
            pivot_px=self._active.pivot_px,
            max_abs_shift_px=self._active.max_abs_shift_px,
            max_abs_rotation_deg=self._active.max_abs_rotation_deg,
        )

    # ------------------------------------------------------------------
    def _nudge(self, sign_x: float, sign_y: float) -> None:
        step = NUDGE_STEPS[max(0, self.nudgeStepComboBox.currentIndex())]
        self.globalOffsetXSpinBox.setValue(
            self.globalOffsetXSpinBox.value() + sign_x * step
        )
        self.globalOffsetYSpinBox.setValue(
            self.globalOffsetYSpinBox.value() + sign_y * step
        )

    def _reset_offsets(self) -> None:
        self.globalOffsetXSpinBox.setValue(0.0)
        self.globalOffsetYSpinBox.setValue(0.0)

    def _on_value_changed(self, _value: float) -> None:
        self._update_draft_label()
        self._render_preview()

    def _update_draft_label(self) -> None:
        draft = self.collect()
        if self._is_dirty(draft):
            self.fineAdjustmentDraftLabel.setText(
                "未保存の変更があります (保存するまで検査には適用されません)"
            )
        else:
            self.fineAdjustmentDraftLabel.setText("変更なし")

    def _is_dirty(self, draft: GlobalAlignment) -> bool:
        return (
            abs(draft.cam2_x_px - self._active.cam2_x_px) > 1e-9
            or abs(draft.cam2_y_px - self._active.cam2_y_px) > 1e-9
            or abs(draft.cam2_rotation_deg - self._active.cam2_rotation_deg) > 1e-9
        )

    # ------------------------------------------------------------------
    def save(self) -> bool:
        draft = self.collect()
        errors = [i for i in validate_global_alignment(draft) if i.severity is Severity.ERROR]
        if errors:
            show_error(self, "保存できません", "\n".join(i.message_ja for i in errors))
            return False
        try:
            self.settings.save_global_alignment(draft, note="fine adjustment UI")
        except SettingsValidationError as exc:
            show_error(self, "保存できません", "\n".join(i.message_ja for i in exc.issues))
            return False
        except OSError as exc:
            show_error(self, "保存できません", f"ファイルに書き込めません: {exc}")
            return False

        self._active = draft
        self._update_draft_label()
        if self.on_saved is not None:
            self.on_saved("global_alignment")
        show_info(self, "保存しました", "全体微調整を保存しました。")
        return True

    def restore_defaults(self) -> None:
        self.globalOffsetXSpinBox.setValue(0.0)
        self.globalOffsetYSpinBox.setValue(0.0)
        self.globalRotationSpinBox.setValue(0.0)

    # ------------------------------------------------------------------
    def set_reference_image(self, image: np.ndarray) -> None:
        """Supply the merged preview image as a numpy array."""

        self._reference = image
        self._reference_pixmap = None
        self._render_preview()

    def set_reference_qimage(self, image) -> None:
        """Supply the merged preview straight from a signal (QImage)."""

        if image is None or image.isNull():
            self._reference = self._reference_pixmap = None
            self.finePreviewLabel.clear()
            return
        self._reference = None
        self._reference_pixmap = QPixmap.fromImage(image)
        self._render_preview()

    def _render_preview(self, *_args) -> None:
        if self._reference_pixmap is not None:
            self._render_pixmap_preview()
            return
        if self._reference is None:
            return
        draft = self.collect()
        image = self._reference
        shift_x = 0.0 if self.beforeAfterCheckBox.isChecked() else draft.cam2_x_px
        shift_y = 0.0 if self.beforeAfterCheckBox.isChecked() else draft.cam2_y_px

        if shift_x or shift_y:
            matrix = np.array(
                [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            import cv2

            image = cv2.warpAffine(
                image, matrix[:2], (image.shape[1], image.shape[0])
            )

        rgb = image[:, :, ::-1] if image.ndim == 3 else image
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        zoom = max(0.1, self.zoomSlider.value() / 100.0)
        target = self.finePreviewLabel.size() * zoom
        self.finePreviewLabel.setPixmap(
            pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _render_pixmap_preview(self) -> None:
        """Scale the incoming frame and draw a reference crosshair.

        The crosshair follows the draft offset (and shows the un-corrected
        position when 補正前を表示 is checked), so the operator gets immediate
        feedback on direction and magnitude before saving.
        """

        base = self._reference_pixmap
        if base is None:
            return

        zoom = max(0.1, self.zoomSlider.value() / 100.0)
        scaled = base.scaled(
            self.finePreviewLabel.size() * zoom, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if scaled.isNull():
            return

        draft = self.collect()
        shift_x = 0.0 if self.beforeAfterCheckBox.isChecked() else draft.cam2_x_px
        shift_y = 0.0 if self.beforeAfterCheckBox.isChecked() else draft.cam2_y_px

        displayed_w = scaled.width()
        displayed_h = scaled.height()
        if base.width() > 0 and base.height() > 0:
            ratio_x = displayed_w / float(base.width())
            ratio_y = displayed_h / float(base.height())
        else:  # pragma: no cover - defensive
            ratio_x = ratio_y = 1.0

        canvas = QPixmap(scaled.size())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.setOpacity(self.overlayOpacitySlider.value() / 100.0)
        painter.drawPixmap(0, 0, scaled)
        painter.setOpacity(1.0)

        center_x = displayed_w / 2.0 + shift_x * ratio_x
        center_y = displayed_h / 2.0 + shift_y * ratio_y
        pen = QPen(QColor(252, 233, 79))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(int(center_x) - 30, int(center_y), int(center_x) + 30, int(center_y))
        painter.drawLine(int(center_x), int(center_y) - 30, int(center_x), int(center_y) + 30)
        painter.end()

        self.finePreviewLabel.setPixmap(canvas)
