"""Per-part / per-widget adjustment page controller (品番別設定)."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QInputDialog, QWidget

from aikensa.core.settings.schema import CropRect, Severity, WidgetAlignment
from aikensa.core.settings.service import SettingsService
from aikensa.core.settings.validator import validate_widget_alignment
from aikensa.core.settings.yaml_store import SettingsValidationError
from aikensa.core.settings.part_registry import load_part_registry_entries
from aikensa.gui.settings_pages.common import load_settings_ui, show_error, show_info


class WidgetAdjustmentPage(QWidget):
    """Edit one widget's camera-2 offset and crop, isolated per part."""

    def __init__(
        self,
        settings: Optional[SettingsService] = None,
        on_saved: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        load_settings_ui("widget_adjustment.ui", self)
        self.settings = settings or SettingsService()
        self.on_saved = on_saved

        self._parts = load_part_registry_entries(self.settings.paths.root)
        self._defaults = self.settings.repository.shared_widget_defaults()
        self._overrides: Dict[int, WidgetAlignment] = self.settings.active_widget_overrides()
        self._current: Optional[WidgetAlignment] = None
        self._reference: Optional[np.ndarray] = None
        self._reference_pixmap = None

        for widget_id, part_number, label in self._parts:
            self.partComboBox.addItem(f"{widget_id}: {label} ({part_number})", widget_id)

        self.partComboBox.currentIndexChanged.connect(self._on_part_changed)
        for spin in (
            self.widgetOffsetXSpinBox,
            self.widgetOffsetYSpinBox,
            self.widgetRotationSpinBox,
            self.cropXSpinBox,
            self.cropYSpinBox,
            self.cropWidthSpinBox,
            self.cropHeightSpinBox,
        ):
            spin.valueChanged.connect(self._update_draft_label)

        self.saveWidgetAdjustmentButton.clicked.connect(self.save)
        self.revertWidgetAdjustmentButton.clicked.connect(self.load_current)
        self.restoreWidgetDefaultButton.clicked.connect(self.restore_default)
        self.copyFromPartButton.clicked.connect(self.copy_from_part)

        if self._parts:
            self.partComboBox.setCurrentIndex(0)
            self.load_current()

    # ------------------------------------------------------------------
    def _on_part_changed(self, _index: int) -> None:
        self.load_current()

    def current_widget_id(self) -> Optional[int]:
        return self.partComboBox.currentData()

    def load_current(self) -> None:
        widget_id = self.current_widget_id()
        if widget_id is None:
            return
        effective = self.settings.effective_widget_alignment(
            widget_id, self._overrides, self._defaults
        )
        if effective is None:
            effective = WidgetAlignment(widget_id=widget_id)
        self._current = effective

        self.widgetIdLabel.setText(f"ウィジェットID: {widget_id}")
        self.partNumberLabel.setText(f"品番: {effective.part_number or '-'}")
        self.widgetSourceLabel.setText(
            "適用元: 品番別上書き" if effective.is_overridden() else "適用元: 共通既定値"
        )

        self.widgetOffsetXSpinBox.setValue(float(effective.cam2_offset_x_px))
        self.widgetOffsetYSpinBox.setValue(float(effective.cam2_offset_y_px))
        self.widgetRotationSpinBox.setValue(float(effective.cam2_rotation_deg))

        crop = effective.crop or CropRect(0, 0, 0, 0)
        self.cropXSpinBox.setValue(int(crop.x_px))
        self.cropYSpinBox.setValue(int(crop.y_px))
        self.cropWidthSpinBox.setValue(int(crop.width_px))
        self.cropHeightSpinBox.setValue(int(crop.height_px))

        katabu = effective.katabu_crop
        self.katabuCropXSpinBox.setValue(int(katabu.x_px) if katabu else 0)
        self.katabuCropYSpinBox.setValue(int(katabu.y_px) if katabu else 0)
        self.katabuCropWidthSpinBox.setValue(int(katabu.width_px) if katabu else 0)
        self.katabuCropHeightSpinBox.setValue(int(katabu.height_px) if katabu else 0)

        self._update_draft_label()
        self._render_preview()

    # ------------------------------------------------------------------
    def collect(self) -> Optional[WidgetAlignment]:
        widget_id = self.current_widget_id()
        if widget_id is None:
            return None

        crop = CropRect(
            x_px=int(self.cropXSpinBox.value()),
            y_px=int(self.cropYSpinBox.value()),
            width_px=int(self.cropWidthSpinBox.value()),
            height_px=int(self.cropHeightSpinBox.value()),
        )
        katabu_width = int(self.katabuCropWidthSpinBox.value())
        katabu_height = int(self.katabuCropHeightSpinBox.value())
        katabu = None
        if katabu_width > 0 and katabu_height > 0:
            katabu = CropRect(
                x_px=int(self.katabuCropXSpinBox.value()),
                y_px=int(self.katabuCropYSpinBox.value()),
                width_px=katabu_width,
                height_px=katabu_height,
            )

        part_number = self._current.part_number if self._current else ""
        return WidgetAlignment(
            widget_id=widget_id,
            part_number=part_number,
            cam2_offset_x_px=float(self.widgetOffsetXSpinBox.value()),
            cam2_offset_y_px=float(self.widgetOffsetYSpinBox.value()),
            cam2_rotation_deg=float(self.widgetRotationSpinBox.value()),
            crop=crop if crop.is_valid() else None,
            katabu_crop=katabu,
            source="machine_override",
        )

    def _is_dirty(self, draft: WidgetAlignment) -> bool:
        if self._current is None:
            return True
        return not draft.same_as(self._current)

    def _update_draft_label(self, *_args) -> None:
        draft = self.collect()
        if draft is None:
            return
        if self._is_dirty(draft):
            self.widgetDraftLabel.setText(
                "未保存の変更があります (保存するまで検査には適用されません)"
            )
        else:
            self.widgetDraftLabel.setText("変更なし")

    # ------------------------------------------------------------------
    def save(self) -> bool:
        draft = self.collect()
        if draft is None:
            return False
        errors = [i for i in validate_widget_alignment(draft) if i.severity is Severity.ERROR]
        if errors:
            show_error(self, "保存できません", "\n".join(i.message_ja for i in errors))
            return False

        overrides = dict(self._overrides)
        overrides[draft.widget_id] = draft
        try:
            self.settings.save_widget_adjustments(overrides, note="widget adjustment UI")
        except SettingsValidationError as exc:
            show_error(self, "保存できません", "\n".join(i.message_ja for i in exc.issues))
            return False
        except OSError as exc:
            show_error(self, "保存できません", f"ファイルに書き込めません: {exc}")
            return False

        self._overrides = overrides
        self._current = draft
        self.load_current()
        if self.on_saved is not None:
            self.on_saved("widget_adjustments")
        show_info(
            self,
            "保存しました",
            f"ウィジェット {draft.widget_id} の設定を保存しました。",
        )
        return True

    def restore_default(self) -> None:
        widget_id = self.current_widget_id()
        if widget_id is None:
            return
        if widget_id not in self._overrides:
            show_info(self, "変更なし", "この品番は既に共通既定値です。")
            return
        overrides = dict(self._overrides)
        overrides.pop(widget_id, None)
        try:
            self.settings.save_widget_adjustments(overrides, note="restore shared default")
        except Exception as exc:
            show_error(self, "戻せません", str(exc))
            return
        self._overrides = overrides
        self.load_current()
        if self.on_saved is not None:
            self.on_saved("widget_adjustments")

    def copy_from_part(self) -> None:
        if len(self._parts) < 2:
            return
        labels = [f"{wid}: {label}" for wid, _pn, label in self._parts]
        choice, accepted = QInputDialog.getItem(
            self, "他品番からコピー", "コピー元の品番を選択してください", labels, 0, False
        )
        if not accepted:
            return
        source_id = self._parts[labels.index(choice)][0]
        source = self.settings.effective_widget_alignment(source_id, self._overrides, self._defaults)
        if source is None:
            show_error(self, "コピーできません", "コピー元の設定が見つかりません。")
            return
        target_id = self.current_widget_id()
        if target_id is None:
            return
        copied = self.settings.import_widget_alignment(
            source, target_id, self._current.part_number if self._current else ""
        )
        self.widgetOffsetXSpinBox.setValue(copied.cam2_offset_x_px)
        self.widgetOffsetYSpinBox.setValue(copied.cam2_offset_y_px)
        self.widgetRotationSpinBox.setValue(copied.cam2_rotation_deg)
        if copied.crop is not None:
            self.cropXSpinBox.setValue(copied.crop.x_px)
            self.cropYSpinBox.setValue(copied.crop.y_px)
            self.cropWidthSpinBox.setValue(copied.crop.width_px)
            self.cropHeightSpinBox.setValue(copied.crop.height_px)
        self._update_draft_label()

    # ------------------------------------------------------------------
    def set_reference_image(self, image: np.ndarray) -> None:
        self._reference = image
        self._reference_pixmap = None
        self._render_preview()

    def set_reference_qimage(self, image) -> None:
        """Supply the merged image straight from a thread signal (QImage)."""

        if image is None or image.isNull():
            return
        self._reference = None
        self._reference_pixmap = QPixmap.fromImage(image)
        self._render_preview()

    def _render_preview(self) -> None:
        if self._reference_pixmap is not None:
            self._render_pixmap_preview()
            return
        if self._reference is None:
            return
        image = self._reference
        height, width = image.shape[:2]
        rgb = image[:, :, ::-1] if image.ndim == 3 else image
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        self.widgetFullPreviewLabel.setPixmap(
            pixmap.scaled(
                self.widgetFullPreviewLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        draft = self.collect()
        if draft is None or draft.crop is None:
            return
        safe = draft.crop.clamp_to(width, height)
        cropped = image[
            safe.y_px : safe.y_px + safe.height_px, safe.x_px : safe.x_px + safe.width_px
        ]
        if cropped.size == 0:
            return
        crop_rgb = cropped[:, :, ::-1] if cropped.ndim == 3 else cropped
        crop_q = QImage(
            crop_rgb.data, crop_rgb.shape[1], crop_rgb.shape[0], crop_rgb.strides[0], QImage.Format_RGB888
        )
        self.widgetCropPreviewLabel.setPixmap(
            QPixmap.fromImage(crop_q.copy()).scaled(
                self.widgetCropPreviewLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _render_pixmap_preview(self) -> None:
        """Show the incoming frame with the draft crop rectangle overlaid."""

        base = self._reference_pixmap
        if base is None or base.width() == 0 or base.height() == 0:
            return

        scaled = base.scaled(
            self.widgetFullPreviewLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if scaled.isNull():
            return

        canvas = QPixmap(scaled.size())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.drawPixmap(0, 0, scaled)

        rect = None
        draft = self.collect()
        if draft is not None and draft.crop is not None and draft.crop.is_valid():
            ratio_x = scaled.width() / float(base.width())
            ratio_y = scaled.height() / float(base.height())
            rect = QRect(
                int(draft.crop.x_px * ratio_x),
                int(draft.crop.y_px * ratio_y),
                max(1, int(draft.crop.width_px * ratio_x)),
                max(1, int(draft.crop.height_px * ratio_y)),
            )
            pen = QPen(QColor(252, 233, 79))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
        painter.end()

        crop_pixmap = canvas.copy(rect) if rect is not None else canvas
        self.widgetFullPreviewLabel.setPixmap(canvas)
        self.widgetCropPreviewLabel.setPixmap(
            crop_pixmap.scaled(
                self.widgetCropPreviewLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
