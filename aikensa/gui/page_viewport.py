"""Resolve page content and fit images into their existing display area."""

from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QScrollArea, QWidget

from aikensa.core.settings.schema import CropRect


def page_content(page: QWidget | None) -> QWidget | None:
    return page.widget() if isinstance(page, QScrollArea) else page


def display_image(label: QLabel, image: QImage) -> None:
    if image.isNull():
        label.clear()
        return
    label.setPixmap(
        QPixmap.fromImage(image).scaled(
            label.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    )


def preview_crop_rect(crop: CropRect, full_size: QSize, preview_size: QSize) -> QRect:
    """Map a full-resolution merged ROI to pixels in the preview image itself."""
    if not crop.is_valid() or full_size.isEmpty() or preview_size.isEmpty():
        return QRect()
    x, y, width, height = crop.as_tuple()
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(full_size.width(), x + width), min(full_size.height(), y + height)
    if x1 <= x0 or y1 <= y0:
        return QRect()
    src_w, src_h = full_size.width(), full_size.height()
    dst_w, dst_h = preview_size.width(), preview_size.height()
    left, top = x0 * dst_w // src_w, y0 * dst_h // src_h
    right = (x1 * dst_w + src_w - 1) // src_w
    bottom = (y1 * dst_h + src_h - 1) // src_h
    return QRect(left, top, right - left, bottom - top)
