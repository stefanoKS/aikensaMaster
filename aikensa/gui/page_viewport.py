"""Scrollable viewports for the existing fixed-coordinate pages."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QLabel, QScrollArea, QWidget


def legacy_page_viewport(page: QWidget) -> QScrollArea:
    page.setMinimumSize(page.size())
    viewport = QScrollArea()
    viewport.setFrameShape(QFrame.NoFrame)
    viewport.setWidgetResizable(True)
    viewport.setWidget(page)
    return viewport


def display_image(label: QLabel, image: QImage) -> None:
    if image.isNull():
        label.clear()
        return
    label.setPixmap(
        QPixmap.fromImage(image).scaled(
            label.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    )
