"""Render inspection equipment status without acquiring or retaining camera frames."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter
from PyQt5.QtWidgets import QLabel

from aikensa.core.camera.placeholder import make_camera_placeholder
from aikensa.core.camera.preview_state import EquipmentPreview, PreviewReadiness
from aikensa.gui.page_viewport import display_image


def display_equipment_preview(label: QLabel, state: EquipmentPreview) -> None:
    width, height = 690, 120
    image = QImage(2 * width, height, QImage.Format_RGB888)
    image.fill(Qt.black)
    painter = QPainter(image)
    painter.setPen(QColor("#e0e0e0"))
    painter.setFont(QFont("Sans Serif", 30))
    description = state.message
    if state.readiness is PreviewReadiness.CAMERA_UNAVAILABLE:
        for camera, side in ((0, image.rect().adjusted(0, 0, -width, 0)),
                             (1, image.rect().adjusted(width, 0, 0, 0))):
            if camera in state.missing_cameras:
                tile = make_camera_placeholder(camera, width, height)
                painter.drawImage(side, QImage(tile.data, width, height, tile.strides[0], QImage.Format_RGB888))
            else:
                painter.drawText(side, Qt.AlignCenter, f"cam {camera + 1} available")
    else:
        description = state.message or "カメラ・校正を確認中…"
        painter.drawText(image.rect(), Qt.AlignCenter | Qt.TextWordWrap, description)
    painter.end()
    display_image(label, image)
    label.setAccessibleDescription(description)
