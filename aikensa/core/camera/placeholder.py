"""Display-only camera placeholders, independent of the camera SDK."""

import cv2
import numpy as np


def make_camera_placeholder(camera_index: int, width: int, height: int) -> np.ndarray:
    if camera_index not in (0, 1):
        raise ValueError(f"Unknown logical camera index: {camera_index}")
    if min(width, height) < 1:
        raise ValueError("Placeholder dimensions must be positive")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    text = f"cam {camera_index + 1} placeholder"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (base_w, base_h), baseline = cv2.getTextSize(text, font, 1.0, 1)
    scale = min(0.86 * width / base_w, 0.70 * height / (base_h + baseline))
    thickness = max(1, round(1.5 * scale))
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = ((width - text_w) // 2, (height + text_h) // 2)
    cv2.putText(image, text, origin, font, scale, (245, 245, 245), thickness, cv2.LINE_AA)
    return image
