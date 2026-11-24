"""Explicit, read-only diagnostic exports from the merge camera owner."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2

from aikensa.core.geometry.transforms import compose_into_base, rotation_about_pivot, translation_matrix


def inspection_cam2_transform(matrix, offset, global_alignment, *, scale=1.0, pivot=None):
    """Preserve the legacy offset and matrix-based pivot convention."""
    dx, dy = offset
    rotation = 0.0
    if global_alignment is not None:
        dx += float(global_alignment.cam2_x_px)
        dy += float(global_alignment.cam2_y_px)
        rotation = float(global_alignment.cam2_rotation_deg)
    transforms = []
    if rotation:
        if pivot is None:
            pivot = (matrix.shape[1] / (2 * scale), matrix.shape[0] / (2 * scale))
        transforms.append(rotation_about_pivot(rotation, pivot))
    if dx or dy:
        transforms.append(translation_matrix(dx / scale, dy / scale))
    return compose_into_base(matrix, transforms) if transforms else matrix


def export_merge_diagnostic(root: Path, images, report, metadata) -> Path:
    now = datetime.now().astimezone()
    destination = root / f"{now:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    dimensions = {}
    with TemporaryDirectory(dir=root, prefix=".merge-") as staging:
        working = Path(staging)
        for name, image in images.items():
            if not cv2.imwrite(str(working / f"{name}.png"), image):
                raise OSError(f"診断画像を保存できません: {name}")
            dimensions[name] = [image.shape[1], image.shape[0]]
        payload = {**metadata, "captured_at": now.isoformat(), "image_sizes_wh": dimensions, "coverage": asdict(report)}
        (working / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        working.rename(destination)
    return destination
