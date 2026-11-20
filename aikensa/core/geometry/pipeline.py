"""Null-aware geometry pipeline shared by calibration and inspection.

Processing concept (each stage is optional and guarded)::

    raw frames -> consistent orientation -> undistortion -> base homography
        -> global correction -> per-part correction -> image merge
        -> ROI crop -> inspection or preview

The legacy order is preserved: frames are already rotated by the acquisition
layer when ``rotate_180`` is enabled, undistortion happens before the
homography, and the merge composites camera 1 first and camera 2 second.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from aikensa.core.geometry.transforms import (
    compose_into_base,
    homography_for_scale,
    is_valid_homography,
    merge_warped,
    rotation_about_pivot,
    translation_matrix,
    warp_with_mask,
)
from aikensa.core.settings.schema import (
    CropRect,
    GlobalAlignment,
    HomographyCalibration,
    IntrinsicCalibration,
    MachineSettingsSnapshot,
    WidgetAlignment,
)

LOGGER = logging.getLogger(__name__)


class UncalibratedError(RuntimeError):
    """Raised when a geometry operation needs calibration that does not exist."""

    def __init__(self, stage: str, camera_index: Optional[int] = None):
        self.stage = stage
        self.camera_index = camera_index
        camera_text = "" if camera_index is None else f"カメラ{camera_index + 1} の"
        super().__init__(f"{camera_text}{stage}が未校正のため処理できません。")


@dataclass
class GeometryPipeline:
    """Holds the calibrated geometry for the current machine profile."""

    canvas_size: Tuple[int, int]  # (height, width) like numpy arrays
    intrinsics: Dict[int, IntrinsicCalibration] = field(default_factory=dict)
    homographies: Dict[int, HomographyCalibration] = field(default_factory=dict)
    global_alignment: GlobalAlignment = field(default_factory=GlobalAlignment)
    widget_alignments: Mapping[int, WidgetAlignment] = field(default_factory=dict)
    _rectification_maps: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_snapshot(
        cls,
        snapshot: MachineSettingsSnapshot,
        canvas_size: Tuple[int, int],
        widget_alignments: Optional[Mapping[int, WidgetAlignment]] = None,
    ) -> "GeometryPipeline":
        return cls(
            canvas_size=(int(canvas_size[0]), int(canvas_size[1])),
            intrinsics=dict(snapshot.intrinsics),
            homographies=dict(snapshot.homographies),
            global_alignment=snapshot.global_alignment,
            widget_alignments=widget_alignments
            if widget_alignments is not None
            else snapshot.widget_alignments,
        )

    # ------------------------------------------------------------------
    # Calibration availability
    # ------------------------------------------------------------------
    def has_intrinsics(self, camera_index: int) -> bool:
        intrinsic = self.intrinsics.get(camera_index)
        return (
            intrinsic is not None
            and intrinsic.camera_matrix is not None
            and intrinsic.distortion_coefficients is not None
        )

    def has_homography(self, camera_index: int, *, scaled: bool = False) -> bool:
        calibration = self.homographies.get(camera_index)
        if calibration is None:
            return False
        matrix = calibration.scaled_matrix if scaled else calibration.matrix
        return is_valid_homography(matrix)

    def uncalibrated_cameras(self) -> list[int]:
        return [
            index
            for index in sorted(self.homographies)
            if not (self.has_intrinsics(index) and self.has_homography(index))
        ]

    # ------------------------------------------------------------------
    # Stage 1: undistortion
    # ------------------------------------------------------------------
    def rectification_maps(
        self, camera_index: int, image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cached ``(map1, map2)`` for ``cv2.remap``. Raises if uncalibrated."""

        if not self.has_intrinsics(camera_index):
            raise UncalibratedError("内部パラメータ", camera_index)

        key = (camera_index, int(image_size[0]), int(image_size[1]))
        cached = self._rectification_maps.get(key)
        if cached is not None:
            return cached

        intrinsic = self.intrinsics[camera_index]
        camera_matrix = np.asarray(intrinsic.camera_matrix, dtype=np.float64)
        dist_coeffs = np.asarray(intrinsic.distortion_coefficients, dtype=np.float64)
        # Shape the distortion coefficients the way OpenCV expects (1, N).
        dist_coeffs = dist_coeffs.reshape(1, -1)

        width, height = int(image_size[0]), int(image_size[1])
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, (width, height), cv2.CV_16SC2
        )
        self._rectification_maps[key] = (map1, map2)
        return map1, map2

    def undistort(self, frame: np.ndarray, camera_index: int) -> np.ndarray:
        if frame is None:
            raise ValueError("frame is None")
        map1, map2 = self.rectification_maps(camera_index, (frame.shape[1], frame.shape[0]))
        return cv2.remap(
            frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    # ------------------------------------------------------------------
    # Stage 2: effective homography
    # ------------------------------------------------------------------
    def effective_homography(
        self, camera_index: int, widget_id: Optional[int] = None, *, scaled: bool = False
    ) -> np.ndarray:
        """``T_widget @ T_global @ H_base`` for camera 2 (base only for camera 1)."""

        calibration = self.homographies.get(camera_index)
        if calibration is None:
            raise UncalibratedError("ホモグラフィ", camera_index)
        base = calibration.scaled_matrix if scaled else calibration.matrix
        if not is_valid_homography(base):
            raise UncalibratedError("ホモグラフィ", camera_index)

        transforms = []
        if camera_index == 1:
            alignment = self.global_alignment
            if not alignment.is_identity:
                if abs(alignment.cam2_rotation_deg) > 1e-9:
                    pivot = alignment.pivot_px or (
                        self.canvas_size[1] / 2.0,
                        self.canvas_size[0] / 2.0,
                    )
                    transforms.append(rotation_about_pivot(alignment.cam2_rotation_deg, pivot))
                transforms.append(
                    translation_matrix(alignment.cam2_x_px, alignment.cam2_y_px)
                )

            widget = self.widget_alignments.get(widget_id) if widget_id is not None else None
            if widget is not None:
                if abs(widget.cam2_rotation_deg) > 1e-9:
                    pivot = (self.canvas_size[1] / 2.0, self.canvas_size[0] / 2.0)
                    transforms.append(rotation_about_pivot(widget.cam2_rotation_deg, pivot))
                if widget.cam2_offset_x_px or widget.cam2_offset_y_px:
                    transforms.append(
                        translation_matrix(widget.cam2_offset_x_px, widget.cam2_offset_y_px)
                    )

        return compose_into_base(base, transforms)

    def preview_homography(self, camera_index: int, *, widget_id: Optional[int] = None) -> np.ndarray:
        """Scaled (preview) homography derived from the authoritative full one."""

        full = self.effective_homography(camera_index, widget_id=widget_id, scaled=False)
        calibration = self.homographies[camera_index]
        source_full = (calibration.width_px, calibration.height_px)
        if source_full[0] is None or source_full[1] is None:
            raise UncalibratedError("校正画像サイズ", camera_index)

        scale = 5
        source_size = (source_full[0] // scale, source_full[1] // scale)
        destination_full = (self.canvas_size[1], self.canvas_size[0])
        destination_size = (destination_full[0] // scale, destination_full[1] // scale)
        return homography_for_scale(
            full,
            source_full_size=source_full,
            destination_full_size=destination_full,
            source_size=source_size,
            destination_size=destination_size,
        )

    # ------------------------------------------------------------------
    # Stage 3: merge
    # ------------------------------------------------------------------
    def merge_frames(
        self,
        frame1: Optional[np.ndarray],
        frame2: Optional[np.ndarray],
        *,
        camera1: int = 0,
        camera2: int = 1,
        widget_id: Optional[int] = None,
        canvas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composite both cameras onto the template canvas."""

        if canvas is not None:
            canvas_size = (canvas.shape[1], canvas.shape[0])
        else:
            canvas_size = (self.canvas_size[1], self.canvas_size[0])

        result = (
            canvas.copy()
            if canvas is not None
            else np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        )

        for frame, index in ((frame1, camera1), (frame2, camera2)):
            if frame is None:
                continue
            matrix = self.effective_homography(index, widget_id=widget_id)
            warped, mask = warp_with_mask(frame, matrix, canvas_size)
            result = merge_warped(result, warped, mask)
        return result

    # ------------------------------------------------------------------
    # Stage 4: ROI
    # ------------------------------------------------------------------
    @staticmethod
    def crop(image: np.ndarray, crop: Optional[CropRect]) -> np.ndarray:
        if crop is None or not crop.is_valid():
            return image
        height, width = image.shape[:2]
        safe = crop.clamp_to(width, height)
        return image[safe.y_px : safe.y_px + safe.height_px, safe.x_px : safe.x_px + safe.width_px]

    def widget_crop(self, image: np.ndarray, widget_id: int, *, katabu: bool = False) -> np.ndarray:
        alignment = self.widget_alignments.get(widget_id)
        if alignment is None:
            return image
        crop = alignment.katabu_crop if katabu else alignment.crop
        return self.crop(image, crop)
