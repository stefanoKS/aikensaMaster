import cv2
import os
from datetime import datetime
from networkx import jaccard_coefficient
import numpy as np
from sympy import fu
import yaml
import time
import logging
from pathlib import Path

from PyQt5.QtCore import QSize, QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from aikensa.core.scripts.cam_init.cam_init import initialize_camera
from aikensa.core.camera.camera_service import CameraRole, CameraStream
from aikensa.core.camera.placeholder import make_camera_placeholder
from aikensa.core.camera.registry import get_camera_service
from aikensa.core.settings.service import SettingsService
from aikensa.core.geometry.pipeline import GeometryPipeline, UncalibratedError
from aikensa.core.geometry.coverage import analyze_merge_coverage, render_merge_coverage
from aikensa.core.geometry.merge_diagnostic import export_merge_diagnostic, inspection_cam2_transform
from aikensa.core.inspection.daily_identity import daily_geometry_digest
from aikensa.core.scripts.img_processing.cameracalibrate import detectCharucoBoard, detectCharucoBoard_scaledImage, calculatecameramatrix, calculatecameramatrix_scaledImage, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.core.scripts.img_processing.arucoplanarize import planarize_image_wide, planarize_image_narrow
from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.core.config.sound.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound


@dataclass
class CalibrationConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected
    
    calculateSingeFrameMatrix: bool = False
    calculateCamMatrix: bool = False
    delCamMatrix: bool = False
    savecalculatedCamImage: bool = False

    calibrationMatrix: np.ndarray = field(default=None)
    mapCalculated: list = field(default_factory=lambda: [False]*10)     #max for 10 cameras
    map1: list = field(default_factory=lambda: [None]*10)               #max for 10 cameras
    map2: list = field(default_factory=lambda: [None]*10)               #max for 10 cameras

    calibrationMatrix_scaled: np.ndarray = field(default=None)
    map1_downscaled: list = field(default_factory=lambda: [None]*10)    #max for 10 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*10)    #max for 10 cameras

    calculateHomo_cam1: bool = False
    calculateHomo_cam2: bool = False

    calculateHomo_cam1_high: bool = False
    calculateHomo_cam2_high: bool = False
    
    deleteHomo: bool = False

    mergeCam: bool = False
    saveImage: bool = False

    savePlanarize: bool = False
    savePlanarizeHigh: bool = False

    delPlanarize: bool = False

    HDRes: bool = False
    coverage_request: tuple[int, int | None, bool] | None = None

class CalibrationThread(QThread):

    CalibCamStream = pyqtSignal(int, QImage)
    CamMerge1 = pyqtSignal(QImage)
    CamMerge2 = pyqtSignal(QImage)
    CamMergeAll = pyqtSignal(QImage, QSize)
    mergeCoverageSignal = pyqtSignal(int, QImage, str)
    cameraErrorSignal = pyqtSignal(str)

    def __init__(self, calib_config: CalibrationConfig = None):
        super(CalibrationThread, self).__init__()
        self.running = True

        if calib_config is None:
            self.calib_config = CalibrationConfig()    
        else:
            self.calib_config = calib_config

        self.widget_dir_map={
            1: "CamCalibration1",
            2: "CamCalibration2"
        }

        self.cameraMatrix = None
        self.kanjiFontPath = "aikensa/gui/resources/font/NotoSansJP-ExtraBold.ttf"
        self.frame  = None
        self.frame_downsampled = None
        self.frame_scaled = None


        self.mergeframe1 = None
        self.mergeframe2 = None

        self.mergeframe1_scaled = None
        self.mergeframe2_scaled = None

        self.mergeframe1_downsampled = None
        self.mergeframe2_downsampled = None

        self.homography_template = None
        self.homography_matrix1 = None
        self.homography_matrix2 = None
        self.homography_matrix1_high = None
        self.homography_matrix2_high = None

        self.homography_template_scaled = None
        self.homography_matrix1_scaled = None
        self.homography_matrix2_scaled = None
        self.homography_matrix1_high_scaled= None
        self.homography_matrix2_high_scaled = None

        self.H1 = None
        self.H2 = None
        self.H1_high = None
        self.H2_high = None

        self.H1_scaled = None
        self.H2_scaled = None
        self.H1_high_scaled = None
        self.H2_high_scaled = None

        self.homography_size = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.narrow_planarize = (531, 2646)
        self.wide_planarize = (1342, 5672)

        self.combinedImage = None
        self.combinedImage_scaled = None

        self.combinedImage_high = None
        self.combinedImage_high_scaled = None

        self.combinedImage_narrow = None
        self.combinedImage_narrow_scaled = None
        self.combinedImage_wide = None
        self.combinedImage_wide_scaled = None

        self.combinedImage_high_narrow = None
        self.combinedImage_high_narrow_scaled = None
        self.combinedImage_high_wide = None
        self.combinedImage_high_wide_scaled = None

        self.scale_factor = 5.0
        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.planarizeTransform_narrow = None
        self.planarizeTransform_narrow_scaled = None
        self.planarizeTransform_high_narrow = None
        self.planarizeTransform_high_narrow_scaled = None

        self.planarizeTransform_wide = None
        self.planarizeTransform_wide_scaled = None
        self.planarizeTransform_high_wide = None
        self.planarizeTransform_high_wide_scaled = None

        this_dir = os.path.dirname(__file__)
        cam_config_path = os.path.abspath(os.path.join(this_dir, '..', 'config', 'cam'))

        self.cam_config_file = os.path.join(cam_config_path, 'cam_config.yaml')
        
        with open(self.cam_config_file, 'r') as file:
            self.cam_map = yaml.safe_load(file)
        self.ic4_rotate_180 = bool((self.cam_map or {}).get("ic4_rotate_180", False))

        # Central settings: camera serials and hardware parameters are machine
        # specific and must never come from hardcoded literals.
        self.settings = SettingsService()
        self.camera_service = get_camera_service(self.settings)
        self.camera_stream = CameraStream(self.camera_service, CameraRole.CALIBRATION)
        self._camera_diagnostic = ""

    def release_all_camera(self):
        self.camera_stream.close()

    def _report_camera_status(self, message):
        if message != self._camera_diagnostic:
            if not message:
                self._load_calibration_data_from_settings()
            self._camera_diagnostic = message
            self.cameraErrorSignal.emit(message)

    def _capture_single_calibration_frame(self, camera_index):
        frame = self.camera_stream.read(camera_index)
        if frame is None:
            self.frame = self.frame_scaled = self.frame_downsampled = None
            self.calib_config.calculateSingeFrameMatrix = False
            self.calib_config.calculateCamMatrix = False
            if self._camera_diagnostic != self.camera_stream.errors[camera_index]:
                self.CalibCamStream.emit(camera_index, self._placeholder_preview(camera_index, 614, 409))
            self._report_camera_status(self.camera_stream.errors[camera_index])
            return False
        self._report_camera_status("")
        self.frame = frame.image if self.ic4_rotate_180 else cv2.rotate(frame.image, cv2.ROTATE_180)
        self.frame_scaled = self.downSampling(self.frame, self.scaled_width, self.scaled_height)
        return True

    def _capture_merge_calibration_frames(self):
        first, second = self.camera_stream.read_pair()
        if first is not None and second is not None:
            self._report_camera_status("")
            self.mergeframe1, self.mergeframe2 = first.image, second.image
            return True
        self.mergeframe1 = self.mergeframe2 = None
        self.mergeframe1_scaled = self.mergeframe2_scaled = None
        self.combinedImage_scaled = self.combinedImage_high_scaled = None
        for flag in (
            "calculateHomo_cam1", "calculateHomo_cam2", "calculateHomo_cam1_high",
            "calculateHomo_cam2_high", "savePlanarize", "savePlanarizeHigh",
        ):
            setattr(self.calib_config, flag, False)
        for index, frame, signal in ((0, first, self.CamMerge1), (1, second, self.CamMerge2)):
            preview = (
                self._placeholder_preview(index, 246, 163) if frame is None
                else self.convertQImage(self.downSampling(frame.image, 246, 163)).copy()
            )
            signal.emit(preview)
        self.CamMergeAll.emit(QImage(), QSize())
        self._report_camera_status(" / ".join(self.camera_stream.errors.values()))
        return False

    def _placeholder_preview(self, index, width, height):
        return self.convertQImage(make_camera_placeholder(index, width, height)).copy()

    @pyqtSlot()
    def start(self, priority=QThread.InheritPriority):
        if not self.isRunning():
            self.running = True
            super().start(priority)

    def run(self):
        try:
            self._run_calibration()
        finally:
            self.release_all_camera()

    def _run_calibration(self):

        active_page = None
        #print thread started
        print("Calibration Thread Started")

        self._save_dir = f"aikensa/core/param/camera/"

        self.homography_template = cv2.imread("aikensa/core/tools/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])

        #make dark blank image with same size as homography_template
        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)

        #INIT all variables - calibration data is machine specific and may be
        # null (not yet calibrated).  The settings repository never converts
        # null into a plausible matrix.
        self._load_calibration_data_from_settings()


        while self.running:
            page = self.calib_config.widget
            if page != active_page:
                self.release_all_camera()
                active_page = page
                self._camera_diagnostic = ""
                cameras, _issues = self.settings.repository.load_camera_settings()
                self.ic4_rotate_180 = cameras.rotate_180_global
                if page == 3:
                    self._merge_geometry_diagnostic = ""
                    self._load_calibration_data_from_settings()
            if page not in (1, 2, 3):
                self.calib_config.cameraID = -1
                self.msleep(50)
                continue

            if page in (1, 2):
                camera_index = page - 1
                self.calib_config.cameraID = camera_index
                if not self._capture_single_calibration_frame(camera_index):
                    self.msleep(100)
                    continue
                if not self.running or page != self.calib_config.widget:
                    continue
            
                if self.calib_config.calculateSingeFrameMatrix:
                    self.frame, _, _ = detectCharucoBoard(self.frame)
                    self.frame_scaled, _, _ = detectCharucoBoard_scaledImage(self.frame_scaled)
                    self.calib_config.calculateSingeFrameMatrix = False

                if self.calib_config.calculateCamMatrix:
                    self.calib_config.calculateCamMatrix = False
                    self.calib_config.calibrationMatrix = calculatecameramatrix()
                    self.calib_config.calibrationMatrix_scaled = calculatecameramatrix_scaledImage()

                    print(f"Calibration Matrix Value: {self.calib_config.calibrationMatrix}")
                    print(f"Calibration Matrix Scaled Value: {self.calib_config.calibrationMatrix_scaled}")

                    # A failed calculation must never overwrite the previous valid
                    # calibration: the repository validates before writing.
                    for matrix, scaled in (
                        (self.calib_config.calibrationMatrix, False),
                        (self.calib_config.calibrationMatrix_scaled, True),
                    ):
                        if not isinstance(matrix, dict):
                            print(
                                "[Calibration] No calibration matrix calculated; "
                                "keeping the previous file."
                            )
                            continue
                        try:
                            self.settings.save_intrinsic(
                                camera_index,
                                np.array(matrix["camera_matrix"], dtype=np.float64),
                                np.array(matrix["distortion_coefficients"], dtype=np.float64),
                                scaled=scaled,
                            )
                            print(
                                f"Saved {'scaled ' if scaled else ''}calibration for camera {camera_index}"
                            )
                        except Exception as e:
                            print(
                                f"[Calibration] Refused to save {'scaled ' if scaled else ''}"
                                f"calibration: {e}"
                            )

                if self.frame is not None:
                    self.frame_downsampled = self.downSampling(self.frame, 1229, 819)

                if self.frame is not None:
                    self.CalibCamStream.emit(camera_index, self.convertQImage(self.frame_downsampled).copy())
            
            if page == 3:
                if not self._capture_merge_calibration_frames():
                    self._reject_coverage_request("カメラ画像を取得できません。")
                    self.msleep(100)
                    continue
                if not self.running or page != self.calib_config.widget:
                    continue

                frames = (self.mergeframe1, self.mergeframe2)
                try:
                    self.mergeframe1, self.mergeframe2 = (
                        self._merge_geometry.undistort(frame, index)
                        for index, frame in enumerate(frames)
                    )
                except UncalibratedError as exc:
                    for flag in (
                        "calculateHomo_cam1", "calculateHomo_cam2", "calculateHomo_cam1_high",
                        "calculateHomo_cam2_high", "savePlanarize", "savePlanarizeHigh",
                    ):
                        setattr(self.calib_config, flag, False)
                    for frame, signal in zip(frames, (self.CamMerge1, self.CamMerge2)):
                        signal.emit(self.convertQImage(self.downSampling(frame, 246, 163)).copy())
                    self.CamMergeAll.emit(QImage(), QSize())
                    self._report_merge_geometry(str(exc))
                    self._reject_coverage_request(str(exc))
                    self.msleep(100)
                    continue
                self._report_merge_geometry("")

                # self.mergeframe1 = cv2.cvtColor(self.mergeframe1, cv2.COLOR_BGR2RGB)
                # self.mergeframe2 = cv2.cvtColor(self.mergeframe2, cv2.COLOR_BGR2RGB)
                # Rotation for merge stream is handled by IC4 at camera initialization.

                #original res
                self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
                self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)

                #Calculate Homography matrix
                if self.calib_config.calculateHomo_cam1 is True:

                    self.calib_config.calculateHomo_cam1 = False
                    if self.mergeframe1 is not None:
                        result_img, matrix = calculateHomography_template(self.homography_template, self.mergeframe1)
                        if result_img is not None:
                            cv2.imwrite("resultmergeframe1.png", result_img)
                        if self._persist_homography(0, matrix, scaled=False):
                            self.homography_matrix1 = np.asarray(matrix).tolist()
                            self.H1 = np.asarray(matrix)
                            print(f"Homography matrix is calculated for Camera 1 with value {self.H1}")
                    else:
                        print("mergeframe1 is empty")

                    if self.mergeframe1_scaled is not None:
                        _, matrix_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe1_scaled)
                        if self._persist_homography(0, matrix_scaled, scaled=True):
                            self.homography_matrix1_scaled = np.asarray(matrix_scaled).tolist()
                            self.H1_scaled = np.asarray(matrix_scaled)
                            print(f"Homography scaled matrix is calculated for Camera 1 with value {self.H1_scaled}")
                    else:
                        print("mergeframe1 scaled is empty")

                if self.calib_config.calculateHomo_cam2 is True:

                    self.calib_config.calculateHomo_cam2 = False
                    result_img, matrix = calculateHomography_template(self.homography_template, self.mergeframe2)
                    if result_img is not None:
                        cv2.imwrite("resultmergeframe2.png", result_img)
                    if self._persist_homography(1, matrix, scaled=False):
                        self.homography_matrix2 = np.asarray(matrix).tolist()
                        self.H2 = np.asarray(matrix)
                        print(f"Homography matrix is calculated for Camera 2 with value {self.H2}")

                    _, matrix_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe2_scaled)
                    if self._persist_homography(1, matrix_scaled, scaled=True):
                        self.homography_matrix2_scaled = np.asarray(matrix_scaled).tolist()
                        self.H2_scaled = np.asarray(matrix_scaled)
                        print(f"Homography scaled matrix is calculated for Camera 2 with value {self.H2_scaled}")

                if self.calib_config.calculateHomo_cam1_high is True:

                    self.calib_config.calculateHomo_cam1_high = False
                    _, matrix = calculateHomography_template(self.homography_template, self.mergeframe1)
                    if matrix is not None:
                        self.homography_matrix1_high = np.asarray(matrix).tolist()
                        self.H1_high = np.asarray(matrix)
                        print(f"Homography matrix is calculated for Camera 1 (high) with value {self.H1_high}")
                    _, matrix_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe1_scaled)
                    if matrix_scaled is not None:
                        self.homography_matrix1_high_scaled = np.asarray(matrix_scaled).tolist()
                        self.H1_high_scaled = np.asarray(matrix_scaled)
                        print(f"Homography scaled matrix is calculated for Camera 1 (high) with value {self.H1_high_scaled}")

                if self.calib_config.calculateHomo_cam2_high is True:

                        self.calib_config.calculateHomo_cam2_high = False
                        _, matrix = calculateHomography_template(self.homography_template, self.mergeframe2)
                        if matrix is not None:
                            self.homography_matrix2_high = np.asarray(matrix).tolist()
                            self.H2_high = np.asarray(matrix)
                            print(f"Homography matrix is calculated for Camera 2 (high) with value {self.H2_high}")
                        _, matrix_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe2_scaled)
                        if matrix_scaled is not None:
                            self.homography_matrix2_high_scaled = np.asarray(matrix_scaled).tolist()
                            self.H2_high_scaled = np.asarray(matrix_scaled)
                            print(f"Homography scaled matrix is calculated for Camera 2 (high) with value {self.H2_high_scaled}")

                # Homography availability is decided by the null-aware settings
                # loader at thread start; null means "not calibrated" and must
                # never be turned into a matrix here.
                if self.H1 is not None and self.H2 is not None:
                    self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                    self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                else:
                    self.combinedImage = self.homography_blank_canvas

                if self.H1_scaled is not None and self.H2_scaled is not None:
                    self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                    self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)
                else:
                    self.combinedImage_scaled = self.homography_blank_canvas_scaled

                if self.H1_high is not None and self.H2_high is not None:
                    self.combinedImage_high = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1_high)
                    self.combinedImage_high = warpTwoImages_template(self.combinedImage_high, self.mergeframe2, self.H2_high)
                else:
                    self.combinedImage_high = self.homography_blank_canvas

                if self.H1_high_scaled is not None and self.H2_high_scaled is not None:
                    self.combinedImage_high_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_high_scaled)
                    self.combinedImage_high_scaled = warpTwoImages_template(self.combinedImage_high_scaled, self.mergeframe2_scaled, self.H2_high_scaled)
                else:
                    self.combinedImage_high_scaled = self.homography_blank_canvas_scaled

                if self.calib_config.savePlanarize is True:
                    self.calib_config.savePlanarize = False
                    if any(matrix is None for matrix in (self.H1, self.H2, self.H1_scaled, self.H2_scaled)):
                        self.cameraErrorSignal.emit("ホモグラフィが未計算です。先にカメラマージ校正を実行してください。")
                        continue
                    print("Saving planarize")
                    # self.combinedImage_narrow, self.planarizeTransform_narrow = planarize_image_narrow(self.combinedImage, 
                    #                                                               target_width=self.narrow_planarize[1], target_height=self.narrow_planarize[0], 
                    #                                                               top_offset=0, bottom_offset=0, side_offset=0)
                    
                    # self.combinedImage_narrow_scaled, self.planarizeTransform_narrow_scaled = planarize_image_narrow(self.combinedImage_scaled,
                    #                                                                               target_width=int(self.narrow_planarize[1]/self.scale_factor), target_height=int(self.narrow_planarize[0]/self.scale_factor),
                    #                                                                               top_offset=0, bottom_offset=0, side_offset=0)
                    cv2.imwrite("combinedImagelatest2.png", self.combinedImage)
                    self.combinedImage_wide, self.planarizeTransform_wide = planarize_image_wide(self.combinedImage, 
                                                                                  target_width=self.wide_planarize[1], target_height=self.wide_planarize[0], 
                                                                                  top_offset=0, bottom_offset=0, side_offset=0)
                    cv2.imwrite("combinedImage_wide_latest.png", self.combinedImage_wide)
                    
                    self.combinedImage_wide_scaled, self.planarizeTransform_wide_scaled = planarize_image_wide(self.combinedImage_scaled,
                                                                                                  target_width=int(self.wide_planarize[1]/self.scale_factor), target_height=int(self.wide_planarize[0]/self.scale_factor),
                                                                                                  top_offset=0, bottom_offset=0, side_offset=int(0/self.scale_factor))
                    os.makedirs(self._save_dir, exist_ok=True)
                    # with open("./aikensa/core/param/camera/planarizeTransform_narrow.yaml", "w") as file:  
                    #     yaml.dump(self.planarizeTransform_narrow.tolist(), file)
                    # with open("./aikensa/core/param/camera/planarizeTransform_narrow_scaled.yaml", "w") as file:
                    #     yaml.dump(self.planarizeTransform_narrow_scaled.tolist(), file)
                    self._persist_planarize("wide", self.planarizeTransform_wide)
                    self._persist_planarize("wide_scaled", self.planarizeTransform_wide_scaled)

                    # print(f"Image size after narrow warping is {self.combinedImage_narrow.shape}")
                    # print(f"Image size after narrow scaled warping is {self.combinedImage_narrow_scaled.shape}")
                    # print(f"Image size after wide warping is {self.combinedImage_wide.shape}")
                    # print(f"Image size after wide scaled warping is {self.combinedImage_wide_scaled.shape}")

                    # cv2.imwrite("combinedImage_narrow.png", self.combinedImage_narrow)
                    # cv2.imwrite("combinedImage_narrow_scaled.png", self.combinedImage_narrow_scaled)
                    # cv2.imwrite("combinedImage_wide.png", self.combinedImage_wide)
                    # cv2.imwrite("combinedImage_wide_scaled.png", self.combinedImage_wide_scaled)

                if self.calib_config.savePlanarizeHigh is True:
                    self.calib_config.savePlanarizeHigh = False
                    if any(matrix is None for matrix in (self.H1_high, self.H2_high, self.H1_high_scaled, self.H2_high_scaled)):
                        self.cameraErrorSignal.emit("高位置のホモグラフィが未計算です。先にカメラマージ校正を実行してください。")
                        continue
                    print("Saving planarize high")
                    self.combinedImage_high_narrow, self.planarizeTransform_high_narrow = planarize_image_narrow(self.combinedImage_high,
                                                                                                    target_width=self.narrow_planarize[1], target_height=self.narrow_planarize[0],
                                                                                                    top_offset=0, bottom_offset=0, side_offset=0)       
                    self.combinedImage_high_narrow_scaled, self.planarizeTransform_high_narrow_scaled = planarize_image_narrow(self.combinedImage_high_scaled,
                                                                                                    target_width=int(self.narrow_planarize[1]/self.scale_factor), target_height=int(self.narrow_planarize[0]/self.scale_factor),
                                                                                                    top_offset=0, bottom_offset=0, side_offset=0)
                    self.combinedImage_high_wide, self.planarizeTransform_high_wide = planarize_image_wide(self.combinedImage_high,
                                                                                                    target_width=self.wide_planarize[1], target_height=self.wide_planarize[0],
                                                                                                    top_offset=0, bottom_offset=0, side_offset=0)             
                    self.combinedImage_high_wide_scaled, self.planarizeTransform_high_wide_scaled = planarize_image_wide(self.combinedImage_high_scaled,
                                                                                                    target_width=int(self.wide_planarize[1]/self.scale_factor), target_height=int(self.wide_planarize[0]/self.scale_factor),
                                                                                                    top_offset=0, bottom_offset=0, side_offset=int(0/self.scale_factor))
                    os.makedirs(self._save_dir, exist_ok=True)
                    self._persist_planarize("high_narrow", self.planarizeTransform_high_narrow)
                    self._persist_planarize("high_narrow_scaled", self.planarizeTransform_high_narrow_scaled)
                    self._persist_planarize("high_wide", self.planarizeTransform_high_wide)
                    self._persist_planarize("high_wide_scaled", self.planarizeTransform_high_wide_scaled)
                    
                    print(f"Image size after narrow warping is {self.combinedImage_high_narrow.shape}")
                    print(f"Image size after narrow scaled warping is {self.combinedImage_high_narrow_scaled.shape}")
                    print(f"Image size after wide warping is {self.combinedImage_high_wide.shape}")
                    print(f"Image size after wide scaled warping is {self.combinedImage_high_wide_scaled.shape}")

                    cv2.imwrite("combinedImage_high_narrow.png", self.combinedImage_high_narrow)
                    cv2.imwrite("combinedImage_high_narrow_scaled.png", self.combinedImage_high_narrow_scaled)
                    cv2.imwrite("combinedImage_high_wide.png", self.combinedImage_high_wide)
                    cv2.imwrite("combinedImage_high_wide_scaled.png", self.combinedImage_high_wide_scaled)

                    
                # if self.planarizeTransform is not None:
                #     self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform, (self.homography_size[1],self.homography_size[0]))

                # if self.planarizeTransform_scaled is not None:
                #     self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_scaled, (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))


                self.mergeframe1_downsampled = self.downSampling(self.mergeframe1, 246, 163)
                self.mergeframe2_downsampled = self.downSampling(self.mergeframe2, 246, 163)

                if self.mergeframe1_downsampled is not None:
                    self.CamMerge1.emit(self.convertQImage(self.mergeframe1_downsampled).copy())
                if self.mergeframe2_downsampled is not None:
                    self.CamMerge2.emit(self.convertQImage(self.mergeframe2_downsampled).copy())
                if self.H1_scaled is not None and self.H2_scaled is not None:
                    if os.environ.get("AIKENSA_TRACE_MERGE") == "1":
                        trace = tuple((frame.shape[1], frame.shape[0]) for frame in (
                            *frames, self.mergeframe1, self.mergeframe2, self.mergeframe1_scaled,
                            self.mergeframe2_scaled, self.combinedImage, self.combinedImage_scaled,
                        ))
                        if trace != getattr(self, "_merge_dimension_trace", None):
                            self._merge_dimension_trace = trace
                            logging.getLogger(__name__).warning(
                                "Merge worker W×H: native=%s corrected=%s scaled source=%s fixed=%s published=%s",
                                trace[:2], trace[2:4], trace[4:6], trace[6], trace[7],
                            )
                    published = cv2.cvtColor(self.combinedImage_scaled, cv2.COLOR_BGR2RGB)
                    self.CamMergeAll.emit(
                        self.convertQImage(published).copy(),
                        QSize(self.homography_size[1], self.homography_size[0]),
                    )
                else:
                    self.CamMergeAll.emit(QImage(), QSize())
                request = self.calib_config.coverage_request
                if request is not None:
                    self.calib_config.coverage_request = None
                    self._provide_merge_coverage(request, frames)

            #wait for 5ms
            # self.msleep(2)
            
        print(f"Camera {self.calib_config.cameraID} released.")

    def _reject_coverage_request(self, message):
        request = self.calib_config.coverage_request
        if request is not None:
            self.calib_config.coverage_request = None
            self.mergeCoverageSignal.emit(request[0], QImage(), message)

    def _provide_merge_coverage(self, request, native_frames):
        serial, widget, save = request
        try:
            corrected = (self.mergeframe1, self.mergeframe2)
            matrices = (self.H1, self.H2)
            canvas = (self.homography_size[1], self.homography_size[0])
            crop_points = None
            rotation = None
            if widget is not None:
                repository = self.settings.repository
                default = repository.shared_widget_defaults().get(widget)
                if default is None or default.crop is None or self.H2 is None:
                    raise ValueError("検査範囲またはホモグラフィが未設定です。")
                override = repository.load_widget_adjustments().get(widget)
                alignment = override if override and (override.cam2_offset_x_px or override.cam2_offset_y_px) else default
                matrices = (self.H1, inspection_cam2_transform(
                    self.H2, (alignment.cam2_offset_x_px, alignment.cam2_offset_y_px),
                    repository.load_global_alignment(),
                ))
                x, y, w, h = default.crop.as_tuple()
                if not (0 <= x < x + w <= canvas[0] and 0 <= y < y + h <= canvas[1]):
                    raise ValueError("検査切り出し範囲が既存マージ範囲の外にあります。")
                crop_points = (x, y, x + w, y, x + w, y + h, x, y + h)
                if not self.ic4_rotate_180:
                    rotation = cv2.ROTATE_180
                    corrected = tuple(cv2.rotate(frame, rotation) for frame in corrected)
            sizes = tuple((frame.shape[1], frame.shape[0]) for frame in corrected)
            coverage = analyze_merge_coverage(sizes, matrices, canvas)
            image = render_merge_coverage(corrected, matrices, coverage, crop_points)
            lines = [
                f"診断スナップショット / {'校正マージ' if widget is None else f'検査位置 {widget}'} / "
                f"既存 {canvas[0]}×{canvas[1]} (黄) / 検査切り出し (緑) / 重なり {coverage.overlap_px2:.0f} px²",
                "Cam 1: 水色 / Cam 2: ピンク / 左・右・上・下は既存キャンバス外の画素数",
            ]
            for camera, footprint in enumerate(coverage.footprints, 1):
                bounds = ", ".join(f"{value:.1f}" for value in footprint.bounds)
                overflow = ", ".join(f"{value:.1f}" for value in footprint.overflow)
                lines.append(f"Cam {camera}: ({bounds}) / 範囲外 左・右・上・下: {overflow} px")
            if save:
                fixed = self.homography_blank_canvas
                for frame, matrix in zip(corrected, matrices):
                    fixed = warpTwoImages_template(fixed, frame, matrix)
                images = {
                    "camera1_native": native_frames[0], "camera2_native": native_frames[1],
                    "camera1_corrected": self.mergeframe1, "camera2_corrected": self.mergeframe2,
                    "fixed_merge": fixed, "published_calibration_preview": self.combinedImage_scaled,
                    "full_coverage": image,
                }
                if crop_points:
                    images["inspection_crop"] = fixed[y:y + h, x:x + w]
                cameras = []
                configured, _ = self.settings.repository.load_camera_settings()
                for index in range(2):
                    handle = self.camera_service.handle(index)
                    hw = configured.hardware_for(index)
                    cameras.append({
                        "serial": handle.serial if handle else None,
                        "requested_size_wh": [hw.width_px, hw.height_px],
                        "configured_rotate_180": hw.rotate_180,
                        "readback": handle.readback if handle else {},
                    })
                digest = daily_geometry_digest(
                    self._merge_geometry.intrinsics, tuple(frame.shape for frame in native_frames),
                    matrices, crop_points, self.homography_blank_canvas.shape, rotation,
                )
                path = export_merge_diagnostic(
                    self.settings.paths.root / "aikensa/diagnostic_results", images, coverage,
                    {"widget": widget, "cameras": cameras, "geometry_digest": digest,
                     "settings_revision": self.settings.revisions.active_revision_id(),
                     "daily_revision": self.settings.daily_revisions.active_revision_id()},
                )
                lines.append(f"保存: {path.name}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.mergeCoverageSignal.emit(serial, self.convertQImage(rgb).copy(), "\n".join(lines))
        except (ValueError, cv2.error, OSError) as exc:
            self.mergeCoverageSignal.emit(serial, QImage(), f"診断できません: {exc}")

    def _persist_planarize(self, name: str, matrix) -> bool:
        """Save a planarisation transform only when ArUco detection succeeded."""

        if matrix is None:
            print(f"[Calibration] planarizeTransform_{name}: detection failed; keeping previous file.")
            return False
        try:
            from aikensa.core.settings.yaml_store import write_yaml_atomic

            write_yaml_atomic(
                self.settings.paths.camera_file(f"planarizeTransform_{name}.yaml"),
                np.asarray(matrix, dtype=np.float64).tolist(),
            )
            return True
        except Exception as e:
            print(f"[Calibration] planarizeTransform_{name}: refused to save: {e}")
            return False

    def _persist_homography(self, camera_index: int, matrix, scaled: bool) -> bool:
        """Save a homography only when it is valid.

        Returns ``True`` when the file was written.  A failed detection keeps the
        previously working file untouched.
        """

        if matrix is None:
            print(
                f"[Calibration] camera {camera_index + 1}: homography detection failed; "
                "keeping the previous file."
            )
            return False
        try:
            self.settings.save_homography(camera_index, np.asarray(matrix, dtype=np.float64), scaled=scaled)
            return True
        except Exception as e:
            print(f"[Calibration] camera {camera_index + 1}: refused to save homography: {e}")
            return False

    def _load_calibration_data_from_settings(self):
        """Load homography / planarisation data through the settings repository.

        ``None`` stays ``None``: an uncalibrated machine must never be given a
        plausible-looking identity transform.
        """

        snapshot = self.settings.load_snapshot()
        self._merge_geometry = GeometryPipeline.from_snapshot(snapshot, self.homography_size)
        repository = self.settings.repository
        for camera_index, suffix in ((0, "1"), (1, "2")):
            full = snapshot.homographies[camera_index]
            scaled = repository.load_homography(camera_index, scaled=True)
            setattr(self, f"homography_matrix{suffix}", None if full.matrix is None else full.matrix.tolist())
            setattr(self, f"H{suffix}", None if full.matrix is None else full.matrix)
            setattr(self, f"homography_matrix{suffix}_scaled", None if scaled.matrix is None else scaled.matrix.tolist())
            setattr(self, f"H{suffix}_scaled", None if scaled.matrix is None else scaled.matrix)

        for name, attr in (
            ("wide", "planarizeTransform_wide"),
            ("wide_scaled", "planarizeTransform_wide_scaled"),
            ("narrow", "planarizeTransform_narrow"),
            ("narrow_scaled", "planarizeTransform_narrow_scaled"),
            ("high_wide", "planarizeTransform_high_wide"),
            ("high_wide_scaled", "planarizeTransform_high_wide_scaled"),
            ("high_narrow", "planarizeTransform_high_narrow"),
            ("high_narrow_scaled", "planarizeTransform_high_narrow_scaled"),
        ):
            matrix, _state = repository.load_planarize_transform(name)
            setattr(self, attr, matrix)

    def _report_merge_geometry(self, message):
        if message != getattr(self, "_merge_geometry_diagnostic", ""):
            self._merge_geometry_diagnostic = message
            self.cameraErrorSignal.emit(message)

    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def undistortFrame(self, frame,cameraMatrix, distortionCoeff):
        if not self.ic4_rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.undistort(frame, cameraMatrix, distortionCoeff, None, cameraMatrix)
        return frame

    def stop(self):
        self.running = False
        print("Calibration thread stopped.")
    
    def downSampling(self, image, width=384, height=256):
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def save_calibration_to_yaml(self, calibrationMatrix, filename):
        with open(filename, 'w') as file:
            yaml.dump(calibrationMatrix, file)

    def load_matrix_from_yaml(self, filename):
        """Load an intrinsic calibration, refusing to invent data from null."""

        with open(filename, 'r') as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader) or {}

        camera_matrix = calibration_param.get('camera_matrix')
        distortion_coeff = calibration_param.get('distortion_coefficients')
        if camera_matrix is None or distortion_coeff is None:
            raise ValueError(f"{filename}: calibration is not configured (null)")

        return (
            np.array(camera_matrix, dtype=np.float64),
            np.array(distortion_coeff, dtype=np.float64).reshape(1, -1),
        )
        
    def initialize_maps(camera_matrix, dist_coeffs, image_size):
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, image_size, cv2.CV_16SC2)
        return map1, map2
