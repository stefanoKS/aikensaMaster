import cv2
import os
from datetime import datetime
from networkx import jaccard_coefficient
import numpy as np
from sympy import fu
import yaml
import time
import logging

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from aikensa.core.scripts.cam_init.cam_init import initialize_camera
from aikensa.core.scripts.cam_init.cam_init_ic4 import open_ic4_camera_or_placeholder, DummyCapture
from aikensa.core.camera.camera_service import CameraRole, CameraUnavailableError, CameraBusyError
from aikensa.core.camera.registry import get_camera_service
from aikensa.core.settings.service import SettingsService
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

class CalibrationThread(QThread):

    CalibCamStream = pyqtSignal(QImage)
    CamMerge1 = pyqtSignal(QImage)
    CamMerge2 = pyqtSignal(QImage)
    CamMergeAll = pyqtSignal(QImage)

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
        self.cap_cam = None
        self.frame  = None
        self.frame_downsampled = None
        self.frame_scaled = None

        self.multiCam_stream = False
        self.cap_cam1 = None
        self.cap_cam2 = None

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

        self.cap_cam_ic4_1 = None
        self.cap_cam_ic4_2 = None


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
        self.camera_error = {0: None, 1: None}

    def _open_configured_camera(self, camera_index: int):
        """Open *camera_index* through the shared camera service.

        Returns ``None`` when the camera cannot be opened (unassigned serial,
        hardware missing, or another owner).  The reason is stored in
        ``self.camera_error`` so the UI can explain it in Japanese.
        """

        try:
            handle = self.camera_service.acquire(camera_index, CameraRole.CALIBRATION)
        except (CameraUnavailableError, CameraBusyError) as exc:
            self.camera_error[camera_index] = exc.reason_ja
            print(f"[Calibration] camera {camera_index}: {exc.reason_ja}")
            return None

        self.camera_error[camera_index] = None
        cameras, _issues = self.settings.repository.load_camera_settings()
        self.ic4_rotate_180 = cameras.rotate_180_global
        cam_type = "Placeholder" if handle.is_placeholder else "Real"
        print(f"Initialized Camera {camera_index + 1} (serial {handle.serial}): {cam_type}")
        return handle.capture

    def initialize_single_camera(self, widgetID: int):

        if self.cap_cam is not None:
            self.camera_service.release_all(CameraRole.CALIBRATION)
            self.cap_cam = None
            print(f"Camera {self.calib_config.cameraID} released.")

        # widget 1 == logical camera index 0 (Calibration_camera_0.yaml)
        camera_index = widgetID - 1
        self.cap_cam = self._open_configured_camera(camera_index)

    def release_all_camera(self):
        self.camera_service.release_all(CameraRole.CALIBRATION)
        self.cap_cam_ic4_1 = None
        self.cap_cam_ic4_2 = None
        self.cap_cam1 = None
        self.cap_cam2 = None
        self.cap_cam = None

    def initialize_all_camera(self):
        """Initialize both cameras through the shared camera service."""

        self.cap_cam_ic4_1 = self._open_configured_camera(0)
        self.cap_cam_ic4_2 = self._open_configured_camera(1)

        # Keep legacy aliases in sync for code paths that still reference cap_cam1/2
        self.cap_cam1 = self.cap_cam_ic4_1
        self.cap_cam2 = self.cap_cam_ic4_2

    def read_camera_frame(self, camera_index: int):
        """Read one frame while the camera lock is held by the service."""

        return self.camera_service.read_frame(camera_index, CameraRole.CALIBRATION)

    def run(self):

        self.current_cameraID = -1
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

            if self.calib_config.widget == 0:
                self.calib_config.cameraID = -1

            if self.calib_config.widget in [1, 2]:
                # Leaving merge mode: release merge cameras so single-camera open does not collide.
                if self.multiCam_stream:
                    self.camera_service.release_all(CameraRole.CALIBRATION)
                    self.cap_cam_ic4_1 = None
                    self.cap_cam_ic4_2 = None
                    self.cap_cam1 = None
                    self.cap_cam2 = None
                    self.multiCam_stream = False
                
                if self.calib_config.cameraID != self.current_cameraID:
                    # Camera ID has changed, reinitialize the camera
                    if self.current_cameraID != -1:
                        self.camera_service.release_all(CameraRole.CALIBRATION)
                        print(f"Camera {self.current_cameraID} released.")
                    self.current_cameraID = self.calib_config.cameraID
                    # self.initialize_single_camera(self.current_cameraID)
                    if self.calib_config.widget == 1:
                        self.initialize_single_camera(self.calib_config.widget)
                        print("Initializing Camera 0")
                    if self.calib_config.widget == 2:
                        self.initialize_single_camera(self.calib_config.widget)
                        print("Initializing Camera 1")
                  
                if self.cap_cam is not None:
                    try:
                        camera_index = max(self.calib_config.widget - 1, 0)
                        ret, self.frame = self.read_camera_frame(camera_index)
                        if not ret or self.frame is None:
                            print("Failed to capture frame")
                            self.frame = None
                            continue
                        # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                        if not self.ic4_rotate_180:
                            self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)

                        self.frame_scaled = cv2.resize(self.frame, (self.scaled_width, self.scaled_height), interpolation=cv2.INTER_LINEAR)
                    except cv2.error as e:
                        print("An error occurred while reading frames from the cameras:", str(e))

                self.calib_config.cameraID = self.calib_config.widget - 1 #Don't forget to clean this up in deployment !!
            
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
                    camera_index = max(self.calib_config.widget - 1, 0)
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
                    self.CalibCamStream.emit(self.convertQImage(self.frame_downsampled))
            
            if self.calib_config.widget == 3:
                if self.multiCam_stream is False:
                    self.multiCam_stream = True

                    # Leaving single-camera mode: release currently opened single camera first.
                    if self.cap_cam is not None:
                        self.camera_service.release_all(CameraRole.CALIBRATION)
                        self.cap_cam = None
                        self.current_cameraID = -1

                    self.initialize_all_camera()

                # Ensure merge camera handles exist, then read from the IC4 handles.
                if self.cap_cam_ic4_1 is None or self.cap_cam_ic4_2 is None:
                    self.initialize_all_camera()

                ok1, mergeframe1 = self.read_camera_frame(0)
                ok2, mergeframe2 = self.read_camera_frame(1)
                if not (ok1 and isinstance(mergeframe1, np.ndarray)) or not (
                    ok2 and isinstance(mergeframe2, np.ndarray)
                ):
                    # One or both cameras are unavailable; keep the previous frame
                    # instead of feeding None into the geometry pipeline.
                    if self.mergeframe1 is None or self.mergeframe2 is None:
                        self.msleep(50)
                        continue
                else:
                    self.mergeframe1 = mergeframe1
                    self.mergeframe2 = mergeframe2

                #Calculate all map from calibration matrix for 5 cameras, thus i in range(1, 6)
                for i in range(0, 2):
                    if self.calib_config.mapCalculated[i] is False:
                        if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                            try:
                                camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                            except ValueError as e:
                                print(f"[Calibration] Camera {i}: {e}")
                                continue
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                            self.calib_config.map1[i], self.calib_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated")
                            self.calib_config.mapCalculated[i] = True
                            print(f"Calibration map is calculated for Camera {i}")

                if all(self.calib_config.mapCalculated[i] for i in range(0, 2)):
                    # print("All calibration maps are calculated.")
                    self.mergeframe1 = cv2.remap(self.mergeframe1, self.calib_config.map1[0], self.calib_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe2 = cv2.remap(self.mergeframe2, self.calib_config.map1[1], self.calib_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

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

                cv2.imwrite("combinedImagev2.png", self.combinedImage)

                if self.calib_config.savePlanarize is True:
                    self.calib_config.savePlanarize = False
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


                # self.combinedImage = cv2.resize(self.combinedImage, (self.homography_size[1], int(self.homography_size[0]/1.26)))
                self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.5)), int(self.homography_size[0]/(self.scale_factor*1.5))))
                self.combinedImage_high_scaled = cv2.resize(self.combinedImage_high_scaled, (int(self.homography_size[1]/(self.scale_factor*1.5)), int(self.homography_size[0]/(self.scale_factor*1.5))))
                

                self.mergeframe1_downsampled = self.downSampling(self.mergeframe1, 246, 163)
                self.mergeframe2_downsampled = self.downSampling(self.mergeframe2, 246, 163)

                if self.mergeframe1_downsampled is not None:
                    self.CamMerge1.emit(self.convertQImage(self.mergeframe1_downsampled))
                if self.mergeframe2_downsampled is not None:
                    self.CamMerge2.emit(self.convertQImage(self.mergeframe2_downsampled))
                if self.combinedImage_scaled is not None:
                    self.CamMergeAll.emit(self.convertQImage(self.combinedImage_scaled))

            #wait for 5ms
            # self.msleep(2)
            
        print(f"Camera {self.calib_config.cameraID} released.")

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

        try:
            self.settings.bootstrap()
        except Exception as e:  # pragma: no cover - defensive
            print(f"[Calibration] settings bootstrap failed: {e}")

        repository = self.settings.repository
        for camera_index, suffix in ((0, "1"), (1, "2")):
            full = repository.load_homography(camera_index)
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

    def create_placeholder_image(self):
        # Create a small black image with a white dot in the center
        size = 100
        placeholder = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(placeholder, (size // 2, size // 2), 10, (255, 255, 255), -1)
        return placeholder

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
        self.release_all_camera()
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
