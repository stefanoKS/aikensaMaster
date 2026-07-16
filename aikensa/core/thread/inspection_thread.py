import inspect
import cv2
import os
from datetime import datetime
import numpy as np
import yaml
import time
import logging
import sqlite3
import mysql.connector
from ast import literal_eval

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from aikensa.core.scripts.cam_init.cam_init_ic4 import open_ic4_camera_or_placeholder, DummyCapture
from aikensa.core.scripts.img_processing.cameracalibrate import warpTwoImages_template
from dataclasses import dataclass, field
from typing import List

from aikensa.core.config.sound.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound, play_ok_sound, play_ng_sound, play_ok_count_10_sound

from ultralytics import YOLO

# NEW: Registry-based part inspection system
from aikensa.core.scripts.part_inspector_registry import get_registry

# LEGACY: Keep old imports for backward compatibility during transition
from aikensa.core.config.parts.MMC.M_5A45.P828XXW0X0P_CTRPLR import partcheck as P828XXW0X0P_check
from aikensa.core.config.parts.NISSAN.M_JC2D.P808387UA1A import partcheck as P808387UA1A_check
from aikensa.core.config.parts.NISSAN.M_JC2D.P828447UA0A import partcheck as P828447UA0A_check
from aikensa.core.config.parts.NISSAN.M_J42U.P731957YA0A_SEALROOF import partcheck as P731957YA0A_check
from aikensa.core.config.parts.NISSAN.M_J42U.P808387YA0A_SEALFRDOORPARTING import partcheck as P808387YA0A_check
from aikensa.core.config.parts.NISSAN.M_J42U.P808387YA0A_P828387YA0A_KEYPOINT import partcheck as P808387YA0A_P828387YA0A_keypoint_check
from aikensa.core.config.parts.NISSAN.M_J42U.P828387YA6A_SEALRRDOORPARTINGRRSTEP import partcheck as P828387YA6A_check
from aikensa.core.config.parts.NISSAN.M_J42U.P828387YA1A_SEALRRDOORPARTINGLOCK import partcheck as P828387YA1A_check
from aikensa.core.config.parts.NISSAN.M_J42U.P828387YA6A_SEALRRDOORPARTINGRRSTEP_KATABU_NASHI import partcheck as P828387YA6A_KATABU_NASHI_check
from aikensa.core.config.parts.NISSAN.M_J42U.P658107Y0A_SEALASSYRADCORE import partcheck as P658107YA0A_check
from aikensa.core.config.parts.NICHIJOU_TENKEN import debug_images_enabled as NICHIJOU_TENKEN_debug_images_enabled
from aikensa.core.config.parts.P658207LE0A import debug_images_enabled as P658207LE0A_debug_images_enabled
from aikensa.core.config.parts.P658207LE0A import partcheck as P658207LE0A_check
from aikensa.core.config.parts.NICHIJOU_TENKEN import partcheck as dailyTenken

from PIL import ImageFont, ImageDraw, Image

@dataclass
class InspectionConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected

    mapCalculated: list = field(default_factory=lambda: [False]*30) #for 30 cameras
    map1: list = field(default_factory=lambda: [None]*30) #for 30 cameras
    map2: list = field(default_factory=lambda: [None]*30) #for 30 cameras

    map1_downscaled: list = field(default_factory=lambda: [None]*30) #for 30 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*30) #for 30 cameras

    doInspection: bool = False
    button_sensor: int = 0

    kensainNumber: str = None
    ppmsnumber : str = None
    furyou_plus: bool = False
    furyou_minus: bool = False
    kansei_plus: bool = False
    kansei_minus: bool = False
    furyou_plus_10: bool = False #to add 10
    furyou_minus_10: bool = False
    kansei_plus_10: bool = False
    kansei_minus_10: bool = False

    counterReset: bool = False
    nichijoutenken_enabled: list = field(default_factory=lambda: [False for _ in range(50)])

    today_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(50)])
    current_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(50)])

class InspectionThread(QThread):

    partCam = pyqtSignal(QImage)
    partKatabuL = pyqtSignal(QImage)
    partKatabuR = pyqtSignal(QImage)

    modelErrorSignal = pyqtSignal(str)  # Signal for model error messages

    P82833W050P_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W040P_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82833W090P_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W080P_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P808387UA1A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828447UA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    
    P82833W050PKENGEN_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W040PKENGEN_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82833W090PKENGEN_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W080PKENGEN_InspectionResult_PitchMeasured = pyqtSignal(list, list)

    P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured = pyqtSignal(list, list)

    P658217UA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P658207UA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P658217UJ0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P658207UJ0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P731957YA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P808387YA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828387YA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828387YA1A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828397YA1A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828387YA6A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828397YA6A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828387YA6A_KATABU_NASHI_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828397YA6A_KATABU_NASHI_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P658107YA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)

    P658207LE0A_InspectionResult = pyqtSignal(list, list)  # P13C Part

    today_numofPart_signal = pyqtSignal(list)
    current_numofPart_signal = pyqtSignal(list)

    ethernetStatus = pyqtSignal(list)

    pickingOrderSignal = pyqtSignal(list)

    def __init__(self, inspection_config: InspectionConfig = None):
        super(InspectionThread, self).__init__()
        self.running = True

        # ---- Config ----
        self.inspection_config = inspection_config or InspectionConfig()

        # ---- NEW: Initialize Part Registry ----
        try:
            self.part_registry = get_registry()
            logging.info(f"Part registry loaded: {len(self.part_registry.get_all_parts())} parts registered")
        except Exception as e:
            logging.warning(f"Failed to load part registry: {e}. Using legacy mode.")
            self.part_registry = None

        # ---- Simple constants / defaults ----
        self.kanjiFontPath = "aikensa/gui/resources/font/NotoSansJP-ExtraBold.ttf"
        self._today_str = datetime.now().strftime("%Y%m%d")
        self.multiCam_stream = False
        self._last_color_debug_ts = 0.0
        self._color_debug_interval_sec = 5.0

        # ---- Timing ----
        self.pickingTimerStart = time.time()
        self.pickingWaitTime = 3.0

        self.InspectionWaitTime = 1.0
        self.InspectionTimeStart = None

        self.bool_keep_measurement = False
        self._inspection_start_image = None

        self._init_geometry_defaults()
        self._init_placeholders()

        # ---- Core state groups ----
        self._init_cameras()
        self._init_frames()
        self._init_homography()
        self._init_planarize()
        self._init_images_and_crops()
        self._init_hand_state()
 
        # ---- Derived geometry ----
        self._init_scaled_geometry()

        # ---- Inspection buffers/results ----
        self._init_inspection_images()
        self._init_results()

        # ---- Widget maps / indices ----
        self._init_widget_maps()
        self._load_widget_crop_points()
        self._load_widget_h2_offsets()

        # ---- Trigger/order ----
        self._init_clip_order()

        # ---- Sounyuuki specific ----
        self._init_sounyuuki()

        # ---- Optional defaults YAML (safe: only if file exists) ----
        # You can create this file later to further reduce hardcoded values.
        self._load_optional_defaults("aikensa/core/config/inspection_defaults.yaml")

        # ---- Camera map YAML ----
        self.cam_config_file = "aikensa/core/config/cam/cam_config.yaml"
        self._load_cam_map()

        # ---- MySQL credentials YAML ----
        self._load_mysql_credentials()



    def release_all_camera(self):
        if self.cap_cam_ic4_1 is not None:
            self.cap_cam_ic4_1.release()
            print(f"Camera 1 released.")
        if self.cap_cam_ic4_2 is not None:
            self.cap_cam_ic4_2.release()
            print(f"Camera 2 released.")

    def initialize_all_camera(self):
        """
        Initialize both cameras. Now uses DummyCapture for missing cameras.
        """
        self.cap_cam_ic4_1 = open_ic4_camera_or_placeholder(
            "11620526", width=3072, height=2048, fps=15,
            wb_temperature=4500,
            auto_exposure=False,
            auto_gain=False,
            auto_wb=False,
            rotate_180=self.ic4_rotate_180
        )
        self.cap_cam_ic4_2 = open_ic4_camera_or_placeholder(
            "11620167", width=3072, height=2048, fps=15,
            wb_temperature=4500,
            auto_exposure=False,
            auto_gain=False,
            auto_wb=False,
            rotate_180=self.ic4_rotate_180
        )

        # Cameras are always valid (either real or dummy), so just report status
        cam1_type = "Real" if not isinstance(self.cap_cam_ic4_1, DummyCapture) else "Placeholder"
        cam2_type = "Real" if not isinstance(self.cap_cam_ic4_2, DummyCapture) else "Placeholder"
        
        print(f"Camera 1: {cam1_type}")
        print(f"Camera 2: {cam2_type}")


    def run(self):
        #initialize the database
        if not os.path.exists("./aikensa/inspection_results"):
            os.makedirs("./aikensa/inspection_results")

        self.conn = sqlite3.connect('./aikensa/inspection_results/database_results.db')
        self.cursor = self.conn.cursor()

        # Create the table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            partName TEXT,
            numofPart TEXT,
            currentnumofPart TEXT,
            timestampHour TEXT,
            timestampDate TEXT,
            deltaTime REAL,
            kensainName TEXT,
            detected_pitch TEXT,
            delta_pitch TEXT,
            total_length REAL,
            resultpitch TEXT,
            status TEXT,
            NGreason TEXT,
            ClipInsertionMachine TEXT,
            PPMS TEXT
        )
        ''')

        # List of columns to add
        columns_to_add = [
            ("resultpitch", "TEXT"),
            ("status", "TEXT"),
            ("NGreason", "TEXT"),
            ("PPMS", "TEXT"),
        ]

        # Using the function to add columns
        self.add_columns(self.cursor, "inspection_results", columns_to_add)

        self.conn.commit()

        #Initialize connection to mysql server if available
        try:
            self.mysql_conn = mysql.connector.connect(
                host=self.mysqlHost,
                user=self.mysqlID,
                password=self.mysqlPassword,
                port=self.mysqlHostPort,
                database="AIKENSAresults"
            )
            print(f"Connected to MySQL database at {self.mysqlHost}")
        except Exception as e:
            print(f"Error connecting to MySQL database: {e}")
            self.mysql_conn = None

        #try adding data to the schema in mysql
        if self.mysql_conn is not None:
            self.mysql_cursor = self.mysql_conn.cursor()
            self.mysql_cursor.execute('''
            CREATE TABLE IF NOT EXISTS inspection_results (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                partName TEXT,
                numofPart TEXT,
                currentnumofPart TEXT,
                timestampHour TEXT,
                timestampDate TEXT,
                deltaTime REAL,
                kensainName TEXT,
                detected_pitch TEXT,
                delta_pitch TEXT,
                total_length REAL,
                resultpitch TEXT,
                status TEXT,
                NGreason TEXT,
                ClipInsertionMachine TEXT,
                PPMS TEXT
            )
            ''')
            self.mysql_conn.commit()

        print("Inspection Thread Started")
        self.initialize_model()
        print("AI Models Initialized")

        self.current_cameraID = self.inspection_config.cameraID
        self._save_dir = f"aikensa/core/param/camera/"

        self.homography_template = cv2.imread("aikensa/core/tools/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])
        self.homography_size_scaled = (self.homography_template.shape[0]//5, self.homography_template.shape[1]//5)

        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        for key, value in self.widget_dir_map.items():
            self.inspection_config.current_numofPart[key] = self.get_last_entry_currentnumofPart(value)
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        if os.path.exists("./aikensa/core/param/camera/homography_param_cam1.yaml"):
            with open("./aikensa/core/param/camera/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)
                print(f"Loaded homography matrix for camera 1")

        if os.path.exists("./aikensa/core/param/camera/homography_param_cam2.yaml"):
            with open("./aikensa/core/param/camera/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)
                print(f"Loaded homography matrix for camera 2")

        if os.path.exists("./aikensa/core/param/camera/homography_param_cam1_scaled.yaml"):
            with open("./aikensa/core/param/camera/homography_param_cam1_scaled.yaml") as file:
                self.homography_matrix1_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_scaled = np.array(self.homography_matrix1_scaled)
                print(f"Loaded scaled homography matrix for camera 1")

        if os.path.exists("./aikensa/core/param/camera/homography_param_cam2_scaled.yaml"):
            with open("./aikensa/core/param/camera/homography_param_cam2_scaled.yaml") as file:
                self.homography_matrix2_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_scaled = np.array(self.homography_matrix2_scaled)
                print(f"Loaded scaled homography matrix for camera 2")


        if os.path.exists("./aikensa/core/param/camera/planarizeTransform_wide.yaml"):
            with open("./aikensa/core/param/camera/planarizeTransform_wide.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_wide = np.array(transform_list)

        if os.path.exists("./aikensa/core/param/camera/planarizeTransform_wide_scaled.yaml"):
            with open("./aikensa/core/param/camera/planarizeTransform_wide_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_wide_scaled = np.array(transform_list)

        while self.running:

            self._check_date_rollover()

            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget > 0:

                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()
                    # print("initialize all camera")    

                ret1, frame1 = self.cap_cam_ic4_1.read()
                ret2, frame2 = self.cap_cam_ic4_2.read()

                self.mergeframe1 = frame1 if ret1 and isinstance(frame1, np.ndarray) else self._placeholder_cam1.copy()
                self.mergeframe2 = frame2 if ret2 and isinstance(frame2, np.ndarray) else self._placeholder_cam2.copy()

                # Periodically log color channel and camera setting values for quick tint diagnosis.
                now_ts = time.time()
                if now_ts - self._last_color_debug_ts >= self._color_debug_interval_sec:
                    self._log_camera_color_debug(self.mergeframe1, self.cap_cam_ic4_1, "Cam1")
                    self._log_camera_color_debug(self.mergeframe2, self.cap_cam_ic4_2, "Cam2")
                    self._last_color_debug_ts = now_ts

                #debug save image

                #Downsampled the image
                self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
                self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)

                if self.inspection_config.mapCalculated[1] is False:  # Only checking the first camera for efficiency
                    for i in range(0, 2): #Make sure to check the camID
                        calib_file = self._save_dir + f"Calibration_camera_{i}.yaml"
                        if os.path.exists(calib_file):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(calib_file)
                            h, w = self.mergeframe1.shape[:2]
                            self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(
                                camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2
                            )
                            self.inspection_config.mapCalculated[i] = True
                            print(f"Calibration map calculated for Camera {i}")

                            scaled_file = self._save_dir + f"Calibration_camera_scaled_{i}.yaml"

                            if os.path.exists(scaled_file):
                                camera_matrix, dist_coeffs = self.load_matrix_from_yaml(scaled_file)
                                h, w = self.mergeframe1_scaled.shape[:2]
                                self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(
                                    camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2
                                )
                                print(f"Downscaled calibration map calculated for Camera {i}")
                            else:
                                print(f"Error: Scaled calibration file {scaled_file} does not exist.")

                if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time

                    has_downscaled_maps = all(v is not None for v in [
                        self.inspection_config.map1_downscaled[0],
                        self.inspection_config.map2_downscaled[0],
                        self.inspection_config.map1_downscaled[1],
                        self.inspection_config.map2_downscaled[1],
                    ])

                    if has_downscaled_maps:
                        self.mergeframe1_scaled = cv2.remap(
                            self.mergeframe1_scaled,
                            self.inspection_config.map1_downscaled[0],
                            self.inspection_config.map2_downscaled[0],
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                        )
                        self.mergeframe2_scaled = cv2.remap(
                            self.mergeframe2_scaled,
                            self.inspection_config.map1_downscaled[1],
                            self.inspection_config.map2_downscaled[1],
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                        )

                    if self.inspection_config.widget in self.inspection_widget_indices: # no need to emit katabu
                        h2_scaled_for_widget = self._get_cam2_homography_with_widget_offset(
                            self.H2_scaled,
                            widget_id=self.inspection_config.widget,
                            scaled=True,
                        )
                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, h2_scaled_for_widget)
                        crop_points_scaled = self._get_crop_points_for_widget(self.inspection_config.widget, scaled=True)
                        self.combinedImage_scaled = self._crop_image_by_4points(self.combinedImage_scaled, crop_points_scaled)
                       
                        if self.inspection_config.widget in self.inspection_widget_katabu: # emit katabu
                            if self.inspection_config.widget in self.inspection_widget_katabu_L: #5,7,9,11
                                #katabu L is blank
                                #katabu R is cropped image
                                katabu_crop_r = self._get_katabu_crop_for_widget(self.inspection_config.widget, side="R")
                                self.katabuImageL_scaled = self.createBlackImage(width=256, height=128)
                                self.katabuImageR_scaled = self.frameCrop(self.combinedImage_scaled, katabu_crop_r[0]/self.scale_factor, katabu_crop_r[1]/self.scale_factor, katabu_crop_r[2]/self.scale_factor, katabu_crop_r[3]/self.scale_factor, katabu_crop_r[4], katabu_crop_r[5])
    
                            if self.inspection_config.widget in self.inspection_widget_katabu_R: #6,8,10,12
                                #katabu L is cropped image
                                #katabu R is blank
                                katabu_crop_l = self._get_katabu_crop_for_widget(self.inspection_config.widget, side="L")
                                self.katabuImageL_scaled = self.frameCrop(self.combinedImage_scaled, katabu_crop_l[0]/self.scale_factor, katabu_crop_l[1]/self.scale_factor, katabu_crop_l[2]/self.scale_factor, katabu_crop_l[3]/self.scale_factor, katabu_crop_l[4], katabu_crop_l[5])
                                self.katabuImageR_scaled = self.createBlackImage(width=256, height=128)

                            self.katabuImageL_scaled = self.convertQImage(self.katabuImageL_scaled)
                            self.katabuImageR_scaled = self.convertQImage(self.katabuImageR_scaled)

                            self.partKatabuL.emit(self.katabuImageL_scaled)
                            self.partKatabuR.emit(self.katabuImageR_scaled)

                    self.InspectionResult_PitchMeasured = [None]*50
                    self.InspectionResult_PitchResult = [None]*50
                    self.InspectionResult_DeltaPitch = [None]*50

                    if self.combinedImage_scaled is not None:
                        preview_width, preview_height = self._get_partcam_preview_size()
                        self.combinedImage_scaled = self.downSampling(
                            self.combinedImage_scaled,
                            width=preview_width,
                            height=preview_height,
                        )

                    self.partCam.emit(self.convertQImage(self.combinedImage_scaled))
        
                    self.P82833W050P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W040P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82833W090P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W080P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                    self.P82833W050PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W040PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82833W090PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W080PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                    self.P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                    self.P808387UA1A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828447UA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                    self.P658217UA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P658207UA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P658217UJ0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P658207UJ0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                    self.P808387YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828387YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828387YA1A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828397YA1A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828387YA6A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828397YA6A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828387YA6A_KATABU_NASHI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828397YA6A_KATABU_NASHI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

            #for the kengen
            if self.inspection_config.widget in [9, 10, 11, 12]:    
                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)

                if self._begin_inspection_cycle(print_start=True):
                    print(self.inspection_config.kensainNumber)


                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()
                            self._prepare_wide_inspection_images(status_text="検査中", x_offset=-200, y_offset=-100)

                            if self.inspection_config.widget in [5, 6, 7, 8, 9, 10, 11, 12]: # emit katabu
                                if self.inspection_config.widget in [5, 7, 9, 11]:
                                    #katabu L is blank
                                    #katabu R is cropped image
                                    self.katabuImageL = self.createBlackImage(width=256, height=128)
                                    self.katabuImageR = self.frameCrop(self.combinedImage, self.katabuImageR_Crop[0], self.katabuImageR_Crop[1], self.katabuImageR_Crop[2], self.katabuImageR_Crop[3], self.katabuImageR_Crop[4], self.katabuImageR_Crop[5])
                                    self.katabuImage = self.katabuImageR.copy()
                                    self.katabuImage_init = self.katabuImageR.copy()
                                if self.inspection_config.widget in [6, 8, 10, 12]: 
                                    #katabu L is cropped image
                                    #katabu R is blank
                                    self.katabuImageL = self.frameCrop(self.combinedImage, self.katabuImageL_Crop[0], self.katabuImageL_Crop[1], self.katabuImageL_Crop[2], self.katabuImageL_Crop[3], self.katabuImageL_Crop[4], self.katabuImageL_Crop[5])
                                    self.katabuImageR = self.createBlackImage(width=256, height=128)
                                    self.katabuImage = self.katabuImageL.copy()
                                    self.katabuImage_init = self.katabuImageL.copy()

                                self.partKatabuL.emit(self.convertQImage(self.katabuImageL))
                                self.partKatabuR.emit(self.convertQImage(self.katabuImageR))

                            for i in range(len(self.InspectionImages)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P828XXW0X0P_CLIP_Model'):
                                    self.inspection_config.doInspection = False
                                    break
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P828XXW0X0P_CLIP_Model, 
                                                slice_height=1920, slice_width=1920, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P828XXW0X0P_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    self.inspection_config.doInspection = False
                                    break
                                
                                if self.inspection_config.widget in [5, 7, 9, 11]:
                                    # Check if KATABU model is available
                                    if not self._check_model_available('P828XXW0X0P_KATABU_Model'):
                                        continue
                                    
                                    try:
                                        self.InspectionResult_KatabuDetection = self.P828XXW0X0P_KATABU_Model(cv2.cvtColor(self.katabuImage, cv2.COLOR_BGR2RGB),
                                                                                                            stream=True,
                                                                                                            verbose=False,
                                                                                                            conf=0.1,
                                                                                                            iou=0.5)
                                    except Exception as e:
                                        print(f"[Inference Error] P828XXW0X0P_KATABU_Model: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.inspection_config.doInspection = False
                                        break

                                if self.inspection_config.widget in [6, 8, 10, 12]:
                                    # Check if KATABU model is available
                                    if not self._check_model_available('P828XXW0X0P_KATABU_Model'):
                                        continue
                                    
                                    try:
                                        self.InspectionResult_KatabuDetection = self.P828XXW0X0P_KATABU_Model(cv2.cvtColor(self.katabuImage, cv2.COLOR_BGR2RGB),
                                                                                                            stream=True,
                                                                                                            verbose=False,
                                                                                                            conf=0.1,
                                                                                                            iou=0.5)
                                    except Exception as e:
                                        print(f"[Inference Error] P828XXW0X0P_KATABU_Model: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.inspection_config.doInspection = False
                                        break
                                    
                                self.InspectionImages[i], self.InspectionImagesKatabu[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i]  = P828XXW0X0P_check(self.InspectionImages[i], self.katabuImage,
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_KatabuDetection,
                                                                                                                                                                                                                self.widget_name_map[self.inspection_config.widget])


                                for i in range(len(self.InspectionResult_Status)):
                                    if self.InspectionResult_Status[i] == "OK": 
                                        # Increment the 'OK' count at the appropriate index (1)
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1
                                        play_ok_sound()

                                    elif self.InspectionResult_Status[i] == "NG": 
                                        # Increment the 'NG' count at the appropriate index (0)
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1
                                        play_ng_sound()


                            self.emitImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=428)
                            self.partCam.emit(self.convertQImage(self.emitImages[0]))
                            if self.inspection_config.widget in [5, 7, 9, 11]:
                                self.partKatabuR.emit(self.convertQImage(self.InspectionImagesKatabu[0]))
                            if self.inspection_config.widget in [6, 8, 10, 12]: 
                                self.partKatabuL.emit(self.convertQImage(self.InspectionImagesKatabu[0]))

                            self.P82833W050PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82832W040PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82833W090PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82832W080PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

                            # self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])
                            self.save_image_result_withKatabu(self.combinedImage, self.InspectionImages[0], self.katabuImage_init, self.InspectionImagesKatabu[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0,
                                    resultPitch = self.InspectionResult_PitchResult[0], 
                                    status = self.InspectionResult_Status[0], 
                                    NGreason = self.InspectionResult_NGReason[0])

                            time.sleep(1.5)
          
            #for clip insertion  machine
            if self.inspection_config.widget in [13, 14, 15, 16]:    
                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)

                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()
                            self._prepare_wide_inspection_images(status_text="検査中", x_offset=-200, y_offset=-100)

                            for i in range(len(self.InspectionImages)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P828XXW0X0P_CLIP_Model'):
                                    continue
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P828XXW0X0P_CLIP_Model, 
                                                slice_height=1920, slice_width=1920, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P828XXW0X0P_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    continue

                                    
                                self.InspectionImages[i], _, self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i]  = P828XXW0X0P_check(self.InspectionImages[i], None,
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                None,
                                                                                                                                                                                                                self.widget_name_map[self.inspection_config.widget])


                                for i in range(len(self.InspectionResult_Status)):
                                    if self.InspectionResult_Status[i] == "OK": 
                                        # Increment the 'OK' count at the appropriate index (1)
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1
                                        play_ok_sound()

                                    elif self.InspectionResult_Status[i] == "NG": 
                                        # Increment the 'NG' count at the appropriate index (0)
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1
                                        play_ng_sound()

                            self.emitImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=428)
                            self.partCam.emit(self.convertQImage(self.emitImages[0]))

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

                            self.P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0,
                                    resultPitch = self.InspectionResult_PitchResult[0], 
                                    status = self.InspectionResult_Status[0], 
                                    NGreason = self.InspectionResult_NGReason[0],
                                    PPMS = self.inspection_config.ppmsnumber)

                            self.bool_keep_measurement = True

                            time.sleep(1.5)

                if self.inspection_config.doInspection is True and self.bool_keep_measurement is True:
                    self.bool_keep_measurement = False

                if self.bool_keep_measurement == True:
                    self.P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.partCam.emit(self.convertQImage(self.InspectionImages[0]))
          
            #for the P808387UA1A (widget 17)
            if self.inspection_config.widget in [17]:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-90)

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                                                                    planarize_transform=self.planarizeTransform_wide,
                                                                                    planarize_size=self.wide_planarize,
                                                                                    h1=self.H1,
                                                                                    h2=self.H2,
                                                                                    blank_canvas=self.homography_blank_canvas
                                                                                )

                            for i in range(len(self.InspectionImages_bgr)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P808387UA1A_CLIP_Model'):
                                    continue
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P808387UA1A_CLIP_Model, 
                                                slice_height=512, slice_width=1980, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P808387UA1A_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    continue

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, :512, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -512:, :]
                                
                                # Check if keypoint model is available
                                if not self._check_model_available('P808387UA1A_keypoint'):
                                    self.InspectionResult_keypoint_Left[i] = None
                                    self.InspectionResult_keypoint_Right[i] = None
                                else:
                                    try:
                                        self.InspectionResult_keypoint_Left[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Left[i], conf=0.6, imgsz=512, verbose=False)
                                        self.InspectionResult_keypoint_Right[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Right[i], conf=0.6, imgsz=512, verbose=False)
                                    except Exception as e:
                                        print(f"[Inference Error] P808387UA1A_keypoint: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.InspectionResult_keypoint_Left[i] = None
                                        self.InspectionResult_keypoint_Right[i] = None

                                # Check if anomaly classification model is available
                                anomaly_model = self.P828447UA0A_ANOMALY_CLASSIFICATION_Model if self._check_model_available('P828447UA0A_ANOMALY_CLASSIFICATION_Model') else None

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DeltaPitch[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P808387UA1A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Left[i],
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Right[i],
                                                                                                                                                                                                                                                anomaly_model)

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 169),
                                pitch_signal=self.P808387UA1A_InspectionResult_PitchMeasured
                            )
                            time.sleep(1.5)

            #for the P828447UA0A (widget 18)
            if self.inspection_config.widget in [18]:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-90)

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                                                                    planarize_transform=self.planarizeTransform_wide,
                                                                                    planarize_size=self.wide_planarize,
                                                                                    h1=self.H1,
                                                                                    h2=self.H2,
                                                                                    blank_canvas=self.homography_blank_canvas
                                                                                )

                            for i in range(len(self.InspectionImages_bgr)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P828447UA0A_CLIP_Model'):
                                    continue
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P828447UA0A_CLIP_Model, 
                                                slice_height=256, slice_width=1980, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P828447UA0A_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    continue

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, :512, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -512:, :]
                                
                                # Check if keypoint model is available
                                if not self._check_model_available('P808387UA1A_keypoint'):
                                    self.InspectionResult_keypoint_Left[i] = None
                                    self.InspectionResult_keypoint_Right[i] = None
                                else:
                                    try:
                                        self.InspectionResult_keypoint_Left[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Left[i], conf=0.6, imgsz=648, verbose=False)
                                        self.InspectionResult_keypoint_Right[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Right[i], conf=0.6, imgsz=648, verbose=False)
                                    except Exception as e:
                                        print(f"[Inference Error] P808387UA1A_keypoint: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.InspectionResult_keypoint_Left[i] = None
                                        self.InspectionResult_keypoint_Right[i] = None

                                # Check if anomaly classification model is available
                                anomaly_model = self.P828447UA0A_ANOMALY_CLASSIFICATION_Model if self._check_model_available('P828447UA0A_ANOMALY_CLASSIFICATION_Model') else None

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DeltaPitch[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P828447UA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Left[i],
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Right[i],
                                                                                                                                                                                                                                                anomaly_model)

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 169),
                                pitch_signal=self.P828447UA0A_InspectionResult_PitchMeasured
                            )
                            time.sleep(1.5)

            #for P658217UA0A (widget 30)
            if self.inspection_config.widget in [30]:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-90)

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                                                                    planarize_transform=self.planarizeTransform_wide,
                                                                                    planarize_size=self.wide_planarize,
                                                                                    h1=self.H1,
                                                                                    h2=self.H2_high,
                                                                                    blank_canvas=self.homography_blank_canvas
                                                                                )

                            for i in range(len(self.InspectionImages)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P658217UA0A_CLIP_Model'):
                                    continue
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P658217UA0A_CLIP_Model, 
                                                slice_height=256, slice_width=1980, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P658217UA0A_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    continue

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, :512, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -512:, :]
                                
                                # Check if keypoint model is available
                                if not self._check_model_available('P808387UA1A_keypoint'):
                                    self.InspectionResult_keypoint_Left[i] = None
                                    self.InspectionResult_keypoint_Right[i] = None
                                else:
                                    try:
                                        self.InspectionResult_keypoint_Left[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Left[i], conf=0.6, imgsz=512, verbose=False)
                                        self.InspectionResult_keypoint_Right[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Right[i], conf=0.6, imgsz=512, verbose=False)
                                    except Exception as e:
                                        print(f"[Inference Error] P808387UA1A_keypoint: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.InspectionResult_keypoint_Left[i] = None
                                        self.InspectionResult_keypoint_Right[i] = None

                                # Check if anomaly classification model is available
                                anomaly_model = self.P828447UA0A_ANOMALY_CLASSIFICATION_Model if self._check_model_available('P828447UA0A_ANOMALY_CLASSIFICATION_Model') else None

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DeltaPitch[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P828447UA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Left[i],
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Right[i],
                                                                                                                                                                                                                                                anomaly_model)
                        
                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 169),
                                pitch_signal=self.P658217UA0A_InspectionResult_PitchMeasured
                            )
                            time.sleep(1.5)

            #for P658217UA0A (widget 31)
            if self.inspection_config.widget in [31]:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-90)

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                                                                    planarize_transform=self.planarizeTransform_wide,
                                                                                    planarize_size=self.wide_planarize,
                                                                                    h1=self.H1,
                                                                                    h2=self.H2_high,
                                                                                    blank_canvas=self.homography_blank_canvas
                                                                                )

                            for i in range(len(self.InspectionImages)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P658217UA0A_CLIP_Model'):
                                    continue
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P658217UA0A_CLIP_Model, 
                                                slice_height=256, slice_width=1980, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P658217UA0A_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    continue

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, :512, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -512:, :]
                                
                                # Check if keypoint model is available
                                if not self._check_model_available('P808387UA1A_keypoint'):
                                    self.InspectionResult_keypoint_Left[i] = None
                                    self.InspectionResult_keypoint_Right[i] = None
                                else:
                                    try:
                                        self.InspectionResult_keypoint_Left[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Left[i], conf=0.6, imgsz=512, verbose=False)
                                        self.InspectionResult_keypoint_Right[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Right[i], conf=0.6, imgsz=512, verbose=False)
                                    except Exception as e:
                                        print(f"[Inference Error] P808387UA1A_keypoint: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.InspectionResult_keypoint_Left[i] = None
                                        self.InspectionResult_keypoint_Right[i] = None

                                # Check if anomaly classification model is available
                                anomaly_model = self.P828447UA0A_ANOMALY_CLASSIFICATION_Model if self._check_model_available('P828447UA0A_ANOMALY_CLASSIFICATION_Model') else None

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DeltaPitch[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P828447UA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Left[i],
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Right[i],
                                                                                                                                                                                                                                                anomaly_model)

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 169),
                                pitch_signal=self.P658217UA0A_InspectionResult_PitchMeasured
                            )
                            time.sleep(1.5)

            #for P658207UA0A (widget 32)
            if self.inspection_config.widget in [32]:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-90)

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                                                                    planarize_transform=self.planarizeTransform_wide,
                                                                                    planarize_size=self.wide_planarize,
                                                                                    h1=self.H1,
                                                                                    h2=self.H2_high,
                                                                                    blank_canvas=self.homography_blank_canvas
                                                                                )

                            for i in range(len(self.InspectionImages)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P658207UA0A_CLIP_Model'):
                                    continue
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P658207UA0A_CLIP_Model, 
                                                slice_height=256, slice_width=1980, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P658207UA0A_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    continue

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, :512, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -512:, :]
                                
                                # Check if keypoint model is available
                                if not self._check_model_available('P808387UA1A_keypoint'):
                                    self.InspectionResult_keypoint_Left[i] = None
                                    self.InspectionResult_keypoint_Right[i] = None
                                else:
                                    try:
                                        self.InspectionResult_keypoint_Left[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Left[i], conf=0.6, imgsz=512, verbose=False)
                                        self.InspectionResult_keypoint_Right[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Right[i], conf=0.6, imgsz=512, verbose=False)
                                    except Exception as e:
                                        print(f"[Inference Error] P808387UA1A_keypoint: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.InspectionResult_keypoint_Left[i] = None
                                        self.InspectionResult_keypoint_Right[i] = None

                                # Check if anomaly classification model is available
                                anomaly_model = self.P828447UA0A_ANOMALY_CLASSIFICATION_Model if self._check_model_available('P828447UA0A_ANOMALY_CLASSIFICATION_Model') else None

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DeltaPitch[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P828447UA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Left[i],
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Right[i],
                                                                                                                                                                                                                                                anomaly_model)

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 169),
                                pitch_signal=self.P658207UA0A_InspectionResult_PitchMeasured
                            )
                            time.sleep(1.5)

            #for P658217UJ0A (widget 33)
            if self.inspection_config.widget in [33]:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-90)

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                                                                    planarize_transform=self.planarizeTransform_wide,
                                                                                    planarize_size=self.wide_planarize,
                                                                                    h1=self.H1,
                                                                                    h2=self.H2_high,
                                                                                    blank_canvas=self.homography_blank_canvas
                                                                                )

                            for i in range(len(self.InspectionImages)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P658217UJ0A_CLIP_Model'):
                                    continue
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P658217UJ0A_CLIP_Model, 
                                                slice_height=256, slice_width=1980, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P658217UJ0A_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    continue

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, :512, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -512:, :]
                                
                                # Check if keypoint model is available
                                if not self._check_model_available('P808387UA1A_keypoint'):
                                    self.InspectionResult_keypoint_Left[i] = None
                                    self.InspectionResult_keypoint_Right[i] = None
                                else:
                                    try:
                                        self.InspectionResult_keypoint_Left[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Left[i], conf=0.6, imgsz=512, verbose=False)
                                        self.InspectionResult_keypoint_Right[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Right[i], conf=0.6, imgsz=512, verbose=False)
                                    except Exception as e:
                                        print(f"[Inference Error] P808387UA1A_keypoint: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.InspectionResult_keypoint_Left[i] = None
                                        self.InspectionResult_keypoint_Right[i] = None

                                # Check if anomaly classification model is available
                                anomaly_model = self.P828447UA0A_ANOMALY_CLASSIFICATION_Model if self._check_model_available('P828447UA0A_ANOMALY_CLASSIFICATION_Model') else None

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DeltaPitch[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P828447UA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Left[i],
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Right[i],
                                                                                                                                                                                                                                                anomaly_model)

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 169),
                                pitch_signal=self.P658217UJ0A_InspectionResult_PitchMeasured
                            )
                            time.sleep(1.5)

            #for P658207UJ0A (widget 34)
            if self.inspection_config.widget in [34]:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-90)

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                                                                    planarize_transform=self.planarizeTransform_wide,
                                                                                    planarize_size=self.wide_planarize,
                                                                                    h1=self.H1,
                                                                                    h2=self.H2_high,
                                                                                    blank_canvas=self.homography_blank_canvas
                                                                                )

                            for i in range(len(self.InspectionImages)):
                                # Check if CLIP model is available
                                if not self._check_model_available('P658207UJ0A_CLIP_Model'):
                                    continue
                                
                                try:
                                    self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                                self.InspectionImages_bgr[i], 
                                                self.P658207UJ0A_CLIP_Model, 
                                                slice_height=256, slice_width=1980, 
                                                overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                                postprocess_match_metric="IOS",
                                                postprocess_match_threshold=0.2,
                                                postprocess_class_agnostic=True,
                                                postprocess_type="GREEDYNMM",
                                                verbose=0,
                                                perform_standard_pred=False
                                            )
                                except Exception as e:
                                    print(f"[Inference Error] P658207UJ0A_CLIP_Model: {e}")
                                    self.modelErrorSignal.emit("no ai model found in the image")
                                    continue

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, :512, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -512:, :]
                                
                                # Check if keypoint model is available
                                if not self._check_model_available('P808387UA1A_keypoint'):
                                    self.InspectionResult_keypoint_Left[i] = None
                                    self.InspectionResult_keypoint_Right[i] = None
                                else:
                                    try:
                                        self.InspectionResult_keypoint_Left[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Left[i], conf=0.6, imgsz=512, verbose=False)
                                        self.InspectionResult_keypoint_Right[i] = self.P808387UA1A_keypoint(source=self.InspectionImages_keypoint_Right[i], conf=0.6, imgsz=512, verbose=False)
                                    except Exception as e:
                                        print(f"[Inference Error] P808387UA1A_keypoint: {e}")
                                        self.modelErrorSignal.emit("no ai model found in the image")
                                        self.InspectionResult_keypoint_Left[i] = None
                                        self.InspectionResult_keypoint_Right[i] = None

                                # Check if anomaly classification model is available
                                anomaly_model = self.P828447UA0A_ANOMALY_CLASSIFICATION_Model if self._check_model_available('P828447UA0A_ANOMALY_CLASSIFICATION_Model') else None

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DeltaPitch[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P828447UA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Left[i],
                                                                                                                                                                                                                                                self.InspectionResult_keypoint_Right[i],
                                                                                                                                                                                                                                                anomaly_model)

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 169),
                                pitch_signal=self.P658207UJ0A_InspectionResult_PitchMeasured
                            )
                            time.sleep(1.5)

            # P731957YA0A keypoint-based inspection
            if self.inspection_config.widget == 36:
                nichijoutenken_mode = self._is_nichijoutenken_mode(self.inspection_config.widget)

                if not nichijoutenken_mode:
                    self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-100, fallback_shape=(137, 1791, 3))

                            if not self.inspection_config.mapCalculated[1]:
                                continue

                            if any(v is None for v in [
                                self.inspection_config.map1[0],
                                self.inspection_config.map2[0],
                                self.inspection_config.map1[1],
                                self.inspection_config.map2[1],
                                self.H1,
                                self.H2,
                                self.planarizeTransform_wide,
                            ]):
                                continue

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                planarize_transform=self.planarizeTransform_wide,
                                planarize_size=self.wide_planarize,
                                h1=self.H1,
                                h2=self.H2,
                                blank_canvas=self.homography_blank_canvas,
                            )

                            crop_points = self._get_crop_points_for_widget(self.inspection_config.widget, scaled=False)
                            h2_for_widget = self._get_cam2_homography_with_widget_offset(
                                self.H2,
                                widget_id=self.inspection_config.widget,
                                scaled=False,
                            )
                            h2_dx = float(h2_for_widget[0, 2] - self.H2[0, 2]) if isinstance(h2_for_widget, np.ndarray) and isinstance(self.H2, np.ndarray) else 0.0
                            h2_dy = float(h2_for_widget[1, 2] - self.H2[1, 2]) if isinstance(h2_for_widget, np.ndarray) and isinstance(self.H2, np.ndarray) else 0.0
                            print(
                                "[P828387YA1A thread] geometry "
                                f"widget={self.inspection_config.widget} "
                                f"image_shape={self.InspectionImages[0].shape if isinstance(self.InspectionImages[0], np.ndarray) else None} "
                                f"crop_points={crop_points} "
                                f"cam2_offset=({h2_dx:.1f}, {h2_dy:.1f})"
                            )

                            for i in range(len(self.InspectionImages)):
                                if not self._check_model_available('P731957YA0A_CLIP_Model'):
                                    continue

                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages_bgr[i],
                                    self.P731957YA0A_CLIP_Model,
                                    slice_height=497,
                                    slice_width=1960,
                                    overlap_height_ratio=0.0,
                                    overlap_width_ratio=0.2,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=True,
                                )

                                if not self._check_model_available('P731957YA0A_END_KEYPOINT_Model'):
                                    continue

                                self.InspectionImages_endKeypoint_Left[i] = self.InspectionImages[i][:, :840, :]
                                self.InspectionImages_endKeypoint_Right[i] = self.InspectionImages[i][:, -840:, :]

                                self.InspectionResult_EndKeypoint_Left[i] = self.P731957YA0A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Left[i],
                                    conf=0.5,
                                    imgsz=960,
                                    verbose=False,
                                )
                                self.InspectionResult_EndKeypoint_Right[i] = self.P731957YA0A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Right[i],
                                    conf=0.5,
                                    imgsz=960,
                                    verbose=False,
                                )

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P731957YA0A_check(
                                    self.InspectionImages[i],
                                    self.InspectionResult_ClipDetection[i].object_prediction_list,
                                    self.InspectionResult_EndKeypoint_Left[i],
                                    self.InspectionResult_EndKeypoint_Right[i],
                                )
                                self.InspectionResult_NGReason[i] = ""

                            if not nichijoutenken_mode:
                                self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                                self._maybe_apply_packaging_prompt(
                                    widget=self.inspection_config.widget,
                                    idx=0,
                                    interval=120,
                                    text="梱包してください",
                                )
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 137),
                                pitch_signal=self._get_pitch_signal_for_widget(self.inspection_config.widget),
                                emit_count_signals=not nichijoutenken_mode,
                                save_partname=self._get_effective_partname(self.inspection_config.widget),
                            )
                            time.sleep(1.5)

            #P808387YA0A / P828387YA0A
            if self.inspection_config.widget in [37, 38]:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if isinstance(self.combinedImage, np.ndarray) and self.combinedImage.size > 0:
                        self._inspection_start_image = self.combinedImage.copy()

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-100, fallback_shape=(137, 1791, 3))

                            if not self.inspection_config.mapCalculated[1]:
                                continue

                            if any(v is None for v in [
                                self.inspection_config.map1[0],
                                self.inspection_config.map2[0],
                                self.inspection_config.map1[1],
                                self.inspection_config.map2[1],
                                self.H1,
                                self.H2,
                                self.planarizeTransform_wide,
                            ]):
                                continue

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                planarize_transform=self.planarizeTransform_wide,
                                planarize_size=self.wide_planarize,
                                h1=self.H1,
                                h2=self.H2,
                                blank_canvas=self.homography_blank_canvas,
                            )

                            for i in range(len(self.InspectionImages)):
                                if not self._check_model_available('P808387UA0A_828387YA0A_detect_Model'):
                                    continue

                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages[i],
                                    self.P808387UA0A_828387YA0A_detect_Model,
                                    slice_height=497,
                                    slice_width=1960,
                                    overlap_height_ratio=0.0,
                                    overlap_width_ratio=0.2,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=True,
                                )

                                if not self._check_model_available('P808387UA0A_828387YA0A_keypoint_Model'):
                                    continue

                                if not self._check_model_available('P808387UA0A_828387YA0A_hanire_clip_Model'):
                                    continue

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, :840, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -840:, :]

                                self.InspectionResult_keypoint_Left[i] = self.P808387UA0A_828387YA0A_keypoint_Model(
                                    source=self.InspectionImages_keypoint_Left[i],
                                    conf=0.6,
                                    imgsz=864,
                                    verbose=False,
                                )
                                self.InspectionResult_keypoint_Right[i] = self.P808387UA0A_828387YA0A_keypoint_Model(
                                    source=self.InspectionImages_keypoint_Right[i],
                                    conf=0.6,
                                    imgsz=864,
                                    verbose=False,
                                )

                                part_id = "P808387YA0A" if self.inspection_config.widget == 37 else "P828387YA0A"
                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P808387YA0A_P828387YA0A_keypoint_check(
                                    self.InspectionImages[i],
                                    self.InspectionResult_ClipDetection[i].object_prediction_list,
                                    self.InspectionResult_keypoint_Left[i],
                                    self.InspectionResult_keypoint_Right[i],
                                    self.P808387UA0A_828387YA0A_hanire_clip_Model,
                                    part_id=part_id,
                                    keypoint_crop_px=840,
                                )

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._maybe_play_widget_37_38_count_sound(self.inspection_config.widget)
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 137),
                                pitch_signal=self._get_pitch_signal_for_widget(self.inspection_config.widget),
                            )
                            time.sleep(1.5)

            # P828387YA1A / P828397YA1A keypoint-based inspection
            if self.inspection_config.widget in [39, 40]:

                keypoint_crop_px = 1080
                if self.part_registry:
                    try:
                        part_cfg = self.part_registry.get_part_by_widget(self.inspection_config.widget)
                        keypoint_crop_px = int(part_cfg.keypoint_crop_px) if part_cfg and part_cfg.keypoint_crop_px else 1080
                    except Exception:
                        keypoint_crop_px = 1080

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if isinstance(self.combinedImage, np.ndarray) and self.combinedImage.size > 0:
                        self._inspection_start_image = self.combinedImage.copy()

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-100, fallback_shape=(137, 1791, 3))

                            if not self.inspection_config.mapCalculated[1]:
                                continue

                            if any(v is None for v in [
                                self.inspection_config.map1[0],
                                self.inspection_config.map2[0],
                                self.inspection_config.map1[1],
                                self.inspection_config.map2[1],
                                self.H1,
                                self.H2,
                                self.planarizeTransform_wide,
                            ]):
                                continue

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                planarize_transform=self.planarizeTransform_wide,
                                planarize_size=self.wide_planarize,
                                h1=self.H1,
                                h2=self.H2,
                                blank_canvas=self.homography_blank_canvas,
                            )

                            for i in range(len(self.InspectionImages)):
                                if not self._check_model_available('P828387YA1A_CLIP_Model'):
                                    continue

                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages_bgr[i],
                                    self.P828387YA1A_CLIP_Model,
                                    slice_height=497,
                                    slice_width=1960,
                                    overlap_height_ratio=0.0,
                                    overlap_width_ratio=0.2,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=True,
                                )

                                if not self._check_model_available('P828387YA1A_END_KEYPOINT_Model'):
                                    continue

                                if not self._check_model_available('P828387YA1A_hanire_clip_Model'):
                                    continue
                                

                                self.InspectionImages_endKeypoint_Left[i] = self.InspectionImages[i][:, :keypoint_crop_px, :]
                                self.InspectionImages_endKeypoint_Right[i] = self.InspectionImages[i][:, -keypoint_crop_px:, :]

                                self.InspectionResult_EndKeypoint_Left[i] = self.P828387YA1A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Left[i],
                                    conf=0.3,
                                    imgsz=960,
                                    verbose=False,
                                )
                                self.InspectionResult_EndKeypoint_Right[i] = self.P828387YA1A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Right[i],
                                    conf=0.3,
                                    imgsz=960,
                                    verbose=False,
                                )

                                expected_side = "RH" if self.inspection_config.widget == 39 else "LH"

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P828387YA1A_check(
                                    self.InspectionImages[i],
                                    self.InspectionResult_ClipDetection[i].object_prediction_list,
                                    self.InspectionResult_EndKeypoint_Left[i],
                                    self.InspectionResult_EndKeypoint_Right[i],
                                    expected_side=expected_side,
                                    ws_clip_hanire_model=self.P828387YA1A_hanire_clip_Model,

                                    clip_classifier_crop_px=128,
                                    clip_classifier_imgsz=128,
                                    classifier_convert_bgr_to_rgb=True,

                                    debug_save_crops=False,
                                    debug_crop_dir=r"C:\Users\AIKENSA8GOU\Documents\aikensaMaster\debug_crops\P828387YA1A",
                                )


                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._maybe_play_widget_39_40_count_sound(self.inspection_config.widget)
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 137),
                                pitch_signal=self._get_pitch_signal_for_widget(self.inspection_config.widget),
                            )
                            time.sleep(1.5)

            # P658107YA0A (widget 43)
            if self.inspection_config.widget == 43:

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-100, fallback_shape=(137, 1791, 3))

                            if not self.inspection_config.mapCalculated[1]:
                                continue

                            if any(v is None for v in [
                                self.inspection_config.map1[0],
                                self.inspection_config.map2[0],
                                self.inspection_config.map1[1],
                                self.inspection_config.map2[1],
                                self.H1,
                                self.H2,
                                self.planarizeTransform_wide,
                            ]):
                                continue

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                planarize_transform=self.planarizeTransform_wide,
                                planarize_size=self.wide_planarize,
                                h1=self.H1,
                                h2=self.H2,
                                blank_canvas=self.homography_blank_canvas,
                            )

                            for i in range(len(self.InspectionImages)):
                                if not self._check_model_available('P658107YA0A_CLIP_Model'):
                                    continue

                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages_bgr[i],
                                    self.P658107YA0A_CLIP_Model,
                                    slice_height=497,
                                    slice_width=1960,
                                    overlap_height_ratio=0.0,
                                    overlap_width_ratio=0.2,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=True,
                                )

                                if not self._check_model_available('P658107YA0A_END_KEYPOINT_Model'):
                                    continue

                                self.InspectionImages_endKeypoint_Left[i] = self.InspectionImages[i][:, :840, :]
                                self.InspectionImages_endKeypoint_Right[i] = self.InspectionImages[i][:, -840:, :]

                                self.InspectionResult_EndKeypoint_Left[i] = self.P658107YA0A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Left[i],
                                    conf=0.5,
                                    imgsz=840,
                                    verbose=False,
                                )
                                self.InspectionResult_EndKeypoint_Right[i] = self.P658107YA0A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Right[i],
                                    conf=0.5,
                                    imgsz=840,
                                    verbose=False,
                                )

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P658107YA0A_check(
                                    self.InspectionImages[i],
                                    self.InspectionResult_ClipDetection[i].object_prediction_list,
                                    self.InspectionResult_EndKeypoint_Left[i],
                                    self.InspectionResult_EndKeypoint_Right[i],
                                )
                                self.InspectionResult_NGReason[i] = ""

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._maybe_play_widget_43_count_sound(self.inspection_config.widget)
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 137),
                                pitch_signal=self._get_pitch_signal_for_widget(self.inspection_config.widget),
                            )
                            time.sleep(1.5)

            # P13C PP658207LE0A (widget 35) keypoint-based inspection
            if self.inspection_config.widget == 35:

                keypoint_crop_px = 2200
                keypoint_model_imgsz = 1080
                if self.part_registry:
                    try:
                        part_cfg = self.part_registry.get_part_by_widget(self.inspection_config.widget)
                        keypoint_crop_px = int(part_cfg.keypoint_crop_px) if part_cfg and part_cfg.keypoint_crop_px else 2200
                        keypoint_model_imgsz = int(part_cfg.keypoint_model_imgsz) if part_cfg and part_cfg.keypoint_model_imgsz else 1080
                    except Exception:
                        keypoint_crop_px = 2200
                        keypoint_model_imgsz = 1080

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if isinstance(self.combinedImage, np.ndarray) and self.combinedImage.size > 0:
                        self._inspection_start_image = self.combinedImage.copy()

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-100, fallback_shape=(137, 1791, 3))

                            if not self.inspection_config.mapCalculated[1]:
                                continue

                            if any(v is None for v in [
                                self.inspection_config.map1[0],
                                self.inspection_config.map2[0],
                                self.inspection_config.map1[1],
                                self.inspection_config.map2[1],
                                self.H1,
                                self.H2,
                                self.planarizeTransform_wide,
                            ]):
                                continue

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                planarize_transform=self.planarizeTransform_wide,
                                planarize_size=self.wide_planarize,
                                h1=self.H1,
                                h2=self.H2,
                                blank_canvas=self.homography_blank_canvas,
                            )

                            for i in range(len(self.InspectionImages)):
                                if not self._check_model_available('P658207LE0A_CLIP_Model'):
                                    continue

                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages_bgr[i],
                                    self.P658207LE0A_CLIP_Model,
                                    slice_height=497,
                                    slice_width=1960,
                                    overlap_height_ratio=0.0,
                                    overlap_width_ratio=0.2,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=True,
                                )

                                if P658207LE0A_debug_images_enabled():
                                    clip_detection_debug = self._render_sahi_detection_debug_image(
                                        self.InspectionImages[i],
                                        self.InspectionResult_ClipDetection[i].object_prediction_list,
                                        target_class_id=0,
                                    )
                                    self.save_debug_image(clip_detection_debug, "clip_detection", suffix=f"idx{i}")

                                if not self._check_model_available('P658207LE0A_END_KEYPOINT_Model'):
                                    continue
                                if not self._check_model_available('P658207LE0A_CLIPHEIGHT_Model'):
                                    continue

                                self.InspectionImages_endKeypoint_Left[i] = self.InspectionImages[i][:, :keypoint_crop_px, :]
                                self.InspectionImages_endKeypoint_Right[i] = self.InspectionImages[i][:, -keypoint_crop_px:, :]

                                left_keypoint = self.P658207LE0A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Left[i],
                                    conf=0.5,
                                    imgsz=keypoint_model_imgsz,
                                    verbose=False,
                                )
                                right_keypoint = self.P658207LE0A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Right[i],
                                    conf=0.5,
                                    imgsz=keypoint_model_imgsz,
                                    verbose=False,
                                )

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i], self.InspectionResult_DeltaPitch[i] = P658207LE0A_check(
                                    self.InspectionImages[i],
                                    self.InspectionResult_ClipDetection[i].object_prediction_list,
                                    left_keypoint=left_keypoint,
                                    right_keypoint=right_keypoint,
                                    keypoint_crop_px=keypoint_crop_px,
                                    clipheight_model=self.P658207LE0A_CLIPHEIGHT_Model,
                                )

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 137),
                                pitch_signal=self._get_pitch_signal_for_widget(self.inspection_config.widget),
                            )
                            time.sleep(1.5)

            # P828387YA6A / P828397YA6A KATABU NASHI keypoint-based inspection
            if self.inspection_config.widget in [48, 49]:

                keypoint_crop_px = 1260
                if self.part_registry:
                    try:
                        part_cfg = self.part_registry.get_part_by_widget(self.inspection_config.widget)
                        keypoint_crop_px = int(part_cfg.keypoint_crop_px) if part_cfg and part_cfg.keypoint_crop_px else 1260
                    except Exception:
                        keypoint_crop_px = 1260

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if isinstance(self.combinedImage, np.ndarray) and self.combinedImage.size > 0:
                        self._inspection_start_image = self.combinedImage.copy()

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-100, fallback_shape=(137, 1791, 3))

                            if not self.inspection_config.mapCalculated[1]:
                                continue

                            if any(v is None for v in [
                                self.inspection_config.map1[0],
                                self.inspection_config.map2[0],
                                self.inspection_config.map1[1],
                                self.inspection_config.map2[1],
                                self.H1,
                                self.H2,
                                self.planarizeTransform_wide,
                            ]):
                                continue

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                planarize_transform=self.planarizeTransform_wide,
                                planarize_size=self.wide_planarize,
                                h1=self.H1,
                                h2=self.H2,
                                blank_canvas=self.homography_blank_canvas,
                            )

                            for i in range(len(self.InspectionImages)):
                                if not self._check_model_available('P828387YA6A_CLIP_Model'):
                                    continue

                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages[i],
                                    self.P828387YA6A_CLIP_Model,
                                    slice_height=1280,
                                    slice_width=1280,
                                    overlap_height_ratio=0.0,
                                    overlap_width_ratio=0.2,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.05,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=False,
                                )

                                if not self._check_model_available('P828387YA6A_END_KEYPOINT_Model'):
                                    continue

                                self.InspectionImages_endKeypoint_Left[i] = self.InspectionImages[i][:, :keypoint_crop_px, :]
                                self.InspectionImages_endKeypoint_Right[i] = self.InspectionImages[i][:, -keypoint_crop_px:, :]

                                self.InspectionResult_EndKeypoint_Left[i] = self.P828387YA6A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Left[i],
                                    conf=0.5,
                                    imgsz=864,
                                    verbose=False,
                                )
                                self.InspectionResult_EndKeypoint_Right[i] = self.P828387YA6A_END_KEYPOINT_Model(
                                    source=self.InspectionImages_endKeypoint_Right[i],
                                    conf=0.5,
                                    imgsz=864,
                                    verbose=False,
                                )

                                side = "RH" if self.inspection_config.widget == 48 else "LH"
                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P828387YA6A_KATABU_NASHI_check(
                                    self.InspectionImages[i],
                                    self.InspectionResult_ClipDetection[i].object_prediction_list,
                                    self.InspectionResult_EndKeypoint_Left[i],
                                    self.InspectionResult_EndKeypoint_Right[i],
                                    side,
                                    keypoint_crop_px=keypoint_crop_px,
                                )
                                self.InspectionResult_NGReason[i] = ""

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 137),
                                pitch_signal=self._get_pitch_signal_for_widget(self.inspection_config.widget),
                            )
                            time.sleep(1.5)

            # P828387YA6A / P828397YA6A hybrid inspection
            if self.inspection_config.widget in [41, 42]:

                keypoint_crop_px = 1360
                if self.part_registry:
                    try:
                        part_cfg = self.part_registry.get_part_by_widget(self.inspection_config.widget)
                        keypoint_crop_px = int(part_cfg.keypoint_crop_px) if part_cfg and part_cfg.keypoint_crop_px else 1360
                    except Exception:
                        keypoint_crop_px = 1360

                self._handle_manual_adjustment_and_reset(widget=self.inspection_config.widget, use_ppms=True)
                if self._begin_inspection_cycle(require_keep_measurement_false=True, print_start=True):

                    if isinstance(self.combinedImage, np.ndarray) and self.combinedImage.size > 0:
                        self._inspection_start_image = self.combinedImage.copy()

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            self.InspectionTimeStart = time.time()
                            self.emit = self.combinedImage_scaled
                            self._emit_kensa_status(x_offset=-200, y_offset=-100, fallback_shape=(137, 1791, 3))

                            if not self.inspection_config.mapCalculated[1]:
                                continue

                            if any(v is None for v in [
                                self.inspection_config.map1[0],
                                self.inspection_config.map2[0],
                                self.inspection_config.map1[1],
                                self.inspection_config.map2[1],
                                self.H1,
                                self.H2,
                                self.planarizeTransform_wide,
                            ]):
                                continue

                            self.InspectionImages[0], self.InspectionImages_bgr[0] = self._build_inspection_image(
                                planarize_transform=self.planarizeTransform_wide,
                                planarize_size=self.wide_planarize,
                                h1=self.H1,
                                h2=self.H2,
                                blank_canvas=self.homography_blank_canvas,
                            )

                            for i in range(len(self.InspectionImages)):
                                if not self._check_model_available('P828387YA6A_CLIP_Model'):
                                    continue
                                
                                if not self._check_model_available('P828387YA6A_KATABU_CLIP_FLIP_Model'):
                                    continue

                                if not self._check_model_available('P828387YA6A_WS_CLIP_HANIRE_Model'):
                                    continue

                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages[i],
                                    self.P828387YA6A_CLIP_Model,
                                    slice_height=497,
                                    slice_width=1960,
                                    overlap_height_ratio=0.0,
                                    overlap_width_ratio=0.2,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=True,
                                )

                                if not self._check_model_available('P828387YA6A_END_KEYPOINT_Model'):
                                    continue

                                if not self._check_model_available('P828387YA6A_KATABUMARKING_Model'):
                                    continue

                                side = "RH" if self.inspection_config.widget == 41 else "LH"
                                use_left_keypoint = side == "RH"

                                left_keypoint = None
                                right_keypoint = None

                                if use_left_keypoint:
                                    self.InspectionImages_endKeypoint_Left[i] = self.InspectionImages[i][:, :keypoint_crop_px, :]
                                    left_keypoint = self.P828387YA6A_END_KEYPOINT_Model(
                                        source=self.InspectionImages_endKeypoint_Left[i],
                                        conf=0.5,
                                        imgsz=864,
                                        verbose=False,
                                    )
                                else:
                                    self.InspectionImages_endKeypoint_Right[i] = self.InspectionImages[i][:, -keypoint_crop_px:, :]
                                    right_keypoint = self.P828387YA6A_END_KEYPOINT_Model(
                                        source=self.InspectionImages_endKeypoint_Right[i],
                                        conf=0.5,
                                        imgsz=864,
                                        verbose=False,
                                    )

                                katabu_crop = self._get_katabu_crop_for_widget(
                                    self.inspection_config.widget,
                                    side="R" if side == "RH" else "L",
                                )
                                katabu_image = self.frameCrop(
                                    self.InspectionImages[i],
                                    katabu_crop[0],
                                    katabu_crop[1],
                                    katabu_crop[2],
                                    katabu_crop[3],
                                    katabu_crop[4],
                                    katabu_crop[5],
                                )
                                katabu_detection = list(self.P828387YA6A_KATABUMARKING_Model(
                                    source=katabu_image,
                                    stream=True,
                                    verbose=False,
                                    conf=0.3,
                                    iou=0.5,
                                ))

                                self.katabuImage_init = katabu_image.copy()
                                if side == "RH":
                                    self.partKatabuR.emit(self.convertQImage(self.katabuImage_init))
                                else:
                                    self.partKatabuL.emit(self.convertQImage(self.katabuImage_init))

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P828387YA6A_check(
                                    self.InspectionImages[i],
                                    self.InspectionResult_ClipDetection[i].object_prediction_list,
                                    left_keypoint=left_keypoint,
                                    right_keypoint=right_keypoint,
                                    katabu_detection=katabu_detection,
                                    side=side,
                                    keypoint_crop_px=keypoint_crop_px,
                                    katabu_crop_width=int(katabu_crop[2]),

                                    katabu_clip_flip_model=self.P828387YA6A_KATABU_CLIP_FLIP_Model,
                                    ws_clip_hanire_model=self.P828387YA6A_WS_CLIP_HANIRE_Model,

                                    clip_classifier_crop_px=128,
                                    clip_classifier_imgsz=128,

                                    # Use True if this matches the behavior you already confirmed.
                                    classifier_convert_bgr_to_rgb=False,

                                    debug_save_crops=True,
                                    debug_crop_dir="./debug_crops/P828387YA6A",
                                )


                                self.InspectionImagesKatabu[i] = self._render_katabu_result_image(
                                    katabu_image,
                                    katabu_detection,
                                    self.InspectionResult_Status[i],
                                )
                                if side == "RH":
                                    self.partKatabuR.emit(self.convertQImage(self.InspectionImagesKatabu[i]))
                                else:
                                    self.partKatabuL.emit(self.convertQImage(self.InspectionImagesKatabu[i]))

                            self._update_ok_ng_counts_single(self.inspection_config.widget, idx=0, status=self.InspectionResult_Status[0])
                            self._maybe_play_widget_41_42_count_sound(self.inspection_config.widget)
                            self._finalize_inspection_common(
                                widget=self.inspection_config.widget,
                                preview_size=(1791, 137),
                                pitch_signal=self._get_pitch_signal_for_widget(self.inspection_config.widget),
                                save_with_katabu=True,
                            )
                            time.sleep(1.5)

            #for daily inspection
            if self.inspection_config.widget in [21, 22, 23]:

                if self._begin_inspection_cycle(print_start=True):

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()
                            self._prepare_wide_inspection_images(status_text="校正確認中", x_offset=-280, y_offset=-100, h2=self.H2_high)

                            keypoint_model_imgsz = 1980
                            if self.part_registry:
                                try:
                                    part_cfg = self.part_registry.get_part_by_widget(self.inspection_config.widget)
                                    keypoint_model_imgsz = int(part_cfg.keypoint_model_imgsz) if part_cfg and part_cfg.keypoint_model_imgsz else 1280
                                except Exception:
                                    keypoint_model_imgsz = 1980

                            for i in range(len(self.InspectionImages)):
                                crop_image, crop_offset = self._crop_daily_tenken_inference_image(self.InspectionImages[i])

                                if NICHIJOU_TENKEN_debug_images_enabled():
                                    self.save_debug_image(
                                        crop_image,
                                        "dailytenken_inference_crop",
                                        suffix=f"widget{self.inspection_config.widget}_idx{i}",
                                    )

                                if not self._check_model_available('NICHIJOU_TENKEN_Model'):
                                    self.InspectionResult_keypoint_Left[i] = None
                                else:
                                    try:
                                        self.InspectionResult_keypoint_Left[i] = self.NICHIJOU_TENKEN_Model(
                                            source=crop_image,
                                            conf=0.4,
                                            imgsz=keypoint_model_imgsz,
                                            verbose=False,
                                        )
                                    except Exception as e:
                                        print(f"[Inference Error] NICHIJOU_TENKEN_Model: {e}")
                                        self.InspectionResult_keypoint_Left[i] = None

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = dailyTenken(
                                    self.InspectionImages[i],
                                    self.InspectionResult_keypoint_Left[i],
                                    crop_offset=crop_offset,
                                )

                                if self.InspectionResult_Status[i] == "OK":
                                    play_ok_sound()
                                elif self.InspectionResult_Status[i] == "NG":
                                    play_ng_sound()

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0,
                                    resultPitch = self.InspectionResult_PitchResult[0], 
                                    status = self.InspectionResult_Status[0], 
                                    NGreason = self.InspectionResult_NGReason[0],
                                    PPMS="Null")


                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=131)

                            self.partCam.emit(self.convertQImage(self.InspectionImages[0]))

                            time.sleep(1.5)

            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

        # self.msleep(5)
        time.sleep(0.02)


    # =========================
    # Supporting methods
    # =========================

    def _begin_inspection_cycle(self, require_keep_measurement_false=False, print_start=False):
        """
        Shared gate for all inspection branches.
        Keeps the legacy behavior while removing repeated boilerplate.
        """
        if self.InspectionTimeStart is None:
            self.InspectionTimeStart = time.time()

        if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
            self.inspection_config.doInspection = False

        if self.inspection_config.doInspection is not True:
            return False

        if require_keep_measurement_false and self.bool_keep_measurement is True:
            return False

        self.inspection_config.doInspection = False

        if print_start:
            print("Inspection Started")

        if self.InspectionTimeStart is None:
            return False

        return (time.time() - self.InspectionTimeStart) > self.InspectionWaitTime

    def setCounterFalse(self):
        self.inspection_config.furyou_plus = False
        self.inspection_config.furyou_minus = False
        self.inspection_config.kansei_plus = False
        self.inspection_config.kansei_minus = False
        self.inspection_config.furyou_plus_10 = False
        self.inspection_config.furyou_minus_10 = False
        self.inspection_config.kansei_plus_10 = False
        self.inspection_config.kansei_minus_10 = False

    def manual_adjustment(self, currentPart, Totalpart,
                          furyou_plus, furyou_minus, 
                          furyou_plus_10, furyou_minus_10,
                          kansei_plus, kansei_minus,
                          kansei_plus_10, kansei_minus_10):
        
        ok_count_current = currentPart[0]
        ng_count_current = currentPart[1]
        ok_count_total = Totalpart[0]
        ng_count_total = Totalpart[1]
        
        if furyou_plus:
            ng_count_current += 1
            ng_count_total += 1

        if furyou_plus_10:
            ng_count_current += 10
            ng_count_total += 10

        if furyou_minus and ng_count_current > 0 and ng_count_total > 0:
            ng_count_current -= 1
            ng_count_total -= 1
        
        if furyou_minus_10 and ng_count_current > 9 and ng_count_total > 9:
            ng_count_current -= 10
            ng_count_total -= 10

        if kansei_plus:
            ok_count_current += 1
            ok_count_total += 1

        if kansei_plus_10:
            ok_count_current += 10
            ok_count_total += 10

        if kansei_minus and ok_count_current > 0 and ok_count_total > 0:
            ok_count_current -= 1
            ok_count_total -= 1

        if kansei_minus_10 and ok_count_current > 9 and ok_count_total > 9:
            ok_count_current -= 10
            ok_count_total -= 10

        self.setCounterFalse()

        self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                numofPart = [ok_count_total, ng_count_total], 
                currentnumofPart = [ok_count_current, ng_count_current],
                deltaTime = 0.0,
                kensainName = self.inspection_config.kensainNumber, 
                detected_pitch_str = "MANUAL", 
                delta_pitch_str = "MANUAL", 
                total_length=0,
                resultPitch = "MANUAL",
                status = "MANUAL",
                NGreason = "MANUAL",
                PPMS="MANUAL")

        return [ok_count_current, ng_count_current], [ok_count_total, ng_count_total]
    
    def save_result_database(self, partname, numofPart, 
                             currentnumofPart, deltaTime, 
                             kensainName, detected_pitch_str, 
                             delta_pitch_str, total_length, 
                             resultPitch, status, NGreason, PPMS="Null"):
        # Ensure all inputs are strings or compatible types

        timestamp = datetime.now()
        timestamp_date = timestamp.strftime("%Y%m%d")
        timestamp_hour = timestamp.strftime("%H:%M:%S")

        partname = str(partname)
        numofPart = str(numofPart)
        currentnumofPart = str(currentnumofPart)
        timestamp_hour = str(timestamp_hour)
        timestamp_date = str(timestamp_date)
        deltaTime = float(deltaTime)  # Ensure this is a float
        kensainName = str(kensainName)
        detected_pitch_str = str(detected_pitch_str)
        delta_pitch_str = str(delta_pitch_str)
        total_length = float(total_length)  # Ensure this is a float
        resultPitch = str(resultPitch)
        status = str(status)
        NGreason = str(NGreason)

        if PPMS != "Null":
            PPMS = str(PPMS)
        else:
            PPMS = "Null"


        self.cursor.execute('''
        INSERT INTO inspection_results (partname, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason, PPMS)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason, PPMS))
        self.conn.commit()
    
        # Update the totatl part number (Maybe the day has been changed)
        for key, value in self.widget_dir_map.items():
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        if getattr(self, "mysql_conn", None) is not None and hasattr(self, "mysql_cursor"):
            try:
                # Also save to MySQL cursor when available.
                self.mysql_cursor.execute('''
                INSERT INTO inspection_results (partName, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason, PPMS)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason, PPMS))
                self.mysql_conn.commit()
            except Exception as e:
                print(f"Error saving to MySQL database: {str(e)}")

    def get_last_entry_currentnumofPart(self, part_name):
        self.cursor.execute('''
            SELECT currentnumofPart
            FROM inspection_results
            WHERE partName = ?
            ORDER BY id DESC
            LIMIT 1
        ''', (part_name,))
        row = self.cursor.fetchone()
        if row:
            try:
                return list(literal_eval(row[0]))
            except Exception:
                return [0, 0]
        return [0, 0]

    def get_last_entry_total_numofPart(self, part_name):
        today_date = datetime.now().strftime("%Y%m%d")
        self.cursor.execute('''
            SELECT numofPart
            FROM inspection_results
            WHERE partName = ? AND timestampDate = ?
            ORDER BY id DESC
            LIMIT 1
        ''', (part_name, today_date))
        row = self.cursor.fetchone()
        if row:
            try:
                return list(literal_eval(row[0]))
            except Exception:
                return [0, 0]
        return [0, 0]

    def draw_status_text_PIL(self, image, text, color, size = "normal", x_offset = 0, y_offset = 0):

        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2

        if size == "large":
            font_scale = 130.0

        if size == "normal":
            font_scale = 100.0

        elif size == "small":
            font_scale = 50.0
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.kanjiFontPath, font_scale)

        draw.text((center_x + x_offset, center_y + y_offset), text, font=font, fill=color)  
        # Convert back to BGR for OpenCV compatibility
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return image

    def draw_bottom_center_status_text_PIL(self, image, text, color):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.kanjiFontPath, 110)

        text = str(text).strip().upper()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = max((img_pil.width - text_width) // 2, 0)
        text_y = max(img_pil.height - text_height - 54, 0)

        outline_color = (0, 0, 0)
        for offset_x in (-3, -2, -1, 0, 1, 2, 3):
            for offset_y in (-3, -2, -1, 0, 1, 2, 3):
                if offset_x == 0 and offset_y == 0:
                    continue
                draw.text((text_x + offset_x, text_y + offset_y), text, font=font, fill=outline_color)

        draw.text((text_x, text_y), text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _maybe_apply_packaging_prompt(self, widget, idx=0, interval=120, text="梱包してください", target_counts=None, sound_callback=play_konpou_sound, pre_silence=False):
        try:
            status_text = str(self.InspectionResult_Status[idx]).strip().upper()
            if status_text != "OK":
                return False

            ok_count = int(self.inspection_config.current_numofPart[widget][0])
            should_trigger = False
            if target_counts is not None:
                should_trigger = ok_count in {int(count) for count in target_counts}
            elif interval is not None:
                should_trigger = ok_count > 0 and ok_count % interval == 0

            if not should_trigger:
                return False

            if text:
                image = self.InspectionImages[idx]
                if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                    return False

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(image_rgb)
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype(self.kanjiFontPath, 120)

                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = max((img_pil.width - text_width) // 2, 0)
                text_y = max(((img_pil.height - text_height) // 2) + 20, 0)

                outline_color = (0, 0, 0)
                fill_color = (5, 80, 160)
                for offset_x in (-3, -2, -1, 0, 1, 2, 3):
                    for offset_y in (-3, -2, -1, 0, 1, 2, 3):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        draw.text((text_x + offset_x, text_y + offset_y), text, font=font, fill=outline_color)

                draw.text((text_x, text_y), text, font=font, fill=fill_color)
                self.InspectionImages[idx] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            if sound_callback is not None:
                try:
                    sound_callback(pre_silence=pre_silence)
                except TypeError:
                    sound_callback()
            return True
        except Exception as e:
            print(f"[packaging] failed to apply prompt for widget={widget}: {e}")
            return False

    def save_image(self, image):
        # Skip saving if image is None or invalid
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return
        
        dir = "aikensa/inspection/" + self.widget_dir_map[self.inspection_config.widget]
        os.makedirs(dir, exist_ok=True)
        filename = dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        
        # Check if the file already exists and add an identifier if it does
        counter = 1
        while os.path.exists(filename):
            filename = dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{counter}.png"
            counter += 1
        
        cv2.imwrite(filename, self._prepare_image_for_cv_save(image))

    def _prepare_image_for_cv_save(self, image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return image

        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def save_image_result(self, image_initial, image_result, result, keep_full_resolution=False):
        # Keep normal folder split while being tolerant to missing start/result buffers.
        valid_initial = isinstance(image_initial, np.ndarray) and image_initial.size > 0
        valid_result = isinstance(image_result, np.ndarray) and image_result.size > 0

        if not valid_initial and not valid_result:
            return

        if not valid_initial and isinstance(self._inspection_start_image, np.ndarray) and self._inspection_start_image.size > 0:
            image_initial = self._inspection_start_image.copy()
            valid_initial = True

        if not valid_result and valid_initial:
            image_result = image_initial.copy()
            valid_result = True

        if not valid_initial or not valid_result:
            return
        
        result_str = str(result).strip().upper()
        if result_str == "OK":
            status_folder = "OK"
        elif result_str == "NG":
            status_folder = "NG"
        else:
            status_folder = str(result)
        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" +  status_folder + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" + status_folder + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        # Keep default storage-saving behavior unless full resolution is requested.
        if not keep_full_resolution:
            image_result = self.downSampling(image_result, width=image_result.shape[1] // 2, height=image_result.shape[0] // 2)

        image_initial_to_save = self._prepare_image_for_cv_save(image_initial)
        image_result_to_save = self._prepare_image_for_cv_save(image_result)

        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_initial_to_save)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_result_to_save)

    def save_image_result_withKatabu(self, image_initial, image_result, katabu_initial, katabu_result, result, keep_full_resolution=False):
        # Skip saving if any image is None or invalid
        if (image_initial is None or not isinstance(image_initial, np.ndarray) or image_initial.size == 0 or
            image_result is None or not isinstance(image_result, np.ndarray) or image_result.size == 0 or
            katabu_initial is None or not isinstance(katabu_initial, np.ndarray) or katabu_initial.size == 0 or
            katabu_result is None or not isinstance(katabu_result, np.ndarray) or katabu_result.size == 0):
            return

        result_str = str(result).strip().upper()
        if result_str == "OK":
            status_folder = "OK"
        elif result_str == "NG":
            status_folder = "NG"
        else:
            status_folder = str(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_dir = datetime.now().strftime("%Y%m%d")
        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + date_dir +  "/" +  status_folder + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + date_dir +  "/" + status_folder + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        image_result_to_save = image_result
        if not keep_full_resolution:
            image_result_to_save = self.downSampling(image_result, width=image_result.shape[1] // 2, height=image_result.shape[0] // 2)

        cv2.imwrite(raw_dir + "/" + timestamp + ".png", self._prepare_image_for_cv_save(image_initial))
        cv2.imwrite(raw_dir + "/" + timestamp + "_katabu.png", self._prepare_image_for_cv_save(katabu_initial))
        cv2.imwrite(result_dir + "/" + timestamp + ".png", self._prepare_image_for_cv_save(image_result_to_save))
        cv2.imwrite(result_dir + "/" + timestamp + "_katabu.png", self._prepare_image_for_cv_save(katabu_result))

    def save_debug_image(self, image, debug_name, suffix=None):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return

        part_dir = self.widget_dir_map.get(self.inspection_config.widget, f"unknown_{self.inspection_config.widget}")
        date_dir = datetime.now().strftime("%Y%m%d")
        debug_dir = os.path.join("aikensa", "inspection_results", part_dir, date_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{debug_name}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".png"

        cv2.imwrite(os.path.join(debug_dir, filename), self._prepare_image_for_cv_save(image))

    def _render_sahi_detection_debug_image(self, image, prediction_list, target_class_id=None):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None

        debug_image = image.copy()
        for index, prediction in enumerate(prediction_list or []):
            try:
                class_id = int(prediction.category.id)
                bbox = prediction.bbox
            except Exception:
                continue

            if target_class_id is not None and class_id != target_class_id:
                continue

            x1 = max(int(round(bbox.minx)), 0)
            y1 = max(int(round(bbox.miny)), 0)
            x2 = min(int(round(bbox.maxx)), debug_image.shape[1] - 1)
            y2 = min(int(round(bbox.maxy)), debug_image.shape[0] - 1)
            color = (0, 255, 0) if class_id == 0 else (0, 165, 255)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                debug_image,
                f"id={class_id} #{index}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        return debug_image

    def _render_katabu_result_image(self, katabu_image, katabu_detection, status):
        if katabu_image is None or not isinstance(katabu_image, np.ndarray) or katabu_image.size == 0:
            return None

        rendered = katabu_image.copy()
        for result in katabu_detection or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                x_val = float(box.xywh[0][0].cpu())
                y_val = float(box.xywh[0][1].cpu())
                w_val = float(box.xywh[0][2].cpu())
                h_val = float(box.xywh[0][3].cpu())
                cls_id = int(box.cls.cpu())
                color = (0, 255, 0) if cls_id == 0 else (100, 100, 200)
                x1 = int(x_val - w_val / 2)
                y1 = int(y_val - h_val / 2)
                x2 = int(x_val + w_val / 2)
                y2 = int(y_val + h_val / 2)
                cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)

        status_color = (10, 210, 60) if str(status).strip().upper() == "OK" else (200, 30, 50)
        cv2.putText(rendered, str(status).strip().upper(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        return rendered

    def convertQImage(self, image):
        if image is None:
            # Return a blank black image if input is None
            preview_width, preview_height = self._get_partcam_preview_size()
            blank_image = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
            h, w, ch = blank_image.shape
            bytesPerLine = ch * w
            processed_image = QImage(blank_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
            return processed_image

        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def _log_camera_color_debug(self, frame, cap, cam_label):
        return
    
    def downScaledImage(self, image, scaleFactor=1.0):
        resized_image = cv2.resize(image, (0, 0), fx=1/scaleFactor, fy=1/scaleFactor, interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    def downSampling(self, image, width=384, height=256):
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def _resize_with_aspect_pad(self, image, target_w, target_h):
        """Resize to fit target while preserving aspect ratio, then pad to exact size."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        channels = 1 if image.ndim == 2 else image.shape[2]
        if channels == 1:
            canvas = np.zeros((target_h, target_w), dtype=image.dtype)
        else:
            canvas = np.zeros((target_h, target_w, channels), dtype=image.dtype)

        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def load_matrix_from_yaml(self, filename):
        with open(filename, 'r') as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_param.get('camera_matrix'))
            distortion_coeff = np.array(calibration_param.get('distortion_coefficients'))
        return camera_matrix, distortion_coeff

    def frameCrop(self,img, x=0, y=0, w=640, h=480, wout=640, hout=480):
        x, y, w, h, wout, hout = int(x), int(y), int(w), int(h), int(wout), int(hout)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        img = img[y:y+h, x:x+w]
        try:
            img = cv2.resize(img, (wout, hout), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print("An error occurred while cropping the image:", str(e))
        return img

    def createBlackImage(self, width, height): #create a black image with width and height
        return np.zeros((height, width, 3), dtype=np.uint8)

    def initialize_model(self):
        print("Model Dummy Loaded")

        config_path = "./aikensa/core/config/models.yaml"

        # Default: in case YAML is missing
        self.model_device = "cuda:0"

        # -------------------------
        # Safe YAML load
        # -------------------------
        if not os.path.exists(config_path):
            print(f"[Model Config Missing] {config_path}")
            return

        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Model Config Load Failed] {config_path} -> {e}")
            return

        self.model_device = cfg.get("device", "cuda:0")
        models_cfg = cfg.get("models", {}) or {}

        # -------------------------
        # Local safe loaders
        # -------------------------
        def _exists(path):
            try:
                return bool(path) and os.path.exists(path)
            except Exception:
                return False

        def _load_yolo(path, attr_name):
            if not _exists(path):
                print(f"[Model Missing] {attr_name}: {path}")
                return None
            try:
                return YOLO(path)
            except Exception as e:
                print(f"[Model Load Failed] {attr_name}: {path} -> {e}")
                return None

        def _load_sahi(path, attr_name, model_type, conf, device, image_size=None):
            if not _exists(path):
                print(f"[Model Missing] {attr_name}: {path}")
                return None
            try:
                sahi_kwargs = {
                    "model_type": model_type,
                    "model_path": path,
                    "confidence_threshold": float(conf),
                    "device": device,
                }
                if image_size:
                    sahi_kwargs["image_size"] = int(image_size)

                return AutoDetectionModel.from_pretrained(**sahi_kwargs)
            except Exception as e:
                print(f"[Model Load Failed] {attr_name}: {path} -> {e}")
                return None

        # -------------------------
        # Load all models defined in YAML
        # -------------------------
        loaded = []
        missing = []

        for attr_name, spec in models_cfg.items():
            loader = (spec.get("loader") or "").lower()
            path = spec.get("path")

            model = None

            if loader == "yolo":
                model = _load_yolo(path, attr_name)

            elif loader == "sahi":
                model_type = spec.get("model_type", "yolov8")
                conf = spec.get("conf", 0.5)
                image_size = spec.get("image_size", None)
                model = _load_sahi(path, attr_name, model_type, conf, self.model_device, image_size=image_size)

            else:
                print(f"[Unknown Loader] {attr_name}: loader='{spec.get('loader')}'")
                model = None

            setattr(self, attr_name, model)

            if model is None:
                missing.append(attr_name)
            else:
                loaded.append(attr_name)

        print(f"[Models Loaded] {len(loaded)} -> {loaded}")
        print(f"[Models Missing/Failed] {len(missing)} -> {missing}")

    def _check_model_available(self, model_name):
        """Check if a model is available (not None). If not, emit error signal and return False."""
        model = getattr(self, model_name, None)
        if model is None:
            error_msg = f"no ai model found in the image"
            print(f"[Model Error] {model_name}: {error_msg}")
            self.modelErrorSignal.emit(error_msg)
            return False
        return True

    def stop(self):
        print("Releasing all cameras.")
        print("Inspection thread stopped.")
        self.inspection_config.widget = -1
        self.running = False
        self.release_all_camera()

    def add_columns(self, cursor, table_name, columns):
        for column_name, column_type in columns:
            try:
                cursor.execute(f'''
                ALTER TABLE {table_name}
                ADD COLUMN {column_name} {column_type};
                ''')
                print(f"Added column: {column_name}")
            except sqlite3.OperationalError as e:
                print(f"Could not add column {column_name}: {e}")

    def _check_date_rollover(self):
        today = datetime.now().strftime("%Y%m%d")
        if today != self._today_str:
            self._today_str = today
            # reset only the daily totals, keep current (session) counts as-is
            for k in self.widget_dir_map.keys():
                self.inspection_config.today_numofPart[k] = [0, 0]

    def _handle_manual_adjustment_and_reset(self, widget, use_ppms=False):
        cfg = self.inspection_config
        counts_changed = False

        if (
            cfg.furyou_plus or cfg.furyou_minus or
            cfg.kansei_plus or cfg.kansei_minus or
            cfg.furyou_plus_10 or cfg.furyou_minus_10 or
            cfg.kansei_plus_10 or cfg.kansei_minus_10
        ):
            cfg.current_numofPart[widget], cfg.today_numofPart[widget] = self.manual_adjustment(
                cfg.current_numofPart[widget],
                cfg.today_numofPart[widget],
                cfg.furyou_plus, cfg.furyou_minus,
                cfg.furyou_plus_10, cfg.furyou_minus_10,
                cfg.kansei_plus, cfg.kansei_minus,
                cfg.kansei_plus_10, cfg.kansei_minus_10,
            )
            counts_changed = True
            print("Manual Adjustment Done")

        # Counter reset
        if cfg.counterReset:
            cfg.counterReset = False
            self.setCounterFalse()
            cfg.current_numofPart[widget] = [0, 0]
            counts_changed = True

            partname = self.widget_dir_map[widget]
            ppms = cfg.ppmsnumber if use_ppms else "COUNTERRESET"

            self.save_result_database(
                partname=partname,
                numofPart=cfg.today_numofPart[widget],
                currentnumofPart=[0, 0],
                deltaTime=0.0,
                kensainName=cfg.kensainNumber,
                detected_pitch_str="COUNTERRESET",
                delta_pitch_str="COUNTERRESET",
                total_length=0,
                resultPitch="COUNTERRESET",
                status="COUNTERRESET",
                NGreason="COUNTERRESET",
                PPMS=ppms,
            )

        if counts_changed:
            self.today_numofPart_signal.emit(cfg.today_numofPart)
            self.current_numofPart_signal.emit(cfg.current_numofPart)

    # =========================
    # Init helper methods
    # =========================

    def _init_cameras(self):
        # Use your known camera resolution
        w = getattr(self, "frame_width", 3072)
        h = getattr(self, "frame_height", 2048)

        self._placeholder_cam1 = make_camera_placeholder(w, h, 1)
        self._placeholder_cam2 = make_camera_placeholder(w, h, 2)

        # Start as DummyCapture so read() is always valid
        self.cap_cam = DummyCapture(make_camera_placeholder(w, h, 0))
        self.cap_cam_ic4_1 = DummyCapture(self._placeholder_cam1)
        self.cap_cam_ic4_2 = DummyCapture(self._placeholder_cam2)

    def _init_frames(self):
        self.mergeframe1 = None
        self.mergeframe2 = None

        self.mergeframe1_scaled = None
        self.mergeframe2_scaled = None

        self.mergeframe1_downsampled = None
        self.mergeframe2_downsampled = None

    def _init_homography(self):
        self.homography_template = None
        self.homography_matrix1 = None
        self.homography_matrix2 = None

        self.homography_template_scaled = None
        self.homography_matrix1_scaled = None
        self.homography_matrix2_scaled = None

        self.H1 = None
        self.H2 = None

        self.H1_scaled = None
        self.H2_scaled = None

        self.homography_size = None
        self.homography_size_scaled = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

    def _init_planarize(self):
        self.planarizeTransform_narrow = None
        self.planarizeTransform_narrow_scaled = None
        self.planarizeTransform_high_narrow = None
        self.planarizeTransform_high_narrow_scaled = None

        self.planarizeTransform_wide = None
        self.planarizeTransform_wide_scaled = None
        self.planarizeTransform_high_wide = None
        self.planarizeTransform_high_wide_scaled = None

    def _init_images_and_crops(self):
        self.combinedImage = None
        self.combinedImage_scaled = None

        self.katabuImageL = None
        self.katabuImageR = None
        self.katabuImageL_scaled = None
        self.katabuImageR_scaled = None

        self.katabuImage = None
        self.katabuImage_init = None

        # Default crops (safe fallback if no optional YAML)
        self.katabuImageL_Crop = np.array([620, 360, 320, 160, 320, 160])
        self.katabuImageR_Crop = np.array([4800, 360, 320, 160, 320, 160])

        self.clipImage1 = None
        self.clipImage2 = None
        self.clipImage3 = None

        self.clipImage1_Crop = np.array([1750, 1600, 600, 600, 128, 128])
        self.clipImage2_Crop = np.array([600, 1600, 600, 600, 128, 128])
        self.clipImage3_Crop = np.array([1880, 1600, 600, 600, 128, 128])

    def _init_hand_state(self):
        self.HandinFrame1 = None
        self.HandinFrame2 = None
        self.HandinFrame3 = None

    def _init_scaled_geometry(self):
        # Derived values from frame + scale_factor
        self.scaled_height = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)

    def _init_inspection_images(self):
        # Currently batch size is 1
        self.InspectionImages = [None] * 1
        self.InspectionImages_bgr = [None] * 1
        self.emitImages = [None] * 1

        self.InspectionImagesKatabu = [None] * 1

        self.InspectionImages_keypoint_Left = [None] * 1
        self.InspectionImages_keypoint_Right = [None] * 1

        self.InspectionImages_endKeypoint_Left = [None] * 1
        self.InspectionImages_endKeypoint_Right = [None] * 1

    def _init_results(self):
        MAX = 50

        # Keypoint/seg results are length 5 in your current design
        self.InspectionResult_keypoint_Left = [None] * 5
        self.InspectionResult_keypoint_Right = [None] * 5

        self.InspectionResult_EndKeypoint_Left = [None] * 5
        self.InspectionResult_EndKeypoint_Right = [None] * 5

        # 50-slot per-widget result groups
        result_fields = [
            "ClipDetection",
            "KatabuDetection",
            "Segmentation",
            "Hanire",
            "PitchMeasured",
            "PitchResult",
            "DetectionID",
            "Status",
            "DeltaPitch",
            "NGReason",
        ]
        for field in result_fields:
            setattr(self, "InspectionResult_" + field, [None] * MAX)

        self.InspectionImages_prev = [None] * MAX
        self._test = [0] * MAX

    def _init_widget_maps(self):
        """
        Initialize widget mappings and indices.
        Now uses Part Registry for dynamic configuration.
        Legacy hardcoded values kept as fallback.
        """
        # Basic lists
        self.widget_indices_list = list(range(19))

        # NEW: Try loading from registry first
        if self.part_registry:
            try:
                self.widget_dir_map = self.part_registry.get_widget_to_directory_map()
                self.widget_name_map = self.part_registry.get_widget_to_name_map()
                self.inspection_widget_indices = self.part_registry.get_widget_group('inspection_widgets')
                self.inspection_widget_indices_without_dailytenken = self.part_registry.get_widget_group('inspection_without_dailytenken')
                self.inspection_widget_katabu = self.part_registry.get_widget_group('katabu_widgets')
                self.inspection_widget_katabu_L = self.part_registry.get_widget_group('katabu_L')
                self.inspection_widget_katabu_R = self.part_registry.get_widget_group('katabu_R')
                self.narrow_height_widget = self.part_registry.get_widget_group('narrow_height')
                self.wide_height_widget = self.part_registry.get_widget_group('wide_height')

                # Backward compatibility for legacy widget 30 -> same family as 31 (658217UA0A)
                if 30 not in self.widget_dir_map and 31 in self.widget_dir_map:
                    self.widget_dir_map[30] = self.widget_dir_map[31]
                    self.widget_name_map[30] = "P{}".format(self.widget_dir_map[31])
                    self.inspection_widget_indices = list(self.inspection_widget_indices) + [30]
                    self.inspection_widget_indices_without_dailytenken = list(self.inspection_widget_indices_without_dailytenken) + [30]
                    self.narrow_height_widget = list(self.narrow_height_widget) + [30]

                # Keep preview-only legacy pages active even if the registry has not
                # been updated yet. Widget 47 depends on this to receive partCam.
                legacy_preview_widgets = {
                    44: "5902A510",
                    45: "5902A509",
                    46: "5819A107",
                    47: "8462284S00",
                }
                for widget_id, directory_name in legacy_preview_widgets.items():
                    if widget_id not in self.widget_dir_map:
                        self.widget_dir_map[widget_id] = directory_name
                    if widget_id not in self.widget_name_map:
                        self.widget_name_map[widget_id] = "P{}".format(directory_name)
                    if widget_id not in self.inspection_widget_indices:
                        self.inspection_widget_indices = list(self.inspection_widget_indices) + [widget_id]
                    if widget_id not in self.inspection_widget_indices_without_dailytenken:
                        self.inspection_widget_indices_without_dailytenken = list(self.inspection_widget_indices_without_dailytenken) + [widget_id]
                    if widget_id not in self.wide_height_widget:
                        self.wide_height_widget = list(self.wide_height_widget) + [widget_id]

                self.widget_group_jc2d = [w for w in [17, 18] if w in self.widget_dir_map]
                self.widget_group_j42u = [w for w in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 48, 49] if w in self.widget_dir_map]
                self.widget_group_p13c = [w for w, v in self.widget_dir_map.items() if "LE0A" in v]
                self.widget_group_5a45 = [w for w in [44, 45, 46] if w in self.widget_dir_map]
                self.widget_group_yt3 = [w for w in [47] if w in self.widget_dir_map]
                
                logging.info("Widget maps loaded from registry")
                return  # Success - skip legacy code
            except Exception as e:
                logging.warning(f"Failed to load from registry: {e}. Using legacy widget maps.")
        
        # LEGACY: Fallback to hardcoded values if registry fails
        self.inspection_widget_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        self.inspection_widget_indices_without_dailytenken = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        self.inspection_widget_katabu = [5, 6, 7, 8, 9, 10, 11, 12]
        self.inspection_widget_katabu_L = [5, 7, 9, 11 ]
        self.inspection_widget_katabu_R = [6, 8, 10, 12]
       
        self.narrow_height_widget = [17, 18, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 48, 49]
        self.wide_height_widget = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 31, 32, 33, 34, 44, 45, 46, 47]

        # Widget → directory name
        self.widget_dir_map = {
            5: "82833W050P",
            6: "82832W040P",
            7: "82833W090P",
            8: "82832W080P",
            9: "82833W050PKENGEN",
            10: "82832W040PKENGEN",
            11: "82833W090PKENGEN",
            12: "82832W080PKENGEN",
            13: "82833W050PCLIPSOUNYUUKI",
            14: "82832W040PCLIPSOUNYUUKI",
            15: "82833W090PCLIPSOUNYUUKI",
            16: "82832W080PCLIPSOUNYUUKI",
            17: "808387UA1A",
            18: "828447UA0A",
            21: "dailyTenken_01",
            22: "dailyTenken_02",
            23: "dailyTenken_03",
            30: "658217UA0A",
            31: "658217UA0A",
            32: "658207UA0A",
            33: "658217UJ0A",
            34: "658207UJ0A",
            35: "658207LE0A",
            36: "731957YA0A",
            37: "808387YA0A",
            38: "828387YA0A",
            39: "828387YA1A",
            40: "828397YA1A",
            41: "828387YA6A",
            42: "828397YA6A",
            43: "658107YA0A",
            44: "5902A510",
            45: "5902A509",
            46: "5819A107",
            47: "8462284S00",
            48: "828387YA6A_KATABU_NASHI",
            49: "828397YA6A_KATABU_NASHI",
        }

        # Widget → UI name prefix
        self.widget_name_map = {k: "P{}".format(v) for k, v in self.widget_dir_map.items()}
        self.widget_group_jc2d = [17, 18]
        self.widget_group_j42u = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 48, 49]
        self.widget_group_p13c = [w for w, v in self.widget_dir_map.items() if "LE0A" in v]
        self.widget_group_5a45 = [44, 45, 46]
        self.widget_group_yt3 = [47]

    def execute_part_inspection(self, widget_id, image, clip_result=None, seg_result=None, 
                                keypoint_result=None, katabu_image=None):
        """
        Execute part inspection using registry-based dynamic dispatch.
        Falls back to legacy hardcoded logic if registry is unavailable.
        
        Args:
            widget_id: Widget ID to inspect
            image: Main inspection image
            clip_result: Clip detection results (optional)
            seg_result: Segmentation results (optional)
            keypoint_result: Keypoint detection results (optional)
            katabu_image: Katabu image for widgets that need it (optional)
        
        Returns:
            Tuple: (annotated_image, katabu_image, pitch_measured, pitch_result, 
                    detection_id, status, ng_reason)
        """
        # Try registry-based execution first
        if self.part_registry:
            part_config = self.part_registry.get_part_by_widget(widget_id)
            if part_config:
                try:
                    result = self.part_registry.execute_inspection(
                        widget_id=widget_id,
                        image=image,
                        clip_detection_result=clip_result,
                        segmentation_result=seg_result,
                        keypoint_result=keypoint_result
                    )
                    return result
                except Exception as e:
                    logging.warning(f"Registry execution failed for widget {widget_id}: {e}. Falling back to legacy.")
        
        # LEGACY: Fallback to hardcoded inspection logic
        part_name = self.widget_name_map.get(widget_id, f"WIDGET_{widget_id}")
        
        # For now, use P828XXW0X0P_check as default (most common)
        # In future, we can expand this with more specific logic if needed
        if widget_id in [5, 6, 7, 8, 9, 10, 11, 12]:  # Katabu-based widgets
            return P828XXW0X0P_check(
                image, katabu_image, clip_result, seg_result, part_name
            )
        elif widget_id in [17, 18, 30, 31, 32, 33, 34]:  # M_JC2D + J42U UA/UJ variants
            if widget_id == 17:
                return P808387UA1A_check(
                    image, None, clip_result, keypoint_result, part_name
                )
            elif widget_id in [18, 30, 31, 32, 33, 34]:
                return P828447UA0A_check(
                    image, None, clip_result, keypoint_result, part_name
                )
        elif widget_id in [21, 22, 23]:  # Daily Tenken
            return dailyTenken(
                image, None, clip_result, None, part_name
            )
        else:
            # Default fallback
            return P828XXW0X0P_check(
                image, katabu_image, clip_result, seg_result, part_name
            )

    def _init_clip_order(self):
        MAX = 50
        self.ethernetTrigger = [0] * 5

        # Each widget has its own list; safe (no shared inner list bug)
        self.clipPickingOrder = [[0] * 10 for _ in range(MAX)]

        self.OrderTargetMore = [1, 1, 1, 1, 1, 1]
        self.OrderTargetLess = [1, 1, 1, 1, 1]

    def _init_sounyuuki(self):
        MAX = 50
        self.InspectionResult_PitchResult_sounyuuki = [None] * MAX
        self.InspectionResult_PitchMeasured_sounyuuki = [None] * MAX
        self.InspectionImages_sounyuuki = [None] * 1

    def _init_geometry_defaults(self):
        # Safe fallback values even if YAML doesn't exist
        self.scale_factor = 5.0
        self.frame_width = 3072
        self.frame_height = 2048

        self.narrow_planarize = (531, 2646)
        self.wide_planarize = (1342, 5672)

    # =========================
    # YAML loading helpers
    # =========================

    def _load_optional_defaults(self, path):
        """
        Optional YAML-based defaults loader.
        Safe behavior:
        - If file not found, do nothing.
        - If partial keys missing, only update what exists.
        This lets you move hardcoded init data into YAML gradually.
        """
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print("Failed to load optional defaults YAML:", path, e)
            return

        # Geometry
        geo = cfg.get("geometry", {})
        if "frame_width" in geo:
            self.frame_width = int(geo["frame_width"])
        if "frame_height" in geo:
            self.frame_height = int(geo["frame_height"])
        if "scale_factor" in geo:
            self.scale_factor = float(geo["scale_factor"])

        # Recompute derived geometry
        self._init_scaled_geometry()

        # Planarize sizes (optional)
        planar = cfg.get("planarize", {})
        if "narrow_planarize" in planar:
            self.narrow_planarize = tuple(planar["narrow_planarize"])
        if "wide_planarize" in planar:
            self.wide_planarize = tuple(planar["wide_planarize"])

        # Crops
        crops = cfg.get("crops", {})
        kat = crops.get("katabu", {})
        clip = crops.get("clip", {})

        if "L" in kat:
            self.katabuImageL_Crop = np.array(kat["L"])
        if "R" in kat:
            self.katabuImageR_Crop = np.array(kat["R"])

        # clip keys may be strings in YAML; handle both
        for key, attr in [("1", "clipImage1_Crop"), ("2", "clipImage2_Crop"), ("3", "clipImage3_Crop")]:
            if key in clip:
                setattr(self, attr, np.array(clip[key]))
            elif int(key) in clip:
                setattr(self, attr, np.array(clip[int(key)]))

        # Widget maps / indices
        widgets = cfg.get("widgets", {})
        wmap = widgets.get("map")
        if isinstance(wmap, dict):
            self.widget_dir_map = {int(k): v for k, v in wmap.items()}
            self.widget_name_map = {k: "P{}".format(v) for k, v in self.widget_dir_map.items()}

        idx = widgets.get("inspection_indices")
        if isinstance(idx, list):
            self.inspection_widget_indices = list(idx)

        idx2 = widgets.get("inspection_indices_without_dailytenken")
        if isinstance(idx2, list):
            self.inspection_widget_indices_without_dailytenken = list(idx2)

    def _get_katabu_crop_for_widget(self, widget_id, side):
        if self.part_registry:
            try:
                part_cfg = self.part_registry.get_part_by_widget(widget_id)
                if part_cfg and part_cfg.katabu_crop:
                    return np.array(part_cfg.katabu_crop)
            except Exception:
                pass

        return self.katabuImageR_Crop if side == "R" else self.katabuImageL_Crop

    def _load_cam_map(self):
        self.cam_map = {}
        self.ic4_rotate_180 = False
        if not os.path.exists(self.cam_config_file):
            print("Camera config YAML not found:", self.cam_config_file)
            return

        try:
            with open(self.cam_config_file, "r") as file:
                self.cam_map = yaml.safe_load(file) or {}
                self.ic4_rotate_180 = bool(self.cam_map.get("ic4_rotate_180", False))
        except Exception as e:
            print("Failed to read camera config YAML:", self.cam_config_file, e)
            self.cam_map = {}
            self.ic4_rotate_180 = False

    def _load_mysql_credentials(self):
        self.mysqlID = None
        self.mysqlPassword = None
        self.mysqlHost = None
        self.mysqlHostPort = None

        cred_path = "aikensa/mysql/id.yaml"
        if not os.path.exists(cred_path):
            print("MySQL credential YAML not found:", cred_path)
            return

        try:
            with open(cred_path, "r") as file:
                credentials = yaml.load(file, Loader=yaml.FullLoader) or {}
        except Exception as e:
            print("Failed to read MySQL credential YAML:", cred_path, e)
            return

        self.mysqlID = credentials.get("id")
        self.mysqlPassword = credentials.get("pass")
        self.mysqlHost = credentials.get("host")
        self.mysqlHostPort = credentials.get("port")

    def _load_widget_crop_points(self):
        """
        Load per-widget crop points from part_registry.yaml.
        Format per widget: [x1, y1, x2, y2, x3, y3, x4, y4]
        """
        self.widget_crop_points = {}
        registry_path = "aikensa/core/config/part_registry.yaml"
        if not os.path.exists(registry_path):
            return

        try:
            with open(registry_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            raw_map = cfg.get("widget_crop_points", {}) or {}
            for k, v in raw_map.items():
                try:
                    wid = int(k)
                except Exception:
                    continue
                if isinstance(v, list) and len(v) == 8:
                    self.widget_crop_points[wid] = [int(x) for x in v]
        except Exception as e:
            print("Failed to load widget crop points:", e)
            self.widget_crop_points = {}

    def _load_widget_h2_offsets(self):
        """
        Load per-widget cam2 homography XY translation offsets from part_registry.yaml.
        Supports either mapping style:
          widget_h2_offsets_cam2:
            43: [dx, dy]
        or:
          widget_h2_offsets_cam2:
            43: {x: dx, y: dy}

        Units are full-resolution pixels. Scaled preview offsets are derived automatically.
        """
        self.widget_h2_offsets = {}
        registry_path = "aikensa/core/config/part_registry.yaml"
        if not os.path.exists(registry_path):
            return

        try:
            with open(registry_path, "r") as f:
                cfg = yaml.safe_load(f) or {}

            raw_offsets = cfg.get("widget_h2_offsets_cam2", {}) or {}
            for key, val in raw_offsets.items():
                try:
                    wid = int(key)
                except Exception:
                    continue

                dx = 0.0
                dy = 0.0
                if isinstance(val, list) and len(val) >= 2:
                    dx = float(val[0])
                    dy = float(val[1])
                elif isinstance(val, dict):
                    dx = float(val.get("x", 0.0))
                    dy = float(val.get("y", 0.0))
                else:
                    continue

                self.widget_h2_offsets[wid] = (dx, dy)
        except Exception as e:
            print("Failed to load widget cam2 homography offsets:", e)
            self.widget_h2_offsets = {}

    def _get_cam2_homography_with_widget_offset(self, h2_matrix, widget_id=None, scaled=False):
        """Return a cam2 homography matrix with per-widget XY translation offset applied."""
        if h2_matrix is None:
            return None

        if widget_id is None:
            widget_id = self.inspection_config.widget

        try:
            wid = int(widget_id)
        except Exception:
            return h2_matrix

        dx, dy = self.widget_h2_offsets.get(wid, (0.0, 0.0))
        if dx == 0.0 and dy == 0.0:
            return h2_matrix

        adjusted = np.array(h2_matrix, dtype=np.float64, copy=True)

        if scaled:
            sf = float(self.scale_factor) if getattr(self, "scale_factor", 0) else 1.0
            if sf <= 0.0:
                sf = 1.0
            dx = dx / sf
            dy = dy / sf

        adjusted[0, 2] += float(dx)
        adjusted[1, 2] += float(dy)
        return adjusted

    def _get_crop_points_for_widget(self, widget_id, scaled=False):
        pts = self.widget_crop_points.get(int(widget_id))
        if not pts:
            return None
        if not scaled:
            return pts

        sf = float(self.scale_factor) if getattr(self, "scale_factor", 0) else 1.0
        if sf <= 0:
            sf = 1.0
        scaled_pts = []
        for i, p in enumerate(pts):
            if i % 2 == 0:  # x
                scaled_pts.append(int(round(p / sf)))
            else:           # y
                scaled_pts.append(int(round(p / sf)))
        return scaled_pts

    def _crop_image_by_4points(self, image, points):
        """Simple crop from 4-point polygon by using its bounding rectangle."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return image
        if not points or len(points) != 8:
            return image

        h, w = image.shape[:2]
        xs = [int(points[0]), int(points[2]), int(points[4]), int(points[6])]
        ys = [int(points[1]), int(points[3]), int(points[5]), int(points[7])]

        x_min = max(0, min(xs))
        y_min = max(0, min(ys))
        x_max = min(w, max(xs))
        y_max = min(h, max(ys))

        if x_max <= x_min or y_max <= y_min:
            return image

        return image[y_min:y_max, x_min:x_max].copy()

    def _init_placeholders(self):
        w = getattr(self, "frame_width", 3072)
        h = getattr(self, "frame_height", 2048)

        self._placeholder_cam1 = make_camera_placeholder(w, h, 1)
        self._placeholder_cam2 = make_camera_placeholder(w, h, 2)

    # def _init_cameras(self):
    #     self.cap_cam_ic4_1 = DummyCapture(self._placeholder_cam1)
    #     self.cap_cam_ic4_2 = DummyCapture(self._placeholder_cam2)

    def _get_partcam_preview_size(self, widget_id=None):
        widget_id = self.inspection_config.widget if widget_id is None else widget_id

        if widget_id in [21, 22, 23]:
            return 1791, 131
        if widget_id in [35, 37, 38]:
            return 1791, 137
        if widget_id in self.narrow_height_widget:
            return 1791, 169
        return 1791, 428

    def _crop_daily_tenken_inference_image(self, image, widget_id=None):
        if image is None:
            return None, (0, 0)

        widget_id = self.inspection_config.widget if widget_id is None else widget_id
        image_height, image_width = image.shape[:2]
        crop = None

        if self.part_registry:
            try:
                part_cfg = self.part_registry.get_part_by_widget(widget_id)
                crop = part_cfg.inference_crop if part_cfg else None
            except Exception:
                crop = None

        if not crop or len(crop) != 4:
            return image.copy(), (0, 0)

        x, y, width, height = [int(value) for value in crop]
        x = max(0, min(x, image_width))
        y = max(0, min(y, image_height))
        width = max(1, min(width, image_width - x))
        height = max(1, min(height, image_height - y))

        return image[y:y + height, x:x + width].copy(), (x, y)

    def _emit_kensa_status(self, x_offset=-200, y_offset=-90, fallback_shape=None):
        img = self.combinedImage_scaled
        if img is None:
            if fallback_shape is None:
                preview_width, preview_height = self._get_partcam_preview_size()
                fallback_shape = (preview_height, preview_width, 3)
            img = np.zeros(fallback_shape, dtype=np.uint8)

        img = self.draw_status_text_PIL(img, "検査中", (50, 150, 10),
                                        size="large", x_offset=x_offset, y_offset=y_offset)
        self.partCam.emit(self.convertQImage(img))

    def _get_pitch_signal_for_widget(self, widget_id):
        """Resolve per-widget pitch signal dynamically (supports expanded widget families)."""
        signal_map = {
            35: self.P658207LE0A_InspectionResult,
            5: self.P82833W050P_InspectionResult_PitchMeasured,
            6: self.P82832W040P_InspectionResult_PitchMeasured,
            7: self.P82833W090P_InspectionResult_PitchMeasured,
            8: self.P82832W080P_InspectionResult_PitchMeasured,
            9: self.P82833W050PKENGEN_InspectionResult_PitchMeasured,
            10: self.P82832W040PKENGEN_InspectionResult_PitchMeasured,
            11: self.P82833W090PKENGEN_InspectionResult_PitchMeasured,
            12: self.P82832W080PKENGEN_InspectionResult_PitchMeasured,
            13: self.P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured,
            14: self.P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured,
            15: self.P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured,
            16: self.P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured,
            17: self.P808387UA1A_InspectionResult_PitchMeasured,
            18: self.P828447UA0A_InspectionResult_PitchMeasured,
            # Legacy alias + J42U UA/UJ variants
            30: self.P658217UA0A_InspectionResult_PitchMeasured,
            31: self.P658217UA0A_InspectionResult_PitchMeasured,
            32: self.P658207UA0A_InspectionResult_PitchMeasured,
            33: self.P658217UJ0A_InspectionResult_PitchMeasured,
            34: self.P658207UJ0A_InspectionResult_PitchMeasured,
            36: self.P731957YA0A_InspectionResult_PitchMeasured,
            37: self.P808387YA0A_InspectionResult_PitchMeasured,
            38: self.P828387YA0A_InspectionResult_PitchMeasured,
            39: self.P828387YA1A_InspectionResult_PitchMeasured,
            40: self.P828397YA1A_InspectionResult_PitchMeasured,
            41: self.P828387YA6A_InspectionResult_PitchMeasured,
            42: self.P828397YA6A_InspectionResult_PitchMeasured,
            43: self.P658107YA0A_InspectionResult_PitchMeasured,
            48: self.P828387YA6A_KATABU_NASHI_InspectionResult_PitchMeasured,
            49: self.P828397YA6A_KATABU_NASHI_InspectionResult_PitchMeasured,
        }
        return signal_map.get(widget_id)

    def _prepare_wide_inspection_images(self, status_text="検査中", x_offset=-200, y_offset=-100, h2=None):
        """
        Shared setup for wide widgets:
        - Emit status frame
        - Build combined inspection image (BGR/RGB)
        - Populate `InspectionImages[0]` and `InspectionImages_bgr[0]`
        """
        self.emit = self.combinedImage_scaled
        if self.emit is None:
            preview_width, preview_height = self._get_partcam_preview_size()
            self.emit = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)

        self.emit = self.draw_status_text_PIL(
            self.emit,
            status_text,
            (50, 150, 10),
            size="large",
            x_offset=x_offset,
            y_offset=y_offset,
        )
        self.partCam.emit(self.convertQImage(self.emit))

        h2_matrix = self.H2 if h2 is None else h2
        self.combinedImage, self.InspectionImages_bgr[0] = self._build_inspection_image(
            planarize_transform=self.planarizeTransform_wide,
            planarize_size=self.wide_planarize,
            h1=self.H1,
            h2=h2_matrix,
            blank_canvas=self.homography_blank_canvas,
        )
        self.InspectionImages[0] = self.combinedImage.copy()

    def _build_inspection_image(
            self,
            planarize_transform,
            planarize_size,   # (height, width) OR tuple like your wide_planarize (h, w)
            h1=None,
            h2=None,
            blank_canvas=None,
            rotate_code=cv2.ROTATE_180
        ):
        """
        Build combined image from mergeframe1/2 using:
        remap -> rotate -> warpTwoImages_template -> warpPerspective -> optional ROI crop.
        Returns:
            combined_bgr (BGR image)
            combined_rgb (RGB image)
        """

        ph, pw = int(planarize_size[0]), int(planarize_size[1])

        # Build a safe blank fallback to avoid hard crashes when state is not initialized yet.
        # Keep fallback as pure NumPy to avoid entering OpenCV native calls in this stage.
        fallback_bgr = np.zeros((ph, pw, 3), dtype=np.uint8)

        def _fallback():
            fallback_rgb_local = cv2.cvtColor(fallback_bgr, cv2.COLOR_BGR2RGB)
            return fallback_bgr, fallback_rgb_local

        def _valid_h(m):
            return (
                isinstance(m, np.ndarray)
                and m.shape == (3, 3)
                and np.isfinite(m).all()
            )

        def _valid_img(img):
            return (
                isinstance(img, np.ndarray)
                and img.ndim == 3
                and img.shape[0] > 0
                and img.shape[1] > 0
                and img.shape[2] in (1, 3, 4)
            )

        # Apply per-part cam2 translation offset while preserving cam1 as anchor.
        h2 = self._get_cam2_homography_with_widget_offset(
            h2,
            widget_id=self.inspection_config.widget,
            scaled=False,
        )

        if any(v is None for v in [
            self.mergeframe1,
            self.mergeframe2,
            self.inspection_config.map1[0],
            self.inspection_config.map2[0],
            self.inspection_config.map1[1],
            self.inspection_config.map2[1],
            h1,
            h2,
        ]):
            return _fallback()

        if not _valid_img(self.mergeframe1) or not _valid_img(self.mergeframe2):
            print("[_build_inspection_image] invalid merge frame shape; using fallback")
            return _fallback()

        if not _valid_h(h1) or not _valid_h(h2):
            print("[_build_inspection_image] invalid homography matrix; using fallback")
            return _fallback()

        # Make all heavy OpenCV inputs contiguous and owned to avoid native access violations.
        frame1 = np.ascontiguousarray(self.mergeframe1.copy())
        frame2 = np.ascontiguousarray(self.mergeframe2.copy())

        map10 = self.inspection_config.map1[0]
        map20 = self.inspection_config.map2[0]
        map11 = self.inspection_config.map1[1]
        map21 = self.inspection_config.map2[1]

        if not all(isinstance(m, np.ndarray) for m in [map10, map20, map11, map21]):
            print("[_build_inspection_image] calibration map type invalid; using fallback")
            return _fallback()

        if map10.shape[:2] != frame1.shape[:2] or map20.shape[:2] != frame1.shape[:2] or map11.shape[:2] != frame2.shape[:2] or map21.shape[:2] != frame2.shape[:2]:
            print("[_build_inspection_image] calibration map size mismatch; using fallback")
            return _fallback()

        map10 = np.ascontiguousarray(map10)
        map20 = np.ascontiguousarray(map20)
        map11 = np.ascontiguousarray(map11)
        map21 = np.ascontiguousarray(map21)

        h1 = np.ascontiguousarray(h1.astype(np.float64, copy=False))
        h2 = np.ascontiguousarray(h2.astype(np.float64, copy=False))
        local_blank_canvas = blank_canvas
        if isinstance(local_blank_canvas, np.ndarray):
            local_blank_canvas = np.ascontiguousarray(local_blank_canvas.copy())
        else:
            local_blank_canvas = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)

        # ---- Remap ----
        try:
            mf1 = cv2.remap(
                frame1,
                map10,
                map20,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            mf2 = cv2.remap(
                frame2,
                map11,
                map21,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
        except cv2.error as e:
            print(f"[_build_inspection_image] remap failed: {e}")
            return _fallback()

        # ---- Rotate ----
        if rotate_code is not None and not self.ic4_rotate_180:
            mf1 = cv2.rotate(mf1, rotate_code)
            mf2 = cv2.rotate(mf2, rotate_code)

        # ---- Merge by homography ----
        try:
            combined = warpTwoImages_template(local_blank_canvas, mf1, h1)
            combined = warpTwoImages_template(combined, mf2, h2)
        except cv2.error as e:
            print(f"[_build_inspection_image] homography merge failed: {e}")
            return _fallback()
        except Exception as e:
            print(f"[_build_inspection_image] homography merge unexpected error: {e}")
            return _fallback()

        # ---- 4-point crop (replacement for planarize warpPerspective) ----
        crop_points = self._get_crop_points_for_widget(self.inspection_config.widget, scaled=False)
        combined = self._crop_image_by_4points(combined, crop_points)

        # Keep native cropped resolution for all widgets (no padding, no forced resize).
        if combined is None or not isinstance(combined, np.ndarray) or combined.size == 0:
            return _fallback()

        combined_bgr = combined
        # Keep a pristine copy for raw (nama) saving before partcheck draws on images.
        self._inspection_start_image = combined_bgr.copy()
        combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

        return combined_bgr, combined_rgb

    def _finalize_inspection_common(
            self,
            widget,
            preview_size=(1791, 169),
            pitch_signal=None,  # e.g. self.P808387UA1A_InspectionResult_PitchMeasured
            save_with_katabu=False,
            emit_count_signals=True,
            save_partname=None,
        ):
        """Common end-of-inspection UI emit + signals + save. Assumes index 0 result."""
        idx = 0

        # ---- Preview image ----
        try:
            status_text = str(self.InspectionResult_Status[idx]).strip().upper()
            if status_text in ("OK", "NG"):
                status_color = (10, 210, 60) if status_text == "OK" else (200, 30, 50)
                self.InspectionImages[idx] = self.draw_bottom_center_status_text_PIL(
                    self.InspectionImages[idx],
                    status_text,
                    status_color,
                )

            w, h = preview_size
            self.emitImages[idx] = self.downSampling(self.InspectionImages[idx], width=w, height=h)
            self.partCam.emit(self.convertQImage(self.emitImages[idx]))
        except Exception as e:
            print(f"[finalize] preview emit failed: {e}")

        # ---- Count signals ----
        if emit_count_signals:
            try:
                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            except Exception as e:
                print(f"[finalize] count signals failed: {e}")

        # ---- Pitch signal (optional) ----
        if pitch_signal is not None:
            try:
                pitch_signal.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
            except Exception as e:
                print(f"[finalize] pitch signal failed: {e}")

        # ---- Save image ----
        try:
            image_initial_for_save = self.combinedImage
            if isinstance(self._inspection_start_image, np.ndarray) and self._inspection_start_image.size > 0:
                image_initial_for_save = self._inspection_start_image

            if save_with_katabu:
                katabu_initial_for_save = self.katabuImage_init
                katabu_result_for_save = self.InspectionImagesKatabu[idx]
                self.save_image_result_withKatabu(
                    image_initial_for_save,
                    self.InspectionImages[idx],
                    katabu_initial_for_save,
                    katabu_result_for_save,
                    self.InspectionResult_Status[idx],
                    keep_full_resolution=(widget in [37, 38]),
                )
            else:
                self.save_image_result(
                                    image_initial_for_save,
                                    self.InspectionImages[idx],
                                    self.InspectionResult_Status[idx],
                                    keep_full_resolution=(widget in [37, 38])
                                )
        except Exception as e:
            print(f"[finalize] save_image_result failed: {e}")

        # ---- Save DB ----
        try:
            self.save_result_database(
                partname=save_partname or self.widget_dir_map[widget],
                numofPart=self.inspection_config.today_numofPart[widget],
                currentnumofPart=self.inspection_config.current_numofPart[widget],
                deltaTime=0.0,
                kensainName=self.inspection_config.kensainNumber,
                detected_pitch_str=self.InspectionResult_PitchMeasured[idx],
                delta_pitch_str=self.InspectionResult_DeltaPitch[idx],
                total_length=0,
                resultPitch=self.InspectionResult_PitchResult[idx],
                status=self.InspectionResult_Status[idx],
                NGreason=self.InspectionResult_NGReason[idx],
                PPMS=self.inspection_config.ppmsnumber
            )
        except Exception as e:
            print(f"[finalize] save_result_database failed: {e}")

        self._inspection_start_image = None

        self.bool_keep_measurement = False

    def _is_nichijoutenken_mode(self, widget):
        try:
            return bool(self.inspection_config.nichijoutenken_enabled[widget])
        except (AttributeError, IndexError, TypeError):
            return False

    def _maybe_play_widget_interval_count_sound(
        self,
        widget,
        priority_interval,
        priority_sound_callback,
        priority_label,
        fallback_interval=10,
        fallback_sound_callback=play_ok_count_10_sound,
        fallback_label="10",
    ):
        played_priority = self._maybe_apply_packaging_prompt(
            widget=widget,
            idx=0,
            interval=priority_interval,
            text=None,
            sound_callback=priority_sound_callback,
            pre_silence=True,
        )
        played_fallback = False
        if not played_priority:
            played_fallback = self._maybe_apply_packaging_prompt(
                widget=widget,
                idx=0,
                interval=fallback_interval,
                text=None,
                sound_callback=fallback_sound_callback,
                pre_silence=True,
            )

        if played_fallback:
            print(f"[count-sound] Played OK count {fallback_label} sound for widget={widget}")
        if played_priority:
            print(f"[count-sound] Played OK count {priority_label} sound for widget={widget}")

    def _maybe_play_widget_37_38_count_sound(self, widget):
        if widget not in [37, 38]:
            return

        self._maybe_play_widget_interval_count_sound(
            widget=widget,
            priority_interval=50,
            priority_sound_callback=play_konpou_sound,
            priority_label="50",
        )

    def _maybe_play_widget_39_40_count_sound(self, widget):
        if widget not in [39, 40]:
            return

        self._maybe_play_widget_interval_count_sound(
            widget=widget,
            priority_interval=250,
            priority_sound_callback=play_konpou_sound,
            priority_label="250",
        )

    def _maybe_play_widget_41_42_count_sound(self, widget):
        if widget not in [41, 42]:
            return

        self._maybe_play_widget_interval_count_sound(
            widget=widget,
            priority_interval=50,
            priority_sound_callback=play_konpou_sound,
            priority_label="50",
        )

    def _maybe_play_widget_43_count_sound(self, widget):
        if widget != 43:
            return

        self._maybe_play_widget_interval_count_sound(
            widget=widget,
            priority_interval=140,
            priority_sound_callback=play_konpou_sound,
            priority_label="140",
        )

    def _get_effective_partname(self, widget):
        partname = self.widget_dir_map[widget]
        if self._is_nichijoutenken_mode(widget):
            return f"{partname}NICHIJOUTENKEN"
        return partname

    def _update_ok_ng_counts_single(self, widget, idx=0, status=None, play_sound=True):
        """
        Update OK/NG counters for a single inspection result.

        Supports BOTH:
        - list-based storage: current_numofPart[widget][0/1]
        - dict-based storage: current_numofPart[widget][0/1]

        Convention assumed (same as your old working code):
        [0] = OK
        [1] = NG
        """
        # Resolve status
        st = status
        if st is None:
            try:
                st = self.InspectionResult_Status[idx]
            except Exception:
                st = None

        if st not in ("OK", "NG"):
            return

        try:
            if st == "OK":
                self.inspection_config.current_numofPart[widget][0] += 1
                self.inspection_config.today_numofPart[widget][0] += 1
                if play_sound:
                    play_ok_sound()

            else:  # "NG"
                self.inspection_config.current_numofPart[widget][1] += 1
                self.inspection_config.today_numofPart[widget][1] += 1
                if play_sound:
                    play_ng_sound()

        except (KeyError, IndexError, TypeError) as e:
            print(f"[count] failed to update counts for widget={widget}, status={st}: {e}")

class DummyCapture:
    """Fallback camera-like object that always returns a placeholder frame."""
    def __init__(self, frame: np.ndarray):
        self._frame = frame

    def read(self):
        # mimic cv2.VideoCapture.read() signature
        return True, self._frame.copy()

    def isOpened(self):
        return True

    def release(self):
        pass


def make_camera_placeholder(width: int, height: int, cam_id: int):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Simple message. You can add your logo later.
    text1 = "CAMERA OFFLINE"
    text2 = f"ID: {cam_id}"
    text3 = time.strftime("%Y-%m-%d %H:%M:%S")

    # Big readable text
    cv2.putText(img, text1, (60, int(height * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text2, (60, int(height * 0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text3, (60, int(height * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

    return img