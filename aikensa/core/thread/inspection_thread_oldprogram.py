import inspect
from tabnanny import verbose
import cv2
import os
from datetime import datetime
from networkx import center, jaccard_coefficient
import numpy as np
from shapely import length
from sympy import fu
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

from aikensa.camscripts.cam_init import initialize_camera, initialize_gaikan_camera
from aikensa.camscripts.cam_hole_init import initialize_hole_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image
from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound, play_ok_sound, play_ng_sound

from ultralytics import YOLO
# from aikensa.parts_config.ctrplr_8283XW0W0P import partcheck as ctrplrCheck
# from aikensa.parts_config.ctrplr_8283XW0W0P import dailytenkencheck
from aikensa.parts_config.P658107Y0A_SEALASSYRADCORE import partcheck as P658107Y0A_check               #5
from aikensa.parts_config.P808387YA0A_SEALFRDOORPARTING import partcheck as P808387YA0A_check           #6
from aikensa.parts_config.P828387YA0A_SEALRRDOORPARTING import partcheck as P828387YA0A_check           #7
from aikensa.parts_config.P828387YA6A_SEALRRDOORPARTINGRRSTEP import partcheck as P828387YA6A_check   #8
# from aikensa.parts_config.P828397YA6A_SEALRRDOORPARTINGRRSTEPLH import partcheck as P828397YA6A_check   #9
from aikensa.parts_config.P828387YA1A_SEALRRDOORPARTINGLOCK import partcheck as P828387YA1A_check     #10 #11
from aikensa.parts_config.P731957YA0A_SEALROOF import partcheck as P731957YA0A_check                    #12
from aikensa.parts_config.P8462284S00_RRDOORDUST import partcheck as P8462284S00_check                  #13

from aikensa.parts_config.dailyTenken import dailyTenken

from PIL import ImageFont, ImageDraw, Image

from aikensa.tools.yolo_tools import remove_imageborder_yolo
from aikensa.tools.opencv_tools import add_imageborder

@dataclass
class InspectionConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected

    mapCalculated: list = field(default_factory=lambda: [False]*30) #for 10 cameras
    map1: list = field(default_factory=lambda: [None]*30) #for 10 cameras
    map2: list = field(default_factory=lambda: [None]*30) #for 10 cameras

    map1_downscaled: list = field(default_factory=lambda: [None]*30) #for 10 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*30) #for 10 cameras

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

    clipSounyuuNumber: str = None

    counterReset: bool = False

    today_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(30)])
    current_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(30)])


class InspectionThread(QThread):

    part1Cam = pyqtSignal(QImage)
    partKatabu = pyqtSignal(QImage)
    gaikanCam = pyqtSignal(QImage)

    P658107YA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P808387YA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828387YA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828387YA6A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828397YA6A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828387YA1A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P828397YA1A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P731957YA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P8462284S00_InspectionResult_PitchMeasured = pyqtSignal(list, list)

    today_numofPart_signal = pyqtSignal(list)
    current_numofPart_signal = pyqtSignal(list)

    ethernetStatus = pyqtSignal(list)
    

    def __init__(self, inspection_config: InspectionConfig = None):
        super(InspectionThread, self).__init__()
        self.running = True

        if inspection_config is None:
            self.inspection_config = InspectionConfig()    
        else:
            self.inspection_config = inspection_config

        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"
        self._today_str = datetime.now().strftime("%Y%m%d")


        self.multiCam_stream = False

        self.cap_cam = None
        self.cap_cam1 = None
        self.cap_cam2 = None
        self.cap_cam3 = None

        self.emit = None

        self.bottomframe = None
        self.mergeframe1 = None
        self.mergeframe2 = None
        self.gaikanframe = None

        self.mergeframe1_scaled = None
        self.mergeframe2_scaled = None
        self.gaikanframe_scaled = None

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

        self.part1Crop = None
        self.part2Crop = None
        
        self.part1Crop_scaled = None

        self.homography_size = None
        self.homography_size_scaled = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.combinedImage = None
        self.combinedImage_scaled = None

        self.katabuImage = None
        self.katabuImage_scaled = None

        self.katabuImageRightPos = [35, 57, 1090, 1154]
        self.katabuImageLeftPos = [35, 57, 1298 - self.katabuImageRightPos[3], 1298 - self.katabuImageRightPos[2]]


        self.combinedImage_narrow = None
        self.combinedImage_narrow_scaled = None
        self.combinedImage_wide = None
        self.combinedImage_wide_scaled = None

        self.combinedImage_high_narrow = None
        self.combinedImage_high_narrow_scaled = None
        self.combinedImage_high_wide = None
        self.combinedImage_high_wide_scaled = None


        self.scale_factor = 5.0 #Scale Factor, might increase this later
        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.narrow_planarize = (531, 2646)
        self.wide_planarize = (531, 6491)

        self.planarizeTransform_narrow = None
        self.planarizeTransform_narrow_scaled = None
        self.planarizeTransform_high_narrow = None
        self.planarizeTransform_high_narrow_scaled = None

        self.planarizeTransform_wide = None
        self.planarizeTransform_wide_scaled = None
        self.planarizeTransform_high_wide = None
        self.planarizeTransform_high_wide_scaled = None

        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*1
        self.InspectionImages_bgr = [None]*1

        self.emitImages = [None]*1

        self.InspectionImagesKatabu = [None]*1

        self.InspectionImages_endSegmentation_Left = [None]*1
        self.InspectionImages_endSegmentation_Right = [None]*1

        self.InspectionImages_keypoint_Left = [None]*1
        self.InspectionImages_keypoint_Right = [None]*1

        self.InspectionResult_EndSegmentation_Left = [None]*5
        self.InspectionResult_EndSegmentation_Right = [None]*5

        self.InspectionResult_keypoint_Left = [None]*5
        self.InspectionResult_keypoint_Right = [None]*5

        self.InspectionResult_ClipDetection = [None]*30
        self.InspectioNResult_KatabuDetection = [None]*30
        self.InspectionResult_Segmentation = [None]*30
        self.InspectionResult_Hanire = [None]*30

        self.InspectionResult_PitchMeasured = [None]*30
        self.InspectionResult_PitchResult = [None]*30
        self.InspectionResult_DetectionID = [None]*30
        self.InspectionResult_Status = [None]*30
        self.InspectionResult_DeltaPitch = [None]*30
        self.InspectionResult_NGReason = [None]*30

        self.DetectionResult_HoleDetection = [None]*30

        self.InspectionImages_prev = [None]*30
        self._test = [0]*30
        self.widget_dir_map = {
            5: "658107YA0A",
            6: "808387YA0A",
            7: "828387YA0A",
            8: "828387YA6A",
            9: "828397YA6A",
            10: "828387YA1A",
            11: "828397YA1A",
            12: "731957YA0A",
            13: "8462284S00",
            21: "dailyTenken01",
            22: "dailyTenken02",
            23: "dailyTenken03"
        }

        self.InspectionWaitTime = 1.0
        self.InspectionTimeStart = None

        self.ethernetTrigger = [0]*5

        #Timer for gaikan
        self.gaikanStart = False
        self.gaikanTimer = 0.0
        self.gaikanTimerPeriod = 4.0

        "Read mysql id and password from yaml file"

        with open("aikensa/mysql/id.yaml") as file:
            credentials = yaml.load(file, Loader=yaml.FullLoader)
            self.mysqlID = credentials["id"]
            self.mysqlPassword = credentials["pass"]
            self.mysqlHost = credentials["host"]
            self.mysqlHostPort = credentials["port"]
            

    def release_all_camera(self):
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")
        if self.cap_cam3 is not None:
            self.cap_cam3.release()
            print(f"Camera 3 released.")

    def initialize_single_camera(self, camID):
        if self.cap_cam is not None:
            self.cap_cam.release()  # Release the previous camera if it's already open
            print(f"Camera {self.inspection_config.cameraID} released.")

        if camID == -1:
            print("No valid camera selected, displaying placeholder.")
            self.cap_cam = None  # No camera initialized
            # self.frame = self.create_placeholder_image()
            self.cap_cam = initialize_camera(camID)
            if not self.cap_cam.isOpened():
                print(f"Failed to open camera with ID {camID}")
                self.cap_cam = None
            else:
                print(f"Initialized Camera on ID {camID}")

    def initialize_all_camera(self):
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")
        # if self.cap_cam3 is not None:
        #     self.cap_cam3.release()
        #     print(f"Camera 3 released.")

        self.cap_cam1 = initialize_camera(0)
        self.cap_cam2 = initialize_camera(1)
        # self.cap_cam3 = initialize_camera(1) #This is the gaikanCamera

        if not self.cap_cam1.isOpened():
            print(f"Failed to open camera with ID 1, problem with camera 1.")
            self.cap_cam1 = None
        else:
            print(f"Initialized Camera on ID 1")

        if not self.cap_cam2.isOpened():
            print(f"Failed to open camera with ID 2, problem with camera 2.")
            self.cap_cam2 = None
        else:
            print(f"Initialized Camera on ID 2")

        # if not self.cap_cam3.isOpened():
        #     print(f"Failed to open camera with ID 3, problem with camera 3.")
        #     self.cap_cam3 = None


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
            total_length REAL
        )
        ''')

        # List of columns to add
        columns_to_add = [
            ("resultpitch", "TEXT"),
            ("status", "TEXT"),
            ("NGreason", "TEXT"),
            ("ClipInsertionMachine", "TEXT"),
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
        # self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])
        self.homography_size_scaled = (self.homography_template.shape[0]//5, self.homography_template.shape[1]//5)

        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        for key, value in self.widget_dir_map.items():
            self.inspection_config.current_numofPart[key] = self.get_last_entry_currentnumofPart(value)
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)
                print(f"Loaded homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)
                print(f"Loaded homography matrix for camera 2")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml") as file:
                self.homography_matrix1_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_scaled = np.array(self.homography_matrix1_scaled)
                print(f"Loaded scaled homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml") as file:
                self.homography_matrix2_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_scaled = np.array(self.homography_matrix2_scaled)
                print(f"Loaded scaled homography matrix for camera 2")
 

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_high.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1_high.yaml") as file:
                self.homography_matrix1_high = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_high = np.array(self.homography_matrix1_high)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_high.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2_high.yaml") as file:
                self.homography_matrix2_high = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_high = np.array(self.homography_matrix2_high)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_high_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1_high_scaled.yaml") as file:
                self.homography_matrix1_high_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_high_scaled = np.array(self.homography_matrix1_high_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_high_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2_high_scaled.yaml") as file:
                self.homography_matrix2_high_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_high_scaled = np.array(self.homography_matrix2_high_scaled)


        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_narrow.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_narrow.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_narrow = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_narrow_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_narrow_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_narrow_scaled = np.array(transform_list)
        
        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_wide.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_wide.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_wide = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_wide_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_wide_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_wide_scaled = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_high_narrow.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_high_narrow.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_high_narrow = np.array(transform_list)
        
        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_high_narrow_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_high_narrow_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_high_narrow_scaled = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_high_wide.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_high_wide.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_high_wide = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_high_wide_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_high_wide_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_high_wide_scaled = np.array(transform_list)


        while self.running:

            self._check_date_rollover()

            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget > 0:

                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()
                    # print("initialize all camera")    
                    # self.cap_cam1 = initialize_camera("/dev/v4l/by-id/usb-The_Imaging_Source_Europe_GmbH_DFK_33UX178_35420835-video-index0")
                    # self.cap_cam2 = initialize_camera("/dev/v4l/by-id/usb-The_Imaging_Source_Europe_GmbH_DFK_33UX178_30320216-video-index0")

                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()
                

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

                    self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[0], self.inspection_config.map2_downscaled[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                    self.mergeframe1_scaled = cv2.rotate(self.mergeframe1_scaled, cv2.ROTATE_180)
                    self.mergeframe2_scaled = cv2.rotate(self.mergeframe2_scaled, cv2.ROTATE_180)

                    if self.inspection_config.widget in [5, 6, 7, 12]:
                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)

                        self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_wide_scaled, (int(self.wide_planarize[1]/(self.scale_factor)), int(self.wide_planarize[0]/(self.scale_factor))))
                        self.combinedImage_scaled = self.downScaledImage(self.combinedImage_scaled, scaleFactor=0.724734785036293)

                    if self.inspection_config.widget in [8, 9, 10, 11]:
                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_high_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_high_scaled)

                        self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_high_wide_scaled, (int(self.wide_planarize[1]/(self.scale_factor)), int(self.wide_planarize[0]/(self.scale_factor))))
                        
                        if self.inspection_config.widget == 8:
                            self.katabuImage_scaled = self.combinedImage_scaled[self.katabuImageRightPos[0]:self.katabuImageRightPos[1], self.katabuImageRightPos[2]:self.katabuImageRightPos[3]]
                            self.katabuImage_scaled = self.downScaledImage(self.katabuImage_scaled, scaleFactor=.3)
                            self.katabuImage_scaled = self.convertQImageKatabu(self.katabuImage_scaled)
                            self.partKatabu.emit(self.katabuImage_scaled)

                        if self.inspection_config.widget == 9:
                            self.katabuImage_scaled = self.combinedImage_scaled[self.katabuImageLeftPos[0]:self.katabuImageLeftPos[1], self.katabuImageLeftPos[2]:self.katabuImageLeftPos[3]]
                            self.katabuImage_scaled = self.downScaledImage(self.katabuImage_scaled, scaleFactor=.3)
                            self.katabuImage_scaled = self.convertQImageKatabu(self.katabuImage_scaled)
                            self.partKatabu.emit(self.katabuImage_scaled)

                    if self.inspection_config.widget in [13, 21, 22, 23]:
                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)

                        self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_wide_scaled, (int(self.wide_planarize[1]/(self.scale_factor)), int(self.wide_planarize[0]/(self.scale_factor))))
                        self.combinedImage_scaled = self.downScaledImage(self.combinedImage_scaled, scaleFactor=0.724734785036293)

                        # _, self.gaikanframe = self.cap_cam3.read()
                        # self.gaikanframe = cv2.rotate(self.gaikanframe, cv2.ROTATE_180)
                        self.gaikanframe = np.zeros((147, 110, 3), dtype=np.uint8)

                        if self.gaikanframe is not None:
                            self.gaikanframe_scaled = self.downSampling(self.gaikanframe, 147, 110) #w147, h110 is the qt window size
                            self.gaikanCam.emit(self.convertQImage(self.gaikanframe_scaled))


                    self.InspectionResult_PitchMeasured = [None]*30
                    self.InspectionResult_PitchResult = [None]*30
                    self.InspectionResult_DeltaPitch = [None]*30

                    if self.combinedImage_scaled is not None:
                        self.part1Cam.emit(self.convertQImage(self.combinedImage_scaled))
        
                    self.P658107YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P808387YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828387YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828387YA6A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828397YA6A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828387YA1A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P828397YA1A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P731957YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P8462284S00_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

            if self.inspection_config.widget == 5:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])


                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P658107YA0A_CLIP_Model, 
                                            slice_height=497, slice_width=1980, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )

                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1680, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1680:, :]
                                self.InspectionResult_EndSegmentation_Left[i] = self.P658107YA0A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False, retina_masks=True)
                                self.InspectionResult_EndSegmentation_Right[i] = self.P658107YA0A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False, retina_masks=True)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P658107Y0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Right[i])


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

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)
                                
                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P658107YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            # self.InspectionImages_prev[0] = self.InspectionImages[0]
                            # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                            # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)

            if self.inspection_config.widget == 6:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])


                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P828387YA0A_CLIP_Model,
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )
                                        
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1280, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1280:, :]
                                self.InspectionResult_EndSegmentation_Left[i] = self.P828387YA0A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False, retina_masks=True)
                                self.InspectionResult_EndSegmentation_Right[i] = self.P828387YA0A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False, retina_masks=True)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P808387YA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Right[i])


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

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)
                                
                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P808387YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            # self.InspectionImages_prev[0] = self.InspectionImages[0]
                            # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                            # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)

            if self.inspection_config.widget == 7:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])


                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P828387YA0A_CLIP_Model,
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )
                                        
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1280, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1280:, :]
                                self.InspectionResult_EndSegmentation_Left[i] = self.P828387YA0A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False, retina_masks=True)
                                self.InspectionResult_EndSegmentation_Right[i] = self.P828387YA0A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False, retina_masks=True)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P828387YA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Right[i])


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

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)
                                
                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P828387YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            # self.InspectionImages_prev[0] = self.InspectionImages[0]
                            # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                            # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)

            if self.inspection_config.widget == 8:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1_high)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2_high)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_high_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            self.katabuImage = self.combinedImage.copy()
                            self.katabuImage = self.katabuImage[self.katabuImageRightPos[0]*int(self.scale_factor) :self.katabuImageRightPos[1]*int(self.scale_factor), self.katabuImageRightPos[2]*int(self.scale_factor):self.katabuImageRightPos[3]*int(self.scale_factor)]
                            # self.katabuImage = self.combinedImage[self.katabuImageLeftPos[0]*int(self.scale_factor) :self.katabuImageLeftPos[1]*int(self.scale_factor), self.katabuImageLeftPos[2]*int(self.scale_factor):self.katabuImageLeftPos[3]*int(self.scale_factor)]
                            
                            
                            # # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])
                            # time.sleep(2.5)
                            # self.save_image(self.katabuImage)


                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P828387YA6A_CLIP_Model,
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )
                                self.InspectioNResult_KatabuDetection[i] = self.P828387YA6A_KATABUMARKING_MODEL(source=self.katabuImage, stream=True, verbose=False, conf=0.3, iou=0.5)
                                        
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1640, :]
                                # self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1640:, :]
                                self.InspectionResult_EndSegmentation_Left[i] = self.P828387YA6A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.3, retina_masks=True, imgsz=1960, verbose=False)
                                # self.InspectionResult_EndSegmentation_Right[i] = self.P828387YA6A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.3, retina_masks=True, imgsz=1960, verbose=False)

                                self.InspectionImages[i], self.InspectionImagesKatabu[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P828387YA6A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.katabuImage,
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                self.InspectioNResult_KatabuDetection[i], side="RH")


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

                            self.save_image_result_withKatabu(self.combinedImage, self.InspectionImages[0], self.katabuImage, self.InspectionImagesKatabu[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)
                                
                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P828387YA6A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            # self.InspectionImages_prev[0] = self.InspectionImages[0]
                            # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                            # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)

            if self.inspection_config.widget == 9:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1_high)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2_high)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_high_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            self.katabuImage = self.combinedImage.copy()
                            # self.katabuImage = self.combinedImage[self.katabuImageRightPos[0]*int(self.scale_factor) :self.katabuImageRightPos[1]*int(self.scale_factor), self.katabuImageRightPos[2]*int(self.scale_factor):self.katabuImageRightPos[3]*int(self.scale_factor)]
                            self.katabuImage = self.katabuImage[self.katabuImageLeftPos[0]*int(self.scale_factor) :self.katabuImageLeftPos[1]*int(self.scale_factor), self.katabuImageLeftPos[2]*int(self.scale_factor):self.katabuImageLeftPos[3]*int(self.scale_factor)]
                            
                            
                            # # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])
                            # time.sleep(2.5)
                            # self.save_image(self.katabuImage)

                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P828387YA6A_CLIP_Model,
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )
                                self.InspectioNResult_KatabuDetection[i] = self.P828387YA6A_KATABUMARKING_MODEL(source=self.katabuImage, stream=True, verbose=False, conf=0.3, iou=0.5)
                                        
                                # self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1640, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1640:, :]
                                # self.InspectionResult_EndSegmentation_Left[i] = self.P828387YA6A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.3, retina_masks=True, imgsz=1960, verbose=False)
                                self.InspectionResult_EndSegmentation_Right[i] = self.P828387YA6A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.3, retina_masks=True, imgsz=1960, verbose=False)

                                self.InspectionImages[i], self.InspectionImagesKatabu[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P828387YA6A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.katabuImage,
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Right[i],
                                                                                                                                                                                                                self.InspectioNResult_KatabuDetection[i], side="LH")


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

                            self.save_image_result_withKatabu(self.combinedImage, self.InspectionImages[0], self.katabuImage, self.InspectionImagesKatabu[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)
                                
                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P828397YA6A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            # self.InspectionImages_prev[0] = self.InspectionImages[0]
                            # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                            # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)

            if self.inspection_config.widget == 10:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")  

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1_high)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2_high)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_high_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])

                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P828387YA1A_CLIP_Model,
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )
                                        
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :2400, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -2400:, :]
                                self.InspectionResult_EndSegmentation_Left[i] = self.P828387YA1A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.3, retina_masks=True, imgsz=1960, verbose=False)
                                self.InspectionResult_EndSegmentation_Right[i] = self.P828387YA1A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.3, retina_masks=True, imgsz=1960, verbose=False)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P828387YA1A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Right[i], side="RH")


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

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)
                                
                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P828387YA1A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            # self.InspectionImages_prev[0] = self.InspectionImages[0]
                            # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                            # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)

            if self.inspection_config.widget == 11:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1_high)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2_high)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_high_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])

                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P828387YA1A_CLIP_Model,
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )
                                        
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :2400, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -2400:, :]
                                self.InspectionResult_EndSegmentation_Left[i] = self.P828387YA1A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.3, retina_masks=True, imgsz=1960, verbose=False)
                                self.InspectionResult_EndSegmentation_Right[i] = self.P828387YA1A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.3, retina_masks=True, imgsz=1960, verbose=False)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P828387YA1A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Right[i], side="LH")


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

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)
                                
                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P828397YA1A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            # self.InspectionImages_prev[0] = self.InspectionImages[0]
                            # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                            # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)

            if self.inspection_config.widget == 12:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])


                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P731957YA0A_CLIP_Model,
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )
                                        
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1680, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1680:, :]
                                self.InspectionResult_EndSegmentation_Left[i] = self.P731957YA0A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False, retina_masks=True)
                                self.InspectionResult_EndSegmentation_Right[i] = self.P731957YA0A_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False, retina_masks=True)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P731957YA0A_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Right[i])


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

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)
                                
                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P731957YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                            # self.InspectionImages_prev[0] = self.InspectionImages[0]
                            # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                            # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)

            if self.inspection_config.widget == 13:


                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            ClipInsertionMachine = "COUNTERRESET",
                            PPMS = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                # print(self.inspection_config.doInspection)
                # print(time.time() - self.InspectionTimeStart)
                
                self.ethernetTrigger[0] = 0
                self.ethernetStatus.emit(self.ethernetTrigger)

                if self.gaikanStart == True:
                    if time.time() - self.gaikanTime > self.gaikanTimerPeriod:
                        self.gaikanStart = False
                        self.gaikanTime = 0
                        #write the gaikanframe to folder with cv2
                    self.save_image(self.gaikanframe)
                    print(True)


                    

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            # #do imwrite with date as name
                            # self.save_image(self.InspectionImages[0])


                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P8462284S00_CLIP_Model,
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )
                                        
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1640, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1640:, :]

                                self.InspectionImages_keypoint_Left[i] = self.InspectionImages[i][:, 412:1692, :]
                                self.InspectionImages_keypoint_Right[i] = self.InspectionImages[i][:, -1692:-412, :]

                                self.InspectionImages_endSegmentation_Left[i] = add_imageborder(img = self.InspectionImages_endSegmentation_Left[i], width = 200)
                                self.InspectionImages_endSegmentation_Right[i] = add_imageborder(img = self.InspectionImages_endSegmentation_Right[i], width = 200)

                                self.InspectionResult_keypoint_Left[i] = self.P8462284S00_KEYPOINT_Model(source=self.InspectionImages_keypoint_Left[i], conf=0.6, imgsz=1280, verbose=False)
                                self.InspectionResult_keypoint_Right[i] = self.P8462284S00_KEYPOINT_Model(source=self.InspectionImages_keypoint_Right[i], conf=0.6, imgsz=1280, verbose=False)
   
                                self.InspectionResult_EndSegmentation_Left[i] = self.P8462284S00_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=1680, verbose=False)
                                self.InspectionResult_EndSegmentation_Right[i] = self.P8462284S00_SEGMENT_Model(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=1680, verbose=False)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i]   = P8462284S00_check(self.InspectionImages[i], 
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                self.InspectionResult_EndSegmentation_Right[i],
                                                                                                                                                                                                                self.InspectionResult_keypoint_Left[i],
                                                                                                                                                                                                                self.InspectionResult_keypoint_Right[i]
                                                                                                                                                                                                                )


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

                        

                            self.emitImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.emitImages[0] = cv2.cvtColor(self.emitImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.emitImages[0]))

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)
                            self.P8462284S00_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

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
                                    ClipInsertionMachine = self.inspection_config.clipSounyuuNumber,
                                    PPMS = self.inspection_config.ppmsnumber)
                                
                            time.sleep(1.5)

            if self.inspection_config.widget in [21, 22, 23]:

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False


                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((137, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "校正確認中", (50,150,10), size="large", x_offset = -280, y_offset = -100)
                            self.part1Cam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)


                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.NICHIJOU_TENKEN_Model, 
                                            slice_height=1980, slice_width=1980, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.2,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=False
                                        )
                                        


                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = dailyTenken(self.InspectionImages[i], self.InspectionResult_ClipDetection[i].object_prediction_list)


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
                                    ClipInsertionMachine = self.inspection_config.clipSounyuuNumber)
             

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=137)

                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            time.sleep(1.5)


                continue

            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

        self.msleep(1)

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
                ClipInsertionMachine = "MANUAL")

        return [ok_count_current, ng_count_current], [ok_count_total, ng_count_total]

    def save_result_database(self, partname, numofPart, 
                             currentnumofPart, deltaTime, 
                             kensainName, detected_pitch_str, 
                             delta_pitch_str, total_length, 
                             resultPitch, status, NGreason, ClipInsertionMachine="Null", PPMS="Null"):
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
        
        if ClipInsertionMachine != "Null":
            ClipInsertionMachine = str(ClipInsertionMachine)
        else:
            ClipInsertionMachine = "Null"

        if PPMS != "Null":
            PPMS = str(PPMS)
        else:
            PPMS = "Null"

        self.cursor.execute('''
        INSERT INTO inspection_results (partname, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason, ClipInsertionMachine, PPMS)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason, ClipInsertionMachine, PPMS))
        self.conn.commit()

        # Update the totatl part number (Maybe the day has been changed)
        for key, value in self.widget_dir_map.items():
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        try:
            #Also save to mysql cursor
            self.mysql_cursor.execute('''
            INSERT INTO inspection_results (partName, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason, ClipInsertionMachine, PPMS)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason, ClipInsertionMachine, PPMS))
            self.mysql_conn.commit()
        except Exception as e:
            print(f"Error saving to MySQL: {e}")


    # def get_last_entry_currentnumofPart(self, part_name):
    #     self.cursor.execute('''
    #     SELECT currentnumofPart 
    #     FROM inspection_results 
    #     WHERE partName = ? 
    #     ORDER BY id DESC 
    #     LIMIT 1
    #     ''', (part_name,))
        
    #     row = self.cursor.fetchone()
    #     if row:
    #         currentnumofPart = eval(row[0])
    #         return currentnumofPart
    #     else:
    #         return [0, 0]
            
    # def get_last_entry_total_numofPart(self, part_name):
    #     # Get today's date in yyyymmdd format
    #     today_date = datetime.now().strftime("%Y%m%d")

    #     self.cursor.execute('''
    #         SELECT numofPart, timestampDate 
    #         FROM inspection_results 
    #         WHERE partName = ? 
    #         ORDER BY id DESC 
    #         LIMIT 1
    #     ''', (part_name,))

    #     row = self.cursor.fetchone()
    #     #print last entry and today date
    #     print(f"Part name: {part_name}, Last entry: {row}")
    #     print(f"Today date: {today_date}")
    #     if row:
    #         last_entry_date = row[1]
    #         if last_entry_date == today_date:
    #             # Convert the string (e.g., "[10, 5]") into an actual list
    #             numofPart = eval(row[0])
    #             return numofPart
    #         else:
    #             # If the last entry is not from today, return [0, 0]
    #             return [0, 0]
    #     else:
    #         return [0, 0]  # Default values if no entry is found

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

    def save_image(self, image):
        dir = "aikensa/inspection/" + self.widget_dir_map[self.inspection_config.widget]
        os.makedirs(dir, exist_ok=True)
        filename = dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        
        # Check if the file already exists and add an identifier if it does
        counter = 1
        while os.path.exists(filename):
            filename = dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{counter}.png"
            counter += 1
        
        cv2.imwrite(filename, image)

    def save_image_result(self, image_initial, image_result, result):
        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" +  str(result) + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" + str(result) + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        image_result = self.downSampling(image_result, width=image_result.shape[1] // 2, height=image_result.shape[0] // 2)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_initial)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_result)

    def save_image_result_withKatabu(self, image_initial, image_result, katabu_initial, katabu_result, result):
        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" +  str(result) + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" + str(result) + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        image_result = self.downSampling(image_result, width=image_result.shape[1] // 2, height=image_result.shape[0] // 2)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_initial)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_katabu.png", katabu_initial)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_result)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_katabu.png", katabu_result)

    def minitimerStart(self):
        self.timerStart_mini = time.time()
    
    def minitimerFinish(self, message = "OperationName"):
        self.timerFinish_mini = time.time()
        # self.fps_mini = 1/(self.timerFinish_mini - self.timerStart_mini)
        print(f"Time to {message} : {(self.timerFinish_mini - self.timerStart_mini) * 1000} ms")
        # print(f"FPS of {message} : {self.fps_mini}")

    def convertQImage(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
        return processed_image
    
    def convertQImageKatabu(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        # Ensure image data is converted to bytes
        processed_image = QImage(image.data.tobytes(), w, h, bytesPerLine, QImage.Format_BGR888)
        return processed_image

    def converQImageRGB(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image
    
    def downScaledImage(self, image, scaleFactor=1.0):
        #create a copy of the image
        resized_image = cv2.resize(image, (0, 0), fx=1/scaleFactor, fy=1/scaleFactor, interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    def downSampling(self, image, width=384, height=256):
        #create a copy of the image
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def load_matrix_from_yaml(self, filename):
        with open(filename, 'r') as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_param.get('camera_matrix'))
            distortion_coeff = np.array(calibration_param.get('distortion_coefficients'))
        return camera_matrix, distortion_coeff

    def initialize_model(self):

        print("Model Dummy Loaded")

        # #Change based on the widget
        #05
        P658107YA0A_CLIP_Model = None
        P658107YA0A_SEGMENT_Model = None
        #06 07
        P828387YA0A_CLIP_Model = None
        P828387YA0A_SEGMENT_Model = None
        #08 09
        P828387YA6A_CLIP_Model = None
        P828387YA6A_SEGMENT_Model = None
        #10 11
        P828387YA1A_CLIP_Model = None
        P828387YA1A_SEGMENT_Model = None
        #12 
        P731957YA0A_CLIP_Model = None
        P731957YA0A_SEGMENT_Model = None
        #13
        P8462284S00_CLIP_Model = None
        P8462284S00_SEGMENT_Model = None
        P8462284S00_KEYPOINT_Model = None

        #05
        path_P658107YA0A_CLIP_Model = "./aikensa/models/P658107YA0A_detect.pt"
        path_P658107YA0A_SEGMENT_Model = "./aikensa/models/P658107YA0A_segment.pt"
        P658107YA0A_CLIP_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",model_path=path_P658107YA0A_CLIP_Model,
                                                                            confidence_threshold=0.5,
                                                                            device="cuda:0")
        P658107YA0A_SEGMENT_Model = YOLO(path_P658107YA0A_SEGMENT_Model)
        self.P658107YA0A_CLIP_Model = P658107YA0A_CLIP_Model
        self.P658107YA0A_SEGMENT_Model = P658107YA0A_SEGMENT_Model
        #06 07
        path_P828387YA0A_CLIP_Model = "./aikensa/models/P828387YA0A_detect.pt"
        path_P828387YA0A_SEGMENT_Model = "./aikensa/models/P828387YA0A_segment.pt"
        P828387YA0A_CLIP_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",model_path=path_P828387YA0A_CLIP_Model,
                                                                            confidence_threshold=0.5,   
                                                                            device="cuda:0")
        P828387YA0A_SEGMENT_Model = YOLO(path_P828387YA0A_SEGMENT_Model)
        self.P828387YA0A_CLIP_Model = P828387YA0A_CLIP_Model
        self.P828387YA0A_SEGMENT_Model = P828387YA0A_SEGMENT_Model
        #08 09
        path_P828387YA6A_CLIP_Model = "./aikensa/models/P828387YA6A_detect.pt"
        path_P828387YA6A_SEGMENT_Model = "./aikensa/models/P828387YA6A_segment.pt"
        path_P828387YA6A_KATABUMARKING_MODEL = "./aikensa/models/P828387YA6A_katabu.pt"
        P828387YA6A_CLIP_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",model_path=path_P828387YA6A_CLIP_Model,
                                                                            confidence_threshold=0.7,
                                                                            device="cuda:0")

        P828387YA6A_SEGMENT_Model = YOLO(path_P828387YA6A_SEGMENT_Model)
        P828387YA6A_KATABUMARKING_MODEL = YOLO(path_P828387YA6A_KATABUMARKING_MODEL)
        self.P828387YA6A_CLIP_Model = P828387YA6A_CLIP_Model
        self.P828387YA6A_SEGMENT_Model = P828387YA6A_SEGMENT_Model
        self.P828387YA6A_KATABUMARKING_MODEL = P828387YA6A_KATABUMARKING_MODEL

        #10 11
        path_P828387YA1A_CLIP_Model = "./aikensa/models/P828387YA1A_detect.pt"
        path_P828387YA1A_SEGMENT_Model = "./aikensa/models/P828387YA1A_segment.pt"
        P828387YA1A_CLIP_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",model_path=path_P828387YA1A_CLIP_Model,
                                                                            confidence_threshold=0.35,
                                                                            device="cuda:0")
        P828387YA1A_SEGMENT_Model = YOLO(path_P828387YA1A_SEGMENT_Model)
        self.P828387YA1A_CLIP_Model = P828387YA1A_CLIP_Model
        self.P828387YA1A_SEGMENT_Model = P828387YA1A_SEGMENT_Model

        #12
        path_P731957YA0A_CLIP_Model = "./aikensa/models/P731957YA0A_detect.pt"
        path_P731957YA0A_SEGMENT_Model = "./aikensa/models/P731957YA0A_segment.pt"
        P731957YA0A_CLIP_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",model_path=path_P731957YA0A_CLIP_Model,
                                                                            confidence_threshold=0.5,
                                                                            device="cuda:0")
        P731957YA0A_SEGMENT_Model = YOLO(path_P731957YA0A_SEGMENT_Model)
        self.P731957YA0A_CLIP_Model = P731957YA0A_CLIP_Model
        self.P731957YA0A_SEGMENT_Model = P731957YA0A_SEGMENT_Model
        
        #13
        path_P8462284S00_CLIP_Model = "./aikensa/models/P8462284S00_detect.pt"
        path_P8462284S00_SEGMENT_Model = "./aikensa/models/P8462284S00_segment.pt"
        path_P8462284S00_KEYPOINT_Model = "./aikensa/models/P8462284S00_KEYPOINT.pt"
        P8462284S00_CLIP_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",model_path=path_P8462284S00_CLIP_Model,
                                                                            confidence_threshold=0.80,
                                                                            device="cuda:0")
        P8462284S00_SEGMENT_Model = YOLO(path_P8462284S00_SEGMENT_Model)
        P8462284S00_KEYPOINT_Model = YOLO(path_P8462284S00_KEYPOINT_Model)
        self.P8462284S00_CLIP_Model = P8462284S00_CLIP_Model
        self.P8462284S00_KEYPOINT_Model = P8462284S00_KEYPOINT_Model
        self.P8462284S00_SEGMENT_Model = P8462284S00_SEGMENT_Model

        NICHIJOU_TENKEN_Model = None
        path_NICHIJOU_TENKEN_Model = "./aikensa/models/AIKENSA5GO_NICHIJOU_TENKEN.pt"

        if os.path.exists(path_NICHIJOU_TENKEN_Model):
            NICHIJOU_TENKEN_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                            model_path=path_NICHIJOU_TENKEN_Model,
                                                                            confidence_threshold=0.5,
                                                                            device="cuda:0")
            
        self.NICHIJOU_TENKEN_Model = NICHIJOU_TENKEN_Model

        print("Model Loaded")
        
    def stop(self):
        self.inspection_config.widget = -1
        self.running = False
        print("Releasing all cameras.")
        self.release_all_camera()
        print("Inspection thread stopped.")

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