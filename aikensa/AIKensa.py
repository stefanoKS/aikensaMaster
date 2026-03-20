import re
import cv2
import sys
import yaml
import os
import json
import traceback
import faulthandler
import threading
from enum import Enum
import time
import datetime
from typing import List

from PyQt5 import QtCore

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut, QLineEdit, QComboBox, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QColor
from aikensa.core.scripts.img_processing.cannydetect import canny_edge_detection
from aikensa.core.scripts.img_processing.detectaruco import detectAruco
from aikensa.core.scripts.img_processing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.core.thread.calibration_thread import CalibrationThread, CalibrationConfig
from aikensa.core.thread.inspection_thread import InspectionThread, InspectionConfig

from aikensa.core.thread.sio_thread import ServerMonitorThread
from aikensa.core.thread.time_thread import TimeMonitorThread

from aikensa.core.scripts.scripts_classes import DebouncedButton


# List of UI files to be loaded
UI_FILES = [
    'aikensa/gui/mainPage.ui',         #index 0
    'aikensa/gui/pages/CALIBRATION/calibration_cam1.ui', #index 1
    'aikensa/gui/pages/CALIBRATION/calibration_cam2.ui', #index 2         
    'aikensa/gui/pages/CALIBRATION/camera_merge.ui',     #index 3
    "aikensa/gui/pages/empty.ui",            #empty 4
    "aikensa/gui/pages/MMC/M_5A45/P82833W050P.ui",      #index 5
    "aikensa/gui/pages/MMC/M_5A45/P82832W040P.ui",      #index 6
    "aikensa/gui/pages/MMC/M_5A45/P82833W090P.ui",      #index 7
    "aikensa/gui/pages/MMC/M_5A45/P82832W080P.ui",      #index 8
    "aikensa/gui/pages/MMC/M_5A45/P82833W050PKENGEN.ui",      #index 9
    "aikensa/gui/pages/MMC/M_5A45/P82832W040PKENGEN.ui",      #index 10
    "aikensa/gui/pages/MMC/M_5A45/P82833W090PKENGEN.ui",      #index 11
    "aikensa/gui/pages/MMC/M_5A45/P82832W080PKENGEN.ui",      #index 12
    "aikensa/gui/pages/MMC/M_5A45/P82833W050PCLIPSOUNYUUKI.ui",      #index 13
    "aikensa/gui/pages/MMC/M_5A45/P82832W040PCLIPSOUNYUUKI.ui",      #index 14
    "aikensa/gui/pages/MMC/M_5A45/P82833W090PCLIPSOUNYUUKI.ui",      #index 15
    "aikensa/gui/pages/MMC/M_5A45/P82832W080PCLIPSOUNYUUKI.ui",      #index 16
    "aikensa/gui/pages/NISSAN/M_JC2D/P808387UA1A.ui", #empty 17
    "aikensa/gui/pages/NISSAN/M_JC2D/P828447UA0A.ui", #empty 18
    "aikensa/gui/pages/empty.ui", #empty 19
    "aikensa/gui/pages/empty.ui", #empty 20
    "aikensa/gui/pages/dailyTenken_new_01.ui",  # index 21
    "aikensa/gui/pages/dailyTenken_new_02.ui",  # index 22
    "aikensa/gui/pages/dailyTenken_new_03.ui",  # index 23
    "aikensa/gui/pages/empty.ui",            #empty 24
    "aikensa/gui/pages/empty.ui",            #empty 25
    "aikensa/gui/pages/empty.ui",            #empty 26
    "aikensa/gui/pages/empty.ui",            #empty 27
    "aikensa/gui/pages/empty.ui",            #empty 28
    "aikensa/gui/pages/empty.ui",            #empty 29
    "aikensa/gui/pages/empty.ui",            #empty 30
    "aikensa/gui/pages/NISSAN/M_JC2D/P658217UA0A.ui",            #empty 31
    "aikensa/gui/pages/NISSAN/M_JC2D/P658207UA0A.ui",            #empty 32
    "aikensa/gui/pages/NISSAN/M_JC2D/P658217UJ0A.ui",            #empty 33
    "aikensa/gui/pages/NISSAN/M_JC2D/P658207UJ0A.ui",            #empty 34
    "aikensa/gui/pages/NISSAN/M_P13C/P658207LE0A.ui",  #index 35 - P13C Part
    "aikensa/gui/pages/NISSAN/M_J42U/P731957YA0A.ui",  #index 36 - M_J42U Part (P731957YA0A_button)
    "aikensa/gui/pages/NISSAN/M_J42U/P808387YA0A.ui",  #index 37 - M_J42U Part (P808387YA0A_button)
    "aikensa/gui/pages/NISSAN/M_J42U/P828387YA0A.ui",  #index 38 - M_J42U Part (no button)
    "aikensa/gui/pages/NISSAN/M_J42U/P828387YA1A.ui",  #index 39 - M_J42U Part (P828387YA1A_button)
    "aikensa/gui/pages/NISSAN/M_J42U/P828397YA1A.ui",  #index 40 - M_J42U Part (P828397YA1A_button)
    "aikensa/gui/pages/NISSAN/M_J42U/P828387YA6A.ui",  #index 41 - M_J42U Part (P828387YA6A_button)
    "aikensa/gui/pages/NISSAN/M_J42U/P828397YA6A.ui",  #index 42 - M_J42U Part (P828397YA6A_button)
    "aikensa/gui/pages/NISSAN/M_J42U/P658107YA0A.ui",  #index 43 - M_J42U Part (P658107YA0A_button)
    "aikensa/gui/pages/MMC/M_5A45/P5902A510.ui",       #index 44 - M_5A45 Part (P5902A510_button)
    "aikensa/gui/pages/MMC/M_5A45/P5902A509.ui",       #index 45 - M_5A45 Part (P5902A509_button)
    "aikensa/gui/pages/MMC/M_5A45/P5819A107.ui",       #index 46 - M_5A45 Part (P5819A107_button)
    "aikensa/gui/pages/SUZUKI/YT3/P8462284S00.ui",     #index 47 - YT3 Part (P8462284S00_button)
    "aikensa/gui/pages/NISSAN/M_J42U/P828387YA6A_KATABU_NASHI.ui",  #index 48 - YA6A KATABU NASHI RH
    "aikensa/gui/pages/NISSAN/M_J42U/P828397YA6A_KATABU_NASHI.ui",  #index 49 - YA6A KATABU NASHI LH
]

class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.calibration_thread = CalibrationThread(CalibrationConfig())
        self.inspection_thread = InspectionThread(InspectionConfig())   
        self._setup_ui()

        # Thread for SiO
        HOST = '192.168.0.100'  # Use the IP address from SiO settings
        PORT = 40001  # Use the port number from SiO settings

        self.server_monitor_thread = ServerMonitorThread(
            HOST, PORT, check_interval=0.05)
        self.server_monitor_thread.server_status_signal.connect(self.handle_server_status)
        self.server_monitor_thread.input_states_signal.connect(self.handle_input_states)
        self.server_monitor_thread.start()

        self.button0 = DebouncedButton(debounce_ms=40)

        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self.initial_colors = {}#store initial colors of the labels

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
            31: "658217UA0A",
            32: "658207UA0A",
            33: "658217UJ0A",
            34: "658207UJ0A",
            35: "P658207LE0A",
            36: "P731957YA0A",
            37: "P808387YA0A",
            38: "P828387YA0A",
            39: "P828387YA1A",
            40: "P828397YA1A",
            41: "P828387YA6A",
            42: "P828397YA6A",
            43: "P658107YA0A",
            44: "P5902A510",
            45: "P5902A509",
            46: "P5819A107",
            47: "P8462284S00",
            48: "P828387YA6A_KATABU_NASHI",
            49: "P828397YA6A_KATABU_NASHI",
        }

        self.inspectionWidget_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        self.inspectionWidget_withKatabu  = [5, 6, 7, 8, 9, 10, 11, 12]
        self.inspectionWidget_withClip  = [5, 6, 7, 8]

        self.prevTriggerStates = 0
        self.TriggerWaitTime = 3.0
        self.currentTime = time.time()



    def timeUpdate(self, time):
        for label in self.timeLabel:
            if label:
                label.setText(time)

    def handle_server_status(self, is_up):
        status_text = "ON" if is_up else "OFF"
        status_color = "green" if is_up else "red"

        #to show the label later. Implement later

        for label in self.siostatus_server:
            if label:  # Check if the label was found correctly
                label.setText(status_text)
                label.setStyleSheet(f"color: {status_color};")

    def handle_input_states(self, input_states: List[int]):
        if not input_states:
            return
        if self.button0.update(int(input_states[0])):
            self.trigger_kensa()

    def trigger_kensa(self):
        self.Inspect_button.click()
        # self.button_kensa4.click()

    def trigger_rekensa(self):
        self.button_rekensa.click()

    def _setup_ui(self):

        # LOAD JSON for PARTS CONFIG


        self.calibration_thread.CalibCamStream.connect(self._setCalibFrame)

        self.calibration_thread.CamMerge1.connect(self._setMergeFrame1)
        self.calibration_thread.CamMerge2.connect(self._setMergeFrame2)
        self.calibration_thread.CamMergeAll.connect(self._setMergeFrameAll)

        self.inspection_thread.partCam.connect(self._setPartFrame)
        self.inspection_thread.partKatabuL.connect(self._setFrameKatabuL)
        self.inspection_thread.partKatabuR.connect(self._setFrameKatabuR)

        self.inspection_thread.clip1Signal.connect(self._setClip1Frame)
        self.inspection_thread.clip2Signal.connect(self._setClip2Frame)
        self.inspection_thread.clip3Signal.connect(self._setClip3Frame)

        self.inspection_thread.ethernetStatus.connect(self._setEthernetStatus)

        self.inspection_thread.modelErrorSignal.connect(self._handleModelError)

        self.inspection_thread.P82833W050P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W050P)
        self.inspection_thread.P82832W040P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W040P)
        self.inspection_thread.P82833W090P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W090P)
        self.inspection_thread.P82832W080P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W080P)

        self.inspection_thread.P82833W050PKENGEN_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W050PKENGEN)
        self.inspection_thread.P82832W040PKENGEN_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W040PKENGEN)
        self.inspection_thread.P82833W090PKENGEN_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W090PKENGEN)
        self.inspection_thread.P82832W080PKENGEN_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W080PKENGEN)  

        self.inspection_thread.P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W050PCLIPSOUNYUUKI)
        self.inspection_thread.P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W040PCLIPSOUNYUUKI)
        self.inspection_thread.P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W090PCLIPSOUNYUUKI)
        self.inspection_thread.P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W080PCLIPSOUNYUUKI)

        self.inspection_thread.P808387UA1A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P808387UA1A)
        self.inspection_thread.P828447UA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828447UA0A)

        self.inspection_thread.P658217UA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P658217UA0A)
        self.inspection_thread.P658207UA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P658207UA0A)
        self.inspection_thread.P658217UJ0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P658217UJ0A)
        self.inspection_thread.P658207UJ0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P658207UJ0A)
        self.inspection_thread.P731957YA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P731957YA0A)
        self.inspection_thread.P808387YA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P808387YA0A)
        self.inspection_thread.P828387YA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828387YA0A)
        self.inspection_thread.P828387YA1A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828387YA1A)
        self.inspection_thread.P828397YA1A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828397YA1A)
        self.inspection_thread.P828387YA6A_KATABU_NASHI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828387YA6A_KATABU_NASHI)
        self.inspection_thread.P828397YA6A_KATABU_NASHI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828397YA6A_KATABU_NASHI)
        self.inspection_thread.P658107YA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P658107YA0A)

        # P13C Part
        self.inspection_thread.P658207LE0A_InspectionResult.connect(self._outputMeasurementText_P658207LE0A)

        self.inspection_thread.current_numofPart_signal.connect(self._update_OKNG_label)
        self.inspection_thread.today_numofPart_signal.connect(self._update_todayOKNG_label)

        self.inspection_thread.pickingOrderSignal.connect(self._update_clipPickingOrder)

        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)

        main_widget = self.stackedWidget.widget(0)

        dailyTenken01_widget = self.stackedWidget.widget(21)
        dailyTenken02_widget = self.stackedWidget.widget(22)
        dailyTenken03_widget = self.stackedWidget.widget(23)

        cameraCalibration1_widget = self.stackedWidget.widget(1)
        cameraCalibration2_widget = self.stackedWidget.widget(2)
        mergeCamera_widget = self.stackedWidget.widget(3)

        cameraCalibration1_button = main_widget.findChild(QPushButton, "camcalibrationbutton1")
        cameraCalibration2_button = main_widget.findChild(QPushButton, "camcalibrationbutton2")
        mergeCamera_button = main_widget.findChild(QPushButton, "cameraMerge")

        dailytenken01_button = main_widget.findChild(QPushButton, "dailytenkenbutton")
        dailytenken02_button = dailyTenken01_widget.findChild(QPushButton, "nextButton")
        dailytenken02_back_button = dailyTenken02_widget.findChild(QPushButton, "prevButton")
        dailytenken03_button = dailyTenken02_widget.findChild(QPushButton, "nextButton")
        dailytenken03_back_button = dailyTenken03_widget.findChild(QPushButton, "prevButton")
        dailytenken_kanryou_button = dailyTenken03_widget.findChild(QPushButton, "finishButton")

        if cameraCalibration1_button:
            cameraCalibration1_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
            cameraCalibration1_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 1))
            cameraCalibration1_button.clicked.connect(self.calibration_thread.start)

        if cameraCalibration2_button:
            cameraCalibration2_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
            cameraCalibration2_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 2))
            cameraCalibration2_button.clicked.connect(self.calibration_thread.start)

        if mergeCamera_button:
            mergeCamera_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
            mergeCamera_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 3))
            mergeCamera_button.clicked.connect(self.calibration_thread.start)

        for i in range(1, 3):
            CalibrateSingleFrame = self.stackedWidget.widget(i).findChild(QPushButton, "calibSingleFrame")
            CalibrateSingleFrame.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateSingeFrameMatrix", True))

            CalibrateFinalCameraMatrix = self.stackedWidget.widget(i).findChild(QPushButton, "calibCam")
            CalibrateFinalCameraMatrix.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateCamMatrix", True))

        calcHomoCam1 = mergeCamera_widget.findChild(QPushButton, "calcH_cam1")
        calcHomoCam2 = mergeCamera_widget.findChild(QPushButton, "calcH_cam2")
        calcHHomoCam1_high = mergeCamera_widget.findChild(QPushButton, "calcH_cam1_high")
        calcHHomoCam2_high = mergeCamera_widget.findChild(QPushButton, "calcH_cam2_high")

        calcHomoCam1.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam1", True))
        calcHomoCam2.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam2", True))
        calcHHomoCam1_high.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam1_high", True))
        calcHHomoCam2_high.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam2_high", True))

        planarize_combined = mergeCamera_widget.findChild(QPushButton, "planarize")
        planarize_combined_high = mergeCamera_widget.findChild(QPushButton, "planarize_high")

        planarize_combined.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarize", True))
        planarize_combined_high.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarizeHigh", True))
        
        button_config = {
            "P82833W050Pbutton": {"widget_index": 5, "inspection_param": 5},
            "P82832W040Pbutton": {"widget_index": 6, "inspection_param": 6},
            "P82833W090Pbutton": {"widget_index": 7, "inspection_param": 7},
            "P82832W080Pbutton": {"widget_index": 8, "inspection_param": 8},
               "P82833W050PKENGEN_button": {"widget_index": 9, "inspection_param": 9},
               "P82832W080PKENGEN_button": {"widget_index": 10, "inspection_param": 10},
               "P82833W090PKENGEN_button": {"widget_index": 11, "inspection_param": 11},
               "P82832W040PKENGEN_button": {"widget_index": 12, "inspection_param": 12},
               "P82833W050PCLIPSOUNYUUKI_button": {"widget_index": 13, "inspection_param": 13},
               "P82832W040PCLIPSOUNYUUKI_button": {"widget_index": 14, "inspection_param": 14},
               "P82833W090PCLIPSOUNYUUKI_button": {"widget_index": 15, "inspection_param": 15},
               "P82832W080PCLIPSOUNYUUKI_button": {"widget_index": 16, "inspection_param": 16},
            "P808387UA1A_button": {"widget_index": 17, "inspection_param": 17},
            "P828447UA0A_button": {"widget_index": 18, "inspection_param": 18},
            "P658217UA0A_button": {"widget_index": 30, "inspection_param": 30},  # Widget 30 (same part as 31)
            "P658217UA0A_31_button": {"widget_index": 31, "inspection_param": 31},
            "P658207UA0A_button": {"widget_index": 32, "inspection_param": 32},
            "P658217UJ0A_button": {"widget_index": 33, "inspection_param": 33},
            "P658207UJ0A_button": {"widget_index": 34, "inspection_param": 34},
            "P658207LE0A_button": {"widget_index": 35, "inspection_param": 35},  # P13C Part
            "P731957YA0A_button": {"widget_index": 36, "inspection_param": 36},  # M_J42U Part
            "P808387YA0A_button": {"widget_index": 37, "inspection_param": 37},  # M_J42U Part
            "P828387YA0A_button": {"widget_index": 38, "inspection_param": 38},  # M_J42U Part
            "P828387YA1A_button": {"widget_index": 39, "inspection_param": 39},  # M_J42U Part
            "P828397YA1A_button": {"widget_index": 40, "inspection_param": 40},  # M_J42U Part
            "P828387YA6A_button": {"widget_index": 41, "inspection_param": 41},  # M_J42U Part
            "P828397YA6A_button": {"widget_index": 42, "inspection_param": 42},  # M_J42U Part
            "P658107YA0A_button": {"widget_index": 43, "inspection_param": 43},  # M_J42U Part
            "P5902A510_button": {"widget_index": 44, "inspection_param": 44},  # M_5A45 Part
            "P5902A509_button": {"widget_index": 45, "inspection_param": 45},  # M_5A45 Part
            "P5819A107_button": {"widget_index": 46, "inspection_param": 46},  # M_5A45 Part
            "P8462284S00_button": {"widget_index": 47, "inspection_param": 47},  # YT3 Part
            "P828387YA6A_KATABUNASHI_button": {"widget_index": 48, "inspection_param": 48},  # M_J42U YA6A KATABU NASHI RH
            "P828397YA6A_KATABU_NASHI_button": {"widget_index": 49, "inspection_param": 49},  # M_J42U YA6A KATABU NASHI LH
        }


        for button_name, config in button_config.items():
            button = main_widget.findChild(QPushButton, button_name)
            
            if button:
                # Connect each signal with the necessary parameters
                button.clicked.connect(lambda _, idx=config["widget_index"]: self.stackedWidget.setCurrentIndex(idx))
                button.clicked.connect(lambda _, param=config["inspection_param"]: self._set_inspection_params(self.inspection_thread, 'widget', param))
                button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
                button.clicked.connect(self.calibration_thread.stop)

        dailytenken01_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(21))
        dailytenken01_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 21))
        dailytenken01_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken01_button.clicked.connect(self.calibration_thread.stop)

        dailytenken02_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(22))
        dailytenken02_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 22))

        dailytenken02_back_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(21))
        dailytenken02_back_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 21))

        dailytenken03_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(23))
        dailytenken03_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 23))

        dailytenken03_back_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(22))
        dailytenken03_back_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 22))

        self.widget_indices_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        self.inspection_widget_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        self.inspection_widget_indices_without_dailytenken = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


        self.timeLabel = [self.stackedWidget.widget(i).findChild(QLabel, "timeLabel") for i in self.widget_indices_list]

        self.siostatus_server = [self.stackedWidget.widget(i).findChild(QLabel, "status_sio") for i in self.widget_indices_list]


        for i in self.inspection_widget_indices:
            self.Inspect_button = self.stackedWidget.widget(i).findChild(QPushButton, "InspectButton")
            if self.Inspect_button:
                self.Inspect_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doInspection", True))

        for i in self.inspection_widget_indices_without_dailytenken:
            self.connect_inspectionConfig_button(i, "kansei_plus", "kansei_plus", True)
            self.connect_inspectionConfig_button(i, "kansei_minus", "kansei_minus", True)
            self.connect_inspectionConfig_button(i, "furyou_plus", "furyou_plus", True)
            self.connect_inspectionConfig_button(i, "furyou_minus", "furyou_minus", True)
            self.connect_inspectionConfig_button(i, "kansei_plus_10", "kansei_plus_10", True)
            self.connect_inspectionConfig_button(i, "kansei_minus_10", "kansei_minus_10", True)
            self.connect_inspectionConfig_button(i, "furyou_plus_10", "furyou_plus_10", True)
            self.connect_inspectionConfig_button(i, "furyou_minus_10", "furyou_minus_10", True)
            #connect reset button
            self.connect_inspectionConfig_button(i, "counterReset", "counterReset", True)

            self.connect_line_edit_text_changed(widget_index=i, line_edit_name="kensain_name", inspection_param="kensainNumber")

            #additional logic for ppms number
            if i in [13, 14, 15, 16, 17, 18]:
                self.connect_line_edit_text_changed(widget_index=i, line_edit_name="ppms_number", inspection_param="ppmsnumber")

        
        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")

            if button_quit:
                button_quit.clicked.connect(self._close_app)

            if button_main_menu:
                button_main_menu.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                button_main_menu.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 0))

        self.setCentralWidget(self.stackedWidget)
        self.showFullScreen()

    def connect_button_font_color_change(self, widget_index, qtbutton, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, qtbutton)

        if button:
            button.setStyleSheet("color: black")
            def toggle_font_color_and_param():
                current_value = getattr(self.cam_thread.cam_config, cam_param, False)
                new_value = not current_value
                setattr(self.cam_thread.cam_config, cam_param, new_value)
                self._set_cam_params(self.cam_thread, cam_param, new_value)
                new_color = "red" if new_value else "black"
                button.setStyleSheet(f"color: {new_color}")
            button.pressed.connect(toggle_font_color_and_param)
        else:
            print(f"Button '{qtbutton}' not found.")

    def connect_button_label_color_change(self, widget_index, qtbutton, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, qtbutton)

        if button:
            button.setStyleSheet("color: red")
            def toggle_font_color_and_param():
                current_value = getattr(self.cam_thread.cam_config, cam_param, False)
                new_value = not current_value
                setattr(self.cam_thread.cam_config, cam_param, new_value)
                self._set_cam_params(self.cam_thread, cam_param, new_value)
                new_color = "green" if new_value else "red"
                button.setStyleSheet(f"color: {new_color}")

            button.pressed.connect(toggle_font_color_and_param)
        else:
            print(f"Button '{qtbutton}' not found.")

    def connect_line_edit_text_changed(self, widget_index, line_edit_name, inspection_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_inspection_params(self.inspection_thread, inspection_param, text))

    def connect_inspectionConfig_button(self, widget_index, button_name, cam_param, value):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.pressed.connect(lambda: self._set_inspection_params(self.inspection_thread, cam_param, value))
            # print(f"Button '{button_name}' connected to cam_param '{cam_param}' with value '{value}' in widget {widget_index}")

    def _close_app(self):
        # self.cam_thread.stop()
        self.calibration_thread.stop()
        self.inspection_thread.stop()
        self.server_monitor_thread.stop()
        time.sleep(1.0)
        QCoreApplication.instance().quit()

    def _load_ui(self, filename):
        widget = QMainWindow()
        loadUi(filename, widget)
        return widget

    def _set_frame_raw(self, image):
        for i in [1, 2]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "cameraFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _set_frame_inference(self, image):
        for i in [3, 4]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "cameraFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _set_cam_params(self, thread, key, value):
        setattr(thread.cam_config, key, value)

    def _toggle_param_and_update_label(self, param, label):
        new_value = not getattr(self.cam_thread.cam_config, param)
        self._set_cam_params(self.cam_thread, param, new_value)

        color = "green" if new_value else "red"
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _update_OKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "current_kansei")
                    current_furyou_label = widget.findChild(QLabel, "current_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            # else:
            #     print(f"Widget key {widget_key} is out of bounds for numofPart")

    def _update_todayOKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "status_kansei")
                    current_furyou_label = widget.findChild(QLabel, "status_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            # else:
            #     print(f"Widget key {widget_key} is out of bounds for todaynumofPart")


#5
    def _outputMeasurementText_P82833W050P(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [5]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#6
    def _outputMeasurementText_P82832W040P(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [6]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#7
    def _outputMeasurementText_P82833W090P(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
        for widget_index in [7]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#8
    def _outputMeasurementText_P82832W080P(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
        for widget_index in [8]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#9
    def _outputMeasurementText_P82833W050PKENGEN(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [9]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#10
    def _outputMeasurementText_P82832W040PKENGEN(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [10]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#11
    def _outputMeasurementText_P82833W090PKENGEN(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
        for widget_index in [11]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#12
    def _outputMeasurementText_P82832W080PKENGEN(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
        for widget_index in [12]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#13
    def _outputMeasurementText_P82833W050PCLIPSOUNYUUKI(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [13]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#14
    def _outputMeasurementText_P82832W040PCLIPSOUNYUUKI(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [14]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#15
    def _outputMeasurementText_P82833W090PCLIPSOUNYUUKI(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
        for widget_index in [15]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#16
    def _outputMeasurementText_P82832W080PCLIPSOUNYUUKI(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
        for widget_index in [16]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#17
    def _outputMeasurementText_P808387UA1A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label"]
        for widget_index in [17]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#18
    def _outputMeasurementText_P828447UA0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [18]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#31
    def _outputMeasurementText_P658217UA0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label"]
        for widget_index in [31]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#32
    def _outputMeasurementText_P658207UA0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label"]
        for widget_index in [32]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#33
    def _outputMeasurementText_P658217UJ0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label"]
        for widget_index in [33]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#34
    def _outputMeasurementText_P658207UJ0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label"]
        for widget_index in [34]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#37
    def _outputMeasurementText_P731957YA0A(self, measurementValue, measurementResult):
        label_names_part = [
            "P1label", "P2label", "P3label", "P4label", "P5label", "P6label",
            "P7label", "P8label", "P9label", "P10label", "P11label", "P12label"
        ]
        for widget_index in [36]:
            for label_index, label_name in enumerate(label_names_part):
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"

                    label.setText(str(value))

                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"

                    if result == 1:
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

#37
    def _outputMeasurementText_P808387YA0A(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA0A(measurementValue, measurementResult, [37])

#38
    def _outputMeasurementText_P828387YA0A(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA0A(measurementValue, measurementResult, [38])

#39
    def _outputMeasurementText_P828387YA1A(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA1A(measurementValue, measurementResult, [39])

#40
    def _outputMeasurementText_P828397YA1A(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA1A(measurementValue, measurementResult, [40])

#43
    def _outputMeasurementText_P658107YA0A(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA0A(measurementValue, measurementResult, [43])

#41
    def _outputMeasurementText_P828387YA6A(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA0A(measurementValue, measurementResult, [41])

#42
    def _outputMeasurementText_P828397YA6A(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA0A(measurementValue, measurementResult, [42])

#48
    def _outputMeasurementText_P828387YA6A_KATABU_NASHI(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA0A(measurementValue, measurementResult, [48])

#49
    def _outputMeasurementText_P828397YA6A_KATABU_NASHI(self, measurementValue, measurementResult):
        self._outputMeasurementText_YA0A(measurementValue, measurementResult, [49])

    def _outputMeasurementText_YA0A(self, measurementValue, measurementResult, widget_indices):
        try:
            if not hasattr(self, "stackedWidget") or self.stackedWidget is None:
                return

            label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label"]
            for widget_index in widget_indices:
                widget = self.stackedWidget.widget(widget_index)
                if widget is None:
                    continue

                for label_index, label_name in enumerate(label_names_part):
                    label = widget.findChild(QLabel, label_name)
                    if label:
                        if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0
                            and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):

                            value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                        else:
                            value = "None"

                        label.setText(str(value))

                        if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0
                            and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                            result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                        else:
                            result = "None"

                        if result == 1:
                            label.setStyleSheet("background-color: green;")
                        elif result == 0:
                            label.setStyleSheet("background-color: red;")
                        else:
                            label.setStyleSheet("background-color: white;")
        except Exception:
            print("[UI Error] _outputMeasurementText_YA0A")
            print(traceback.format_exc())

    def _outputMeasurementText_YA1A(self, measurementValue, measurementResult, widget_indices):
        try:
            if not hasattr(self, "stackedWidget") or self.stackedWidget is None:
                return

            label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label"]
            for widget_index in widget_indices:
                widget = self.stackedWidget.widget(widget_index)
                if widget is None:
                    continue

                for label_index, label_name in enumerate(label_names_part):
                    label = widget.findChild(QLabel, label_name)
                    if label:
                        if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0
                            and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):

                            value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                        else:
                            value = "None"

                        label.setText(str(value))

                        if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0
                            and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                            result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                        else:
                            result = "None"

                        if result == 1:
                            label.setStyleSheet("background-color: green;")
                        elif result == 0:
                            label.setStyleSheet("background-color: red;")
                        else:
                            label.setStyleSheet("background-color: white;")
        except Exception:
            print("[UI Error] _outputMeasurementText_YA1A")
            print(traceback.format_exc())


    def _update_clipPickingOrder(self, pickingOrder):

        col050 = ["brown", "brown", "orange", "orange", "yellow"]
        col040 = ["white", "white", "orange", "orange", "yellow"]
        col090 = ["brown", "brown", "orange", "orange", "orange", "yellow"]
        col080 = ["white", "white", "orange", "orange", "orange", "yellow"]

        label050 = ["order1", "order2", "order3", "order4", "order5"]
        label040 = ["order1", "order2", "order3", "order4", "order5"]
        label090 = ["order1", "order2", "order3", "order4", "order5", "order6"]
        label080 = ["order1", "order2", "order3", "order4", "order5", "order6"]

        # print(f"pickingOrder: {pickingOrder}")

        for widget_key, part_name in self.widget_dir_map.items():
            if 0 <= widget_key < len(pickingOrder):
                lightOrder = pickingOrder[widget_key][:6]
                widget = self.stackedWidget.widget(widget_key)
                # print(f"Widget number {widget_key}, LightOrder: {lightOrder}")
                if widget_key in [5]:
                    for i, order in enumerate(col050):
                        label = widget.findChild(QLabel, f"order{i+1}")
                        if lightOrder[i] == 1:
                            label.setStyleSheet("QLabel { background-color: green; }")
                        elif lightOrder[i] == 0:
                            label.setStyleSheet(f"QLabel {{ background-color: {order}; }}")

                if widget_key in [6]:
                    for i, order in enumerate(col040):
                        label = widget.findChild(QLabel, f"order{i+1}")
                        if lightOrder[i] == 1:
                            label.setStyleSheet("QLabel { background-color: green; }")
                        elif lightOrder[i] == 0:
                            label.setStyleSheet(f"QLabel {{ background-color: {order}; }}")
                        
                if widget_key in [7]:
                    for i, order in enumerate(col090):
                        label = widget.findChild(QLabel, f"order{i+1}")
                        if lightOrder[i] == 1:
                            label.setStyleSheet("QLabel { background-color: green; }")
                        elif lightOrder[i] == 0:
                            label.setStyleSheet(f"QLabel {{ background-color: {order}; }}")

                if widget_key in [8]:
                    for i, order in enumerate(col080):
                        label = widget.findChild(QLabel, f"order{i+1}")
                        if lightOrder[i] == 1:
                            label.setStyleSheet("QLabel { background-color: green; }")
                        elif lightOrder[i] == 0:
                            label.setStyleSheet(f"QLabel {{ background-color: {order}; }}")
                

        
    def _update_OKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "current_kansei")
                    current_furyou_label = widget.findChild(QLabel, "current_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            else:
                print(f"Widget key {widget_key} is out of bounds for numofPart")

    def _set_labelFrame(self, widget, paramValue, label_names):
        colorOK = "blue"
        colorNG = "black"
        label = widget.findChild(QLabel, label_names) 
        color = colorNG if paramValue else colorOK
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")
        
    def _set_button_color(self, pitch_data):
        colorOK = "green"
        colorNG = "red"

        label_names = ["P1color", "P2color", "P3color",
                       "P4color", "P5color", "Lsuncolor"]
        labels = [self.stackedWidget.widget(5).findChild(QLabel, name) for name in label_names]
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _setCalibFrame(self, image):
        for i in [1, 2 ]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame1(self, image):
        widget = self.stackedWidget.widget(3)
        label = widget.findChild(QLabel, "camMerge1")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame2(self, image):
        widget = self.stackedWidget.widget(3)
        label = widget.findChild(QLabel, "camMerge2")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrameAll(self, image):
        widget = self.stackedWidget.widget(3)
        label = widget.findChild(QLabel, "camMergeAll")
        label.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame(self, image):
        for i in self.inspectionWidget_indices:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "framePart")
            label.setPixmap(QPixmap.fromImage(image))

    def _setFrameKatabuL(self, image):
        for i in self.inspectionWidget_withKatabu:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "frameKatabuL")
            label.setPixmap(QPixmap.fromImage(image))

    def _setClip1Frame(self, image):
        for i in self.inspectionWidget_withClip:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip1Frame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setClip2Frame(self, image):
        for i in self.inspectionWidget_withClip:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip2Frame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setClip3Frame(self, image):
        for i in self.inspectionWidget_withClip:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip3Frame")
            label.setPixmap(QPixmap.fromImage(image))
    
    def _setFrameKatabuR(self, image):
        for i in self.inspectionWidget_withKatabu:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "frameKatabuR")
            label.setPixmap(QPixmap.fromImage(image))

    def _extract_color(self, stylesheet):
        # Extracts the color value from the stylesheet string
        start = stylesheet.find("background-color: ") + len("background-color: ")
        end = stylesheet.find(";", start)
        return stylesheet[start:end].strip()

    def _store_initial_colors(self, widget_index, label_names):
        if widget_index not in self.initial_colors:
            self.initial_colors[widget_index] = {}
        labels = [self.stackedWidget.widget(widget_index).findChild(QLabel, name) for name in label_names]
        for label in labels:
            color = self._extract_color(label.styleSheet())
            self.initial_colors[widget_index][label.objectName()] = color
            # print(f"Stored initial color for {label.objectName()} in widget {widget_index}: {color}")

    def _set_calib_params(self, thread, key, value):
        setattr(thread.calib_config, key, value)

    def _set_inspection_params(self, thread, key, value):
        setattr(thread.inspection_config, key, value)

    def _setEthernetStatus(self, input):
        self.server_monitor_thread.server_config.eth_flag_0_4 = input

    def _handleModelError(self, error_message):
        """Handle model error messages from the inspection thread."""
        print(f"[GUI Model Error] {error_message}")
        # Display error message on all relevant widgets
        for widget_index in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(widget_index)
            if widget:
                # Try to find a label to display the error
                error_label = widget.findChild(QLabel, "error_label")
                if error_label:
                    error_label.setText(error_message)
                    error_label.setStyleSheet("color: red; font-weight: bold; background-color: yellow;")

    def _load_json(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load JSON:\n{e}")
            return {}

    def _outputMeasurementText_P658207LE0A(self, measurementValue, measurementResult):
        """Callback for P13C Part (P658207LE0A) inspection results"""
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label"]
        for widget_index in [35]:
            for label_index, label_name in enumerate(label_names_part):
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"
                    label.setText(str(value))
                    
                    # Set color based on result
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index]
                        label.setStyleSheet("color: green;" if result else "color: red;")
        

def main():
    crash_log_dir = "aikensa/logs"
    os.makedirs(crash_log_dir, exist_ok=True)
    crash_log_path = os.path.join(crash_log_dir, "crash.log")

    crash_log_file = open(crash_log_path, "a", encoding="utf-8")
    crash_log_file.write("\n===== APP START {} =====\n".format(datetime.datetime.now().isoformat()))
    crash_log_file.flush()

    faulthandler.enable(file=crash_log_file, all_threads=True)

    def _log_exception(prefix, exc_type, exc_value, exc_tb):
        try:
            crash_log_file.write("\n[{}] Unhandled exception\n".format(prefix))
            traceback.print_exception(exc_type, exc_value, exc_tb, file=crash_log_file)
            crash_log_file.flush()
        except Exception:
            pass

    def _sys_excepthook(exc_type, exc_value, exc_tb):
        _log_exception("sys", exc_type, exc_value, exc_tb)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _thread_excepthook(args):
        _log_exception("thread", args.exc_type, args.exc_value, args.exc_traceback)

    def _qt_message_handler(msg_type, context, message):
        try:
            crash_log_file.write(f"[QtMessage] {msg_type}: {message}\n")
            crash_log_file.flush()
        except Exception:
            pass

    sys.excepthook = _sys_excepthook
    threading.excepthook = _thread_excepthook
    QtCore.qInstallMessageHandler(_qt_message_handler)

    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()