import stat
from matplotlib.pylab import f
import numpy as np
import cv2
import math
import yaml
import os
import pygame
import os
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

# Import all utility functions from img_processing
from aikensa.core.scripts.img_processing.img_processing import (
    get_center, draw_bounding_box, create_masks, play_sound,
    check_id, draw_pitch_line, draw_status_text, draw_status_text_PIL,
    check_tolerance, yolo_to_pixel, find_edge_point_mask, find_edge_point,
    drawcircle, drawbox, drawtext, calclength
)

pygame.mixer.init()
ok_sound = pygame.mixer.Sound("aikensa/core/config/sound/positive_interface.wav") 
ok_sound_v2 = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-software-interface-remove-2576.wav")
ng_sound = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-classic-short-alarm-993.wav")  
ng_sound_v2 = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-system-beep-buzzer-fail-2964.wav")
kanjiFontPath = "aikensa/gui/resources/font/NotoSansJP-ExtraBold.ttf"

pitchSpec_050P = [85, 87, 98, 98, 78, 113, 103, 14]
pitchSpec_040P = [103, 113, 78, 98, 98, 87, 85, 14]
pitchSpec_090P = [85, 87, 98, 98, 78, 61, 52, 38, 37, 28, 14]
pitchSpec_080P = [28, 37, 38, 52, 61, 78, 98, 98, 87, 85, 14]

pitchSpec_050PKENGEN = [85, 87, 98, 98, 78, 113, 103, 14]
pitchSpec_040PKENGEN = [103, 113, 78, 98, 98, 87, 85, 14]
pitchSpec_090PKENGEN = [85, 87, 98, 98, 78, 61, 52, 38, 37, 28, 14]
pitchSpec_080PKENGEN = [28, 37, 38, 52, 61, 78, 98, 98, 87, 85, 14]

pitchSpec_050PCLIPSOUNYUUKI = [87, 98, 98, 78, 113, 103]
pitchSpec_040PCLIPSOUNYUUKI = [103, 113, 78, 98, 98, 87]
pitchSpec_090PCLIPSOUNYUUKI = [87, 98, 98, 78, 61, 52, 38, 37, 28]
pitchSpec_080PCLIPSOUNYUUKI = [28, 37, 38, 52, 61, 78, 98, 98, 87]

pitchTolerance_050P = [2.0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
pitchTolerance_040P = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 2.0, 1.8]
pitchTolerance_090P = [2.0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
pitchTolerance_080P = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 2.0, 1.8]

pitchTolerance_050PCLIPSOUNYUUKI = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
pitchTolerance_040PCLIPSOUNYUUKI = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
pitchTolerance_090PCLIPSOUNYUUKI = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
pitchTolerance_080PCLIPSOUNYUUKI = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]

clipSpec_050P = [2, 1, 0, 0, 0, 0, 3, 3, 0, 1] #white is 0, brown is 1, yellow is 2, orange is 3
clipSpec_040P = [0, 1, 3, 3, 1, 1, 1, 1, 0, 2]
clipSpec_090P = [2, 1, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 1]
clipSpec_080P = [0, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 0, 2]

clipSpec_050PCLIPSOUNYUUKI = [0, 0, 0, 0, 4, 4, 0] #white is 0, brown is 1, yellow is 2, orange is 3, 4 is hole
clipSpec_040PCLIPSOUNYUUKI = [1, 4, 4, 1, 1, 1, 1]
clipSpec_090PCLIPSOUNYUUKI = [0, 0, 0, 0, 4, 4, 4, 0, 0, 0]
clipSpec_080PCLIPSOUNYUUKI = [1, 1, 1, 4, 4, 4, 1, 1, 1, 1]

pitchSpec_Katabu = [14]
pitchTolerance_Katabu = [1.8]

color = (0, 255, 0)
text_offset = 40
endoffset_y = 0
bbox_offset = 10

# segmentation_width = 1640

pixelMultiplier = 0.1598
pixelMultiplier_katabumarking = 0.1598

this_dir = os.path.dirname(__file__)

#if model path is exists, load the model
detectFlip_model_path = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "checkpoints", "MMC", "5A45", "P828XXW0X0P_detect_flip.pt"))
if os.path.exists(detectFlip_model_path):
    P828XXW0X0P_CLIPFLIP_DETECT = YOLO(detectFlip_model_path)
else:
    print(f"Model path {detectFlip_model_path} does not exist. Please check the model path.")
    P828XXW0X0P_CLIPFLIP_DETECT = None


def partcheck(image, img_katabumarking, sahi_predictionList, katabumarking_detection, partname):
        
    print(f"Partname: {partname}")


    sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

    katabumarking_lengths = []

    detectedid = []
    idSpec = []

    measuredPitch = []
    resultPitch = []
    deltaPitch = []

    resultid = []

    detectedposX = []
    detectedposY = []

    detectedposX_katabumarking = []
    detectedposY_katabumarking = []

    detectedWidth = []

    prev_center = None
    prev_center_katabumarking = None

    flag_pitch_furyou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0
    flag_hole_notfound = 0

    leftmostPitch = 0
    rightmostPitch = 0

    status = "OK"
    print_status = ""

    combined_infer_mask = None

    color = (0, 255, 0)


    if partname == "P82833W050P":
        pitchSpec = pitchSpec_050P
        tolerance_pitch = pitchTolerance_050P
        idSpec = clipSpec_050P

    elif partname == "P82832W040P":
        pitchSpec = pitchSpec_040P
        tolerance_pitch = pitchTolerance_040P
        idSpec = clipSpec_040P

    elif partname == "P82833W090P":
        pitchSpec = pitchSpec_090P
        tolerance_pitch = pitchTolerance_090P
        idSpec = clipSpec_090P

    elif partname == "P82832W080P":
        pitchSpec = pitchSpec_080P
        tolerance_pitch = pitchTolerance_080P
        idSpec = clipSpec_080P

    elif partname == "P82833W050PKENGEN":
        pitchSpec = pitchSpec_050PKENGEN
        tolerance_pitch = pitchTolerance_050P
        idSpec = clipSpec_050P

    elif partname == "P82832W040PKENGEN":
        pitchSpec = pitchSpec_040PKENGEN
        tolerance_pitch = pitchTolerance_040P
        idSpec = clipSpec_040P

    elif partname == "P82833W090PKENGEN":
        pitchSpec = pitchSpec_090PKENGEN
        tolerance_pitch = pitchTolerance_090P
        idSpec = clipSpec_090P
    
    elif partname == "P82832W080PKENGEN":
        pitchSpec = pitchSpec_080PKENGEN
        tolerance_pitch = pitchTolerance_080P
        idSpec = clipSpec_080P

    elif partname == "P82833W050PCLIPSOUNYUUKI":
        pitchSpec = pitchSpec_050PCLIPSOUNYUUKI
        tolerance_pitch = pitchTolerance_050PCLIPSOUNYUUKI
        idSpec = clipSpec_050PCLIPSOUNYUUKI

    elif partname == "P82832W040PCLIPSOUNYUUKI":
        pitchSpec = pitchSpec_040PCLIPSOUNYUUKI
        tolerance_pitch = pitchTolerance_040PCLIPSOUNYUUKI
        idSpec = clipSpec_040PCLIPSOUNYUUKI

    elif partname == "P82833W090PCLIPSOUNYUUKI":
        pitchSpec = pitchSpec_090PCLIPSOUNYUUKI
        tolerance_pitch = pitchTolerance_090PCLIPSOUNYUUKI
        idSpec = clipSpec_090PCLIPSOUNYUUKI

    elif partname == "P82832W080PCLIPSOUNYUUKI":
        pitchSpec = pitchSpec_080PCLIPSOUNYUUKI
        tolerance_pitch = pitchTolerance_080PCLIPSOUNYUUKI
        idSpec = clipSpec_080PCLIPSOUNYUUKI


    #KATABU MARKING DETECTION
    #only do the katabu marking detection if the part is not ___clipsounyuuki
    if partname not in ["P82833W050PCLIPSOUNYUUKI", "P82832W040PCLIPSOUNYUUKI", "P82833W090PCLIPSOUNYUUKI", "P82832W080PCLIPSOUNYUUKI"]:
        #class 0 is for clip, class 1 is for katabu marking
        for r in katabumarking_detection:
            for box in r.boxes:
                x_marking, y_marking = float(box.xywh[0][0].cpu()), float(box.xywh[0][1].cpu())
                w_marking, h_marking = float(box.xywh[0][2].cpu()), float(box.xywh[0][3].cpu())
                class_id_marking = int(box.cls.cpu())

                if class_id_marking == 0:
                    color = (0, 255, 0)
                elif class_id_marking == 1:
                    color = (100, 100, 200)

                center_katabummarking = draw_bounding_box(img_katabumarking, 
                                        x_marking, y_marking, 
                                        w_marking, h_marking, 
                                        [img_katabumarking.shape[1], img_katabumarking.shape[0]], color=color,
                                        bbox_offset=3, thickness=2)

                if class_id_marking == 1:
                    if partname in ["P82833W050P", "P82833W090P", "P82833W050PKENGEN", "P82833W090PKENGEN"]:
                        center_katabummarking = (int(x_marking - w_marking/2), int(y_marking))
                    elif partname in ["P82832W040P", "P82832W080P", "P82832W040PKENGEN", "P82832W080PKENGEN"]:
                        center_katabummarking = (int(x_marking + w_marking/2), int(y_marking))
                
                if prev_center_katabumarking is not None:
                    length = calclength(prev_center_katabumarking, center_katabummarking)*pixelMultiplier_katabumarking
                    katabumarking_lengths.append(length)
                    line_center = ((prev_center_katabumarking[0] + center_katabummarking[0]) // 2, (prev_center_katabumarking[1] + center_katabummarking[1]) // 2)
                    img_katabumarking = drawbox(img_katabumarking, line_center, length, font_scale=0.8, offset=40, font_thickness=2)
                    img_katabumarking = drawtext(img_katabumarking, line_center, length, font_scale=0.8, offset=40, font_thickness=2)

                prev_center_katabumarking = center_katabummarking

                detectedposX_katabumarking.append(center_katabummarking[0])
                detectedposY_katabumarking.append(center_katabummarking[1])
    
            katabupitchresult = check_tolerance(katabumarking_lengths, pitchSpec_Katabu, pitchTolerance_Katabu)

            xy_pairs_katabumarking = list(zip(detectedposX_katabumarking, detectedposY_katabumarking))
            draw_pitch_line(img_katabumarking, xy_pairs_katabumarking, katabupitchresult, thickness=2)

            #pick only the first element if array consists of more than 1 element -> detection POKAYOKE (if detection is not that great)
            if len(katabumarking_lengths) > 1:
                katabumarking_lengths = katabumarking_lengths[:1]
            #since there is only one katabu marking, we can just use the first element -> detection POKAYOKE (if detection is not that great)
            print(f"Katabu Marking Length: {katabumarking_lengths}")

            #if katabumarking_lengths is empty, then it is NG
            if katabumarking_lengths == []:
                status = "NG"
                print_status = print_status + "型部マーキング認識不良"
                print(f"Status:{print_status}")
                measuredPitch = [0] * len(pitchSpec)
                resultPitch = [0] * len(pitchSpec)
                resultid = [0] * len(idSpec)
                image = draw_status_text_PIL(image, status, print_status, size = "normal")
                ngreason = "KATABU MARKING NOT FOUND"

                return image, img_katabumarking, measuredPitch, resultPitch, resultid, status, ngreason
    
    for i, detection in enumerate(sorted_detections):

        if partname in ["P82833W050PKENGEN", "P82832W040PKENGEN"] and detection.category.id == 4:
            print ("Hole detected")
        else:
            detectedid.append(detection.category.id)
            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny

            detectedposX.append(x)
            detectedposY.append(y)
            detectedWidth.append(w)

            image_copy = image.copy()

            center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)

            if prev_center is not None:
                length = calclength(prev_center, center)*pixelMultiplier
                measuredPitch.append(length)
                line_center = ((prev_center[0] + center[0]) // 2, (prev_center[1] + center[1]) // 2)
                image = drawbox(image, line_center, length, font_scale=2.0, offset=40, font_thickness=2)
                image = drawtext(image, line_center, length, font_scale=2.0, offset=40, font_thickness=2)
            prev_center = center




        #only do the not clipsounyuuki once
        if partname not in ["P82833W050PCLIPSOUNYUUKI", "P82832W040PCLIPSOUNYUUKI", "P82833W090PCLIPSOUNYUUKI", "P82832W080PCLIPSOUNYUUKI"]:
            if detection.category.id == 2:
                #yellow clip is found, do inference check to see if the clip is flipped
                #crop image to a fixed size of 128x128 from the center of yellow clip
                crop_size = 128
                x1 = int(x - crop_size / 2)
                y1 = int(y - crop_size / 2)
                x2 = int(x + crop_size / 2)
                y2 = int(y + crop_size / 2)
                #crop the image
                crop_img = image_copy[y1:y2, x1:x2]
                # cv2.imwrite("crop_img.png", crop_img)
                clipflip_detection = P828XXW0X0P_CLIPFLIP_DETECT(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), stream=True, verbose=False)
                print(clipflip_detection)
                clipflip_detection = list(clipflip_detection)[0].probs.data.argmax().item()
                #0 is flipped, 1 is not flipped
                if clipflip_detection == 0:
                            status = "NG"
                            print_status = print_status + "型部クリップ向き不良"
                            print(f"Status:{print_status}")
                            measuredPitch = [0] * len(pitchSpec)
                            resultPitch = [0] * len(pitchSpec)
                            resultid = [0] * len(idSpec)
                            image = draw_status_text_PIL(image, status, print_status, size = "normal")
                            ngreason = "KATABU CLIP FLIPPED"

                            return image, img_katabumarking, measuredPitch, resultPitch, resultid, status, ngreason


    #First check, check if any clip is detected or not
    if len (detectedid) == 0:
        status = "NG"
        print_status = print_status + "製品認識不良"
        print(f"Status:{print_status}")
        measuredPitch = [0] * len(pitchSpec)
        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        image = draw_status_text_PIL(image, status, print_status, size = "normal")
        ngreason = "PART NOT FOUND"

        return image, img_katabumarking, measuredPitch, resultPitch, resultid, status, ngreason


    #POP The first and last element for the KENGEN and normal if the detectin lost is not empty
    if partname in ["P82833W050P", "P82832W040P", "P82833W090P", "P82832W080P", "P82833W050PKENGEN", "P82832W040PKENGEN", "P82833W090PKENGEN", "P82832W080PKENGEN"]:
        #Pop the first and last element
        detectedposX.pop(0)
        detectedposX.pop(-1)
        detectedposY.pop(0)
        detectedposY.pop(-1)
        detectedWidth.pop(0)
        detectedWidth.pop(-1)
        measuredPitch.pop(0)
        measuredPitch.pop(-1)

        print("Element Popped")

    print(f"Detected ID: {detectedid}")
    print(f"idSpec: {idSpec}")


    if detectedid != idSpec:
        status = "NG"
        print_status = print_status + "NG クリップ入れ間違い"
        # print(f"Status:{print_status}")
        measuredPitch = [0] * len(pitchSpec)
        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        image = draw_status_text_PIL(image, status, print_status, size = "normal")
        # cv2.imwrite("test.png", image)
        ngreason = "CLIP COLOR MISMATCH"
        return image, img_katabumarking, measuredPitch, resultPitch, resultid, status, ngreason
    
    if katabumarking_lengths is not None:
        if katabumarking_lengths and katabumarking_lengths[0] != 0:
            measuredPitch.append(round(katabumarking_lengths[0], 1))

    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    #print measured pitch, print ID
    # print(f"Spec:,{pitchSpec}")

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
        resultid = check_id(detectedid, idSpec)
        # print(f"Result Pitch: {resultPitch}")
        # print(f"Result ID: {resultid}")

    if len(measuredPitch) != len(pitchSpec):
        resultPitch = [0] * len(pitchSpec)
        status = "NG"
        ngreason = "NUMBER OF CLIP MISMATCH"

    
    if any(result != 1 for result in resultPitch):
        print_status = print_status + " ピッチ不良"
        status = "NG"
        # print(f"Status:{print_status}")
        image  = draw_status_text_PIL(image, status, print_status, size = "normal")
        ngreason = "CLIP PITCH NG"

    print(f"Measured Pitch: {measuredPitch}")
    print(f"Detected ID: {detectedid}")
    print(f"Result Pitch: {resultPitch}")

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)
