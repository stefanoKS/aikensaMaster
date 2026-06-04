import numpy as np
import cv2
import math
import yaml
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

from aikensa.core.scripts.img_processing.img_processing import (
    draw_bounding_box, get_center, calclength,
    check_tolerance, check_id, draw_pitch_line,
    draw_status_text_PIL, check_hanire, draw_redCircle,
    map_keypoint_xcrop_to_original
)

part_id = "P828447UA0A"
specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"

with open(specs_yaml_path, "r", encoding="utf-8") as f:
    specs_root = yaml.safe_load(f) or {}

part_spec = specs_root.get("parts", {}).get(part_id, {})

pitchSpec = part_spec.get("pitchSpec", [10, 121.5, 121.5, 121.5, 121.5, 121.5, 10, 627.5])
idSpec = part_spec.get("idSpec", [0, 0, 0, 0, 0, 0])
tolerance_pitch = part_spec.get("tolerance_pitch", [3.0, 1.7, 1.7, 1.7, 1.7, 1.7, 3.0, 10.0])

color = tuple(part_spec.get("color_ok", [0, 255, 0]))
linecolor = tuple(part_spec.get("linecolor", [20, 120, 120]))
text_offset = part_spec.get("text_offset", 40)
endoffset_y = 0
bbox_offset = part_spec.get("bbox_offset", 1)

pixelMultiplier = float(part_spec.get("pixelMultiplier", 1.0))
detected_cropped_size = int(part_spec.get("detected_cropped_size", 512))
border_width = int(part_spec.get("border_width", 10))

segmentation_pixel_start = int(part_spec.get("segmentation_pixel_start", 0))
segmentation_pixel_finish = int(part_spec.get("segmentation_pixel_finish", 512))
segmentation_width = segmentation_pixel_finish - segmentation_pixel_start


def partcheck(image, sahi_predictionList, keypointLeft, keypointRight, YoloHanireModel):

    sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

    detectedid = []
    measuredPitch = []
    resultPitch = []
    deltaPitch = []
    resultid = []
    detectedposX = []
    detectedposY = []
    detectedWidth = []

    prev_center = None

    flag_pitch_furyou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0
    flag_hole_notfound = 0
    leftmostPitch = 0
    rightmostPitch = 0
    leftmostPointX = 0
    leftmostPointY = 0
    rightmostPointX = 0
    rightmostPointY = 0
    flag_hanire = 0 #whether the clip is half inserted

    status = "OK"
    print_status = ""

    combined_lmask = None
    ngreason = ""

    raw_image = image.copy()

    for keypoint in keypointLeft:
        if keypoint.keypoints.xy is None or keypoint.keypoints.xy.shape[0] == 0 or keypoint.keypoints.xy.shape[1] == 0:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            ngreason = "PART IS NOT FOUND"

            return image, measuredPitch, resultPitch, resultid, status, ngreason

        xy = keypoint.keypoints.xy
        x_pos, y_pos = xy[0, 0].tolist()
        # print ("Keypoint left xy: ", xy)
        leftmostPointX, leftmostPointY = map_keypoint_xcrop_to_original(x_start=segmentation_pixel_start, kpt_xy_crop=(x_pos, y_pos), img_width=image.shape[1])
        # print ("Mapped Keypoint left xy to original: ", (leftmostPointX, leftmostPointY))

    for keypoint in keypointRight:
        if keypoint.keypoints.xy is None or keypoint.keypoints.xy.shape[0] == 0 or keypoint.keypoints.xy.shape[1] == 0:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            ngreason = "PART IS NOT FOUND"

            return image, measuredPitch, resultPitch, resultid, status, ngreason
        
        xy = keypoint.keypoints.xy
        # print ("Keypoint right xy: ", xy)
        x_pos, y_pos = xy[0, 0].tolist()
        rightmostPointX, rightmostPointY = map_keypoint_xcrop_to_original(x_start=-segmentation_pixel_finish, kpt_xy_crop=(x_pos, y_pos), img_width=image.shape[1])
        # print ("Mapped Keypoint right xy to original: ", (rightmostPointX, rightmostPointY))
        
    for i, detection in enumerate(sorted_detections):

        detectedid.append(detection.category.id)
        # print("Detected ID: ", detection.category.id)
        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny

        detectedposX.append(x)
        detectedposY.append(y)
        detectedWidth.append(w)


        center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)
      
        hanireResult = check_hanire(raw_image, x, y, YoloHanireModel, detected_cropped_size)


        
        #FORCEFULLY CHANGE THE HANIRE TO OFF
        hanireResult = 1

        
        print (f"Hanire result: {hanireResult}")
        if hanireResult == 0:
            draw_redCircle(image, x, y, w, h, [image.shape[1], image.shape[0]], thickness=6, bbox_offset=20)
            flag_hanire = 1
                
        # print (center)

        if prev_center is not None:
            length = calclength(prev_center, center)*pixelMultiplier
            measuredPitch.append(length)
        prev_center = center

    # print("Detected IDs: ", detectedid)


    if len(detectedposX) > 0:
        leftmostCenter = (detectedposX[0], detectedposY[0])
        rightmostCenter = (detectedposX[-1], detectedposY[-1])

        # leftmostWidth = detectedWidth[0]
        # rightmostWidth = detectedWidth[-1]
      
        # left_edge = find_edge_point_mask(image, combined_mask, leftmostCenter, direction="left", Yoffsetval = -50, Xoffsetval = 0)
        # right_edge = find_edge_point_mask(image, combined_mask, rightmostCenter, direction="right", Yoffsetval = -50, Xoffsetval = 0)

        # leftmostPitch = calclength(leftmostCenter, left_edge)*pixelMultiplier
        # rightmostPitch = calclength(rightmostCenter, right_edge)*pixelMultiplier

        left_edge =  leftmostPointX, detectedposY[0]
        right_edge = rightmostPointX, detectedposY[-1]

        leftmostPitch = calclength(leftmostCenter, left_edge)*pixelMultiplier
        rightmostPitch = calclength(rightmostCenter, right_edge)*pixelMultiplier

        #append the leftmost and rightmost pitch to the measuredPitch
        measuredPitch.insert(0, leftmostPitch)
        measuredPitch.append(rightmostPitch)
        #Reappend the leftmostcetner and rightmostcenter to the detectedposX and detectedposY
        detectedposX.insert(0, left_edge[0])
        detectedposY.insert(0, left_edge[1])
        detectedposX.append(right_edge[0])
        detectedposY.append(right_edge[1])


    #add total length
    #round the value to 1 decimal
    totalLength = sum(measuredPitch)
    measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    if flag_hanire == 1:
        status = "NG"
        ngreason = "CLIP HALF INSERTED"
        print_status = "クリップ半入れ不良"
        image = draw_status_text_PIL(image, status, print_status, size="normal")

        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        measuredPitch = [0] * (len(pitchSpec))

        return image, measuredPitch, resultPitch, resultid, status, ngreason

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
        resultid = check_id(detectedid, idSpec)

    if len(measuredPitch) != len(pitchSpec):
        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        measuredPitch = [0] * (len(pitchSpec))
        ngreason = "NUMBER OF CLIP MISMATCH"
        status = "NG"
        print_status = "クリップ数不足"

        image = draw_status_text_PIL(image, status, print_status, size="normal")

        return image, measuredPitch, resultPitch, resultid, status, ngreason


    if any(result != 1 for result in resultPitch):
        flag_pitch_furyou = 1
        status = "NG"
        ngreason = "CLIP PITCH NG"
        print_status = "クリップピッチ不良"

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)

    image = draw_status_text_PIL(image, status, print_status, size="normal")
    
    return image, measuredPitch, resultPitch, resultid, status, ngreason
