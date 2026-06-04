import stat
import numpy as np
import cv2
import math
import yaml
import os
import pygame
import os
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

# Import all utility functions from img_processing
from aikensa.core.scripts.img_processing.img_processing import (
    get_center, draw_bounding_box, create_masks, play_sound,
    check_id, draw_pitch_line, draw_status_text, draw_status_text_PIL,
    check_tolerance, yolo_to_pixel, find_edge_point_mask, find_edge_point,
    drawcircle, drawbox, drawtext, calclength
)

# Load specifications from YAML
specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"
with open(specs_yaml_path, 'r') as f:
    all_specs = yaml.safe_load(f)
    part_spec = all_specs['parts']['P808387YA0A']
    global_spec = all_specs['global']

kanjiFontPath = global_spec['font_path']

# Load all specifications from YAML
pitchSpec = part_spec.get('pitchSpec', [15, 142, 143, 143, 143, 139, 139, 113, 15, 992])
idSpec = part_spec.get('idSpec', [1, 1, 1, 1, 1, 1, 1, 1])
tolerance_pitch = part_spec.get('tolerance_pitch', [1.7] * 10)

color = tuple(part_spec.get('color_ok', [0, 255, 0]))
text_offset = part_spec.get('text_offset', 40)
endoffset_y = 0
bbox_offset = part_spec.get('bbox_offset', 10)

segmentation_width = part_spec.get('segmentation_width', 1280)
pixelMultiplier = part_spec.get('pixelMultiplier', 0.1654)


def partcheck(image, sahi_predictionList, leftSegmentation, rightSegmentation):

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

    status = "OK"
    print_status = ""

    # cannydetection_image = image.copy() #Make sure to copy the image to avoid modifying the original image

    combined_lmask = None
    for lm in leftSegmentation:
        if lm.masks is not None:
            orig_shape = (image.shape[0], segmentation_width)
            segmentation_xyn = lm.masks.xyn
            lmask = create_masks(segmentation_xyn, orig_shape)
            if combined_lmask is None:
                combined_lmask = np.zeros_like(lmask)
            combined_lmask = cv2.bitwise_or(combined_lmask, lmask)
            # cv2.imwrite("leftmask.jpg", combined_lmask)

        #Checkgate for mask segmentation handling
        if lm.masks is None:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            resultid = [0] * len(idSpec)

            return image, measuredPitch, resultPitch, resultid, status


            
    combined_rmask = None
    for rm in rightSegmentation:
        if rm.masks is not None:
            orig_shape = (image.shape[0], segmentation_width)
            segmentation_xyn = rm.masks.xyn
            rmask = create_masks(segmentation_xyn, orig_shape)
            if combined_rmask is None:
                combined_rmask = np.zeros_like(rmask)
            combined_rmask = cv2.bitwise_or(combined_rmask, rmask)
            # cv2.imwrite("rightmask.jpg", combined_rmask)

        #Checkgate for mask segmentation handling
        if rm.masks is None:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            resultid = [0] * len(idSpec)

            return image, measuredPitch, resultPitch, resultid, status

        

    combined_mask = np.zeros_like(image[:, :, 0])  # Single-channel black mask

    if combined_lmask is not None and combined_rmask is not None:
        combined_mask[:, :segmentation_width] = combined_lmask 
        combined_mask[:, -segmentation_width:] = combined_rmask 
        # cv2.imwrite("combined_mask.jpg", combined_mask)

    for i, detection in enumerate(sorted_detections):
        detectedid.append(detection.category.id)
        if detection.category.id == 1:
            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny
            # class_name = detection.category.name

            detectedposX.append(x)
            detectedposY.append(y)
            detectedWidth.append(w)

            #id 0 object is brown clip
            #id 1 object is white clip
            center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)

            if prev_center is not None:
                length = calclength(prev_center, center)*pixelMultiplier
                measuredPitch.append(length)
            prev_center = center
    #Check if detectedposX is not empty
    if len(detectedposX) > 0:
        leftmostCenter = (detectedposX[0], detectedposY[0])
        leftmostWidth = detectedWidth[0]
        rightmostCenter = (detectedposX[-1], detectedposY[-1])
        rightmostWidth = detectedWidth[-1]
        adjustment_offset = 5 # to make sure it goes above the clip itself
        # left_edge = find_edge_point(cannydetection_image, leftmostCenter, direction="left", Yoffsetval = 0, Xoffsetval = leftmostWidth + adjustment_offset)
        # right_edge = find_edge_point(cannydetection_image, rightmostCenter, direction="right", Yoffsetval = 0, Xoffsetval = rightmostWidth + adjustment_offset)

        # Positive Yoffsetval means going down, negative means going up
        left_edge = find_edge_point_mask(image, combined_mask, leftmostCenter, direction="left", Yoffsetval = -130, Xoffsetval = 0)
        right_edge = find_edge_point_mask(image, combined_mask, rightmostCenter, direction="right", Yoffsetval = -130, Xoffsetval = 0)

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
    if len(measuredPitch) > 1:
        measuredPitch[1] = measuredPitch[1] + 1.0 #add 1mm to the first pitch
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
        resultid = check_id(detectedid, idSpec)

    if len(measuredPitch) != len(pitchSpec):
        resultPitch = [0] * len(pitchSpec)

    if any(result != 1 for result in resultPitch):
        flag_pitch_furyou = 1
        status = "NG"

    # if any(result != 1 for result in resultid):
    #     flag_clip_furyou = 1
    #     status = "NG"

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)
    
    return image, measuredPitch, resultPitch, resultid, status

# class Category:
#     def __init__(self, id, name):
#         self.id = id
#         self.name = name

# class ObjectPrediction:
#     def __init__(self, bbox, score, category):
#         self.bbox = bbox
#         self.score = score
#         self.category = category