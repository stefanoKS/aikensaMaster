import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from aikensa.core.scripts.img_processing.img_processing import (
    check_id,
    check_tolerance,
    draw_pitch_line,
)

# Load specifications from YAML
specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"
with open(specs_yaml_path, "r") as f:
    all_specs = yaml.safe_load(f)
    part_spec = all_specs["parts"].get("P828387YA6A_KATABU_NASHI")
    global_spec = all_specs["global"]

kanjiFontPath = global_spec["font_path"]

pitchSpecRH = part_spec.get("pitchSpecRH", [14, 132, 133, 133, 58.5, 61.5, 100.5, 85.5, 15, 733])
pitchSpecLH = part_spec.get("pitchSpecLH", [15, 85.5, 100.5, 61.5, 58.5, 133, 133, 132, 14, 733])

idSpecRH = part_spec.get("idSpecRH", [0, 0, 0, 0, 0, 0, 0, 0])
idSpecLH = part_spec.get("idSpecLH", [1, 1, 1, 1, 1, 1, 1, 1])

tolerance_pitchRH = part_spec.get("tolerance_pitchRH", [1.7] * 10)
tolerance_pitchLH = part_spec.get("tolerance_pitchLH", [1.7] * 10)

color = tuple(part_spec.get("color_ok", [0, 255, 0]))
pixelMultiplier = float(part_spec.get("pixelMultiplier", 0.16413))
bbox_offset = int(part_spec.get("bbox_offset", 10))


def partcheck(image, sahi_predictionList, leftKeypoint, rightKeypoint, side, keypoint_crop_px=None):
    if keypoint_crop_px is None:
        keypoint_crop_px = int(part_spec.get("keypoint_crop_px", 1260))

    if side == "RH":
        pitchSpec = pitchSpecRH
        tolerance_pitch = tolerance_pitchRH
        idSpec = idSpecRH
        expected_clip_id = int(idSpec[0]) if idSpec else 1
    else:
        pitchSpec = pitchSpecLH
        tolerance_pitch = tolerance_pitchLH
        idSpec = idSpecLH
        expected_clip_id = int(idSpec[0]) if idSpec else 0

    sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

    detectedid = []
    detectedposX = []
    detectedposY = []
    detectedWidth = []

    measuredPitch = []
    resultPitch = []
    resultid = []

    status = "OK"

    prev_center = None

    left_edge = _extract_endpoint_from_keypoints(leftKeypoint, x_start=0, image_width=image.shape[1])
    right_edge = _extract_endpoint_from_keypoints(
        rightKeypoint,
        x_start=-int(keypoint_crop_px),
        image_width=image.shape[1],
    )

    all_detected_ids = [int(d.category.id) for d in sorted_detections]
    print(
        f"[KATABU_NASHI][{side}] all_clip_ids={all_detected_ids}, "
        f"expected_clip_id={expected_clip_id}, keypoint_crop_px={keypoint_crop_px}, "
        f"left_keypoint={'OK' if left_edge is not None else 'MISS'}, "
        f"right_keypoint={'OK' if right_edge is not None else 'MISS'}"
    )

    # Show '製品は見つかりません' only when keypoint endpoint is not detected.
    if left_edge is None or right_edge is None:
        return _return_ng_keypoint_missing(image, pitchSpec, idSpec)

    # Clip class is side-specific for no-katabu flow.
    for detection in sorted_detections:
        cls_id = int(detection.category.id)
        if cls_id != expected_clip_id:
            continue
        detectedid.append(cls_id)

        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny

        detectedposX.append(x)
        detectedposY.append(y)
        detectedWidth.append(w)

        center = draw_bounding_box(
            image,
            x,
            y,
            w,
            h,
            [image.shape[1], image.shape[0]],
            color=color,
        )

        if prev_center is not None:
            length = calclength(prev_center, center) * pixelMultiplier
            measuredPitch.append(length)
        prev_center = center

    print(f"[KATABU_NASHI][{side}] matched_clip_count={len(detectedid)}, matched_clip_ids={detectedid}")

    if len(detectedposX) == 0:
        return _return_ng_clip_missing(image, pitchSpec, idSpec)

    leftmostCenter = (detectedposX[0], detectedposY[0])
    rightmostCenter = (detectedposX[-1], detectedposY[-1])

    # Match Roof/RadCore behavior: both outer pitches come from keypoint-to-clip distances.

    leftmostPitch = calclength(leftmostCenter, left_edge) * pixelMultiplier
    rightmostPitch = calclength(rightmostCenter, right_edge) * pixelMultiplier

    measuredPitch.insert(0, leftmostPitch)
    measuredPitch.append(rightmostPitch)
    detectedposX.insert(0, left_edge[0])
    detectedposY.insert(0, left_edge[1])
    detectedposX.append(right_edge[0])
    detectedposY.append(right_edge[1])

    totalLength = sum(measuredPitch)
    measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
    else:
        resultPitch = [0] * len(pitchSpec)
        status = "NG"

    # idSpec length may differ from detections due model revisions.
    if len(idSpec) == len(detectedid):
        resultid = check_id(detectedid, idSpec)
    else:
        resultid = [1] * len(idSpec)

    if any(result != 1 for result in resultPitch):
        status = "NG"

    if any(result != 1 for result in resultid):
        status = "NG"

    print(
        f"[KATABU_NASHI][{side}] status={status}, measuredPitch={measuredPitch}, "
        f"resultPitch={resultPitch}, resultid={resultid}"
    )

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)

    return image, measuredPitch, resultPitch, resultid, status


def _return_ng_keypoint_missing(image, pitchSpec, idSpec):
    status = "NG"
    image = draw_status_text_PIL(image, status, "製品は見つかりません", size="normal")
    resultPitch = [0] * len(pitchSpec)
    measuredPitch = [0] * len(pitchSpec)
    resultid = [0] * len(idSpec)
    return image, measuredPitch, resultPitch, resultid, status


def _return_ng_clip_missing(image, pitchSpec, idSpec):
    status = "NG"
    # Do not use '製品は見つかりません' for clip miss; reserve that for keypoint miss only.
    resultPitch = [0] * len(pitchSpec)
    measuredPitch = [0] * len(pitchSpec)
    resultid = [0] * len(idSpec)
    return image, measuredPitch, resultPitch, resultid, status


def _extract_endpoint_from_keypoints(keypoint_result, x_start=0, image_width=0):
    result = _normalize_result_object(keypoint_result)
    if result is None or not hasattr(result, "keypoints"):
        return None

    try:
        keypoints = result.keypoints
        if keypoints is None:
            return None
        if not hasattr(keypoints, "xy"):
            return None
        xy = keypoints.xy
        if xy is None or len(xy) == 0:
            return None

        kpt = xy[0][0]
        x_crop = float(kpt[0])
        y_crop = float(kpt[1])

        if x_start < 0:
            x_original = image_width + x_start + x_crop
        else:
            x_original = x_start + x_crop

        return int(x_original), int(y_crop)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _normalize_result_object(keypoint_result):
    if keypoint_result is None:
        return None
    if isinstance(keypoint_result, (list, tuple)):
        if len(keypoint_result) == 0:
            return None
        return keypoint_result[0]
    return keypoint_result


def draw_status_text_PIL(image, status, print_status, size="normal"):
    if size == "large":
        font_scale = 130.0
    elif size == "small":
        font_scale = 50.0
    else:
        font_scale = 100.0

    color_text = (10, 210, 60) if status == "OK" else (200, 30, 50)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(kanjiFontPath, int(font_scale))

    draw.text((120, 5), status, font=font, fill=color_text)
    draw.text((120, 100), print_status, font=font, fill=color_text)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_center(bbox):
    center_x = bbox.minx + (bbox.maxx - bbox.minx) / 2
    center_y = bbox.miny + (bbox.maxy - bbox.miny) / 2
    return center_x, center_y


def calclength(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_bounding_box(image, x, y, w, h, img_size, color=(0, 255, 0), thickness=2, bbox_offset=bbox_offset):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x1, y1 = int(x - w // 2) - bbox_offset, int(y - h // 2) - bbox_offset
    x2, y2 = int(x + w // 2) + bbox_offset, int(y + h // 2) + bbox_offset
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return x, y
