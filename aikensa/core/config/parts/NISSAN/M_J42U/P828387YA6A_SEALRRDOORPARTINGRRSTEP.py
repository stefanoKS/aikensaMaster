import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from aikensa.core.scripts.img_processing.img_processing import (
    check_id,
    check_tolerance,
    draw_bounding_box,
    draw_pitch_line,
    get_center,
)

specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"
with open(specs_yaml_path, "r") as f:
    all_specs = yaml.safe_load(f)
    part_spec = all_specs["parts"]["P828387YA6A"]
    global_spec = all_specs["global"]

kanjiFontPath = global_spec["font_path"]

pitchSpecRH = part_spec.get("pitchSpecRH", [14, 130, 132, 132, 59, 61, 97, 83, 20, 708])
pitchSpecLH = part_spec.get("pitchSpecLH", [83, 97, 61, 59, 132, 132, 130, 14, 20, 708])
tolerance_pitchRH = part_spec.get("tolerance_pitchRH", [3.0, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 2.0, 10.0])
tolerance_pitchLH = part_spec.get("tolerance_pitchLH", [1.7, 1.7, 1.7, 3.0, 1.7, 1.7, 1.7, 1.7, 2.0, 10.0])
idSpecRH = part_spec.get("idSpecRH", [0, 0, 0, 0, 0, 0, 0, 0, 0])
idSpecLH = part_spec.get("idSpecLH", [1, 1, 1, 1, 1, 1, 1, 1, 1])

color = tuple(part_spec.get("color_ok", [0, 255, 0]))
color_ng = tuple(part_spec.get("color_ng", [200, 30, 50]))
pixelMultiplier = float(part_spec.get("pixelMultiplier", 0.16413))
pixelMultiplier_katabumarking = float(part_spec.get("pixelMultiplier_katabumarking", 0.16413))
bbox_offset = int(part_spec.get("bbox_offset", 10))


def partcheck(
    image,
    sahi_predictionList,
    left_keypoint=None,
    right_keypoint=None,
    katabu_detection=None,
    side="RH",
    keypoint_crop_px=None,
    katabu_crop_width=320,
):
    if keypoint_crop_px is None:
        keypoint_crop_px = int(part_spec.get("keypoint_crop_px", 1360))

    if side == "RH":
        pitchSpec = pitchSpecRH
        tolerance_pitch = tolerance_pitchRH
        expected_clip_id = 0
        expected_ids = idSpecRH
        endpoint = _extract_endpoint_from_keypoints(left_keypoint, x_start=0, image_width=image.shape[1])
        katabu_x_offset = image.shape[1] - int(katabu_crop_width)
    else:
        pitchSpec = pitchSpecLH
        tolerance_pitch = tolerance_pitchLH
        expected_clip_id = 1
        expected_ids = idSpecLH
        endpoint = _extract_endpoint_from_keypoints(
            right_keypoint,
            x_start=-int(keypoint_crop_px),
            image_width=image.shape[1],
        )
        katabu_x_offset = 0

    sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

    detectedid = []
    clip_points = []
    measured_inner = []
    prev_center = None

    for detection in sorted_detections:
        cls_id = int(detection.category.id)
        if cls_id != expected_clip_id:
            continue

        expected_id = expected_ids[len(detectedid)] if len(detectedid) < len(expected_ids) else None
        detectedid.append(cls_id)

        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny

        center = draw_bounding_box(
            image,
            x,
            y,
            w,
            h,
            [image.shape[1], image.shape[0]],
            color=color if expected_id == cls_id else color_ng,
            bbox_offset=bbox_offset,
        )
        clip_points.append(center)

        if prev_center is not None:
            measured_inner.append(calclength(prev_center, center) * pixelMultiplier)
        prev_center = center

    print(
        f"[YA6A][{side}] clip_count={len(clip_points)}, clip_ids={detectedid}, "
        f"expected_clip_id={expected_clip_id}, endpoint={'OK' if endpoint is not None else 'MISS'}"
    )

    if endpoint is None:
        return _return_ng(image, pitchSpec, expected_ids, "製品は見つかりません")

    if len(clip_points) == 0:
        return _return_ng(image, pitchSpec, expected_ids, "クリップ未検出")

    katabu_pitch, katabu_mark_point = _extract_katabu_pitch(
        katabu_detection,
        side=side,
        x_offset=katabu_x_offset,
    )
    if katabu_pitch is None or katabu_mark_point is None:
        return _return_ng(image, pitchSpec, expected_ids, "型部未検出")

    if side == "RH":
        endpoint_pitch = calclength(endpoint, clip_points[0]) * pixelMultiplier
        trimmed_inner = measured_inner[:-1] if len(measured_inner) > 0 else []
        measuredPitch = [endpoint_pitch] + trimmed_inner + [katabu_pitch]
        draw_points = [endpoint] + clip_points + [katabu_mark_point]
        skipped_segment_index = len(draw_points) - 2 if len(draw_points) >= 2 else None
    else:
        endpoint_pitch = calclength(clip_points[-1], endpoint) * pixelMultiplier
        trimmed_inner = measured_inner[1:] if len(measured_inner) > 0 else []
        measuredPitch = trimmed_inner + [endpoint_pitch] + [katabu_pitch]
        draw_points = [katabu_mark_point] + clip_points + [endpoint]
        skipped_segment_index = 0 if len(draw_points) >= 2 else None

    totalLength = sum(measuredPitch)
    measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
    else:
        resultPitch = [0] * len(pitchSpec)

    if len(detectedid) == len(expected_ids):
        resultid = check_id(detectedid, expected_ids)
    else:
        resultid = [0] * len(expected_ids)

    status = "OK"
    if any(result != 1 for result in resultPitch):
        status = "NG"
    if any(result != 1 for result in resultid):
        status = "NG"

    line_result_values = _build_line_result_values(
        resultPitch,
        skipped_segment_index=skipped_segment_index,
    )

    if len(draw_points) >= 2:
        draw_pitch_line(image, draw_points, line_result_values, thickness=8)

    print(
        f"[YA6A][{side}] status={status}, measuredPitch={measuredPitch}, "
        f"resultPitch={resultPitch}, resultid={resultid}, katabu_pitch={katabu_pitch}"
    )

    return image, measuredPitch, resultPitch, resultid, status, ""


def _build_line_result_values(result_pitch, skipped_segment_index=None):
    if not result_pitch:
        return []

    segment_results = list(result_pitch[:-1])
    if skipped_segment_index is None:
        return segment_results

    line_results = []
    result_idx = 0
    total_segments = len(segment_results) + 1
    for segment_idx in range(total_segments):
        if segment_idx == skipped_segment_index:
            line_results.append(None)
            continue

        if result_idx < len(segment_results):
            line_results.append(segment_results[result_idx])
            result_idx += 1

    return line_results


def _extract_katabu_pitch(katabu_detection, side, x_offset):
    clip_points = []
    mark_points = []

    for result in _normalize_results(katabu_detection):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            x_val = float(box.xywh[0][0].cpu())
            y_val = float(box.xywh[0][1].cpu())
            w_val = float(box.xywh[0][2].cpu())
            cls_id = int(box.cls.cpu())

            if cls_id == 0:
                point = (int(x_offset + x_val), int(y_val))
                clip_points.append(point)
            elif cls_id == 1:
                point = (int(x_offset + x_val), int(y_val))
                mark_points.append(point)

    if not clip_points or not mark_points:
        return None, None

    clip_points = sorted(clip_points, key=lambda point: point[0])
    mark_points = sorted(mark_points, key=lambda point: point[0])

    if side == "RH":
        clip_point = clip_points[-1]
        mark_point = mark_points[-1]
    else:
        clip_point = clip_points[0]
        mark_point = mark_points[0]

    pitch = round(calclength(clip_point, mark_point) * pixelMultiplier_katabumarking, 1)
    return pitch, mark_point


def _extract_endpoint_from_keypoints(keypoint_result, x_start=0, image_width=0):
    result = _normalize_result_object(keypoint_result)
    if result is None or not hasattr(result, "keypoints"):
        return None

    try:
        keypoints = result.keypoints
        if keypoints is None or not hasattr(keypoints, "xy"):
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


def _normalize_results(results):
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return list(results)
    return list(results)


def _normalize_result_object(keypoint_result):
    if keypoint_result is None:
        return None
    if isinstance(keypoint_result, (list, tuple)):
        if len(keypoint_result) == 0:
            return None
        return keypoint_result[0]
    return keypoint_result


def _return_ng(image, pitchSpec, idSpec, message):
    status = "NG"
    image = draw_status_text_PIL(image, status, message, size="normal")
    resultPitch = [0] * len(pitchSpec)
    measuredPitch = [0] * len(pitchSpec)
    resultid = [0] * len(idSpec)
    return image, measuredPitch, resultPitch, resultid, status, message


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


def calclength(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
