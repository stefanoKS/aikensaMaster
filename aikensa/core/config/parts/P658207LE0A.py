import math
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from aikensa.core.scripts.img_processing.img_processing import (
    check_id,
    check_tolerance,
    draw_bounding_box,
    draw_pitch_line,
    draw_status_text_PIL,
    get_center,
    map_keypoint_xcrop_to_original,
)

specs_yaml_path = Path(__file__).resolve().parents[1] / "parts_specifications.yaml"
with open(specs_yaml_path, "r") as f:
    all_specs = yaml.safe_load(f)
    part_spec = all_specs["parts"]["P658207LE0A"]

pitchSpec = part_spec.get("pitchSpec", [13, 82, 82, 82, 13, 2, 0, 0, 2, 272])
idSpec = part_spec.get("idSpec", [0, 0, 0, 0])
tolerance_pitch = part_spec.get("tolerance_pitch", [3.0, 1.5, 1.5, 1.5, 3.0, 1.0, 1.0, 1.0, 1.0, 10.0])

color = tuple(part_spec.get("color_ok", [0, 255, 0]))
bbox_offset = int(part_spec.get("bbox_offset", 1))
pixelMultiplier = float(part_spec.get("pixelMultiplier", 0.1592))
keypoint_crop_px_default = int(part_spec.get("keypoint_crop_px", 2200))
clip_height_padding_px = int(part_spec.get("clip_height_padding_px", 20))
clip_height_model_imgsz = int(part_spec.get("clip_height_model_imgsz", 256))
debug_save_images = bool(part_spec.get("debug_save_images", False))
debug_root_dir = Path(__file__).resolve().parents[3] / "inspection_results" / "P658207LE0A"


def debug_images_enabled():
    return debug_save_images


def partcheck(image, sahi_prediction_list, left_keypoint=None, right_keypoint=None, keypoint_crop_px=None, clipheight_model=None):
    if keypoint_crop_px is None:
        keypoint_crop_px = keypoint_crop_px_default

    base_image = image.copy()
    sorted_detections = sorted(sahi_prediction_list, key=lambda d: d.bbox.minx)
    print(f"[P658207LE0A] clip detections total={len(sorted_detections)}")

    detectedid = []
    clip_points = []
    clip_boxes = []
    measuredPitch = []
    resultPitch = [0] * len(pitchSpec)
    resultid = [0] * len(idSpec)
    deltaPitch = [0] * len(pitchSpec)
    status = "OK"
    ng_reason = ""

    prev_center = None
    for detection in sorted_detections:
        cls_id = int(detection.category.id)
        if cls_id != 0:
            continue

        detectedid.append(cls_id)
        bbox = detection.bbox
        clip_boxes.append(bbox)
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
            color=color,
            bbox_offset=bbox_offset,
        )
        clip_points.append(center)
        print(
            "[P658207LE0A] clip detected "
            f"idx={len(clip_boxes) - 1} cls_id={cls_id} "
            f"bbox=({bbox.minx:.1f}, {bbox.miny:.1f}, {bbox.maxx:.1f}, {bbox.maxy:.1f}) "
            f"center=({center[0]:.1f}, {center[1]:.1f})"
        )

        if prev_center is not None:
            measuredPitch.append(calclength(prev_center, center) * pixelMultiplier)
        prev_center = center

    if len(clip_points) == 0:
        print("[P658207LE0A] no CLIPBLACK detections found")
        return _return_ng(image, "クリップ未検出")

    left_edge = _extract_endpoint_from_keypoints(left_keypoint, x_start=0, image_width=image.shape[1])
    right_edge = _extract_endpoint_from_keypoints(
        right_keypoint,
        x_start=-int(keypoint_crop_px),
        image_width=image.shape[1],
    )

    if left_edge is None:
        first_bbox = clip_boxes[0]
        left_edge = (int(first_bbox.minx), int(clip_points[0][1]))
    if right_edge is None:
        last_bbox = clip_boxes[-1]
        right_edge = (int(last_bbox.maxx), int(clip_points[-1][1]))

    measuredPitch.insert(0, calclength(left_edge, clip_points[0]) * pixelMultiplier)
    measuredPitch.append(calclength(clip_points[-1], right_edge) * pixelMultiplier)

    base_pitch_measurements = [round(pitch, 1) for pitch in measuredPitch]
    clip_height_measurements = _measure_clip_height_offsets(
        base_image,
        clip_boxes,
        clipheight_model,
    )

    total_length = round(sum(base_pitch_measurements), 1)
    measuredPitch = base_pitch_measurements + clip_height_measurements + [total_length]

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = _evaluate_pitch_results(measuredPitch, pitchSpec, tolerance_pitch)
        deltaPitch = [
            round(measuredPitch[i] - pitchSpec[i], 1) if measuredPitch[i] is not None else None
            for i in range(len(pitchSpec))
        ]
        print(
            f"[P658207LE0A] measuredPitch={measuredPitch} "
            f"resultPitch={resultPitch} deltaPitch={deltaPitch}"
        )

    if len(detectedid) == len(idSpec):
        resultid = check_id(detectedid, idSpec)

    if len(detectedid) != len(idSpec):
        status = "NG"
        ng_reason = "クリップ数不良"
    elif any(result != 1 for result in resultPitch):
        status = "NG"
        ng_reason = "ピッチ不良"
    elif any(result != 1 for result in resultid):
        status = "NG"
        ng_reason = "クリップID不良"

    draw_points = [left_edge] + clip_points + [right_edge]
    draw_pitch_line(image, draw_points, resultPitch[:5], thickness=8)

    if status == "NG":
        image = draw_status_text_PIL(image, status, ng_reason, size="normal")

    return image, measuredPitch, resultPitch, resultid, status, ng_reason, deltaPitch


def _extract_endpoint_from_keypoints(keypoint_result, x_start, image_width):
    result = _normalize_result_object(keypoint_result)
    if result is None or not hasattr(result, "keypoints"):
        return None

    try:
        xy = result.keypoints.xy
        if xy is None or len(xy) == 0:
            return None

        kpt = xy[0][0]
        x_orig, y_orig = map_keypoint_xcrop_to_original(
            x_start=x_start,
            kpt_xy_crop=(float(kpt[0]), float(kpt[1])),
            img_width=image_width,
        )
        return int(round(x_orig)), int(round(y_orig))
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


def _measure_clip_height_offsets(image, clip_boxes, clipheight_model):
    measurements = []
    for clip_index, bbox in enumerate(clip_boxes):
        point = _extract_clip_height_keypoint(image, bbox, clipheight_model, clip_index)
        if point is None:
            measurements.append(None)
            print(f"[P658207LE0A clipheight] idx={clip_index} keypoint=missing")
            continue

        crop_center_y = point["crop_height"] / 2.0
        vertical_distance_mm = abs(point["y"] - crop_center_y) * pixelMultiplier
        measurements.append(round(vertical_distance_mm, 1))
        print(
            "[P658207LE0A clipheight] "
            f"idx={clip_index} crop_shape={point['crop_shape']} "
            f"keypoint=({point['x']:.1f}, {point['y']:.1f}) "
            f"center_y={crop_center_y:.1f} distance_mm={measurements[-1]}"
        )
    return measurements


def _extract_clip_height_keypoint(image, bbox, clipheight_model, clip_index):
    if clipheight_model is None:
        print(f"[P658207LE0A clipheight] idx={clip_index} model missing")
        return None

    image_height, image_width = image.shape[:2]
    x1 = max(int(math.floor(bbox.minx)) - clip_height_padding_px, 0)
    y1 = max(int(math.floor(bbox.miny)) - clip_height_padding_px, 0)
    x2 = min(int(math.ceil(bbox.maxx)) + clip_height_padding_px, image_width)
    y2 = min(int(math.ceil(bbox.maxy)) + clip_height_padding_px, image_height)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        print(f"[P658207LE0A clipheight] idx={clip_index} empty crop")
        return None

    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    _save_clipheight_crop_debug(crop_bgr, clip_index)

    result = _normalize_result_object(
        clipheight_model(
            source=crop_bgr,
            conf=0.2,
            imgsz=clip_height_model_imgsz,
            verbose=True,
        )
    )
    if result is None or not hasattr(result, "keypoints"):
        print(f"[P658207LE0A clipheight] idx={clip_index} no keypoint result object")
        return None

    try:
        xy = result.keypoints.xy
        if xy is None or len(xy) == 0:
            print(f"[P658207LE0A clipheight] idx={clip_index} empty keypoints")
            return None
        kpt = xy[0][0]
        return {
            "x": float(kpt[0]),
            "y": float(kpt[1]),
            "crop_height": crop_bgr.shape[0],
            "crop_shape": crop_bgr.shape[:2],
        }
    except (AttributeError, IndexError, TypeError, ValueError):
        print(f"[P658207LE0A clipheight] idx={clip_index} keypoint parse failed")
        return None


def _save_clipheight_crop_debug(crop, clip_index):
    if not debug_save_images or crop is None or crop.size == 0:
        return

    date_dir = datetime.now().strftime("%Y%m%d")
    debug_dir = debug_root_dir / date_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = debug_dir / f"{timestamp}_clipheight_input_idx{clip_index}.png"
    cv2.imwrite(str(filename), crop)


def _evaluate_pitch_results(measured_pitches, pitch_spec, pitch_tolerance):
    results = [0] * len(pitch_spec)
    for index, (spec, detected) in enumerate(zip(pitch_spec, measured_pitches)):
        if detected is None:
            continue
        if abs(spec - detected) <= pitch_tolerance[index]:
            results[index] = 1
    return results


def _return_ng(image, message):
    status = "NG"
    image = draw_status_text_PIL(image, status, message, size="normal")
    measuredPitch = [0] * len(pitchSpec)
    resultPitch = [0] * len(pitchSpec)
    resultid = [0] * len(idSpec)
    deltaPitch = [0] * len(pitchSpec)
    return image, measuredPitch, resultPitch, resultid, status, message, deltaPitch


def calclength(p1, p2):
    return abs(p1[0] - p2[0])