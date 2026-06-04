import cv2
import yaml
from pathlib import Path

from aikensa.core.scripts.img_processing.img_processing import (
    draw_bounding_box,
    get_center,
    calclength,
    check_tolerance,
    check_id,
    draw_pitch_line,
    draw_status_text_PIL,
    map_keypoint_xcrop_to_original,
)

specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"
with open(specs_yaml_path, "r", encoding="utf-8") as f:
    _specs_root = yaml.safe_load(f) or {}


def _get_part_spec(part_id):
    return _specs_root.get("parts", {}).get(part_id, {})


def partcheck(image, sahi_prediction_list, keypoint_left, keypoint_right, part_id, keypoint_crop_px=840):
    part_spec = _get_part_spec(part_id)

    pitch_spec = part_spec.get("pitchSpec", [])
    id_spec = part_spec.get("idSpec", [])
    tolerance_pitch = part_spec.get("tolerance_pitch", [1.7] * len(pitch_spec))
    pixel_multiplier = float(part_spec.get("pixelMultiplier", 1.0))
    color = tuple(part_spec.get("color_ok", [0, 255, 0]))

    detected_id = []
    measured_pitch = []
    result_pitch = []
    result_id = []

    detected_pos_x = []
    detected_pos_y = []

    status = "OK"
    ng_reason = ""

    sorted_detections = sorted(sahi_prediction_list, key=lambda d: d.bbox.minx)

    expected_class = None
    if isinstance(id_spec, list) and len(id_spec) > 0:
        try:
            unique_ids = set(int(v) for v in id_spec)
            if len(unique_ids) == 1:
                expected_class = int(id_spec[0])
        except Exception:
            expected_class = None

    class_hist = {}
    for det in sorted_detections:
        try:
            cid = int(det.category.id)
        except Exception:
            continue
        class_hist[cid] = class_hist.get(cid, 0) + 1

    if expected_class is not None:
        detections_for_pitch = [d for d in sorted_detections if int(d.category.id) == expected_class]
        wrong_clip_detections = [d for d in sorted_detections if int(d.category.id) != expected_class]
    else:
        detections_for_pitch = list(sorted_detections)
        wrong_clip_detections = []

    print(
        f"[YA_DEBUG] part={part_id} expected_class={expected_class} "
        f"total={len(sorted_detections)} pitch_used={len(detections_for_pitch)} "
        f"wrong_class={len(wrong_clip_detections)} class_hist={class_hist}"
    )

    leftmost_point_x = None
    rightmost_point_x = None

    # Keypoint can be missing on difficult frames; do not fail immediately.
    for kp in keypoint_left:
        if kp.keypoints.xy is None or kp.keypoints.xy.shape[0] == 0 or kp.keypoints.xy.shape[1] == 0:
            continue
        x_pos, _ = kp.keypoints.xy[0, 0].tolist()
        leftmost_point_x, _ = map_keypoint_xcrop_to_original(
            x_start=0,
            kpt_xy_crop=(x_pos, 0),
            img_width=image.shape[1],
        )
        break

    for kp in keypoint_right:
        if kp.keypoints.xy is None or kp.keypoints.xy.shape[0] == 0 or kp.keypoints.xy.shape[1] == 0:
            continue
        x_pos, _ = kp.keypoints.xy[0, 0].tolist()
        rightmost_point_x, _ = map_keypoint_xcrop_to_original(
            x_start=-int(keypoint_crop_px),
            kpt_xy_crop=(x_pos, 0),
            img_width=image.shape[1],
        )
        break

    print(
        f"[YA_DEBUG] part={part_id} keypoint_left_found={leftmost_point_x is not None} "
        f"keypoint_right_found={rightmost_point_x is not None}"
    )

    prev_center = None
    for detection in detections_for_pitch:
        detected_id.append(detection.category.id)

        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny

        detected_pos_x.append(x)
        detected_pos_y.append(y)

        center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)
        if prev_center is not None:
            measured_pitch.append(calclength(prev_center, center) * pixel_multiplier)
        prev_center = center

    # Fallback: use expected-class bbox edges when keypoints are missing.
    if len(detections_for_pitch) > 0:
        if leftmost_point_x is None:
            try:
                leftmost_point_x = int(detections_for_pitch[0].bbox.minx)
            except Exception:
                leftmost_point_x = None
        if rightmost_point_x is None:
            try:
                rightmost_point_x = int(detections_for_pitch[-1].bbox.maxx)
            except Exception:
                rightmost_point_x = None

    if len(detected_pos_x) > 0 and leftmost_point_x is not None and rightmost_point_x is not None:
        left_center = (detected_pos_x[0], detected_pos_y[0])
        right_center = (detected_pos_x[-1], detected_pos_y[-1])

        left_edge = (leftmost_point_x, detected_pos_y[0])
        right_edge = (rightmost_point_x, detected_pos_y[-1])

        measured_pitch.insert(0, calclength(left_center, left_edge) * pixel_multiplier)
        measured_pitch.append(calclength(right_center, right_edge) * pixel_multiplier)

        detected_pos_x.insert(0, left_edge[0])
        detected_pos_y.insert(0, left_edge[1])
        detected_pos_x.append(right_edge[0])
        detected_pos_y.append(right_edge[1])
    elif len(detections_for_pitch) == 0:
        status = "NG"
        if len(sorted_detections) > 0:
            # Objects exist, but none match expected class -> wrong clip type.
            ng_reason = "CLIP TYPE NG"
            print(
                f"[YA_DEBUG] NG CLIP TYPE (no expected class detected) part={part_id} "
                f"expected_class={expected_class} total={len(sorted_detections)}"
            )
            image = draw_status_text_PIL(image, status, "クリップ類不良", size="normal")
        else:
            # Truly no detections.
            ng_reason = "PART IS NOT FOUND"
            image = draw_status_text_PIL(image, status, "製品は見つかりません", size="normal")
        return image, [0] * len(pitch_spec), [0] * len(pitch_spec), [0] * len(id_spec), status, ng_reason

    total_length = sum(measured_pitch)
    measured_pitch.append(round(total_length, 1))
    measured_pitch = [round(p, 1) for p in measured_pitch]

    print(
        f"[YA_DEBUG] part={part_id} measured_len={len(measured_pitch)} "
        f"pitch_spec_len={len(pitch_spec)} detected_ids={detected_id}"
    )

    if len(measured_pitch) == len(pitch_spec):
        result_pitch = check_tolerance(measured_pitch, pitch_spec, tolerance_pitch)
        result_id = check_id(detected_id, id_spec)
    else:
        status = "NG"
        ng_reason = "NUMBER OF CLIP MISMATCH"
        print(
            f"[YA_DEBUG] NG NUMBER OF CLIP MISMATCH part={part_id} "
            f"expected_class={expected_class} used={len(detections_for_pitch)} "
            f"wrong_class={len(wrong_clip_detections)}"
        )
        image = draw_status_text_PIL(image, status, "クリップ数不足", size="normal")
        return image, [0] * len(pitch_spec), [0] * len(pitch_spec), [0] * len(id_spec), status, ng_reason

    if any(r != 1 for r in result_pitch):
        status = "NG"
        ng_reason = "CLIP PITCH NG"

    if any(r != 1 for r in result_id):
        status = "NG"
        ng_reason = "CLIP TYPE NG"

    if len(wrong_clip_detections) > 0:
        status = "NG"
        ng_reason = "CLIP TYPE NG"
        print(
            f"[YA_DEBUG] NG CLIP TYPE part={part_id} expected_class={expected_class} "
            f"wrong_count={len(wrong_clip_detections)}"
        )

    draw_pitch_line(image, list(zip(detected_pos_x, detected_pos_y)), result_pitch, thickness=8)
    image = draw_status_text_PIL(image, status, "" if status == "OK" else "クリップ不良", size="normal")

    return image, measured_pitch, result_pitch, result_id, status, ng_reason
