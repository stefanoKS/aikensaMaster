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


def _safe_category_id(det):
    try:
        return int(det.category.id)
    except Exception:
        return None


def _crop_square_center(image, x, y, crop_px=84):
    """
    Crop fixed square image around center point.

    crop_px=84 means the returned crop is 84x84.
    If the crop goes outside the image, the image is padded.
    """
    crop_px = int(crop_px)
    if crop_px <= 0:
        raise ValueError("crop_px must be larger than 0")

    img_h, img_w = image.shape[:2]

    cx = int(round(x))
    cy = int(round(y))

    half = crop_px // 2

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + crop_px
    y1 = y0 + crop_px

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - img_w)
    pad_bottom = max(0, y1 - img_h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    crop = image[y0:y1, x0:x1]

    if crop.shape[0] != crop_px or crop.shape[1] != crop_px:
        crop = cv2.resize(crop, (crop_px, crop_px), interpolation=cv2.INTER_LINEAR)

    return crop


def _get_class_name_from_result(result, model, class_id):
    """
    Get class name from Ultralytics result/model.

    Example:
        names = {0: "NG", 1: "OK"}
        class_id = 1
        return "OK"
    """
    names = getattr(result, "names", None)

    if names is None:
        names = getattr(model, "names", None)

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def _predict_hanire_class(crop, hanire_detection_model, imgsz=128):
    """
    Run half-insert classifier.

    Important:
        Do not judge by class ID.
        Judge by class name because your model appears to be:

            {0: "NG", 1: "OK"}

        Your log showed:
            OK 1.00, NG 0.00
            class=1

        So class 1 is OK in your model.

    Returns:
        class_id, class_name, confidence
    """
    if hanire_detection_model is None or crop is None:
        return None, None, None

    try:
        # Preferred Ultralytics style.
        if hasattr(hanire_detection_model, "predict"):
            results = hanire_detection_model.predict(
                source=crop,
                imgsz=int(imgsz),
                verbose=False,
                stream=False,
            )
        else:
            # Fallback for callable model wrappers.
            results = hanire_detection_model(
                crop,
                imgsz=int(imgsz),
                stream=True,
                verbose=False,
            )
            results = list(results)

        if len(results) == 0:
            return None, None, None

        result = results[0]

        if getattr(result, "probs", None) is None:
            print("[YA_DEBUG] hanire model result has no classification probs")
            return None, None, None

        probs = result.probs

        class_id = int(probs.top1)
        confidence = float(probs.top1conf.item())

        class_name = _get_class_name_from_result(
            result=result,
            model=hanire_detection_model,
            class_id=class_id,
        )

        return class_id, class_name, confidence

    except Exception as e:
        print(f"[YA_DEBUG] hanire model error: {e}")

    return None, None, None


def _is_hanire_ng_class(class_name, ng_class_name="NG"):
    """
    Compare class name safely.

    This makes:
        "NG"
        "ng"
        " Ng "

    all treated as NG.
    """
    if class_name is None:
        return False

    return str(class_name).strip().upper() == str(ng_class_name).strip().upper()


def _check_hanire(
    raw_image,
    x,
    y,
    hanire_detection_model,
    crop_px=84,
    imgsz=128,
    ng_class_name="NG",
):
    """
    Crop around clip center and classify half-insert status.

    Returns:
        is_hanire_ng, class_id, class_name, confidence
    """
    crop = _crop_square_center(
        image=raw_image,
        x=x,
        y=y,
        crop_px=crop_px,
    )
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    class_id, class_name, confidence = _predict_hanire_class(
        crop=crop,
        hanire_detection_model=hanire_detection_model,
        imgsz=imgsz,
    )

    is_ng = _is_hanire_ng_class(
        class_name=class_name,
        ng_class_name=ng_class_name,
    )

    return is_ng, class_id, class_name, confidence


def _draw_hanire_ng_marker(image, x, y, w, h, thickness=6, bbox_offset=20):
    """
    Draw red circle marker on half-inserted clip.
    """
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))

    radius = int((w + h) / 4) + int(bbox_offset)
    radius = max(radius, 10)

    cv2.circle(
        image,
        center=(x, y),
        radius=radius,
        color=(10, 10, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def partcheck(
    image,
    sahi_prediction_list,
    keypoint_left,
    keypoint_right,
    hanire_detection_model,
    part_id,
    keypoint_crop_px=840,
    hanire_crop_px=84,
    hanire_imgsz=128,
    hanire_ng_class_name="NG",
):
    part_spec = _get_part_spec(part_id)

    pitch_spec = part_spec.get("pitchSpec", [])
    id_spec = part_spec.get("idSpec", [])
    tolerance_pitch = part_spec.get("tolerance_pitch", [1.7] * len(pitch_spec))
    pixel_multiplier = float(part_spec.get("pixelMultiplier", 1.0))
    color = tuple(part_spec.get("color_ok", [0, 255, 0]))

    # Optional per-part override from YAML.
    hanire_crop_px = int(part_spec.get("hanireCropPx", hanire_crop_px))
    hanire_imgsz = int(part_spec.get("hanireImgSz", hanire_imgsz))
    hanire_ng_class_name = str(part_spec.get("hanireNgClassName", hanire_ng_class_name))

    detected_id = []
    measured_pitch = []
    result_pitch = []
    result_id = []

    detected_pos_x = []
    detected_pos_y = []

    status = "OK"
    ng_reason = ""

    # Use raw image for half-insert crop.
    # Do not crop from image after bbox/line/text drawing.
    raw_image = image.copy()

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
        cid = _safe_category_id(det)
        if cid is None:
            continue
        class_hist[cid] = class_hist.get(cid, 0) + 1

    if expected_class is not None:
        detections_for_pitch = [
            d for d in sorted_detections
            if _safe_category_id(d) == expected_class
        ]

        wrong_clip_detections = [
            d for d in sorted_detections
            if _safe_category_id(d) is not None and _safe_category_id(d) != expected_class
        ]
    else:
        detections_for_pitch = list(sorted_detections)
        wrong_clip_detections = []

    print(
        f"[YA_DEBUG] part={part_id} expected_class={expected_class} "
        f"total={len(sorted_detections)} pitch_used={len(detections_for_pitch)} "
        f"wrong_class={len(wrong_clip_detections)} class_hist={class_hist}"
    )

    try:
        print(f"[YA_DEBUG] hanire model names: {hanire_detection_model.names}")
    except Exception:
        pass

    leftmost_point_x = None
    rightmost_point_x = None

    # Left keypoint.
    # Keypoint can be missing on difficult frames, so do not fail immediately.
    for kp in keypoint_left:
        try:
            if (
                kp.keypoints.xy is None
                or kp.keypoints.xy.shape[0] == 0
                or kp.keypoints.xy.shape[1] == 0
            ):
                continue

            x_pos, _ = kp.keypoints.xy[0, 0].tolist()

            leftmost_point_x, _ = map_keypoint_xcrop_to_original(
                x_start=0,
                kpt_xy_crop=(x_pos, 0),
                img_width=image.shape[1],
            )
            break
        except Exception:
            continue

    # Right keypoint.
    for kp in keypoint_right:
        try:
            if (
                kp.keypoints.xy is None
                or kp.keypoints.xy.shape[0] == 0
                or kp.keypoints.xy.shape[1] == 0
            ):
                continue

            x_pos, _ = kp.keypoints.xy[0, 0].tolist()

            rightmost_point_x, _ = map_keypoint_xcrop_to_original(
                x_start=-int(keypoint_crop_px),
                kpt_xy_crop=(x_pos, 0),
                img_width=image.shape[1],
            )
            break
        except Exception:
            continue

    print(
        f"[YA_DEBUG] part={part_id} keypoint_left_found={leftmost_point_x is not None} "
        f"keypoint_right_found={rightmost_point_x is not None}"
    )

    prev_center = None
    hanire_ng_detections = []

    for detection in detections_for_pitch:
        cid = _safe_category_id(detection)
        if cid is None:
            continue

        detected_id.append(cid)

        bbox = detection.bbox

        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny

        detected_pos_x.append(x)
        detected_pos_y.append(y)

        center = draw_bounding_box(
            image,
            x,
            y,
            w,
            h,
            [image.shape[1], image.shape[0]],
            color=color,
        )

        # ------------------------------------------------------------
        # Half-inserted clip detection
        #
        # Crop size:
        #     hanire_crop_px, default 84 px x 84 px
        #
        # Inference image size:
        #     hanire_imgsz, default 128
        #
        # NG judgment:
        #     Uses class name, default "NG"
        #
        # This is important because your model output showed:
        #     OK 1.00, NG 0.00
        #     class=1
        #
        # So class ID 1 is probably OK, not NG.
        # ------------------------------------------------------------
        (
            is_hanire_ng,
            hanire_class_id,
            hanire_class_name,
            hanire_conf,
        ) = _check_hanire(
            raw_image=raw_image,
            x=x,
            y=y,
            hanire_detection_model=hanire_detection_model,
            crop_px=hanire_crop_px,
            imgsz=hanire_imgsz,
            ng_class_name=hanire_ng_class_name,
        )

        print(
            f"[YA_DEBUG] part={part_id} hanire center=({x:.1f},{y:.1f}) "
            f"class_id={hanire_class_id} class_name={hanire_class_name} "
            f"conf={hanire_conf} is_ng={is_hanire_ng} "
            f"crop_px={hanire_crop_px} imgsz={hanire_imgsz}"
        )

        if is_hanire_ng:
            hanire_ng_detections.append(detection)
            _draw_hanire_ng_marker(
                image=image,
                x=x,
                y=y,
                w=w,
                h=h,
                thickness=6,
                bbox_offset=20,
            )

        if prev_center is not None:
            measured_pitch.append(calclength(prev_center, center) * pixel_multiplier)

        prev_center = center

    # Fallback:
    # Use first/last expected-class bbox edges when keypoints are missing.
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

        measured_pitch.insert(
            0,
            calclength(left_center, left_edge) * pixel_multiplier,
        )

        measured_pitch.append(
            calclength(right_center, right_edge) * pixel_multiplier,
        )

        detected_pos_x.insert(0, left_edge[0])
        detected_pos_y.insert(0, left_edge[1])

        detected_pos_x.append(right_edge[0])
        detected_pos_y.append(right_edge[1])

    elif len(detections_for_pitch) == 0:
        status = "NG"

        if len(sorted_detections) > 0:
            # Objects exist, but none match expected class.
            ng_reason = "CLIP TYPE NG"

            print(
                f"[YA_DEBUG] NG CLIP TYPE no expected class detected part={part_id} "
                f"expected_class={expected_class} total={len(sorted_detections)}"
            )

            image = draw_status_text_PIL(
                image,
                status,
                "クリップ類不良",
                size="normal",
            )

        else:
            # Truly no detections.
            ng_reason = "PART IS NOT FOUND"

            image = draw_status_text_PIL(
                image,
                status,
                "製品は見つかりません",
                size="normal",
            )

        return (
            image,
            [0] * len(pitch_spec),
            [0] * len(pitch_spec),
            [0] * len(id_spec),
            status,
            ng_reason,
        )

    total_length = sum(measured_pitch)
    measured_pitch.append(round(total_length, 1))
    measured_pitch = [round(p, 1) for p in measured_pitch]

    print(
        f"[YA_DEBUG] part={part_id} measured_len={len(measured_pitch)} "
        f"pitch_spec_len={len(pitch_spec)} detected_ids={detected_id} "
        f"measured_pitch={measured_pitch}"
    )

    if len(measured_pitch) == len(pitch_spec):
        result_pitch = check_tolerance(
            measured_pitch,
            pitch_spec,
            tolerance_pitch,
        )

        result_id = check_id(
            detected_id,
            id_spec,
        )

    else:
        status = "NG"
        ng_reason = "NUMBER OF CLIP MISMATCH"

        print(
            f"[YA_DEBUG] NG NUMBER OF CLIP MISMATCH part={part_id} "
            f"expected_class={expected_class} used={len(detections_for_pitch)} "
            f"wrong_class={len(wrong_clip_detections)} "
            f"measured_len={len(measured_pitch)} pitch_spec_len={len(pitch_spec)}"
        )

        image = draw_status_text_PIL(
            image,
            status,
            "クリップ数不足",
            size="normal",
        )

        return (
            image,
            [0] * len(pitch_spec),
            [0] * len(pitch_spec),
            [0] * len(id_spec),
            status,
            ng_reason,
        )

    # ------------------------------------------------------------
    # Final NG priority
    # ------------------------------------------------------------
    # 1. Clip type NG
    # 2. Half-insert NG
    # 3. Pitch NG
    #
    # This priority avoids judging half-insert on an already wrong clip type.
    # ------------------------------------------------------------

    if any(r != 1 for r in result_id):
        status = "NG"
        ng_reason = "CLIP TYPE NG"

        print(
            f"[YA_DEBUG] NG CLIP TYPE from id check part={part_id} "
            f"result_id={result_id} detected_id={detected_id} id_spec={id_spec}"
        )

    if len(wrong_clip_detections) > 0:
        status = "NG"
        ng_reason = "CLIP TYPE NG"

        print(
            f"[YA_DEBUG] NG CLIP TYPE wrong detections part={part_id} "
            f"expected_class={expected_class} "
            f"wrong_count={len(wrong_clip_detections)}"
        )

    if status == "OK" and len(hanire_ng_detections) > 0:
        status = "NG"
        ng_reason = "CLIP HALF INSERTED"

        print(
            f"[YA_DEBUG] NG CLIP HALF INSERTED part={part_id} "
            f"hanire_ng_count={len(hanire_ng_detections)}"
        )

    if status == "OK" and any(r != 1 for r in result_pitch):
        status = "NG"
        ng_reason = "CLIP PITCH NG"

        print(
            f"[YA_DEBUG] NG CLIP PITCH part={part_id} "
            f"result_pitch={result_pitch} measured_pitch={measured_pitch} "
            f"pitch_spec={pitch_spec}"
        )

    draw_pitch_line(
        image,
        list(zip(detected_pos_x, detected_pos_y)),
        result_pitch,
        thickness=8,
    )

    if status == "OK":
        image = draw_status_text_PIL(
            image,
            status,
            "",
            size="normal",
        )

    elif ng_reason == "CLIP HALF INSERTED":
        image = draw_status_text_PIL(
            image,
            status,
            "クリップ半入れ不良",
            size="normal",
        )

    elif ng_reason == "CLIP TYPE NG":
        image = draw_status_text_PIL(
            image,
            status,
            "クリップ類不良",
            size="normal",
        )

    elif ng_reason == "CLIP PITCH NG":
        image = draw_status_text_PIL(
            image,
            status,
            "クリップピッチ不良",
            size="normal",
        )

    else:
        image = draw_status_text_PIL(
            image,
            status,
            "クリップ不良",
            size="normal",
        )

    return (
        image,
        measured_pitch,
        result_pitch,
        result_id,
        status,
        ng_reason,
    )

# import cv2
# import yaml
# from pathlib import Path

# from aikensa.core.scripts.img_processing.img_processing import (
#     draw_bounding_box,
#     get_center,
#     calclength,
#     check_tolerance,
#     check_id,
#     draw_pitch_line,
#     draw_status_text_PIL,
#     map_keypoint_xcrop_to_original,
# )

# specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"
# with open(specs_yaml_path, "r", encoding="utf-8") as f:
#     _specs_root = yaml.safe_load(f) or {}


# def _get_part_spec(part_id):
#     return _specs_root.get("parts", {}).get(part_id, {})


# def partcheck(image, sahi_prediction_list, keypoint_left, keypoint_right, hanire_detection_model, part_id, keypoint_crop_px=840):
#     part_spec = _get_part_spec(part_id)

#     pitch_spec = part_spec.get("pitchSpec", [])
#     id_spec = part_spec.get("idSpec", [])
#     tolerance_pitch = part_spec.get("tolerance_pitch", [1.7] * len(pitch_spec))
#     pixel_multiplier = float(part_spec.get("pixelMultiplier", 1.0))
#     color = tuple(part_spec.get("color_ok", [0, 255, 0]))

#     detected_id = []
#     measured_pitch = []
#     result_pitch = []
#     result_id = []

#     detected_pos_x = []
#     detected_pos_y = []

#     status = "OK"
#     ng_reason = ""

#     sorted_detections = sorted(sahi_prediction_list, key=lambda d: d.bbox.minx)

#     expected_class = None
#     if isinstance(id_spec, list) and len(id_spec) > 0:
#         try:
#             unique_ids = set(int(v) for v in id_spec)
#             if len(unique_ids) == 1:
#                 expected_class = int(id_spec[0])
#         except Exception:
#             expected_class = None

#     class_hist = {}
#     for det in sorted_detections:
#         try:
#             cid = int(det.category.id)
#         except Exception:
#             continue
#         class_hist[cid] = class_hist.get(cid, 0) + 1

#     if expected_class is not None:
#         detections_for_pitch = [d for d in sorted_detections if int(d.category.id) == expected_class]
#         wrong_clip_detections = [d for d in sorted_detections if int(d.category.id) != expected_class]
#     else:
#         detections_for_pitch = list(sorted_detections)
#         wrong_clip_detections = []

#     print(
#         f"[YA_DEBUG] part={part_id} expected_class={expected_class} "
#         f"total={len(sorted_detections)} pitch_used={len(detections_for_pitch)} "
#         f"wrong_class={len(wrong_clip_detections)} class_hist={class_hist}"
#     )

#     leftmost_point_x = None
#     rightmost_point_x = None

#     # Keypoint can be missing on difficult frames; do not fail immediately.
#     for kp in keypoint_left:
#         if kp.keypoints.xy is None or kp.keypoints.xy.shape[0] == 0 or kp.keypoints.xy.shape[1] == 0:
#             continue
#         x_pos, _ = kp.keypoints.xy[0, 0].tolist()
#         leftmost_point_x, _ = map_keypoint_xcrop_to_original(
#             x_start=0,
#             kpt_xy_crop=(x_pos, 0),
#             img_width=image.shape[1],
#         )
#         break

#     for kp in keypoint_right:
#         if kp.keypoints.xy is None or kp.keypoints.xy.shape[0] == 0 or kp.keypoints.xy.shape[1] == 0:
#             continue
#         x_pos, _ = kp.keypoints.xy[0, 0].tolist()
#         rightmost_point_x, _ = map_keypoint_xcrop_to_original(
#             x_start=-int(keypoint_crop_px),
#             kpt_xy_crop=(x_pos, 0),
#             img_width=image.shape[1],
#         )
#         break

#     print(
#         f"[YA_DEBUG] part={part_id} keypoint_left_found={leftmost_point_x is not None} "
#         f"keypoint_right_found={rightmost_point_x is not None}"
#     )

#     prev_center = None
#     for detection in detections_for_pitch:
#         detected_id.append(detection.category.id)

#         bbox = detection.bbox
#         x, y = get_center(bbox)
#         w = bbox.maxx - bbox.minx
#         h = bbox.maxy - bbox.miny

#         detected_pos_x.append(x)
#         detected_pos_y.append(y)

#         center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)
#         if prev_center is not None:
#             measured_pitch.append(calclength(prev_center, center) * pixel_multiplier)
#         prev_center = center

#     # Fallback: use expected-class bbox edges when keypoints are missing.
#     if len(detections_for_pitch) > 0:
#         if leftmost_point_x is None:
#             try:
#                 leftmost_point_x = int(detections_for_pitch[0].bbox.minx)
#             except Exception:
#                 leftmost_point_x = None
#         if rightmost_point_x is None:
#             try:
#                 rightmost_point_x = int(detections_for_pitch[-1].bbox.maxx)
#             except Exception:
#                 rightmost_point_x = None

#     if len(detected_pos_x) > 0 and leftmost_point_x is not None and rightmost_point_x is not None:
#         left_center = (detected_pos_x[0], detected_pos_y[0])
#         right_center = (detected_pos_x[-1], detected_pos_y[-1])

#         left_edge = (leftmost_point_x, detected_pos_y[0])
#         right_edge = (rightmost_point_x, detected_pos_y[-1])

#         measured_pitch.insert(0, calclength(left_center, left_edge) * pixel_multiplier)
#         measured_pitch.append(calclength(right_center, right_edge) * pixel_multiplier)

#         detected_pos_x.insert(0, left_edge[0])
#         detected_pos_y.insert(0, left_edge[1])
#         detected_pos_x.append(right_edge[0])
#         detected_pos_y.append(right_edge[1])
#     elif len(detections_for_pitch) == 0:
#         status = "NG"
#         if len(sorted_detections) > 0:
#             # Objects exist, but none match expected class -> wrong clip type.
#             ng_reason = "CLIP TYPE NG"
#             print(
#                 f"[YA_DEBUG] NG CLIP TYPE (no expected class detected) part={part_id} "
#                 f"expected_class={expected_class} total={len(sorted_detections)}"
#             )
#             image = draw_status_text_PIL(image, status, "クリップ類不良", size="normal")
#         else:
#             # Truly no detections.
#             ng_reason = "PART IS NOT FOUND"
#             image = draw_status_text_PIL(image, status, "製品は見つかりません", size="normal")
#         return image, [0] * len(pitch_spec), [0] * len(pitch_spec), [0] * len(id_spec), status, ng_reason

#     total_length = sum(measured_pitch)
#     measured_pitch.append(round(total_length, 1))
#     measured_pitch = [round(p, 1) for p in measured_pitch]

#     print(
#         f"[YA_DEBUG] part={part_id} measured_len={len(measured_pitch)} "
#         f"pitch_spec_len={len(pitch_spec)} detected_ids={detected_id}"
#         f"detected_id={detected_id}"
#         f"measured_pitch={measured_pitch}"
#     )

#     if len(measured_pitch) == len(pitch_spec):
#         result_pitch = check_tolerance(measured_pitch, pitch_spec, tolerance_pitch)
#         result_id = check_id(detected_id, id_spec)
#     else:
#         status = "NG"
#         ng_reason = "NUMBER OF CLIP MISMATCH"
#         print(
#             f"[YA_DEBUG] NG NUMBER OF CLIP MISMATCH part={part_id} "
#             f"expected_class={expected_class} used={len(detections_for_pitch)} "
#             f"wrong_class={len(wrong_clip_detections)}"
#         )
#         image = draw_status_text_PIL(image, status, "クリップ数不足", size="normal")
#         return image, [0] * len(pitch_spec), [0] * len(pitch_spec), [0] * len(id_spec), status, ng_reason

#     if any(r != 1 for r in result_pitch):
#         status = "NG"
#         ng_reason = "CLIP PITCH NG"

#     if any(r != 1 for r in result_id):
#         status = "NG"
#         ng_reason = "CLIP TYPE NG"

#     if len(wrong_clip_detections) > 0:
#         status = "NG"
#         ng_reason = "CLIP TYPE NG"
#         print(
#             f"[YA_DEBUG] NG CLIP TYPE part={part_id} expected_class={expected_class} "
#             f"wrong_count={len(wrong_clip_detections)}"
#         )

#     draw_pitch_line(image, list(zip(detected_pos_x, detected_pos_y)), result_pitch, thickness=8)
#     image = draw_status_text_PIL(image, status, "" if status == "OK" else "クリップ不良", size="normal")

#     return image, measured_pitch, result_pitch, result_id, status, ng_reason
