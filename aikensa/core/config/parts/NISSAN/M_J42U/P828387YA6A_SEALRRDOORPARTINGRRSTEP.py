import math
import time
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


# ============================================================
# YAML SPEC
# ============================================================

specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"

with open(specs_yaml_path, "r", encoding="utf-8") as f:
    all_specs = yaml.safe_load(f) or {}
    part_spec = all_specs.get("parts", {}).get("P828387YA6A", {})
    global_spec = all_specs.get("global", {})


kanjiFontPath = global_spec.get("font_path")


# ============================================================
# PART SPEC DEFAULTS
# ============================================================

pitchSpecRH = part_spec.get(
    "pitchSpecRH",
    [14, 130, 132, 132, 59, 61, 97, 83, 20, 708],
)

pitchSpecLH = part_spec.get(
    "pitchSpecLH",
    [83, 97, 61, 59, 132, 132, 130, 14, 20, 708],
)

tolerance_pitchRH = part_spec.get(
    "tolerance_pitchRH",
    [3.0, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 2.0, 10.0],
)

tolerance_pitchLH = part_spec.get(
    "tolerance_pitchLH",
    [1.7, 1.7, 1.7, 3.0, 1.7, 1.7, 1.7, 1.7, 2.0, 10.0],
)

# Current class ID rule:
# RH is WHITE -> id 1
# LH is BROWN -> id 0
idSpecRH = part_spec.get(
    "idSpecRH",
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
)

idSpecLH = part_spec.get(
    "idSpecLH",
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
)

color = tuple(part_spec.get("color_ok", [0, 255, 0]))
color_ng = tuple(part_spec.get("color_ng", [200, 30, 50]))

pixelMultiplier = float(part_spec.get("pixelMultiplier", 0.16413))
pixelMultiplier_katabumarking = float(
    part_spec.get("pixelMultiplier_katabumarking", 0.16413)
)

bbox_offset = int(part_spec.get("bbox_offset", 10))


# ============================================================
# MAIN CHECK FUNCTION
# ============================================================

def partcheck(
    image,
    sahi_predictionList,
    left_keypoint=None,
    right_keypoint=None,
    katabu_detection=None,
    side="RH",
    keypoint_crop_px=None,
    katabu_crop_width=320,
    *,
    katabu_clip_flip_model=None,
    ws_clip_hanire_model=None,
    clip_classifier_crop_px=128,
    clip_classifier_imgsz=128,
    katabu_flip_ng_class_name="NG",
    hanire_ng_class_name="NG",
    classifier_convert_bgr_to_rgb=False,
    debug_save_crops=False,
    debug_crop_dir=None,
    debug_crop_run_id=None,
):
    """
    P828387YA6A inspection.

    Kept from original:
        - RH/LH pitch spec
        - RH/LH clip class ID check
        - endpoint keypoint logic
        - katabu-marking detection pitch
        - pitch line drawing
        - pitch/id judgment

    Added:
        - RH: last/rightmost expected clip uses katabu clip flip classifier
        - LH: first/leftmost expected clip uses katabu clip flip classifier
        - Other clips use hanire classifier
        - Classifier crop size is adjustable, default 128 x 128
        - Classifier inference imgsz is explicit, default 128
        - Optional BGR -> RGB conversion before inference
        - Debug crop saving for both katabu and normal clip crops
    """

    side = _normalize_side(side)

    if keypoint_crop_px is None:
        keypoint_crop_px = int(part_spec.get("keypoint_crop_px", 1360))

    clip_classifier_crop_px = int(
        part_spec.get("clipClassifierCropPx", clip_classifier_crop_px)
    )

    clip_classifier_imgsz = int(
        part_spec.get("clipClassifierImgSz", clip_classifier_imgsz)
    )

    katabu_flip_ng_class_name = str(
        part_spec.get("katabuFlipNgClassName", katabu_flip_ng_class_name)
    )

    hanire_ng_class_name = str(
        part_spec.get("hanireNgClassName", hanire_ng_class_name)
    )

    classifier_convert_bgr_to_rgb = bool(
        part_spec.get("classifierConvertBgrToRgb", classifier_convert_bgr_to_rgb)
    )

    debug_save_crops = bool(
        part_spec.get("debugSaveCrops", debug_save_crops)
    )

    debug_crop_dir = part_spec.get("debugCropDir", debug_crop_dir)

    if debug_crop_run_id is None:
        debug_crop_run_id = f"{side}_{int(time.time() * 1000)}"

    # ------------------------------------------------------------
    # Side-specific configuration
    # ------------------------------------------------------------

    if side == "RH":
        pitchSpec = pitchSpecRH
        tolerance_pitch = tolerance_pitchRH

        expected_clip_id = int(part_spec.get("expectedClipIdRH", 1))
        expected_ids = part_spec.get("idSpecRH", idSpecRH)

        endpoint = _extract_endpoint_from_keypoints(
            left_keypoint,
            x_start=0,
            image_width=image.shape[1],
        )

        katabu_x_offset = image.shape[1] - int(katabu_crop_width)

    else:
        pitchSpec = pitchSpecLH
        tolerance_pitch = tolerance_pitchLH

        expected_clip_id = int(part_spec.get("expectedClipIdLH", 0))
        expected_ids = part_spec.get("idSpecLH", idSpecLH)

        endpoint = _extract_endpoint_from_keypoints(
            right_keypoint,
            x_start=-int(keypoint_crop_px),
            image_width=image.shape[1],
        )

        katabu_x_offset = 0

    raw_image = image.copy()

    # ------------------------------------------------------------
    # Sort SAHI detections from left to right
    # ------------------------------------------------------------

    sorted_detections = sorted(
        sahi_predictionList,
        key=lambda d: float(d.bbox.minx),
    )

    expected_detections = []
    ignored_detections = []

    for det_index, detection in enumerate(sorted_detections):
        cls_id = _safe_detection_class_id(detection)
        cls_name = _safe_detection_class_name(detection)

        print(
            f"[YA6A][{side}] SAHI det_index={det_index} "
            f"class_id={cls_id} class_name={cls_name} "
            f"bbox=({float(detection.bbox.minx):.1f}, "
            f"{float(detection.bbox.miny):.1f}, "
            f"{float(detection.bbox.maxx):.1f}, "
            f"{float(detection.bbox.maxy):.1f})"
        )

        if cls_id is None:
            ignored_detections.append(detection)
            continue

        if cls_id == expected_clip_id:
            expected_detections.append(detection)
        else:
            ignored_detections.append(detection)

    katabu_clip_index = _get_katabu_clip_index(
        side=side,
        clip_count=len(expected_detections),
    )

    print(
        f"[YA6A][{side}] expected_clip_id={expected_clip_id} "
        f"expected_detection_count={len(expected_detections)} "
        f"ignored_detection_count={len(ignored_detections)} "
        f"katabu_clip_index={katabu_clip_index}"
    )

    # ------------------------------------------------------------
    # Process expected clip detections
    # ------------------------------------------------------------

    detectedid = []
    clip_points = []
    measured_inner = []

    classifier_results = []
    katabu_flip_ng_detections = []
    hanire_ng_detections = []

    prev_center = None

    for clip_index, detection in enumerate(expected_detections):
        cls_id = _safe_detection_class_id(detection)

        if cls_id is None:
            continue

        expected_id = (
            expected_ids[len(detectedid)]
            if len(detectedid) < len(expected_ids)
            else None
        )

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

        is_katabu_clip = clip_index == katabu_clip_index

        if is_katabu_clip:
            clf_model = katabu_clip_flip_model
            clf_ng_class_name = katabu_flip_ng_class_name
            clf_name = "katabu_flip"
        else:
            clf_model = ws_clip_hanire_model
            clf_ng_class_name = hanire_ng_class_name
            clf_name = "hanire"

        debug_crop_prefix = (
            f"{debug_crop_run_id}_"
            f"{side}_"
            f"clip{clip_index:02d}_"
            f"{clf_name}_"
            f"x{int(round(x))}_"
            f"y{int(round(y))}"
        )

        (
            is_ng,
            class_id,
            class_name,
            confidence,
            saved_crop_paths,
        ) = _classify_clip_center(
            raw_image=raw_image,
            x=x,
            y=y,
            model=clf_model,
            crop_px=clip_classifier_crop_px,
            imgsz=clip_classifier_imgsz,
            ng_class_name=clf_ng_class_name,
            convert_bgr_to_rgb=classifier_convert_bgr_to_rgb,
            debug_save_crops=debug_save_crops,
            debug_crop_dir=debug_crop_dir,
            debug_crop_prefix=debug_crop_prefix,
        )

        classifier_results.append(
            {
                "clip_index": clip_index,
                "is_katabu_clip": is_katabu_clip,
                "classifier": clf_name,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "is_ng": is_ng,
                "center": center,
                "saved_crop_paths": saved_crop_paths,
            }
        )

        print(
            f"[YA6A][{side}] classifier={clf_name} "
            f"clip_index={clip_index} "
            f"is_katabu_clip={is_katabu_clip} "
            f"center=({x:.1f},{y:.1f}) "
            f"class_id={class_id} "
            f"class_name={class_name} "
            f"conf={confidence} "
            f"is_ng={is_ng} "
            f"crop_px={clip_classifier_crop_px} "
            f"imgsz={clip_classifier_imgsz} "
            f"bgr_to_rgb={classifier_convert_bgr_to_rgb}"
        )

        if saved_crop_paths:
            for saved_path in saved_crop_paths:
                print(f"[YA6A][{side}][DEBUG_CROP] {saved_path}")

        if is_ng:
            _draw_clip_ng_marker(
                image=image,
                x=x,
                y=y,
                w=w,
                h=h,
                label="KATABU" if is_katabu_clip else "HANIRE",
            )

            if is_katabu_clip:
                katabu_flip_ng_detections.append(detection)
            else:
                hanire_ng_detections.append(detection)

        if prev_center is not None:
            measured_inner.append(
                calclength(prev_center, center) * pixelMultiplier
            )

        prev_center = center

    print(
        f"[YA6A][{side}] clip_count={len(clip_points)} "
        f"clip_ids={detectedid} "
        f"expected_clip_id={expected_clip_id} "
        f"endpoint={'OK' if endpoint is not None else 'MISS'} "
        f"katabu_clip_index={katabu_clip_index} "
        f"ignored_count={len(ignored_detections)}"
    )

    # ------------------------------------------------------------
    # Basic NG checks
    # ------------------------------------------------------------

    if endpoint is None:
        return _return_ng(
            image=image,
            pitchSpec=pitchSpec,
            idSpec=expected_ids,
            message="製品は見つかりません",
        )

    if len(clip_points) == 0:
        return _return_ng(
            image=image,
            pitchSpec=pitchSpec,
            idSpec=expected_ids,
            message="クリップ未検出",
        )

    # ------------------------------------------------------------
    # Katabu marking pitch
    # ------------------------------------------------------------

    katabu_pitch, katabu_mark_point = _extract_katabu_pitch(
        katabu_detection,
        side=side,
        x_offset=katabu_x_offset,
    )

    if katabu_pitch is None or katabu_mark_point is None:
        return _return_ng(
            image=image,
            pitchSpec=pitchSpec,
            idSpec=expected_ids,
            message="型部未検出",
        )

    # ------------------------------------------------------------
    # Original measurement construction
    # ------------------------------------------------------------

    if side == "RH":
        endpoint_pitch = calclength(endpoint, clip_points[0]) * pixelMultiplier

        # Original behavior:
        # skip final clip-to-clip pitch because katabu pitch replaces that side.
        trimmed_inner = measured_inner[:-1] if len(measured_inner) > 0 else []

        measuredPitch = [endpoint_pitch] + trimmed_inner + [katabu_pitch]

        draw_points = [endpoint] + clip_points + [katabu_mark_point]

        skipped_segment_index = (
            len(draw_points) - 2
            if len(draw_points) >= 2
            else None
        )

    else:
        endpoint_pitch = calclength(clip_points[-1], endpoint) * pixelMultiplier

        # Original behavior:
        # skip first clip-to-clip pitch because katabu pitch replaces that side.
        trimmed_inner = measured_inner[1:] if len(measured_inner) > 0 else []

        measuredPitch = trimmed_inner + [endpoint_pitch] + [katabu_pitch]

        draw_points = [katabu_mark_point] + clip_points + [endpoint]

        skipped_segment_index = 0 if len(draw_points) >= 2 else None

    totalLength = sum(measuredPitch)

    measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    # ------------------------------------------------------------
    # Pitch and ID result
    # ------------------------------------------------------------

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(
            measuredPitch,
            pitchSpec,
            tolerance_pitch,
        )
    else:
        resultPitch = [0] * len(pitchSpec)

        print(
            f"[YA6A][{side}] pitch length mismatch "
            f"measured_len={len(measuredPitch)} "
            f"spec_len={len(pitchSpec)} "
            f"measuredPitch={measuredPitch} "
            f"pitchSpec={pitchSpec}"
        )

    if len(detectedid) == len(expected_ids):
        resultid = check_id(
            detectedid,
            expected_ids,
        )
    else:
        resultid = [0] * len(expected_ids)

        print(
            f"[YA6A][{side}] id length mismatch "
            f"detected_len={len(detectedid)} "
            f"expected_len={len(expected_ids)} "
            f"detectedid={detectedid} "
            f"expected_ids={expected_ids}"
        )

    # ------------------------------------------------------------
    # Final NG priority
    # ------------------------------------------------------------

    status = "OK"
    ng_reason = ""

    if any(result != 1 for result in resultid):
        status = "NG"
        ng_reason = "CLIP TYPE NG"

    elif len(katabu_flip_ng_detections) > 0:
        status = "NG"
        ng_reason = "KATABU CLIP FLIP NG"

    elif len(hanire_ng_detections) > 0:
        status = "NG"
        ng_reason = "CLIP HALF INSERTED"

    elif any(result != 1 for result in resultPitch):
        status = "NG"
        ng_reason = "CLIP PITCH NG"

    # ------------------------------------------------------------
    # Draw pitch line
    # ------------------------------------------------------------

    line_result_values = _build_line_result_values(
        resultPitch,
        skipped_segment_index=skipped_segment_index,
    )

    if len(draw_points) >= 2:
        draw_pitch_line(
            image,
            draw_points,
            line_result_values,
            thickness=8,
        )

    print(
        f"[YA6A][{side}] status={status} "
        f"ng_reason={ng_reason} "
        f"measuredPitch={measuredPitch} "
        f"resultPitch={resultPitch} "
        f"resultid={resultid} "
        f"katabu_pitch={katabu_pitch} "
        f"katabu_flip_ng_count={len(katabu_flip_ng_detections)} "
        f"hanire_ng_count={len(hanire_ng_detections)}"
    )

    image = _draw_final_status(
        image=image,
        status=status,
        ng_reason=ng_reason,
    )

    return (
        image,
        measuredPitch,
        resultPitch,
        resultid,
        status,
        ng_reason,
    )


# ============================================================
# SIDE / CLASS ID HELPERS
# ============================================================

def _normalize_side(side):
    side = str(side).upper().strip()

    if side in ["RH", "R", "RIGHT"]:
        return "RH"

    if side in ["LH", "L", "LEFT"]:
        return "LH"

    return "RH"


def _safe_detection_class_id(detection):
    try:
        return int(detection.category.id)
    except Exception:
        return None


def _safe_detection_class_name(detection):
    try:
        return str(detection.category.name)
    except Exception:
        return None


def _get_katabu_clip_index(side, clip_count):
    """
    Decide which clip uses the katabu clip flip classifier.

    Current rule:
        RH: last/rightmost expected clip
        LH: first/leftmost expected clip

    Detections are sorted by bbox.minx before this function.
    """
    if clip_count <= 0:
        return None

    if side == "RH":
        return clip_count - 1

    return 0


# ============================================================
# CROP AND CLASSIFIER
# ============================================================

def _crop_square_center(image, x, y, crop_px=128):
    """
    Fixed-size square crop around clip center.

    The returned crop is always crop_px x crop_px.
    If the crop goes outside the image, BORDER_REPLICATE padding is used.
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
        crop = cv2.resize(
            crop,
            (crop_px, crop_px),
            interpolation=cv2.INTER_LINEAR,
        )

    return crop


def _classify_clip_center(
    raw_image,
    x,
    y,
    model,
    crop_px=128,
    imgsz=128,
    ng_class_name="NG",
    convert_bgr_to_rgb=False,
    debug_save_crops=False,
    debug_crop_dir=None,
    debug_crop_prefix="crop",
):
    """
    Crop around clip center and classify.

    Returns:
        is_ng, class_id, class_name, confidence, saved_crop_paths

    Important:
        raw_image is OpenCV BGR.

        If convert_bgr_to_rgb=False:
            BGR crop is passed directly to YOLO.

        If convert_bgr_to_rgb=True:
            crop is converted BGR -> RGB before YOLO inference.

    Debug crop files:
        *_bgr.png
            Exact OpenCV BGR crop before optional conversion.

        *_inference_preview.png
            The actual image passed to inference, saved in correct visible color.
    """
    saved_crop_paths = []

    if model is None:
        print("[YA6A] classifier model is None, skipping classifier")
        return False, None, None, None, saved_crop_paths

    crop_bgr = _crop_square_center(
        image=raw_image,
        x=x,
        y=y,
        crop_px=crop_px,
    )

    if convert_bgr_to_rgb:
        crop_for_inference = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    else:
        crop_for_inference = crop_bgr

    class_id, class_name, confidence = _predict_classifier(
        model=model,
        crop=crop_for_inference,
        imgsz=imgsz,
    )

    is_ng = _is_ng_class_name(
        class_name=class_name,
        ng_class_name=ng_class_name,
    )

    if debug_save_crops:
        saved_crop_paths = _save_classifier_debug_crop(
            crop_bgr=crop_bgr,
            crop_for_inference=crop_for_inference,
            convert_bgr_to_rgb=convert_bgr_to_rgb,
            debug_crop_dir=debug_crop_dir,
            debug_crop_prefix=debug_crop_prefix,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            is_ng=is_ng,
        )

    return is_ng, class_id, class_name, confidence, saved_crop_paths


def _predict_classifier(model, crop, imgsz=128):
    """
    Ultralytics classification inference.

    Uses explicit imgsz.

    The final NG judgment is done by class name, not class ID.
    This prevents mistakes when model.names is like:
        {0: "NG", 1: "OK"}
    """
    try:
        if hasattr(model, "predict"):
            results = model.predict(
                source=crop,
                imgsz=int(imgsz),
                stream=False,
                verbose=False,
            )
        else:
            results = model(
                source=crop,
                imgsz=int(imgsz),
                stream=True,
                verbose=False,
            )
            results = list(results)

        if results is None or len(results) == 0:
            return None, None, None

        result = results[0]

        if getattr(result, "probs", None) is None:
            print("[YA6A] classifier result has no probs")
            return None, None, None

        probs = result.probs

        class_id = int(probs.top1)
        confidence = float(probs.top1conf.item())

        class_name = _get_class_name_from_result(
            result=result,
            model=model,
            class_id=class_id,
        )

        return class_id, class_name, confidence

    except Exception as e:
        print(f"[YA6A] classifier error: {e}")

    return None, None, None


def _get_class_name_from_result(result, model, class_id):
    names = getattr(result, "names", None)

    if names is None:
        names = getattr(model, "names", None)

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def _is_ng_class_name(class_name, ng_class_name="NG"):
    if class_name is None:
        return False

    return str(class_name).strip().upper() == str(ng_class_name).strip().upper()


# ============================================================
# DEBUG CROP SAVE
# ============================================================

def _safe_filename_text(value):
    text = str(value)

    for bad_char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
        text = text.replace(bad_char, "_")

    return text


def _save_classifier_debug_crop(
    crop_bgr,
    crop_for_inference,
    convert_bgr_to_rgb,
    debug_crop_dir,
    debug_crop_prefix,
    class_id,
    class_name,
    confidence,
    is_ng,
):
    """
    Save classifier crops for visual checking.

    Saves:
        1. *_bgr.png
           Original OpenCV BGR crop.

        2. *_inference_preview.png
           Actual inference image, saved with correct visual colors.

    If convert_bgr_to_rgb=True:
        crop_for_inference is RGB, so convert it back to BGR before cv2.imwrite.
    """
    saved_paths = []

    try:
        if debug_crop_dir is None:
            debug_crop_dir = (
                Path(__file__).resolve().parent
                / "debug_crops"
                / "P828387YA6A"
            )
        else:
            debug_crop_dir = Path(debug_crop_dir)

        debug_crop_dir.mkdir(parents=True, exist_ok=True)

        class_name_safe = _safe_filename_text(class_name)
        class_id_safe = _safe_filename_text(class_id)

        if confidence is None:
            conf_text = "None"
        else:
            conf_text = f"{float(confidence):.4f}"

        filename_base = (
            f"{debug_crop_prefix}_"
            f"pred_id-{class_id_safe}_"
            f"name-{class_name_safe}_"
            f"conf-{conf_text}_"
            f"ng-{int(bool(is_ng))}"
        )

        bgr_path = debug_crop_dir / f"{filename_base}_bgr.png"
        ok_bgr = cv2.imwrite(str(bgr_path), crop_bgr)

        if ok_bgr:
            saved_paths.append(str(bgr_path))
        else:
            print(f"[YA6A][DEBUG_CROP] failed cv2.imwrite: {bgr_path}")

        if convert_bgr_to_rgb:
            inference_preview_bgr = cv2.cvtColor(
                crop_for_inference,
                cv2.COLOR_RGB2BGR,
            )
        else:
            inference_preview_bgr = crop_for_inference

        inference_path = debug_crop_dir / f"{filename_base}_inference_preview.png"
        ok_inf = cv2.imwrite(str(inference_path), inference_preview_bgr)

        if ok_inf:
            saved_paths.append(str(inference_path))
        else:
            print(f"[YA6A][DEBUG_CROP] failed cv2.imwrite: {inference_path}")

    except Exception as e:
        print(f"[YA6A][DEBUG_CROP] failed to save crop: {e}")

    return saved_paths


# ============================================================
# DRAWING HELPERS
# ============================================================

def _draw_clip_ng_marker(image, x, y, w, h, label="NG"):
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))

    radius = max(int((w + h) / 4) + 20, 18)

    cv2.circle(
        image,
        center=(x, y),
        radius=radius,
        color=(200, 25, 25),
        thickness=6,
        lineType=cv2.LINE_AA,
    )

    # Uncomment this if you want visible English marker text.
    # cv2.putText(
    #     image,
    #     str(label),
    #     (x - radius, y - radius - 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1.0,
    #     (200, 25, 25),
    #     3,
    #     cv2.LINE_AA,
    # )


def _draw_final_status(image, status, ng_reason):
    if status == "OK":
        return draw_status_text_PIL(
            image,
            status,
            "",
            size="normal",
        )

    if ng_reason == "KATABU CLIP FLIP NG":
        message = "型部クリップ向き不良"

    elif ng_reason == "CLIP HALF INSERTED":
        message = "クリップ半入れ不良"

    elif ng_reason == "CLIP TYPE NG":
        message = "クリップ類不良"

    elif ng_reason == "CLIP PITCH NG":
        message = "クリップピッチ不良"

    else:
        message = "クリップ不良"

    return draw_status_text_PIL(
        image,
        status,
        message,
        size="normal",
    )


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

    if kanjiFontPath:
        font = ImageFont.truetype(kanjiFontPath, int(font_scale))
    else:
        font = ImageFont.load_default()

    draw.text(
        (120, 5),
        status,
        font=font,
        fill=color_text,
    )

    draw.text(
        (120, 100),
        print_status,
        font=font,
        fill=color_text,
    )

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# PITCH / KATABU MARKING HELPERS
# ============================================================

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
            try:
                x_val = float(box.xywh[0][0].cpu().item())
                y_val = float(box.xywh[0][1].cpu().item())
                cls_id = int(box.cls.cpu().item())
            except Exception:
                try:
                    x_val = float(box.xywh[0][0])
                    y_val = float(box.xywh[0][1])
                    cls_id = int(box.cls)
                except Exception:
                    continue

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

    pitch = round(
        calclength(clip_point, mark_point) * pixelMultiplier_katabumarking,
        1,
    )

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


# ============================================================
# RESULT NORMALIZERS / NG RETURN
# ============================================================

def _normalize_results(results):
    if results is None:
        return []

    if isinstance(results, (list, tuple)):
        return list(results)

    try:
        return list(results)
    except TypeError:
        return [results]


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

    image = draw_status_text_PIL(
        image,
        status,
        message,
        size="normal",
    )

    resultPitch = [0] * len(pitchSpec)
    measuredPitch = [0] * len(pitchSpec)
    resultid = [0] * len(idSpec)

    return (
        image,
        measuredPitch,
        resultPitch,
        resultid,
        status,
        message,
    )


def calclength(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
    )
# import math
# from pathlib import Path

# import cv2
# import numpy as np
# import yaml
# from PIL import Image, ImageDraw, ImageFont

# from aikensa.core.scripts.img_processing.img_processing import (
#     check_id,
#     check_tolerance,
#     draw_bounding_box,
#     draw_pitch_line,
#     get_center,
# )

# specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"
# with open(specs_yaml_path, "r") as f:
#     all_specs = yaml.safe_load(f)
#     part_spec = all_specs["parts"]["P828387YA6A"]
#     global_spec = all_specs["global"]

# kanjiFontPath = global_spec["font_path"]

# pitchSpecRH = part_spec.get("pitchSpecRH", [14, 130, 132, 132, 59, 61, 97, 83, 20, 708])
# pitchSpecLH = part_spec.get("pitchSpecLH", [83, 97, 61, 59, 132, 132, 130, 14, 20, 708])
# tolerance_pitchRH = part_spec.get("tolerance_pitchRH", [3.0, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 2.0, 10.0])
# tolerance_pitchLH = part_spec.get("tolerance_pitchLH", [1.7, 1.7, 1.7, 3.0, 1.7, 1.7, 1.7, 1.7, 2.0, 10.0])
# idSpecRH = part_spec.get("idSpecRH", [0, 0, 0, 0, 0, 0, 0, 0, 0])
# idSpecLH = part_spec.get("idSpecLH", [1, 1, 1, 1, 1, 1, 1, 1, 1])

# color = tuple(part_spec.get("color_ok", [0, 255, 0]))
# color_ng = tuple(part_spec.get("color_ng", [200, 30, 50]))
# pixelMultiplier = float(part_spec.get("pixelMultiplier", 0.16413))
# pixelMultiplier_katabumarking = float(part_spec.get("pixelMultiplier_katabumarking", 0.16413))
# bbox_offset = int(part_spec.get("bbox_offset", 10))


# def partcheck(
#     image,
#     sahi_predictionList,
#     left_keypoint=None,
#     right_keypoint=None,
#     katabu_detection=None,
#     side="RH",
#     keypoint_crop_px=None,
#     katabu_crop_width=320,
# ):
#     if keypoint_crop_px is None:
#         keypoint_crop_px = int(part_spec.get("keypoint_crop_px", 1360))

#     if side == "RH":
#         pitchSpec = pitchSpecRH
#         tolerance_pitch = tolerance_pitchRH
#         expected_clip_id = 0
#         expected_ids = idSpecRH
#         endpoint = _extract_endpoint_from_keypoints(left_keypoint, x_start=0, image_width=image.shape[1])
#         katabu_x_offset = image.shape[1] - int(katabu_crop_width)
#     else:
#         pitchSpec = pitchSpecLH
#         tolerance_pitch = tolerance_pitchLH
#         expected_clip_id = 1
#         expected_ids = idSpecLH
#         endpoint = _extract_endpoint_from_keypoints(
#             right_keypoint,
#             x_start=-int(keypoint_crop_px),
#             image_width=image.shape[1],
#         )
#         katabu_x_offset = 0

#     sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

#     detectedid = []
#     clip_points = []
#     measured_inner = []
#     prev_center = None

#     for detection in sorted_detections:
#         cls_id = int(detection.category.id)
#         if cls_id != expected_clip_id:
#             continue

#         expected_id = expected_ids[len(detectedid)] if len(detectedid) < len(expected_ids) else None
#         detectedid.append(cls_id)

#         bbox = detection.bbox
#         x, y = get_center(bbox)
#         w = bbox.maxx - bbox.minx
#         h = bbox.maxy - bbox.miny

#         center = draw_bounding_box(
#             image,
#             x,
#             y,
#             w,
#             h,
#             [image.shape[1], image.shape[0]],
#             color=color if expected_id == cls_id else color_ng,
#             bbox_offset=bbox_offset,
#         )
#         clip_points.append(center)

#         if prev_center is not None:
#             measured_inner.append(calclength(prev_center, center) * pixelMultiplier)
#         prev_center = center

#     print(
#         f"[YA6A][{side}] clip_count={len(clip_points)}, clip_ids={detectedid}, "
#         f"expected_clip_id={expected_clip_id}, endpoint={'OK' if endpoint is not None else 'MISS'}"
#     )

#     if endpoint is None:
#         return _return_ng(image, pitchSpec, expected_ids, "製品は見つかりません")

#     if len(clip_points) == 0:
#         return _return_ng(image, pitchSpec, expected_ids, "クリップ未検出")

#     katabu_pitch, katabu_mark_point = _extract_katabu_pitch(
#         katabu_detection,
#         side=side,
#         x_offset=katabu_x_offset,
#     )
#     if katabu_pitch is None or katabu_mark_point is None:
#         return _return_ng(image, pitchSpec, expected_ids, "型部未検出")

#     if side == "RH":
#         endpoint_pitch = calclength(endpoint, clip_points[0]) * pixelMultiplier
#         trimmed_inner = measured_inner[:-1] if len(measured_inner) > 0 else []
#         measuredPitch = [endpoint_pitch] + trimmed_inner + [katabu_pitch]
#         draw_points = [endpoint] + clip_points + [katabu_mark_point]
#         skipped_segment_index = len(draw_points) - 2 if len(draw_points) >= 2 else None
#     else:
#         endpoint_pitch = calclength(clip_points[-1], endpoint) * pixelMultiplier
#         trimmed_inner = measured_inner[1:] if len(measured_inner) > 0 else []
#         measuredPitch = trimmed_inner + [endpoint_pitch] + [katabu_pitch]
#         draw_points = [katabu_mark_point] + clip_points + [endpoint]
#         skipped_segment_index = 0 if len(draw_points) >= 2 else None

#     totalLength = sum(measuredPitch)
#     measuredPitch.append(round(totalLength, 1))
#     measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

#     if len(measuredPitch) == len(pitchSpec):
#         resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
#     else:
#         resultPitch = [0] * len(pitchSpec)

#     if len(detectedid) == len(expected_ids):
#         resultid = check_id(detectedid, expected_ids)
#     else:
#         resultid = [0] * len(expected_ids)

#     status = "OK"
#     if any(result != 1 for result in resultPitch):
#         status = "NG"
#     if any(result != 1 for result in resultid):
#         status = "NG"

#     line_result_values = _build_line_result_values(
#         resultPitch,
#         skipped_segment_index=skipped_segment_index,
#     )

#     if len(draw_points) >= 2:
#         draw_pitch_line(image, draw_points, line_result_values, thickness=8)

#     print(
#         f"[YA6A][{side}] status={status}, measuredPitch={measuredPitch}, "
#         f"resultPitch={resultPitch}, resultid={resultid}, katabu_pitch={katabu_pitch}"
#     )

#     return image, measuredPitch, resultPitch, resultid, status, ""


# def _build_line_result_values(result_pitch, skipped_segment_index=None):
#     if not result_pitch:
#         return []

#     segment_results = list(result_pitch[:-1])
#     if skipped_segment_index is None:
#         return segment_results

#     line_results = []
#     result_idx = 0
#     total_segments = len(segment_results) + 1
#     for segment_idx in range(total_segments):
#         if segment_idx == skipped_segment_index:
#             line_results.append(None)
#             continue

#         if result_idx < len(segment_results):
#             line_results.append(segment_results[result_idx])
#             result_idx += 1

#     return line_results


# def _extract_katabu_pitch(katabu_detection, side, x_offset):
#     clip_points = []
#     mark_points = []

#     for result in _normalize_results(katabu_detection):
#         boxes = getattr(result, "boxes", None)
#         if boxes is None:
#             continue

#         for box in boxes:
#             x_val = float(box.xywh[0][0].cpu())
#             y_val = float(box.xywh[0][1].cpu())
#             w_val = float(box.xywh[0][2].cpu())
#             cls_id = int(box.cls.cpu())

#             if cls_id == 0:
#                 point = (int(x_offset + x_val), int(y_val))
#                 clip_points.append(point)
#             elif cls_id == 1:
#                 point = (int(x_offset + x_val), int(y_val))
#                 mark_points.append(point)

#     if not clip_points or not mark_points:
#         return None, None

#     clip_points = sorted(clip_points, key=lambda point: point[0])
#     mark_points = sorted(mark_points, key=lambda point: point[0])

#     if side == "RH":
#         clip_point = clip_points[-1]
#         mark_point = mark_points[-1]
#     else:
#         clip_point = clip_points[0]
#         mark_point = mark_points[0]

#     pitch = round(calclength(clip_point, mark_point) * pixelMultiplier_katabumarking, 1)
#     return pitch, mark_point


# def _extract_endpoint_from_keypoints(keypoint_result, x_start=0, image_width=0):
#     result = _normalize_result_object(keypoint_result)
#     if result is None or not hasattr(result, "keypoints"):
#         return None

#     try:
#         keypoints = result.keypoints
#         if keypoints is None or not hasattr(keypoints, "xy"):
#             return None

#         xy = keypoints.xy
#         if xy is None or len(xy) == 0:
#             return None

#         kpt = xy[0][0]
#         x_crop = float(kpt[0])
#         y_crop = float(kpt[1])

#         if x_start < 0:
#             x_original = image_width + x_start + x_crop
#         else:
#             x_original = x_start + x_crop

#         return int(x_original), int(y_crop)
#     except (AttributeError, IndexError, TypeError, ValueError):
#         return None


# def _normalize_results(results):
#     if results is None:
#         return []
#     if isinstance(results, (list, tuple)):
#         return list(results)
#     return list(results)


# def _normalize_result_object(keypoint_result):
#     if keypoint_result is None:
#         return None
#     if isinstance(keypoint_result, (list, tuple)):
#         if len(keypoint_result) == 0:
#             return None
#         return keypoint_result[0]
#     return keypoint_result


# def _return_ng(image, pitchSpec, idSpec, message):
#     status = "NG"
#     image = draw_status_text_PIL(image, status, message, size="normal")
#     resultPitch = [0] * len(pitchSpec)
#     measuredPitch = [0] * len(pitchSpec)
#     resultid = [0] * len(idSpec)
#     return image, measuredPitch, resultPitch, resultid, status, message


# def draw_status_text_PIL(image, status, print_status, size="normal"):
#     if size == "large":
#         font_scale = 130.0
#     elif size == "small":
#         font_scale = 50.0
#     else:
#         font_scale = 100.0

#     color_text = (10, 210, 60) if status == "OK" else (200, 30, 50)

#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(image_rgb)
#     draw = ImageDraw.Draw(img_pil)
#     font = ImageFont.truetype(kanjiFontPath, int(font_scale))

#     draw.text((120, 5), status, font=font, fill=color_text)
#     draw.text((120, 100), print_status, font=font, fill=color_text)
#     return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# def calclength(p1, p2):
#     return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
