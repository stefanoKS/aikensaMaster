import stat
import time
import math
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
import pygame
from PIL import ImageFont, ImageDraw, Image

from aikensa.core.scripts.img_processing.img_processing import map_keypoint_xcrop_to_original


# ============================================================
# LOAD SPECIFICATIONS
# ============================================================

specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"

with open(specs_yaml_path, "r", encoding="utf-8") as f:
    all_specs = yaml.safe_load(f) or {}
    part_spec = all_specs.get("parts", {}).get("P828387YA1A", {})
    global_spec = all_specs.get("global", {})


# ============================================================
# SOUND / FONT
# ============================================================

pygame.mixer.init()

ok_sound = pygame.mixer.Sound(global_spec["sounds"]["ok"])
ok_sound_v2 = pygame.mixer.Sound(global_spec["sounds"]["ok_v2"])
ng_sound = pygame.mixer.Sound(global_spec["sounds"]["ng"])
ng_sound_v2 = pygame.mixer.Sound(global_spec["sounds"]["ng_v2"])

kanjiFontPath = global_spec["font_path"]


# ============================================================
# PART SPEC DEFAULTS
# ============================================================

pitchSpecRH = part_spec.get(
    "pitchSpecRH",
    [15, 128, 95, 39, 120, 15, 412],
)

pitchSpecLH = part_spec.get(
    "pitchSpecLH",
    [15, 120, 39, 95, 128, 15, 412],
)

idSpecRH = part_spec.get(
    "idSpecRH",
    [0, 2, 0, 0, 0, 0],
)

idSpecLH = part_spec.get(
    "idSpecLH",
    [0, 0, 0, 0, 1, 0],
)

tolerance_pitch = part_spec.get(
    "tolerance_pitch",
    [1.7] * 7,
)

color = tuple(part_spec.get("color_ok", [0, 255, 0]))
text_offset = part_spec.get("text_offset", 40)
endoffset_y = 0
bbox_offset = part_spec.get("bbox_offset", 10)

segmentation_width = part_spec.get("segmentation_width", 1080)
pixelMultiplier = part_spec.get("pixelMultiplier", 0.1655)

DEBUG_ACCEPT_ANY_MARKING = True


# ============================================================
# MAIN CHECK FUNCTION
# ============================================================

def partcheck(
    image,
    sahi_predictionList,
    leftKeypoint,
    rightKeypoint,
    expected_side=None,
    *,
    ws_clip_hanire_model=None,
    clip_classifier_crop_px=None,
    clip_classifier_imgsz=None,
    hanire_ng_class_name=None,
    classifier_convert_bgr_to_rgb=None,
    debug_save_crops=None,
    debug_crop_dir=None,
    debug_crop_run_id=None,
):
    """
    P828387YA1A inspection.

    Original behavior kept:
        - LH/RH variant detection from marker class IDs
        - expected_side override
        - class 0 clip pitch measurement
        - class 1/2 marker drawing
        - endpoint keypoint extraction
        - endpoint fallback to first/last clip bbox edge
        - pitch check
        - ID check
        - status and NG reason handling

    Added:
        - Hanire classifier for clip class 0 only
        - Adjustable classifier crop size, default 128 x 128
        - Explicit classifier imgsz, default 128
        - Optional BGR -> RGB conversion before classification
        - Optional debug crop saving
    """

    # ------------------------------------------------------------
    # Resolve settings
    # Priority:
    #     inspection_thread argument > YAML > default
    # ------------------------------------------------------------

    clip_classifier_crop_px = int(
        _resolve_setting(
            call_value=clip_classifier_crop_px,
            yaml_key="clipClassifierCropPx",
            default=128,
        )
    )

    clip_classifier_imgsz = int(
        _resolve_setting(
            call_value=clip_classifier_imgsz,
            yaml_key="clipClassifierImgSz",
            default=128,
        )
    )

    hanire_ng_class_name = str(
        _resolve_setting(
            call_value=hanire_ng_class_name,
            yaml_key="hanireNgClassName",
            default="NG",
        )
    )

    classifier_convert_bgr_to_rgb = bool(
        _resolve_setting(
            call_value=classifier_convert_bgr_to_rgb,
            yaml_key="classifierConvertBgrToRgb",
            default=True,
        )
    )

    debug_save_crops = bool(
        _resolve_setting(
            call_value=debug_save_crops,
            yaml_key="debugSaveCrops",
            default=False,
        )
    )

    debug_crop_dir = _resolve_setting(
        call_value=debug_crop_dir,
        yaml_key="debugCropDir",
        default=None,
    )

    # ------------------------------------------------------------
    # Determine LH/RH variant
    # Current mapping:
    #     LH marker class = 1
    #     RH marker class = 2
    # ------------------------------------------------------------

    detected_variant_ids = set()

    for d in sahi_predictionList:
        cid = _safe_detection_class_id(d)
        if cid is not None:
            detected_variant_ids.add(cid)

    if expected_side in {"LH", "RH"}:
        side = expected_side
    elif 1 in detected_variant_ids:
        side = "LH"
    elif 2 in detected_variant_ids:
        side = "RH"
    else:
        side = "LH"

    if debug_crop_run_id is None:
        debug_crop_run_id = f"P828387YA1A_{side}_{int(time.time() * 1000)}"

    if side == "RH":
        pitchSpec = pitchSpecRH
        idSpec = idSpecRH
    else:
        pitchSpec = pitchSpecLH
        idSpec = idSpecLH

    raw_image = image.copy()

    sorted_detections = sorted(
        sahi_predictionList,
        key=lambda d: float(d.bbox.minx),
    )

    detectedid = []
    measuredPitch = []
    resultPitch = []
    deltaPitch = []
    resultid = []

    detectedposX = []
    detectedposY = []
    detectedWidth = []
    detectedMinX = []
    detectedMaxX = []

    prev_center = None

    flag_pitch_furyou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0
    flag_hole_notfound = 0

    leftmostPitch = 0
    rightmostPitch = 0

    status = "OK"
    print_status = ""
    ng_reason = ""

    hanire_ng_detections = []
    hanire_classifier_results = []

    # ------------------------------------------------------------
    # Extract endpoint keypoints
    # ------------------------------------------------------------

    left_edge = _extract_endpoint_from_keypoints(
        leftKeypoint,
        x_start=0,
        image_width=image.shape[1],
    )

    right_edge = _extract_endpoint_from_keypoints(
        rightKeypoint,
        x_start=-int(segmentation_width),
        image_width=image.shape[1],
    )

    print(
        "[P828387YA1A partcheck] start "
        f"variant={side} "
        f"clip_total={len(sorted_detections)} "
        f"clip_ids={[_safe_detection_class_id(d) for d in sorted_detections]} "
        f"left_keypoint={_summarize_keypoint_result(leftKeypoint, left_edge)} "
        f"right_keypoint={_summarize_keypoint_result(rightKeypoint, right_edge)} "
        f"hanire_crop_px={clip_classifier_crop_px} "
        f"hanire_imgsz={clip_classifier_imgsz} "
        f"bgr_to_rgb={classifier_convert_bgr_to_rgb} "
        f"debug_save_crops={debug_save_crops}"
    )

    # ------------------------------------------------------------
    # Process detections
    # Class 0 = normal clip
    # Class 1 = LH marker
    # Class 2 = RH marker
    # ------------------------------------------------------------

    clip_index = 0

    for det_index, detection in enumerate(sorted_detections):
        cid = _safe_detection_class_id(detection)
        cname = _safe_detection_class_name(detection)

        if cid is None:
            print(
                "[P828387YA1A partcheck] skip_detection "
                f"det_index={det_index} reason=class_id_none"
            )
            continue

        detectedid.append(cid)

        print(
            "[P828387YA1A partcheck] detection "
            f"det_index={det_index} "
            f"class_id={cid} "
            f"class_name={cname} "
            f"bbox=({float(detection.bbox.minx):.1f}, "
            f"{float(detection.bbox.miny):.1f}, "
            f"{float(detection.bbox.maxx):.1f}, "
            f"{float(detection.bbox.maxy):.1f})"
        )

        # --------------------------------------------------------
        # Clip detection
        # --------------------------------------------------------

        if cid == 0:
            bbox = detection.bbox

            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny

            detectedposX.append(x)
            detectedposY.append(y)
            detectedWidth.append(w)
            detectedMinX.append(bbox.minx)
            detectedMaxX.append(bbox.maxx)

            center = draw_bounding_box(
                image,
                x,
                y,
                w,
                h,
                [image.shape[1], image.shape[0]],
                color=color,
            )

            # ----------------------------------------------------
            # Hanire classifier
            # Only clip class 0 is checked.
            # ----------------------------------------------------

            debug_crop_prefix = (
                f"{debug_crop_run_id}_"
                f"{side}_"
                f"clip{clip_index:02d}_"
                f"hanire_"
                f"x{int(round(x))}_"
                f"y{int(round(y))}"
            )

            (
                is_hanire_ng,
                hanire_class_id,
                hanire_class_name,
                hanire_confidence,
                saved_crop_paths,
            ) = _classify_clip_center(
                raw_image=raw_image,
                x=x,
                y=y,
                model=ws_clip_hanire_model,
                crop_px=clip_classifier_crop_px,
                imgsz=clip_classifier_imgsz,
                ng_class_name=hanire_ng_class_name,
                convert_bgr_to_rgb=classifier_convert_bgr_to_rgb,
                debug_save_crops=debug_save_crops,
                debug_crop_dir=debug_crop_dir,
                debug_crop_prefix=debug_crop_prefix,
            )

            hanire_classifier_results.append(
                {
                    "clip_index": clip_index,
                    "center": center,
                    "class_id": hanire_class_id,
                    "class_name": hanire_class_name,
                    "confidence": hanire_confidence,
                    "is_ng": is_hanire_ng,
                    "saved_crop_paths": saved_crop_paths,
                }
            )

            print(
                "[P828387YA1A partcheck] hanire "
                f"clip_index={clip_index} "
                f"center=({x:.1f},{y:.1f}) "
                f"class_id={hanire_class_id} "
                f"class_name={hanire_class_name} "
                f"conf={hanire_confidence} "
                f"is_ng={is_hanire_ng} "
                f"crop_px={clip_classifier_crop_px} "
                f"imgsz={clip_classifier_imgsz}"
            )

            if saved_crop_paths:
                for saved_path in saved_crop_paths:
                    print(f"[P828387YA1A DEBUG_CROP] {saved_path}")

            if is_hanire_ng:
                flag_clip_hanire = 1
                hanire_ng_detections.append(detection)

                _draw_clip_ng_marker(
                    image=image,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    label="HANIRE",
                )

            if prev_center is not None:
                length = calclength(prev_center, center) * pixelMultiplier
                measuredPitch.append(length)

            prev_center = center
            clip_index += 1

        # --------------------------------------------------------
        # Marker class 1: LH
        # --------------------------------------------------------

        if cid == 1:
            bbox = detection.bbox

            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny

            draw_bounding_box(
                image,
                x,
                y,
                w,
                h,
                [image.shape[1], image.shape[0]],
                color=(255, 255, 0),
            )

        # --------------------------------------------------------
        # Marker class 2: RH
        # --------------------------------------------------------

        if cid == 2:
            bbox = detection.bbox

            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny

            draw_bounding_box(
                image,
                x,
                y,
                w,
                h,
                [image.shape[1], image.shape[0]],
                color=(255, 0, 255),
            )

    print(
        "[P828387YA1A partcheck] clip_summary "
        f"clip_count={len(detectedposX)} "
        f"clip_centers={[(round(x, 1), round(y, 1)) for x, y in zip(detectedposX, detectedposY)]} "
        f"hanire_ng_count={len(hanire_ng_detections)}"
    )

    # ------------------------------------------------------------
    # Endpoint fallback logic
    # ------------------------------------------------------------

    if len(detectedposX) > 0:
        leftmostCenter = (detectedposX[0], detectedposY[0])
        rightmostCenter = (detectedposX[-1], detectedposY[-1])

        left_clip_edge = (
            int(round(detectedMinX[0])),
            int(round(leftmostCenter[1])),
        )

        right_clip_edge = (
            int(round(detectedMaxX[-1])),
            int(round(rightmostCenter[1])),
        )

        fallback_left_edge = None
        fallback_right_edge = None

        if left_edge is None:
            fallback_left_edge = left_clip_edge
            left_edge = fallback_left_edge

        if right_edge is None:
            fallback_right_edge = right_clip_edge
            right_edge = fallback_right_edge

        # Sanity-check endpoint direction.
        # Endpoints must stay outside the clip chain.
        if left_edge is not None and left_edge[0] >= leftmostCenter[0]:
            fallback_left_edge = left_clip_edge

            print(
                "[P828387YA1A partcheck] adjust_left_edge "
                f"detected_left={left_edge} "
                f"leftmost_clip_center={leftmostCenter} "
                f"using_clip_edge={fallback_left_edge}"
            )

            left_edge = fallback_left_edge

        if right_edge is not None and right_edge[0] <= rightmostCenter[0]:
            fallback_right_edge = right_clip_edge

            print(
                "[P828387YA1A partcheck] adjust_right_edge "
                f"detected_right={right_edge} "
                f"rightmost_clip_center={rightmostCenter} "
                f"using_clip_edge={fallback_right_edge}"
            )

            right_edge = fallback_right_edge

        print(
            "[P828387YA1A partcheck] edge_source "
            f"left_edge={left_edge} "
            f"right_edge={right_edge} "
            f"fallback_left={fallback_left_edge} "
            f"fallback_right={fallback_right_edge}"
        )

        if left_edge is None or right_edge is None:
            status = "NG"
            print_status = "端部キーポイント未検出"
            ng_reason = print_status

            print(
                "[P828387YA1A partcheck] early_ng "
                f"reason={print_status} "
                f"left_edge={left_edge} "
                f"right_edge={right_edge} "
                f"clip_count={len(detectedposX)}"
            )

            image = draw_status_text_PIL(
                image,
                status,
                print_status,
                size="normal",
            )

            resultPitch = [0] * len(pitchSpec)
            measuredPitch = [0] * len(pitchSpec)
            resultid = [0] * len(idSpec)

            return image, measuredPitch, resultPitch, resultid, status, ng_reason

        leftmostPitch = calclength(leftmostCenter, left_edge) * pixelMultiplier
        rightmostPitch = calclength(rightmostCenter, right_edge) * pixelMultiplier

        measuredPitch.insert(0, leftmostPitch)
        measuredPitch.append(rightmostPitch)

        detectedposX.insert(0, left_edge[0])
        detectedposY.insert(0, left_edge[1])

        detectedposX.append(right_edge[0])
        detectedposY.append(right_edge[1])

    else:
        status = "NG"

        missing_both_keypoints = left_edge is None and right_edge is None

        if missing_both_keypoints:
            print_status = "製品は見つかりません"
        else:
            print_status = "検査NG"

        ng_reason = print_status

        print(
            "[P828387YA1A partcheck] early_ng "
            f"reason={print_status} "
            f"left_edge={left_edge} "
            f"right_edge={right_edge} "
            f"clip_count={len(detectedposX)}"
        )

        image = draw_status_text_PIL(
            image,
            status,
            print_status,
            size="normal",
        )

        resultPitch = [0] * len(pitchSpec)
        measuredPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)

        return image, measuredPitch, resultPitch, resultid, status, ng_reason

    # ------------------------------------------------------------
    # Add total length
    # ------------------------------------------------------------

    totalLength = sum(measuredPitch)
    measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    print(
        "[P828387YA1A partcheck] measured_pitch "
        f"values={measuredPitch}"
    )

    # ------------------------------------------------------------
    # Pitch and ID check
    # ------------------------------------------------------------

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(
            measuredPitch,
            pitchSpec,
            tolerance_pitch,
        )

        resultid = check_id(
            detectedid,
            idSpec,
        )

        if DEBUG_ACCEPT_ANY_MARKING and any(marker_id in detectedid for marker_id in (1, 2)):
            resultid = [1] * len(idSpec)

    else:
        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)

        print(
            "[P828387YA1A partcheck] pitch_length_mismatch "
            f"measured_len={len(measuredPitch)} "
            f"spec_len={len(pitchSpec)} "
            f"measuredPitch={measuredPitch} "
            f"pitchSpec={pitchSpec}"
        )

    # ------------------------------------------------------------
    # Final NG judgment
    # Original logic is preserved, with hanire added.
    # Priority:
    #     1. Pitch NG
    #     2. Clip / marker type NG
    #     3. Hanire NG
    # ------------------------------------------------------------

    if any(result != 1 for result in resultPitch):
        flag_pitch_furyou = 1
        status = "NG"
        ng_reason = "CLIP PITCH NG"
        print_status = "クリップピッチNG"

    if any(result != 1 for result in resultid):
        flag_clip_furyou = 1
        status = "NG"

        mismatch_indices = [
            index for index, result in enumerate(resultid)
            if result != 1 and index < len(resultPitch) - 1
        ]

        for mismatch_index in mismatch_indices:
            resultPitch[mismatch_index] = 0

        expected_marker_id = 2 if side == "RH" else 1
        opposite_marker_id = 1 if side == "RH" else 2

        if opposite_marker_id in detectedid and expected_marker_id not in detectedid:
            ng_reason = "マーキング色不良"
            print_status = ng_reason

        elif not ng_reason:
            ng_reason = "CLIP TYPE NG"
            print_status = "クリップ類不良"

    if status == "OK" and len(hanire_ng_detections) > 0:
        flag_clip_hanire = 1
        status = "NG"
        ng_reason = "CLIP HALF INSERTED"
        print_status = "クリップ半入れ不良"

    if status == "NG" and print_status:
        image = draw_status_text_PIL(
            image,
            status,
            print_status,
            size="normal",
        )

    print(
        "[P828387YA1A partcheck] final "
        f"pitch_spec={pitchSpec} "
        f"pitch_result={resultPitch} "
        f"detected_ids={detectedid} "
        f"id_spec={idSpec} "
        f"id_result={resultid} "
        f"status={status} "
        f"ng_reason={ng_reason} "
        f"hanire_ng_count={len(hanire_ng_detections)} "
        f"hanire_results={hanire_classifier_results}"
    )

    xy_pairs = list(zip(detectedposX, detectedposY))

    draw_pitch_line(
        image,
        xy_pairs,
        resultPitch,
        thickness=8,
    )

    return image, measuredPitch, resultPitch, resultid, status, ng_reason


# ============================================================
# SETTING HELPER
# ============================================================

def _resolve_setting(call_value, yaml_key, default):
    """
    Priority:
        1. value passed from inspection_thread
        2. YAML value
        3. default
    """
    if call_value is not None:
        return call_value

    return part_spec.get(yaml_key, default)


# ============================================================
# CLASSIFIER HELPERS
# ============================================================

def _crop_square_center(image, x, y, crop_px=128):
    """
    Fixed-size square crop around clip center.

    Returned crop is always crop_px x crop_px.
    If crop goes outside the image, BORDER_REPLICATE padding is used.
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
    convert_bgr_to_rgb=True,
    debug_save_crops=False,
    debug_crop_dir=None,
    debug_crop_prefix="crop",
):
    """
    Crop around clip center and classify hanire.

    Returns:
        is_ng, class_id, class_name, confidence, saved_crop_paths

    Important:
        raw_image is OpenCV BGR.

        If convert_bgr_to_rgb=True:
            crop is converted BGR -> RGB before YOLO classification.
            This is usually correct for Ultralytics classification models.

        If convert_bgr_to_rgb=False:
            BGR crop is passed directly.
    """
    saved_crop_paths = []

    if model is None:
        print("[P828387YA1A] hanire model is None, skipping hanire classifier")
        return False, None, None, None, saved_crop_paths

    crop_bgr = _crop_square_center(
        image=raw_image,
        x=x,
        y=y,
        crop_px=crop_px,
    )

    if convert_bgr_to_rgb:
        crop_for_inference = cv2.cvtColor(
            crop_bgr,
            cv2.COLOR_BGR2RGB,
        )
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

    The final NG judgment is by class name, not class ID.
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
            print("[P828387YA1A] classifier result has no probs")
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
        print(f"[P828387YA1A] classifier error: {e}")

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
    Save classifier crops for checking.

    Saves:
        *_bgr.png
            Original OpenCV BGR crop.

        *_inference_preview.png
            Actual inference image, saved with correct visible colors.
    """
    saved_paths = []

    try:
        if debug_crop_dir is None:
            debug_crop_dir = (
                Path(__file__).resolve().parent
                / "debug_crops"
                / "P828387YA1A"
            )
        else:
            debug_crop_dir = Path(debug_crop_dir)

        debug_crop_dir.mkdir(parents=True, exist_ok=True)

        class_id_safe = _safe_filename_text(class_id)
        class_name_safe = _safe_filename_text(class_name)

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

        if cv2.imwrite(str(bgr_path), crop_bgr):
            saved_paths.append(str(bgr_path))
        else:
            print(f"[P828387YA1A DEBUG_CROP] failed cv2.imwrite: {bgr_path}")

        if convert_bgr_to_rgb:
            inference_preview_bgr = cv2.cvtColor(
                crop_for_inference,
                cv2.COLOR_RGB2BGR,
            )
        else:
            inference_preview_bgr = crop_for_inference

        inference_path = debug_crop_dir / f"{filename_base}_inference_preview.png"

        if cv2.imwrite(str(inference_path), inference_preview_bgr):
            saved_paths.append(str(inference_path))
        else:
            print(f"[P828387YA1A DEBUG_CROP] failed cv2.imwrite: {inference_path}")

    except Exception as e:
        print(f"[P828387YA1A DEBUG_CROP] failed to save crop: {e}")

    return saved_paths


# ============================================================
# DRAW HELPERS
# ============================================================

def _draw_clip_ng_marker(image, x, y, w, h, label="HANIRE"):
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))

    radius = max(int((w + h) / 4) + 20, 18)

    cv2.circle(
        img=image,
        center=(x, y),
        radius=radius,
        color=(60, 60, 200),
        thickness=6,
        lineType=cv2.LINE_AA,
    )

    # Uncomment if you want visible English text.
    # cv2.putText(
    #     image,
    #     str(label),
    #     (x - radius, y - radius - 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1.0,
    #     (60, 60, 200),
    #     3,
    #     cv2.LINE_AA,
    # )


def draw_status_text_PIL(image, status, print_status, size="normal"):
    if size == "large":
        font_scale = 130.0
    elif size == "small":
        font_scale = 50.0
    else:
        font_scale = 100.0

    if status == "OK":
        color_text = (10, 210, 60)
    else:
        color_text = (200, 30, 50)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(
        kanjiFontPath,
        int(font_scale),
    )

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

    image = cv2.cvtColor(
        np.array(img_pil),
        cv2.COLOR_RGB2BGR,
    )

    return image


def draw_flag_status(image, flag_pitchfuryou, flag_clip_furyou, flag_clip_hanire):
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(
        kanjiFontPath,
        40,
    )

    color_text = (200, 10, 10)

    if flag_pitchfuryou == 1:
        draw.text(
            (120, 10),
            "クリップピッチ不良",
            font=font,
            fill=color_text,
        )

    if flag_clip_furyou == 1:
        draw.text(
            (120, 60),
            "クリップ類不良",
            font=font,
            fill=color_text,
        )

    if flag_clip_hanire == 1:
        draw.text(
            (120, 110),
            "クリップ半入れ",
            font=font,
            fill=color_text,
        )

    image = cv2.cvtColor(
        np.array(img_pil),
        cv2.COLOR_RGB2BGR,
    )

    return image


def draw_status_text(image, status, size="normal"):
    center_x = image.shape[1] // 2

    if size == "normal":
        top_y = 50
        font_scale = 5.0
    elif size == "small":
        top_y = 10
        font_scale = 2.0
    else:
        top_y = 50
        font_scale = 5.0

    font_thickness = 8
    outline_thickness = font_thickness + 2

    text_color = (255, 0, 0) if status == "NG" else (0, 255, 0)
    outline_color = (0, 0, 0)

    text_size, _ = cv2.getTextSize(
        status,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    text_x = center_x - text_size[0] // 2
    text_y = top_y + text_size[1]

    cv2.putText(
        image,
        status,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        outline_thickness,
    )

    cv2.putText(
        image,
        status,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
    )

    return image


def drawcircle(image, pos, class_id):
    if class_id == 0:
        circle_color = (60, 200, 60)
    elif class_id == 1:
        circle_color = (60, 60, 200)
    else:
        circle_color = (60, 60, 200)

    pos = (int(pos[0]), int(pos[1]))

    cv2.circle(
        img=image,
        center=pos,
        radius=30,
        color=circle_color,
        thickness=2,
        lineType=cv2.LINE_8,
    )

    return image


def drawbox(image, pos, length, offset=text_offset, font_scale=1.7, font_thickness=4):
    pos = (pos[0], pos[1])
    rectangle_bgr = (255, 255, 255)

    (text_width, text_height), _ = cv2.getTextSize(
        f"{length:.2f}",
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    top_left_x = pos[0] - text_width // 2 - 8
    top_left_y = pos[1] - text_height // 2 - 8 - offset

    bottom_right_x = pos[0] + text_width // 2 + 8
    bottom_right_y = pos[1] + text_height // 2 + 8 - offset

    cv2.rectangle(
        image,
        (top_left_x, top_left_y),
        (bottom_right_x, bottom_right_y),
        rectangle_bgr,
        -1,
    )

    return image


def drawtext(image, pos, length, font_scale=1.7, offset=text_offset, font_thickness=6):
    pos = (pos[0], pos[1])
    text = f"{length:.1f}"

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    text_x = pos[0] - text_width // 2
    text_y = pos[1] + text_height // 2 - offset

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 125, 20),
        font_thickness,
    )

    return image


def draw_bounding_box(
    image,
    x,
    y,
    w,
    h,
    img_size,
    color=(0, 255, 0),
    thickness=2,
    bbox_offset=bbox_offset,
):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x1 = int(x - w // 2) - bbox_offset
    y1 = int(y - h // 2) - bbox_offset
    x2 = int(x + w // 2) + bbox_offset
    y2 = int(y + h // 2) + bbox_offset

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    center_x, center_y = x, y

    return center_x, center_y


# ============================================================
# PITCH / ID / GEOMETRY HELPERS
# ============================================================

def check_id(detectedid, idSpec):
    result = [0] * len(idSpec)

    for i, (spec, detected) in enumerate(zip(idSpec, detectedid)):
        if spec == detected:
            result[i] = 1

    return result


def check_tolerance(checkedPitchResult, pitchSpec, pitchTolerance):
    result = [0] * len(pitchSpec)

    for i, (spec, detected) in enumerate(zip(pitchSpec, checkedPitchResult)):
        if abs(spec - detected) <= pitchTolerance[i]:
            result[i] = 1

    return result


def draw_pitch_line(image, xy_pairs, pitchresult, thickness=2):
    xy_pairs = [(int(x), int(y)) for x, y in xy_pairs]

    if len(xy_pairs) != 0:
        for i in range(len(xy_pairs) - 1):
            if i < len(pitchresult) and pitchresult[i] is not None:
                if pitchresult[i] == 1:
                    lineColor = (0, 255, 0)
                else:
                    lineColor = (255, 0, 0)

                cv2.line(
                    image,
                    xy_pairs[i],
                    xy_pairs[i + 1],
                    lineColor,
                    thickness,
                )

    return None


def calclength(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
    )


def get_center(bbox):
    center_x = bbox.minx + (bbox.maxx - bbox.minx) / 2
    center_y = bbox.miny + (bbox.maxy - bbox.miny) / 2

    return center_x, center_y


def print_bbox_structure(bbox):
    print(f"BoundingBox attributes: {dir(bbox)}")


def yolo_to_pixel(yolo_coords, img_shape):
    class_id, x, y, w, h, confidence = yolo_coords
    x_pixel = int(x * img_shape[1])
    y_pixel = int(y * img_shape[0])

    return x_pixel, y_pixel


# ============================================================
# EDGE / MASK HELPERS
# ============================================================

def create_masks(segmentation_result, orig_shape):
    mask = np.zeros(
        (orig_shape[0], orig_shape[1]),
        dtype=np.uint8,
    )

    for polygon in segmentation_result:
        polygon = np.array(
            [
                [int(x * orig_shape[1]), int(y * orig_shape[0])]
                for x, y in polygon
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255,
        )

    return mask


def find_edge_point_mask(
    image,
    mask,
    center,
    direction="None",
    Xoffsetval=0,
    Yoffsetval=0,
):
    x, y = center[0], center[1]

    min_x = 0
    max_x = image.shape[1] - 1

    if direction == "left":
        while x - Xoffsetval >= 0:
            if mask[int(y + Yoffsetval), int(x - Xoffsetval)] == 0:
                return x - Xoffsetval, y
            x -= 1

        return min_x, y

    if direction == "right":
        while x + Xoffsetval < image.shape[1]:
            if mask[int(y + Yoffsetval), int(x + Xoffsetval)] == 0:
                return x + Xoffsetval, y
            x += 1

        return max_x, y

    return None


def find_edge_point(
    image,
    center,
    direction="None",
    Xoffsetval=0,
    Yoffsetval=0,
):
    x, y = center[0], center[1]

    blur = 11
    brightness = 0
    contrast = 3.0
    lower_canny = 15
    upper_canny = 110

    adjusted_image = cv2.convertScaleAbs(
        image,
        alpha=contrast,
        beta=brightness,
    )

    gray_image = cv2.cvtColor(
        adjusted_image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred_image = cv2.GaussianBlur(
        gray_image,
        (blur | 1, blur | 1),
        0,
    )

    canny_img = cv2.Canny(
        blurred_image,
        lower_canny,
        upper_canny,
    )

    min_x = 0
    max_x = image.shape[1] - 1

    if direction == "left":
        while x - Xoffsetval >= 0:
            if canny_img[int(y + Yoffsetval), int(x - Xoffsetval)] == 255:
                return x - Xoffsetval, y
            x -= 1

        return min_x, y

    if direction == "right":
        while x + Xoffsetval < image.shape[1]:
            if canny_img[int(y + Yoffsetval), int(x + Xoffsetval)] == 255:
                return x + Xoffsetval, y
            x += 1

        return max_x, y

    return None


# ============================================================
# KEYPOINT HELPERS
# ============================================================

def _extract_endpoint_from_keypoints(keypoint_result, x_start=0, image_width=0):
    """
    Extract endpoint from keypoint detection result.

    Maps cropped coordinates back to original image space.
    """
    if keypoint_result is None:
        return None

    if isinstance(keypoint_result, (list, tuple)):
        result_items = keypoint_result
    else:
        result_items = [keypoint_result]

    for item in result_items:
        try:
            keypoints = getattr(item, "keypoints", None)

            if keypoints is None or not hasattr(keypoints, "xy") or keypoints.xy is None:
                continue

            xy = keypoints.xy

            if len(xy) == 0 or xy.shape[1] == 0:
                continue

            x_crop, y_crop = xy[0, 0].tolist()

            x_original, y_original = map_keypoint_xcrop_to_original(
                x_start=x_start,
                kpt_xy_crop=(x_crop, y_crop),
                img_width=image_width,
            )

            return int(round(x_original)), int(round(y_original))

        except Exception:
            continue

    return None


def _summarize_keypoint_result(keypoint_result, endpoint):
    if keypoint_result is None:
        return "none"

    if isinstance(keypoint_result, (list, tuple)):
        result_items = keypoint_result
    else:
        result_items = [keypoint_result]

    box_count = 0
    keypoint_count = 0

    for item in result_items:
        try:
            boxes = getattr(item, "boxes", None)
            box_count += len(boxes) if boxes is not None else 0
        except TypeError:
            pass

        try:
            keypoints = getattr(item, "keypoints", None)

            if keypoints is not None and hasattr(keypoints, "xy") and keypoints.xy is not None:
                keypoint_count += len(keypoints.xy)
        except TypeError:
            pass

    return f"boxes={box_count}, keypoints={keypoint_count}, endpoint={endpoint}"


# ============================================================
# DETECTION SAFE HELPERS
# ============================================================

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


# ============================================================
# SOUND
# ============================================================

def play_sound(status):
    if status == "OK":
        ok_sound_v2.play()
    elif status == "NG":
        ng_sound_v2.play()










# import stat
# import numpy as np
# import cv2
# import math
# import yaml
# import os
# import pygame
# import os
# from pathlib import Path
# from PIL import ImageFont, ImageDraw, Image
# from aikensa.core.scripts.img_processing.img_processing import map_keypoint_xcrop_to_original

# # Load specifications from YAML
# specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"
# with open(specs_yaml_path, 'r') as f:
#     all_specs = yaml.safe_load(f)
#     part_spec = all_specs['parts']['P828387YA1A']
#     global_spec = all_specs['global']

# pygame.mixer.init()
# ok_sound = pygame.mixer.Sound(global_spec['sounds']['ok'])
# ok_sound_v2 = pygame.mixer.Sound(global_spec['sounds']['ok_v2'])
# ng_sound = pygame.mixer.Sound(global_spec['sounds']['ng'])
# ng_sound_v2 = pygame.mixer.Sound(global_spec['sounds']['ng_v2'])
# kanjiFontPath = global_spec['font_path']

# # Load all specifications from YAML (this part has LH/RH variants)
# pitchSpecRH = part_spec.get('pitchSpecRH', [15, 128, 95, 39, 120, 15, 412])
# pitchSpecLH = part_spec.get('pitchSpecLH', [15, 120, 39, 95, 128, 15, 412])
# idSpecRH = part_spec.get('idSpecRH', [0, 2, 0, 0, 0, 0])
# idSpecLH = part_spec.get('idSpecLH', [0, 0, 0, 0, 1, 0])

# tolerance_pitch = part_spec.get('tolerance_pitch', [1.7] * 7)

# idSpec = []

# color = tuple(part_spec.get('color_ok', [0, 255, 0]))
# text_offset = part_spec.get('text_offset', 40)
# endoffset_y = 0
# bbox_offset = part_spec.get('bbox_offset', 10)

# segmentation_width = part_spec.get('segmentation_width', 1080)
# pixelMultiplier = part_spec.get('pixelMultiplier', 0.1655)
# DEBUG_ACCEPT_ANY_MARKING = True


# def partcheck(image, sahi_predictionList, leftKeypoint, rightKeypoint, expected_side=None):
#     # Determine variant based on detected color marking IDs
#     # Current spec mapping:
#     # - LH uses marker class 1
#     # - RH uses marker class 2
#     detected_variant_ids = set([int(d.category.id) for d in sahi_predictionList])
    
#     if expected_side in {"LH", "RH"}:
#         side = expected_side
#     elif 1 in detected_variant_ids:
#         side = "LH"
#     elif 2 in detected_variant_ids:
#         side = "RH"
#     else:
#         side = "LH"  # Default fallback

#     if side == "RH":
#         pitchSpec = pitchSpecRH
#         idSpec = idSpecRH
#     else:
#         pitchSpec = pitchSpecLH
#         idSpec = idSpecLH

#     sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

#     detectedid = []
#     measuredPitch = []
#     resultPitch = []
#     deltaPitch = []
#     resultid = []
#     detectedposX = []
#     detectedposY = []
#     detectedWidth = []
#     detectedMinX = []
#     detectedMaxX = []

#     prev_center = None

#     flag_pitch_furyou = 0
#     flag_clip_furyou = 0
#     flag_clip_hanire = 0
#     flag_hole_notfound = 0

#     leftmostPitch = 0
#     rightmostPitch = 0

#     status = "OK"
#     print_status = ""
#     ng_reason = ""

#     # Extract keypoint endpoints
#     left_edge = _extract_endpoint_from_keypoints(leftKeypoint, x_start=0, image_width=image.shape[1])
#     right_edge = _extract_endpoint_from_keypoints(rightKeypoint, x_start=-int(segmentation_width), image_width=image.shape[1])

#     print(
#         "[P828387YA1A partcheck] start "
#         f"variant={side} "
#         f"clip_total={len(sorted_detections)} "
#         f"clip_ids={[int(d.category.id) for d in sorted_detections]} "
#         f"left_keypoint={_summarize_keypoint_result(leftKeypoint, left_edge)} "
#         f"right_keypoint={_summarize_keypoint_result(rightKeypoint, right_edge)}"
#     )

#     for i, detection in enumerate(sorted_detections):
#         detectedid.append(detection.category.id)
#         if detection.category.id == 0:
#             bbox = detection.bbox
#             x, y = get_center(bbox)
#             w = bbox.maxx - bbox.minx
#             h = bbox.maxy - bbox.miny

#             detectedposX.append(x)
#             detectedposY.append(y)
#             detectedWidth.append(w)
#             detectedMinX.append(bbox.minx)
#             detectedMaxX.append(bbox.maxx)

#             #id 0 object is white clip
#             center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)

#             if prev_center is not None:
#                 length = calclength(prev_center, center)*pixelMultiplier
#                 measuredPitch.append(length)
#             prev_center = center

#         # Draw color marking indicators for variant markers.
#         if detection.category.id == 1:
#             bbox = detection.bbox
#             x, y = get_center(bbox)
#             w = bbox.maxx - bbox.minx
#             h = bbox.maxy - bbox.miny
#             center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=(255, 255, 0))  # Marker class 1 (LH)

#         if detection.category.id == 2:
#             bbox = detection.bbox
#             x, y = get_center(bbox)
#             w = bbox.maxx - bbox.minx
#             h = bbox.maxy - bbox.miny
#             center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=(255, 0, 255))  # Marker class 2 (RH)


#     print(
#         "[P828387YA1A partcheck] clip_summary "
#         f"clip_count={len(detectedposX)} "
#         f"clip_centers={[(round(x, 1), round(y, 1)) for x, y in zip(detectedposX, detectedposY)]}"
#     )

#     # Check if clip detections exist. If one endpoint keypoint is missing, fall back to
#     # the first/last clip bbox edge like widget 36 does.
#     if len(detectedposX) > 0:
#         leftmostCenter = (detectedposX[0], detectedposY[0])
#         rightmostCenter = (detectedposX[-1], detectedposY[-1])
#         left_clip_edge = (int(round(detectedMinX[0])), int(round(leftmostCenter[1])))
#         right_clip_edge = (int(round(detectedMaxX[-1])), int(round(rightmostCenter[1])))

#         fallback_left_edge = None
#         fallback_right_edge = None

#         if left_edge is None:
#             fallback_left_edge = left_clip_edge
#             left_edge = fallback_left_edge

#         if right_edge is None:
#             fallback_right_edge = right_clip_edge
#             right_edge = fallback_right_edge

#         # Sanity-check endpoint direction. Endpoints must remain outside the clip chain.
#         if left_edge is not None and left_edge[0] >= leftmostCenter[0]:
#             fallback_left_edge = left_clip_edge
#             print(
#                 "[P828387YA1A partcheck] adjust_left_edge "
#                 f"detected_left={left_edge} "
#                 f"leftmost_clip_center={leftmostCenter} "
#                 f"using_clip_edge={fallback_left_edge}"
#             )
#             left_edge = fallback_left_edge

#         if right_edge is not None and right_edge[0] <= rightmostCenter[0]:
#             fallback_right_edge = right_clip_edge
#             print(
#                 "[P828387YA1A partcheck] adjust_right_edge "
#                 f"detected_right={right_edge} "
#                 f"rightmost_clip_center={rightmostCenter} "
#                 f"using_clip_edge={fallback_right_edge}"
#             )
#             right_edge = fallback_right_edge

#         print(
#             "[P828387YA1A partcheck] edge_source "
#             f"left_edge={left_edge} "
#             f"right_edge={right_edge} "
#             f"fallback_left={fallback_left_edge} "
#             f"fallback_right={fallback_right_edge}"
#         )

#         if left_edge is None or right_edge is None:
#             status = "NG"
#             print_status = "端部キーポイント未検出"

#             print(
#                 "[P828387YA1A partcheck] early_ng "
#                 f"reason={print_status} "
#                 f"left_edge={left_edge} "
#                 f"right_edge={right_edge} "
#                 f"clip_count={len(detectedposX)}"
#             )

#             image = draw_status_text_PIL(image, status, print_status, size="normal")

#             resultPitch = [0] * len(pitchSpec)
#             measuredPitch = [0] * len(pitchSpec)
#             resultid = [0] * len(idSpec)

#             return image, measuredPitch, resultPitch, resultid, status, ng_reason

#         leftmostPitch = calclength(leftmostCenter, left_edge)*pixelMultiplier
#         rightmostPitch = calclength(rightmostCenter, right_edge)*pixelMultiplier

#         #append the leftmost and rightmost pitch to the measuredPitch
#         measuredPitch.insert(0, leftmostPitch)
#         measuredPitch.append(rightmostPitch)
#         #Reappend the leftmost and rightmost center to the detectedposX and detectedposY
#         detectedposX.insert(0, left_edge[0])
#         detectedposY.insert(0, left_edge[1])
#         detectedposX.append(right_edge[0])
#         detectedposY.append(right_edge[1])
#     else:
#         status = "NG"
#         missing_both_keypoints = left_edge is None and right_edge is None

#         if missing_both_keypoints:
#             print_status = "製品は見つかりません"
#         else:
#             print_status = "検査NG"

#         print(
#             "[P828387YA1A partcheck] early_ng "
#             f"reason={print_status} "
#             f"left_edge={left_edge} "
#             f"right_edge={right_edge} "
#             f"clip_count={len(detectedposX)}"
#         )

#         image = draw_status_text_PIL(image, status, print_status, size="normal")

#         resultPitch = [0] * len(pitchSpec)
#         measuredPitch = [0] * len(pitchSpec)
#         resultid = [0] * len(idSpec)

#         return image, measuredPitch, resultPitch, resultid, status, ng_reason


#     #add total length
#     #round the value to 1 decimal
#     totalLength = sum(measuredPitch)
#     measuredPitch.append(round(totalLength, 1))
#     measuredPitch = [round(pitch, 1) for pitch in measuredPitch]
#     print(
#         "[P828387YA1A partcheck] measured_pitch "
#         f"values={measuredPitch}"
#     )

#     if len(measuredPitch) == len(pitchSpec):
#         resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
#         resultid = check_id(detectedid, idSpec)
#         if DEBUG_ACCEPT_ANY_MARKING and any(marker_id in detectedid for marker_id in (1, 2)):
#             resultid = [1] * len(idSpec)

#     if len(measuredPitch) != len(pitchSpec):
#         resultPitch = [0] * len(pitchSpec)

#     if any(result != 1 for result in resultPitch):
#         flag_pitch_furyou = 1
#         status = "NG"
#         ng_reason = "CLIP PITCH NG"
#         print_status = "クリップピッチNG"

#     # print(f"Result ID: {resultid}")


#     if any(result != 1 for result in resultid):
#         flag_clip_furyou = 1
#         status = "NG"

#         mismatch_indices = [
#             index for index, result in enumerate(resultid)
#             if result != 1 and index < len(resultPitch) - 1
#         ]
#         for mismatch_index in mismatch_indices:
#             resultPitch[mismatch_index] = 0

#         expected_marker_id = 2 if side == "RH" else 1
#         opposite_marker_id = 1 if side == "RH" else 2
#         if opposite_marker_id in detectedid and expected_marker_id not in detectedid:
#             ng_reason = "マーキング色不良"
#             print_status = ng_reason

#         elif not ng_reason:
#             ng_reason = "CLIP TYPE NG"
#             print_status = "クリップ類不良"

#     if status == "NG" and print_status:
#         image = draw_status_text_PIL(image, status, print_status, size="normal")

#     print(
#         "[P828387YA1A partcheck] final "
#         f"pitch_spec={pitchSpec} "
#         f"pitch_result={resultPitch} "
#         f"detected_ids={detectedid} "
#         f"id_spec={idSpec} "
#         f"id_result={resultid} "
#         f"status={status} "
#         f"ng_reason={ng_reason}"
#     )

#     xy_pairs = list(zip(detectedposX, detectedposY))
#     draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)
    
#     return image, measuredPitch, resultPitch, resultid, status, ng_reason

# def draw_status_text_PIL(image, status, print_status, size = "normal"):

#     if size == "large":
#         font_scale = 130.0
#     if size == "normal":
#         font_scale = 100.0
#     elif size == "small":
#         font_scale = 50.0

#     if status == "OK":
#         color = (10, 210, 60)

#     elif status == "NG":
#         color = (200, 30, 50)
    
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(image_rgb)
#     draw = ImageDraw.Draw(img_pil)
#     font = ImageFont.truetype(kanjiFontPath, font_scale)

#     draw.text((120, 5), status, font=font, fill=color)  
#     draw.text((120, 100), print_status, font=font, fill=color)
#     image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

#     return image

# def create_masks(segmentation_result, orig_shape):
#     mask = np.zeros((orig_shape[0], orig_shape[1]), dtype=np.uint8)
#     for polygon in segmentation_result:
#         polygon = np.array([[int(x * orig_shape[1]), int(y * orig_shape[0])] for x, y in polygon], dtype=np.int32)
#         cv2.fillPoly(mask, [polygon], 255)
#     return mask

# def play_sound(status):
#     if status == "OK":
#         # ok_sound.play()
#         ok_sound_v2.play()
#     elif status == "NG":
#         # ng_sound.play()
#         ng_sound_v2.play()

# def get_center(bbox):
#     center_x = bbox.minx + (bbox.maxx - bbox.minx) / 2
#     center_y = bbox.miny + (bbox.maxy - bbox.miny) / 2
#     return center_x, center_y

# def print_bbox_structure(bbox):
#     print(f"BoundingBox attributes: {dir(bbox)}")

# def draw_flag_status(image, flag_pitchfuryou, flag_clip_furyou, flag_clip_hanire):
    
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(image_rgb)
#     draw = ImageDraw.Draw(img_pil)
#     font = ImageFont.truetype(kanjiFontPath, 40)
#     color=(200,10,10)
#     if flag_pitchfuryou == 1:
#         draw.text((120, 10), u"クリップピッチ不良", font=font, fill=color)  
#     if flag_clip_furyou == 1:
#         draw.text((120, 60), u"クリップ類不良", font=font, fill=color)  
#     if flag_clip_hanire == 1:
#         draw.text((120, 110), u"クリップ半入れ", font=font, fill=color)
    
#     # Convert back to BGR for OpenCV compatibility
#     image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

#     return image

# def check_id(detectedid, idSpec):
#     result = [0] * len(idSpec)
#     for i, (spec, detected) in enumerate(zip(idSpec, detectedid)):
#         if spec == detected:
#             result[i] = 1
#     return result

# def draw_pitch_line(image, xy_pairs, pitchresult, thickness=2):
#     xy_pairs = [(int(x), int(y)) for x, y in xy_pairs]

#     if len(xy_pairs) != 0:
#         for i in range(len(xy_pairs) - 1):
#             if i < len(pitchresult) and pitchresult[i] is not None:
#                 if pitchresult[i] == 1:
#                     lineColor = (0, 255, 0)
#                 else:
#                     lineColor = (255, 0, 0)

#                 cv2.line(image, xy_pairs[i], xy_pairs[i+1], lineColor, thickness)

#     return None


# #add "OK" and "NG"
# def draw_status_text(image, status, size = "normal"):
#     # Define the position for the text: Center top of the image
#     center_x = image.shape[1] // 2
#     if size == "normal":
#         top_y = 50  # Adjust this value to change the vertical position
#         font_scale = 5.0  # Increased font scale for bigger text

#     elif size == "small":
#         top_y = 10
#         font_scale = 2.0  # Increased font scale for bigger text
    

#     # Text properties
    
#     font_thickness = 8  # Increased font thickness for bolder text
#     outline_thickness = font_thickness + 2  # Slightly thicker for the outline
#     text_color = (255, 0, 0) if status == "NG" else (0, 255, 0)  # Red for NG, Green for OK
#     outline_color = (0, 0, 0)  # Black for the outline

#     # Calculate text size and position
#     text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#     text_x = center_x - text_size[0] // 2
#     text_y = top_y + text_size[1]

#     # Draw the outline
#     cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness)

#     # Draw the text over the outline
#     cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
#     return image


# def check_tolerance(checkedPitchResult, pitchSpec, pitchTolerance):
#     result = [0] * len(pitchSpec)
#     for i, (spec, detected) in enumerate(zip(pitchSpec, checkedPitchResult)):
#         if abs(spec - detected) <= pitchTolerance[i]:
#             result[i] = 1
#     return result

# def yolo_to_pixel(yolo_coords, img_shape):
#     class_id, x, y, w, h, confidence = yolo_coords
#     x_pixel = int(x * img_shape[1])
#     y_pixel = int(y * img_shape[0])
#     return x_pixel, y_pixel

# def find_edge_point_mask(image, mask, center, direction="None", Xoffsetval = 0, Yoffsetval = 0):
#     x, y = center[0], center[1]

#     min_x = 0
#     max_x = image.shape[1] - 1

#     if direction == "left":
#         while x - Xoffsetval >= 0:
#             if mask[int(y + Yoffsetval), int(x - Xoffsetval)] == 0:  # Found an edge
#                 return x - Xoffsetval, y
#             x -= 1
#         return min_x, y

#     if direction == "right":
#         while x + Xoffsetval < image.shape[1]:
#             if mask[int(y + Yoffsetval), int(x + Xoffsetval)] == 0:  # Found an edge
#                 return x + Xoffsetval, y
#             x += 1
#         return max_x, y

#     return None  # If an invalid direction is provided

# def find_edge_point(image, center, direction="None", Xoffsetval = 0, Yoffsetval = 0):
#     x, y = center[0], center[1]
#     blur = 11
#     brightness = 0
#     contrast = 3.0
#     lower_canny = 15
#     upper_canny = 110

#     # Apply adjustments
#     adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
#     gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
#     blurred_image = cv2.GaussianBlur(gray_image, (blur | 1, blur | 1), 0)
#     canny_img = cv2.Canny(blurred_image, lower_canny, upper_canny)

#     # cv2.imwrite(f"1adjusted_image_{direction}.jpg", adjusted_image)
#     # cv2.imwrite(f"2gray_image_{direction}.jpg", gray_image)
#     # cv2.imwrite(f"3blurred_image_{direction}.jpg", blurred_image)
#     # cv2.imwrite(f"4canny_debug_{direction}.jpg", canny_img)
#     min_x = 0
#     max_x = image.shape[1] - 1

#     if direction == "left":
#         while x - Xoffsetval >= 0:
#             if canny_img[int(y + Yoffsetval), int(x - Xoffsetval)] == 255:  # Found an edge
#                 return x - Xoffsetval, y
#             x -= 1
#         return min_x, y

#     if direction == "right":
#         while x + Xoffsetval < image.shape[1]:
#             if canny_img[int(y + Yoffsetval), int(x + Xoffsetval)] == 255:  # Found an edge
#                 return x + Xoffsetval, y
#             x += 1
#         return max_x, y

#     return None  # If an invalid direction is provided

# def drawcircle(image, pos, class_id): #for ire and hanire
#     #draw either green or red circle depends on the detection
#     if class_id == 0:
#         color = (60, 200, 60)
#     elif class_id == 1:
#         color = (60, 60, 200)
#     #check if pos is tupple
#     pos = (int(pos[0]), int(pos[1]))

#     cv2.circle(img=image, center=pos, radius=30, color=color, thickness=2, lineType=cv2.LINE_8)

#     return image

# def drawbox(image, pos, length, offset = text_offset, font_scale=1.7, font_thickness=4):
#     pos = (pos[0], pos[1])
#     rectangle_bgr = (255, 255, 255)
#     (text_width, text_height), _ = cv2.getTextSize(f"{length:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
#     top_left_x = pos[0] - text_width // 2 - 8
#     top_left_y = pos[1] - text_height // 2 - 8 - offset
#     bottom_right_x = pos[0] + text_width // 2 + 8
#     bottom_right_y = pos[1] + text_height // 2 + 8 - offset
    
#     cv2.rectangle(image, (top_left_x, top_left_y),
#                   (bottom_right_x, bottom_right_y),
#                   rectangle_bgr, -1)
    
#     return image

# def drawtext(image, pos, length, font_scale=1.7, offset = text_offset, font_thickness=6):
#     pos = (pos[0], pos[1])
#     font_scale = font_scale
#     text = f"{length:.1f}"
#     (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
#     text_x = pos[0] - text_width // 2
#     text_y = pos[1] + text_height // 2 - offset
    
#     cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 125, 20), font_thickness)
#     return image

# def calclength(p1, p2):
#     length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
#     return length

# def _extract_endpoint_from_keypoints(keypoint_result, x_start=0, image_width=0):
#     """
#     Extract endpoint (keypoint position) from keypoint detection result.
#     Maps cropped coordinates back to original image space.
    
#     Args:
#         keypoint_result: YOLO keypoint detection result object
#         x_start: x-coordinate offset in original image (0 for left edge, -840 for right edge)
#         image_width: full width of original image
    
#     Returns:
#         Tuple (x, y) of endpoint in original image coordinates, or None if no keypoints found
#     """
#     if keypoint_result is None:
#         return None

#     result_items = keypoint_result if isinstance(keypoint_result, (list, tuple)) else [keypoint_result]

#     for item in result_items:
#         try:
#             keypoints = getattr(item, 'keypoints', None)
#             if keypoints is None or not hasattr(keypoints, 'xy') or keypoints.xy is None:
#                 continue

#             xy = keypoints.xy
#             if len(xy) == 0 or xy.shape[1] == 0:
#                 continue

#             x_crop, y_crop = xy[0, 0].tolist()
#             x_original, y_original = map_keypoint_xcrop_to_original(
#                 x_start=x_start,
#                 kpt_xy_crop=(x_crop, y_crop),
#                 img_width=image_width,
#             )
#             return int(round(x_original)), int(round(y_original))
#         except Exception:
#             continue

#     return None


# def _summarize_keypoint_result(keypoint_result, endpoint):
#     if keypoint_result is None:
#         return "none"

#     result_items = keypoint_result if isinstance(keypoint_result, (list, tuple)) else [keypoint_result]

#     box_count = 0
#     keypoint_count = 0

#     for item in result_items:
#         try:
#             boxes = getattr(item, 'boxes', None)
#             box_count += len(boxes) if boxes is not None else 0
#         except TypeError:
#             pass

#         try:
#             keypoints = getattr(item, 'keypoints', None)
#             if keypoints is not None and hasattr(keypoints, 'xy') and keypoints.xy is not None:
#                 keypoint_count += len(keypoints.xy)
#         except TypeError:
#             pass

#     return f"boxes={box_count}, keypoints={keypoint_count}, endpoint={endpoint}"

# def draw_bounding_box(image, x, y, w, h, img_size, color=(0, 255, 0), thickness=2, bbox_offset=bbox_offset):
#     x = int(x)
#     y = int(y)
#     w = int(w)
#     h = int(h)

#     x1, y1 = int(x - w // 2) - bbox_offset, int(y - h // 2) - bbox_offset
#     x2, y2 = int(x + w // 2) + bbox_offset, int(y + h // 2) + bbox_offset
#     cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
#     center_x, center_y = x, y
#     return (center_x, center_y)

# # class BoundingBox:
# #     def __init__(self, minx, miny, maxx, maxy):
# #         self.minx = minx
# #         self.miny = miny
# #         self.maxx = maxx
# #         self.maxy = maxy

# # class PredictionScore:
# #     def __init__(self, value):
# #         self.value = value

# # class Category:
# #     def __init__(self, id, name):
# #         self.id = id
# #         self.name = name

# # class ObjectPrediction:
# #     def __init__(self, bbox, score, category):
# #         self.bbox = bbox
# #         self.score = score
# #         self.category = category