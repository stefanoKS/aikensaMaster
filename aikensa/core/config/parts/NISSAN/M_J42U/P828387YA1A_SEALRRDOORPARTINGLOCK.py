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
from aikensa.core.scripts.img_processing.img_processing import map_keypoint_xcrop_to_original

# Load specifications from YAML
specs_yaml_path = Path(__file__).resolve().parents[3] / "parts_specifications.yaml"
with open(specs_yaml_path, 'r') as f:
    all_specs = yaml.safe_load(f)
    part_spec = all_specs['parts']['P828387YA1A']
    global_spec = all_specs['global']

pygame.mixer.init()
ok_sound = pygame.mixer.Sound(global_spec['sounds']['ok'])
ok_sound_v2 = pygame.mixer.Sound(global_spec['sounds']['ok_v2'])
ng_sound = pygame.mixer.Sound(global_spec['sounds']['ng'])
ng_sound_v2 = pygame.mixer.Sound(global_spec['sounds']['ng_v2'])
kanjiFontPath = global_spec['font_path']

# Load all specifications from YAML (this part has LH/RH variants)
pitchSpecRH = part_spec.get('pitchSpecRH', [15, 128, 95, 39, 120, 15, 412])
pitchSpecLH = part_spec.get('pitchSpecLH', [15, 120, 39, 95, 128, 15, 412])
idSpecRH = part_spec.get('idSpecRH', [0, 2, 0, 0, 0, 0])
idSpecLH = part_spec.get('idSpecLH', [0, 0, 0, 0, 1, 0])

tolerance_pitch = part_spec.get('tolerance_pitch', [1.7] * 7)

idSpec = []

color = tuple(part_spec.get('color_ok', [0, 255, 0]))
text_offset = part_spec.get('text_offset', 40)
endoffset_y = 0
bbox_offset = part_spec.get('bbox_offset', 10)

segmentation_width = part_spec.get('segmentation_width', 1080)
pixelMultiplier = part_spec.get('pixelMultiplier', 0.1655)
DEBUG_ACCEPT_ANY_MARKING = True


def partcheck(image, sahi_predictionList, leftKeypoint, rightKeypoint, expected_side=None):
    # Determine variant based on detected color marking IDs
    # Current spec mapping:
    # - LH uses marker class 1
    # - RH uses marker class 2
    detected_variant_ids = set([int(d.category.id) for d in sahi_predictionList])
    
    if expected_side in {"LH", "RH"}:
        side = expected_side
    elif 1 in detected_variant_ids:
        side = "LH"
    elif 2 in detected_variant_ids:
        side = "RH"
    else:
        side = "LH"  # Default fallback

    if side == "RH":
        pitchSpec = pitchSpecRH
        idSpec = idSpecRH
    else:
        pitchSpec = pitchSpecLH
        idSpec = idSpecLH

    sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

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

    # Extract keypoint endpoints
    left_edge = _extract_endpoint_from_keypoints(leftKeypoint, x_start=0, image_width=image.shape[1])
    right_edge = _extract_endpoint_from_keypoints(rightKeypoint, x_start=-int(segmentation_width), image_width=image.shape[1])

    print(
        "[P828387YA1A partcheck] start "
        f"variant={side} "
        f"clip_total={len(sorted_detections)} "
        f"clip_ids={[int(d.category.id) for d in sorted_detections]} "
        f"left_keypoint={_summarize_keypoint_result(leftKeypoint, left_edge)} "
        f"right_keypoint={_summarize_keypoint_result(rightKeypoint, right_edge)}"
    )

    for i, detection in enumerate(sorted_detections):
        detectedid.append(detection.category.id)
        if detection.category.id == 0:
            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny

            detectedposX.append(x)
            detectedposY.append(y)
            detectedWidth.append(w)
            detectedMinX.append(bbox.minx)
            detectedMaxX.append(bbox.maxx)

            #id 0 object is white clip
            center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)

            if prev_center is not None:
                length = calclength(prev_center, center)*pixelMultiplier
                measuredPitch.append(length)
            prev_center = center

        # Draw color marking indicators for variant markers.
        if detection.category.id == 1:
            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny
            center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=(255, 255, 0))  # Marker class 1 (LH)

        if detection.category.id == 2:
            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny
            center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=(255, 0, 255))  # Marker class 2 (RH)


    print(
        "[P828387YA1A partcheck] clip_summary "
        f"clip_count={len(detectedposX)} "
        f"clip_centers={[(round(x, 1), round(y, 1)) for x, y in zip(detectedposX, detectedposY)]}"
    )

    # Check if clip detections exist. If one endpoint keypoint is missing, fall back to
    # the first/last clip bbox edge like widget 36 does.
    if len(detectedposX) > 0:
        leftmostCenter = (detectedposX[0], detectedposY[0])
        rightmostCenter = (detectedposX[-1], detectedposY[-1])
        left_clip_edge = (int(round(detectedMinX[0])), int(round(leftmostCenter[1])))
        right_clip_edge = (int(round(detectedMaxX[-1])), int(round(rightmostCenter[1])))

        fallback_left_edge = None
        fallback_right_edge = None

        if left_edge is None:
            fallback_left_edge = left_clip_edge
            left_edge = fallback_left_edge

        if right_edge is None:
            fallback_right_edge = right_clip_edge
            right_edge = fallback_right_edge

        # Sanity-check endpoint direction. Endpoints must remain outside the clip chain.
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

            print(
                "[P828387YA1A partcheck] early_ng "
                f"reason={print_status} "
                f"left_edge={left_edge} "
                f"right_edge={right_edge} "
                f"clip_count={len(detectedposX)}"
            )

            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * len(pitchSpec)
            measuredPitch = [0] * len(pitchSpec)
            resultid = [0] * len(idSpec)

            return image, measuredPitch, resultPitch, resultid, status, ng_reason

        leftmostPitch = calclength(leftmostCenter, left_edge)*pixelMultiplier
        rightmostPitch = calclength(rightmostCenter, right_edge)*pixelMultiplier

        #append the leftmost and rightmost pitch to the measuredPitch
        measuredPitch.insert(0, leftmostPitch)
        measuredPitch.append(rightmostPitch)
        #Reappend the leftmost and rightmost center to the detectedposX and detectedposY
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

        print(
            "[P828387YA1A partcheck] early_ng "
            f"reason={print_status} "
            f"left_edge={left_edge} "
            f"right_edge={right_edge} "
            f"clip_count={len(detectedposX)}"
        )

        image = draw_status_text_PIL(image, status, print_status, size="normal")

        resultPitch = [0] * len(pitchSpec)
        measuredPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)

        return image, measuredPitch, resultPitch, resultid, status, ng_reason


    #add total length
    #round the value to 1 decimal
    totalLength = sum(measuredPitch)
    measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]
    print(
        "[P828387YA1A partcheck] measured_pitch "
        f"values={measuredPitch}"
    )

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
        resultid = check_id(detectedid, idSpec)
        if DEBUG_ACCEPT_ANY_MARKING and any(marker_id in detectedid for marker_id in (1, 2)):
            resultid = [1] * len(idSpec)

    if len(measuredPitch) != len(pitchSpec):
        resultPitch = [0] * len(pitchSpec)

    if any(result != 1 for result in resultPitch):
        flag_pitch_furyou = 1
        status = "NG"
        ng_reason = "CLIP PITCH NG"
        print_status = "クリップピッチNG"

    # print(f"Result ID: {resultid}")


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

    if status == "NG" and print_status:
        image = draw_status_text_PIL(image, status, print_status, size="normal")

    print(
        "[P828387YA1A partcheck] final "
        f"pitch_spec={pitchSpec} "
        f"pitch_result={resultPitch} "
        f"detected_ids={detectedid} "
        f"id_spec={idSpec} "
        f"id_result={resultid} "
        f"status={status} "
        f"ng_reason={ng_reason}"
    )

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)
    
    return image, measuredPitch, resultPitch, resultid, status, ng_reason

def draw_status_text_PIL(image, status, print_status, size = "normal"):

    if size == "large":
        font_scale = 130.0
    if size == "normal":
        font_scale = 100.0
    elif size == "small":
        font_scale = 50.0

    if status == "OK":
        color = (10, 210, 60)

    elif status == "NG":
        color = (200, 30, 50)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(kanjiFontPath, font_scale)

    draw.text((120, 5), status, font=font, fill=color)  
    draw.text((120, 100), print_status, font=font, fill=color)
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return image

def create_masks(segmentation_result, orig_shape):
    mask = np.zeros((orig_shape[0], orig_shape[1]), dtype=np.uint8)
    for polygon in segmentation_result:
        polygon = np.array([[int(x * orig_shape[1]), int(y * orig_shape[0])] for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask

def play_sound(status):
    if status == "OK":
        # ok_sound.play()
        ok_sound_v2.play()
    elif status == "NG":
        # ng_sound.play()
        ng_sound_v2.play()

def get_center(bbox):
    center_x = bbox.minx + (bbox.maxx - bbox.minx) / 2
    center_y = bbox.miny + (bbox.maxy - bbox.miny) / 2
    return center_x, center_y

def print_bbox_structure(bbox):
    print(f"BoundingBox attributes: {dir(bbox)}")

def draw_flag_status(image, flag_pitchfuryou, flag_clip_furyou, flag_clip_hanire):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(kanjiFontPath, 40)
    color=(200,10,10)
    if flag_pitchfuryou == 1:
        draw.text((120, 10), u"クリップピッチ不良", font=font, fill=color)  
    if flag_clip_furyou == 1:
        draw.text((120, 60), u"クリップ類不良", font=font, fill=color)  
    if flag_clip_hanire == 1:
        draw.text((120, 110), u"クリップ半入れ", font=font, fill=color)
    
    # Convert back to BGR for OpenCV compatibility
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return image

def check_id(detectedid, idSpec):
    result = [0] * len(idSpec)
    for i, (spec, detected) in enumerate(zip(idSpec, detectedid)):
        if spec == detected:
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

                cv2.line(image, xy_pairs[i], xy_pairs[i+1], lineColor, thickness)

    return None


#add "OK" and "NG"
def draw_status_text(image, status, size = "normal"):
    # Define the position for the text: Center top of the image
    center_x = image.shape[1] // 2
    if size == "normal":
        top_y = 50  # Adjust this value to change the vertical position
        font_scale = 5.0  # Increased font scale for bigger text

    elif size == "small":
        top_y = 10
        font_scale = 2.0  # Increased font scale for bigger text
    

    # Text properties
    
    font_thickness = 8  # Increased font thickness for bolder text
    outline_thickness = font_thickness + 2  # Slightly thicker for the outline
    text_color = (255, 0, 0) if status == "NG" else (0, 255, 0)  # Red for NG, Green for OK
    outline_color = (0, 0, 0)  # Black for the outline

    # Calculate text size and position
    text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = center_x - text_size[0] // 2
    text_y = top_y + text_size[1]

    # Draw the outline
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness)

    # Draw the text over the outline
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    return image


def check_tolerance(checkedPitchResult, pitchSpec, pitchTolerance):
    result = [0] * len(pitchSpec)
    for i, (spec, detected) in enumerate(zip(pitchSpec, checkedPitchResult)):
        if abs(spec - detected) <= pitchTolerance[i]:
            result[i] = 1
    return result

def yolo_to_pixel(yolo_coords, img_shape):
    class_id, x, y, w, h, confidence = yolo_coords
    x_pixel = int(x * img_shape[1])
    y_pixel = int(y * img_shape[0])
    return x_pixel, y_pixel

def find_edge_point_mask(image, mask, center, direction="None", Xoffsetval = 0, Yoffsetval = 0):
    x, y = center[0], center[1]

    min_x = 0
    max_x = image.shape[1] - 1

    if direction == "left":
        while x - Xoffsetval >= 0:
            if mask[int(y + Yoffsetval), int(x - Xoffsetval)] == 0:  # Found an edge
                return x - Xoffsetval, y
            x -= 1
        return min_x, y

    if direction == "right":
        while x + Xoffsetval < image.shape[1]:
            if mask[int(y + Yoffsetval), int(x + Xoffsetval)] == 0:  # Found an edge
                return x + Xoffsetval, y
            x += 1
        return max_x, y

    return None  # If an invalid direction is provided

def find_edge_point(image, center, direction="None", Xoffsetval = 0, Yoffsetval = 0):
    x, y = center[0], center[1]
    blur = 11
    brightness = 0
    contrast = 3.0
    lower_canny = 15
    upper_canny = 110

    # Apply adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur | 1, blur | 1), 0)
    canny_img = cv2.Canny(blurred_image, lower_canny, upper_canny)

    # cv2.imwrite(f"1adjusted_image_{direction}.jpg", adjusted_image)
    # cv2.imwrite(f"2gray_image_{direction}.jpg", gray_image)
    # cv2.imwrite(f"3blurred_image_{direction}.jpg", blurred_image)
    # cv2.imwrite(f"4canny_debug_{direction}.jpg", canny_img)
    min_x = 0
    max_x = image.shape[1] - 1

    if direction == "left":
        while x - Xoffsetval >= 0:
            if canny_img[int(y + Yoffsetval), int(x - Xoffsetval)] == 255:  # Found an edge
                return x - Xoffsetval, y
            x -= 1
        return min_x, y

    if direction == "right":
        while x + Xoffsetval < image.shape[1]:
            if canny_img[int(y + Yoffsetval), int(x + Xoffsetval)] == 255:  # Found an edge
                return x + Xoffsetval, y
            x += 1
        return max_x, y

    return None  # If an invalid direction is provided

def drawcircle(image, pos, class_id): #for ire and hanire
    #draw either green or red circle depends on the detection
    if class_id == 0:
        color = (60, 200, 60)
    elif class_id == 1:
        color = (60, 60, 200)
    #check if pos is tupple
    pos = (int(pos[0]), int(pos[1]))

    cv2.circle(img=image, center=pos, radius=30, color=color, thickness=2, lineType=cv2.LINE_8)

    return image

def drawbox(image, pos, length, offset = text_offset, font_scale=1.7, font_thickness=4):
    pos = (pos[0], pos[1])
    rectangle_bgr = (255, 255, 255)
    (text_width, text_height), _ = cv2.getTextSize(f"{length:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    top_left_x = pos[0] - text_width // 2 - 8
    top_left_y = pos[1] - text_height // 2 - 8 - offset
    bottom_right_x = pos[0] + text_width // 2 + 8
    bottom_right_y = pos[1] + text_height // 2 + 8 - offset
    
    cv2.rectangle(image, (top_left_x, top_left_y),
                  (bottom_right_x, bottom_right_y),
                  rectangle_bgr, -1)
    
    return image

def drawtext(image, pos, length, font_scale=1.7, offset = text_offset, font_thickness=6):
    pos = (pos[0], pos[1])
    font_scale = font_scale
    text = f"{length:.1f}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    text_x = pos[0] - text_width // 2
    text_y = pos[1] + text_height // 2 - offset
    
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 125, 20), font_thickness)
    return image

def calclength(p1, p2):
    length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return length

def _extract_endpoint_from_keypoints(keypoint_result, x_start=0, image_width=0):
    """
    Extract endpoint (keypoint position) from keypoint detection result.
    Maps cropped coordinates back to original image space.
    
    Args:
        keypoint_result: YOLO keypoint detection result object
        x_start: x-coordinate offset in original image (0 for left edge, -840 for right edge)
        image_width: full width of original image
    
    Returns:
        Tuple (x, y) of endpoint in original image coordinates, or None if no keypoints found
    """
    if keypoint_result is None:
        return None

    result_items = keypoint_result if isinstance(keypoint_result, (list, tuple)) else [keypoint_result]

    for item in result_items:
        try:
            keypoints = getattr(item, 'keypoints', None)
            if keypoints is None or not hasattr(keypoints, 'xy') or keypoints.xy is None:
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

    result_items = keypoint_result if isinstance(keypoint_result, (list, tuple)) else [keypoint_result]

    box_count = 0
    keypoint_count = 0

    for item in result_items:
        try:
            boxes = getattr(item, 'boxes', None)
            box_count += len(boxes) if boxes is not None else 0
        except TypeError:
            pass

        try:
            keypoints = getattr(item, 'keypoints', None)
            if keypoints is not None and hasattr(keypoints, 'xy') and keypoints.xy is not None:
                keypoint_count += len(keypoints.xy)
        except TypeError:
            pass

    return f"boxes={box_count}, keypoints={keypoint_count}, endpoint={endpoint}"

def draw_bounding_box(image, x, y, w, h, img_size, color=(0, 255, 0), thickness=2, bbox_offset=bbox_offset):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x1, y1 = int(x - w // 2) - bbox_offset, int(y - h // 2) - bbox_offset
    x2, y2 = int(x + w // 2) + bbox_offset, int(y + h // 2) + bbox_offset
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    center_x, center_y = x, y
    return (center_x, center_y)

# class BoundingBox:
#     def __init__(self, minx, miny, maxx, maxy):
#         self.minx = minx
#         self.miny = miny
#         self.maxx = maxx
#         self.maxy = maxy

# class PredictionScore:
#     def __init__(self, value):
#         self.value = value

# class Category:
#     def __init__(self, id, name):
#         self.id = id
#         self.name = name

# class ObjectPrediction:
#     def __init__(self, bbox, score, category):
#         self.bbox = bbox
#         self.score = score
#         self.category = category