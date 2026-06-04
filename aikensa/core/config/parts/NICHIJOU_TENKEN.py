from pathlib import Path

import cv2
import numpy as np
import pygame
import yaml
from PIL import Image, ImageDraw, ImageFont

from aikensa.core.scripts.img_processing.img_processing import map_keypoint_xcrop_to_original


specs_yaml_path = Path(__file__).resolve().parents[1] / "parts_specifications.yaml"
with open(specs_yaml_path, "r", encoding="utf-8") as file:
    all_specs = yaml.safe_load(file)
    part_spec = all_specs["parts"]["NICHIJOU_TENKEN"]
    global_spec = all_specs["global"]


pygame.mixer.init()
ok_sound = pygame.mixer.Sound(global_spec["sounds"]["ok"])
ok_sound_v2 = pygame.mixer.Sound(global_spec["sounds"]["ok_v2"])
ng_sound = pygame.mixer.Sound(global_spec["sounds"]["ng"])
ng_sound_v2 = pygame.mixer.Sound(global_spec["sounds"]["ng_v2"])
kanjiFontPath = global_spec["font_path"]

pitchSpec = float(part_spec.get("pitchSpec", 100.0))
pitchTolerance = float(part_spec.get("pitchTolerance", 1.0))
pixelMultiplier = float(part_spec.get("pixelMultiplier", 0.1594))


def debug_images_enabled():
    return bool(part_spec.get("debug_save_images", False))


def debug_mode_enabled():
    return bool(part_spec.get("debug_mode", False))


def partcheck(image, keypoint_results, crop_offset=(0, 0)):
    measuredPitch = []
    resultPitch = []
    resultid = []

    status = "NG"
    print_status = ""
    ngreason = "DETECTION NG"

    detected_points = _extract_daily_tenken_points(
        keypoint_results,
        crop_offset=crop_offset,
        image_width=image.shape[1],
    )

    if len(detected_points) < 2:
        print_status += "基準点検出不良\n"
    else:
        left_point, right_point = detected_points[0], detected_points[-1]
        measured_length = calclength(left_point, right_point) * pixelMultiplier
        rounded_measurement = round(measured_length, 1)
        is_ok = abs(pitchSpec - rounded_measurement) <= pitchTolerance
        if is_ok and debug_mode_enabled():
            rounded_measurement = round(pitchSpec, 1)

        measuredPitch.append(rounded_measurement)
        resultPitch = [0]

        cv2.circle(image, left_point, 18, (10, 10, 255), 4)
        cv2.circle(image, right_point, 18, (10, 10, 255), 4)

        draw_pitch_line(image, [left_point, right_point], [1 if is_ok else 0], thickness=8)

        print(f"Measured Pitch: {measuredPitch[0]:.2f}mm")
        print(f"Pitch Difference: {pitchSpec - measuredPitch[0]:.2f}mm")
        print(f"Pitch Tolerance: {pitchTolerance:.2f}mm")

        if is_ok:
            status = "OK"
            print_status += f"校正良好 {measuredPitch[0]:.2f}mm\n"
            ngreason = ""
            resultPitch = [1]
        else:
            print_status += f"校正不良 {measuredPitch[0]:.2f}mm\n"
            ngreason = "CALIBRATION NG"

    if status == "NG" and not resultPitch:
        resultPitch = [0]

    image = draw_status_text_PIL(image, status, print_status, size="normal")
    return image, measuredPitch, resultPitch, resultid, status, ngreason


dailyTenken = partcheck


def _extract_daily_tenken_points(keypoint_results, crop_offset=(0, 0), image_width=None):
    if keypoint_results is None:
        return []

    x_start, y_start = crop_offset
    detected_points = []

    for result in keypoint_results:
        try:
            keypoints_xy = getattr(getattr(result, "keypoints", None), "xy", None)
            if keypoints_xy is None:
                continue

            if hasattr(keypoints_xy, "cpu"):
                keypoints_xy = keypoints_xy.cpu().numpy()
            else:
                keypoints_xy = np.asarray(keypoints_xy)

            if keypoints_xy.size == 0:
                continue

            for detection_points in keypoints_xy:
                for point in detection_points:
                    x_pos = float(point[0])
                    y_pos = float(point[1])
                    if not np.isfinite(x_pos) or not np.isfinite(y_pos):
                        continue

                    x_orig, y_orig = map_keypoint_xcrop_to_original(
                        x_start=x_start,
                        kpt_xy_crop=(x_pos, y_pos),
                        img_width=image_width,
                    )
                    detected_points.append((int(round(x_orig)), int(round(y_start + y_orig))))
        except Exception:
            continue

    detected_points = sorted(detected_points, key=lambda point: point[0])
    if len(detected_points) >= 2:
        return [detected_points[0], detected_points[-1]]
    return detected_points


def calclength(point_a, point_b):
    return abs(point_a[0] - point_b[0])


def draw_status_text_PIL(image, status, print_status, size="normal"):
    if size == "large":
        font_scale = global_spec.get("text_size_large", 130.0)
    elif size == "small":
        font_scale = global_spec.get("text_size_small", 50.0)
    else:
        font_scale = global_spec.get("text_size_normal", 100.0)

    if status == "OK":
        color = tuple(part_spec.get("color_ok", [10, 210, 60]))
    else:
        color = tuple(part_spec.get("color_ng", [200, 30, 50]))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(kanjiFontPath, font_scale)

    draw.text((120, 5), status, font=font, fill=color)
    draw.text((120, 100), print_status, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def play_sound(status):
    if status == "OK":
        ok_sound_v2.play()
    elif status == "NG":
        ng_sound_v2.play()


def draw_pitch_line(image, xy_pairs, pitchresult, thickness=2):
    xy_pairs = [(int(x), int(y)) for x, y in xy_pairs]

    if len(xy_pairs) != 0:
        for index in range(len(xy_pairs) - 1):
            if index < len(pitchresult) and pitchresult[index] is not None:
                if pitchresult[index] == 1:
                    line_color = (0, 255, 0)
                else:
                    line_color = (255, 0, 0)

                cv2.line(image, xy_pairs[index], xy_pairs[index + 1], line_color, thickness)

    return None