import atexit
from typing import Union, Tuple, Optional, Iterable
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import imagingcontrol4 as ic4

# ---- one-time SDK context ----
_IC4_CTX = None
def _ensure_ic4_context():
    global _IC4_CTX
    if _IC4_CTX is None:
        _IC4_CTX = ic4.Library.init_context(
            api_log_level=ic4.LogLevel.WARNING,
            log_targets=ic4.LogTarget.STDERR
        )
        _IC4_CTX.__enter__()
        atexit.register(_IC4_CTX.__exit__, None, None, None)

def _find_device(index_or_serial: Union[int, str]) -> Optional[ic4.DeviceInfo]:
    """
    Find a camera device. Returns None if no cameras are found instead of raising an error.
    This allows graceful fallback to placeholder images.
    """
    try:
        devs = ic4.DeviceEnum.devices()
    except Exception as e:
        print(f"Warning: Failed to enumerate devices: {e}")
        return None
    
    if not devs:
        print("Warning: No Imaging Source cameras found.")
        return None
    
    if isinstance(index_or_serial, int):
        if not (0 <= index_or_serial < len(devs)):
            print(f"Warning: Camera index {index_or_serial} out of range (0..{len(devs)-1})")
            return None
        return devs[index_or_serial]
    
    for d in devs:
        if d.serial == index_or_serial:
            return d
    
    for d in devs:
        if (index_or_serial or "").lower() in (d.model_name or "").lower():
            return d
    
    print(f"Warning: No camera matched '{index_or_serial}'")
    return None

# handy helpers for tolerant property IO
def _try_set(pm: ic4.PropertyMap, prop_ids: Iterable, value) -> bool:
    for pid in prop_ids:
        if pid is None: continue
        try:
            pm.set_value(pid, value)
            return True
        except ic4.IC4Exception:
            continue
    return False

def _try_get_int(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[int]:
    for pid in prop_ids:
        if pid is None: continue
        try:
            return int(pm.get_value_int(pid))
        except ic4.IC4Exception:
            continue
    return None

def _try_get_float(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[float]:
    for pid in prop_ids:
        if pid is None: continue
        try:
            return float(pm.get_value_float(pid))
        except ic4.IC4Exception:
            continue
    return None

def _try_get_str(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[str]:
    for pid in prop_ids:
        if pid is None: continue
        try:
            return pm.get_value_str(pid)
        except ic4.IC4Exception:
            continue
    return None


def _apply_manual_ae_gain_wb(pm: ic4.PropertyMap,
                             exposure_us: float | None,
                             gain_db: float | None,
                             wb_temperature: int | None) -> None:
    # Force all auto controls off before writing manual values.
    _try_set(pm, (PID_EXPOSURE_AUTO,), "Off")
    _try_set(pm, (PID_GAIN_AUTO,), "Off")
    _try_set(pm, (PID_WB_AUTO,), "Off")

    if exposure_us is not None:
        _try_set(pm, (PID_EXPOSURE_TIME,), float(exposure_us))
    if gain_db is not None:
        _try_set(pm, (PID_GAIN,), float(gain_db))
    if wb_temperature is not None:
        _try_set(pm, (PID_WB_TEMP,), int(wb_temperature))


def _readback_ae_gain_wb(pm: ic4.PropertyMap) -> dict:
    exp_auto = _try_get_str(pm, (PID_EXPOSURE_AUTO,))
    if exp_auto is None:
        exp_auto = str(_try_get_int(pm, (PID_EXPOSURE_AUTO,)))

    gain_auto = _try_get_str(pm, (PID_GAIN_AUTO,))
    if gain_auto is None:
        gain_auto = str(_try_get_int(pm, (PID_GAIN_AUTO,)))

    wb_auto = _try_get_str(pm, (PID_WB_AUTO,))
    if wb_auto is None:
        wb_auto = str(_try_get_int(pm, (PID_WB_AUTO,)))

    return {
        "exp_auto": exp_auto,
        "gain_auto": gain_auto,
        "wb_auto": wb_auto,
        "exposure_us": _try_get_float(pm, (PID_EXPOSURE_TIME,)),
        "gain_db": _try_get_float(pm, (PID_GAIN,)),
        "wb_k": _try_get_int(pm, (PID_WB_TEMP,)),
    }

# PropId aliases (be tolerant across models)
PID_PIXEL_FORMAT       = getattr(ic4.PropId, "PIXEL_FORMAT", None)
PID_WIDTH              = getattr(ic4.PropId, "WIDTH", None)
PID_HEIGHT             = getattr(ic4.PropId, "HEIGHT", None)
PID_OFFSET_X           = getattr(ic4.PropId, "OFFSET_X", None)
PID_OFFSET_Y           = getattr(ic4.PropId, "OFFSET_Y", None)
PID_OFFSET_AUTO_CENTER = getattr(ic4.PropId, "OFFSET_AUTO_CENTER", None)
PID_REVERSE_X          = getattr(ic4.PropId, "REVERSE_X", None)
PID_REVERSE_Y          = getattr(ic4.PropId, "REVERSE_Y", None)

PID_EXPOSURE_AUTO      = getattr(ic4.PropId, "EXPOSURE_AUTO", None)
PID_EXPOSURE_TIME      = getattr(ic4.PropId, "EXPOSURE_TIME", None)
PID_GAIN_AUTO          = getattr(ic4.PropId, "GAIN_AUTO", None)
PID_GAIN               = getattr(ic4.PropId, "GAIN", None)
PID_WB_AUTO            = getattr(ic4.PropId, "BALANCE_WHITE_AUTO", None)
PID_WB_TEMP            = getattr(ic4.PropId, "WHITEBALANCE_TEMPERATURE", None)

# Frame-rate names differ; try both, but it's optional for SnapSink.
PID_FRAME_RATE         = getattr(ic4.PropId, "FRAME_RATE", None)
PID_ACQ_FRAME_RATE     = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE", None)
PID_ACQ_FR_EN          = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE_ENABLE", None)

PF = ic4.PixelFormat

class DummyCapture:
    """
    Placeholder camera that provides black images with text overlay.
    Used when no physical camera is connected.
    """
    def __init__(self, width: int = 3072, height: int = 2048,
                 camera_name: str = "Camera", font_path: str = None,
                 rotate_180: bool = False):
        self._width = width
        self._height = height
        self._camera_name = camera_name
        self._font_path = font_path or "aikensa/gui/resources/font/NotoSansJP-ExtraBold.ttf"
        self._rotate_180 = bool(rotate_180)
        self._open = True
        self._placeholder_image = self._create_placeholder_image()
    
    def _create_placeholder_image(self) -> np.ndarray:
        """Create a black image with 'No Camera Connected' text"""
        # Create black image
        img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        
        # Convert to PIL for better text rendering
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load custom font, fallback to default
        try:
            font_size = int(self._height / 10)  # Dynamic font size
            font = ImageFont.truetype(self._font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        
        # Draw main text
        main_text = "カメラ未接続"  # "No Camera Connected" in Japanese
        sub_text = f"({self._camera_name})"
        
        # Get text bounding box for centering
        try:
            bbox = draw.textbbox((0, 0), main_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width, text_height = 500, 100  # Fallback dimensions
        
        # Calculate center position
        x = (self._width - text_width) // 2
        y = (self._height - text_height) // 2 - 50
        
        # Draw text with white color
        draw.text((x, y), main_text, fill=(255, 255, 255), font=font)
        
        # Draw subtitle
        try:
            small_font = ImageFont.truetype(self._font_path, int(font_size * 0.4))
        except:
            small_font = font
        
        try:
            bbox2 = draw.textbbox((0, 0), sub_text, font=small_font)
            sub_width = bbox2[2] - bbox2[0]
        except:
            sub_width = 200
        
        sub_x = (self._width - sub_width) // 2
        sub_y = y + text_height + 20
        draw.text((sub_x, sub_y), sub_text, fill=(180, 180, 180), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def isOpened(self) -> bool:
        return self._open
    
    def release(self):
        self._open = False
    
    def read(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the placeholder image"""
        if not self._open:
            return False, None
        frame = self._placeholder_image.copy()
        if self._rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return True, frame
    
    def get(self, prop_id: int) -> float:
        """Return camera properties"""
        if prop_id == 3:  # CAP_PROP_FRAME_WIDTH
            return float(self._width)
        elif prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
            return float(self._height)
        elif prop_id == 5:  # CAP_PROP_FPS
            return 30.0
        elif prop_id in (14, 15, 45):  # GAIN, EXPOSURE, WB_TEMP
            return 0.0
        return 0.0
    
    def set(self, prop_id: int, value: float) -> bool:
        """Dummy setter - does nothing but returns True"""
        return True


class IC4Capture:
    _W   = getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)
    _H   = getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)
    _FPS = getattr(cv2, "CAP_PROP_FPS", 5)
    _EXP = getattr(cv2, "CAP_PROP_EXPOSURE", 15)
    _GAIN= getattr(cv2, "CAP_PROP_GAIN", 14)
    _WB  = getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 45)

    def __init__(self,
                 cam: Union[int, str],
                 width: int = 3072, height: int = 2048, fps: float = 5.0,
                 color: bool = True,
                 exposure_us: float | None = 15000,
                 gain_db: float | None = 10,
                 wb_temperature: int | None = 4500,
                 rotate_180: bool = False):
        _ensure_ic4_context()

        # Open device - handle None gracefully
        dev_info = _find_device(cam)
        if dev_info is None:
            # No camera found - this will be handled by initialize_camera_ic4
            # which will return a DummyCapture instead
            raise RuntimeError(f"Camera '{cam}' not found - will use placeholder")
        
        self._grab = ic4.Grabber(dev_info)
        pm = self._grab.device_property_map

        # Pixel format
        if color:
            if not _try_set(pm, (PID_PIXEL_FORMAT,), getattr(PF, "BayerRG8", None)):
                _try_set(pm, (PID_PIXEL_FORMAT,), getattr(PF, "BGR8", None))
        else:
            _try_set(pm, (PID_PIXEL_FORMAT,), getattr(PF, "Mono8", None))

        # ROI / size
        _try_set(pm, (PID_OFFSET_AUTO_CENTER,), "Off")
        _try_set(pm, (PID_OFFSET_X,), 0)
        _try_set(pm, (PID_OFFSET_Y,), 0)
        _try_set(pm, (PID_WIDTH,),  width)
        _try_set(pm, (PID_HEIGHT,), height)
        rotate_180 = bool(rotate_180)
        rx_ok = _try_set(pm, (PID_REVERSE_X,), rotate_180)
        ry_ok = _try_set(pm, (PID_REVERSE_Y,), rotate_180)
        self._rotate_180_fallback = rotate_180 and not (rx_ok and ry_ok)

        # Exposure / Gain / WB: keep deterministic startup order.
        _apply_manual_ae_gain_wb(pm, exposure_us, gain_db, wb_temperature)

        # Frame rate (optional; if out-of-range it will just be ignored by the device)
        _try_set(pm, (PID_ACQ_FR_EN,), True)
        _try_set(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE), float(fps))

        # Start stream with SnapSink (like the sample)
        self._sink = ic4.SnapSink()
        self._grab.stream_setup(self._sink)

        # cache requested; getters will try readback
        self._w, self._h, self._fps = width, height, float(fps)
        self._open = True

        # Decide conversion path
        pf_name = _try_get_str(pm, (PID_PIXEL_FORMAT,))
        self._convert = None
        pf_enum = getattr(PF, pf_name) if pf_name and hasattr(PF, pf_name) else None
        if pf_enum in (getattr(PF, "BayerRG8", None),
                       getattr(PF, "BayerBG8", None),
                       getattr(PF, "BayerGR8", None),
                       getattr(PF, "BayerGB8", None)):
            self._convert = {
                getattr(PF, "BayerRG8", None): cv2.COLOR_BayerRG2BGR,
                getattr(PF, "BayerBG8", None): cv2.COLOR_BayerBG2BGR,
                getattr(PF, "BayerGR8", None): cv2.COLOR_BayerGR2BGR,
                getattr(PF, "BayerGB8", None): cv2.COLOR_BayerGB2BGR,
            }.get(pf_enum, None)
        elif pf_enum == getattr(PF, "RGB8", None):
            self._convert = cv2.COLOR_RGB2BGR
        elif pf_enum in (getattr(PF, "YUV422Packed", None), getattr(PF, "YUY2", None)):
            self._convert = cv2.COLOR_YUV2BGR_YUY2  # adjust to UYVY if needed

    # cv2-like API
    def isOpened(self) -> bool: return self._open

    def release(self):
        if self._open:
            try:
                self._grab.stream_stop()
            finally:
                self._grab.device_close()
                self._open = False

    def read(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._open:
            return False, None
        try:
            buf = self._sink.snap_single(int(timeout_ms))
        except ic4.IC4Exception:
            return False, None
        if buf is None:
            return False, None

        frame = buf.numpy_copy()              # <- independent NumPy array
        # If you prefer zero-copy, use: frame = buf.numpy_wrap()   (but keep 'buf' alive)
        if self._convert is not None:
            frame = cv2.cvtColor(frame, self._convert)
        if self._rotate_180_fallback:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return True, frame

    def get(self, prop_id: int) -> float:
        pm = self._grab.device_property_map
        if prop_id == self._W:
            val = _try_get_int(pm, (PID_WIDTH,));  return float(val if val is not None else self._w)
        if prop_id == self._H:
            val = _try_get_int(pm, (PID_HEIGHT,)); return float(val if val is not None else self._h)
        if prop_id == self._FPS:
            val = _try_get_float(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE))
            return float(val if val is not None else self._fps)
        if prop_id == self._EXP:
            val = _try_get_float(pm, (PID_EXPOSURE_TIME,)); return float(val if val is not None else 0.0)
        if prop_id == self._GAIN:
            val = _try_get_float(pm, (PID_GAIN,));          return float(val if val is not None else 0.0)
        if prop_id == self._WB:
            val = _try_get_int(pm, (PID_WB_TEMP,));         return float(val if val is not None else 0.0)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        pm = self._grab.device_property_map
        try:
            if prop_id == self._FPS:
                _try_set(pm, (PID_ACQ_FR_EN,), True)
                ok = _try_set(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE), float(value))
                if ok: self._fps = float(value)
                return ok
            if prop_id == self._EXP:
                _try_set(pm, (PID_EXPOSURE_AUTO,), "Off")
                _try_set(pm, (PID_EXPOSURE_TIME,), float(value))
                return True
            if prop_id == self._GAIN:
                _try_set(pm, (PID_GAIN_AUTO,), "Off")
                _try_set(pm, (PID_GAIN,), float(value))
                return True
            if prop_id == self._WB:
                _try_set(pm, (PID_WB_AUTO,), "Off")
                _try_set(pm, (PID_WB_TEMP,), int(value))
                return True
        except ic4.IC4Exception:
            return False
        return False

def initialize_camera_ic4(cam_id_or_serial: Union[int, str],
                          width: int = 3072,
                          height: int = 2048,
                          fps: float = 30.0,
                          color: bool = True,
                          exposure_us: float | None = 15000,
                          gain_db: float | None = 10,
                          wb_temperature: int | None = 4500,
                          auto_exposure: bool = False,
                          auto_gain: bool = False,
                          auto_wb: bool = False,
                          rotate_180: bool = False) -> Union[IC4Capture, DummyCapture]:
    """
    Initialize IC4 camera. Returns DummyCapture if camera is not found.
    This allows graceful fallback when no camera is connected.
    """
    try:
        cam = IC4Capture(cam_id_or_serial, width, height, fps, color,
                         exposure_us, gain_db, wb_temperature, rotate_180=rotate_180)
        pm = cam._grab.device_property_map

        # Guard against startup drift: always force manual mode on initialization.
        # Keep the auto_* params for backward compatibility, but ignore True requests here.
        _apply_manual_ae_gain_wb(pm, exposure_us, gain_db, wb_temperature)

        if auto_exposure or auto_gain or auto_wb:
            print(
                "[IC4] Auto AE/Gain/WB request ignored at startup; "
                "manual mode is enforced for stable color/exposure."
            )

        rb = _readback_ae_gain_wb(pm)
        print(
            f"[IC4 Init Readback] cam={cam_id_or_serial} "
            f"auto(exp/gain/wb)=({rb['exp_auto']}/{rb['gain_auto']}/{rb['wb_auto']}) "
            f"exp_us={rb['exposure_us']} gain_db={rb['gain_db']} wb_k={rb['wb_k']}"
        )

        print(f"✓ Camera '{cam_id_or_serial}' initialized successfully")
        return cam
    except (RuntimeError, Exception) as e:
        print(f"⚠ Camera '{cam_id_or_serial}' not available: {e}")
        print(f"  → Using placeholder image instead")
        camera_name = f"Camera {cam_id_or_serial}"
        return DummyCapture(width, height, camera_name, rotate_180=rotate_180)


def open_ic4_camera_or_placeholder(ic4id,
                                   width: int = 3072,
                                   height: int = 2048,
                                   fps: float = 5,
                                   rotate_180: bool = False,
                                   color: bool = True,
                                   exposure_us: float = 10000,
                                   gain_db: float = 10,
                                   wb_temperature: int = 4500,
                                   auto_exposure: bool = False,
                                   auto_gain: bool = False,
                                   auto_wb: bool = False) -> Union[IC4Capture, DummyCapture]:
    """Shared helper for opening an IC4 camera with placeholder fallback."""
    return initialize_camera_ic4(
        ic4id,
        width=width,
        height=height,
        fps=fps,
        color=color,
        exposure_us=exposure_us,
        gain_db=gain_db,
        wb_temperature=wb_temperature,
        auto_exposure=auto_exposure,
        auto_gain=auto_gain,
        auto_wb=auto_wb,
        rotate_180=rotate_180,
    )