"""
Generic Part Inspection Engine
Loads part specifications from YAML and performs standardized inspection
Simplifies part file maintenance by centralizing configuration data
"""

import os
import yaml
import numpy as np
import cv2
import pygame
from PIL import ImageFont, ImageDraw, Image
from typing import Tuple, Dict, List, Optional

from aikensa.core.scripts.img_processing.img_processing import (
    draw_bounding_box, get_center, calclength,
    check_tolerance, check_id, draw_pitch_line,
    draw_status_text_PIL, check_hanire, draw_redCircle,
    map_keypoint_xcrop_to_original
)

# Initialize pygame mixer
pygame.mixer.init()


class PartSpecificationLoader:
    """Loads and manages part specifications from YAML"""
    
    _instance = None
    _specs_cache = {}
    _sounds_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load_specifications(cls, yaml_path: str = "aikensa/core/config/parts_specifications.yaml"):
        """Load all part specifications from YAML file"""
        if not cls._specs_cache:
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"Parts specifications YAML not found: {yaml_path}")
            
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                cls._specs_cache = data
        
        return cls._specs_cache
    
    @classmethod
    def get_part_spec(cls, part_id: str) -> Dict:
        """Get specification for a specific part"""
        specs = cls.load_specifications()
        
        if part_id not in specs.get('parts', {}):
            raise KeyError(f"Part specification not found for: {part_id}")
        
        return specs['parts'][part_id]
    
    @classmethod
    def get_global_settings(cls) -> Dict:
        """Get global settings"""
        specs = cls.load_specifications()
        return specs.get('global', {})
    
    @classmethod
    def get_sound(cls, sound_key: str):
        """Load and cache sound file"""
        if sound_key not in cls._sounds_cache:
            global_settings = cls.get_global_settings()
            sound_file = global_settings.get('sounds', {}).get(sound_key)
            
            if not sound_file:
                raise KeyError(f"Sound file not found for key: {sound_key}")
            
            if not os.path.exists(sound_file):
                raise FileNotFoundError(f"Sound file not found: {sound_file}")
            
            cls._sounds_cache[sound_key] = pygame.mixer.Sound(sound_file)
        
        return cls._sounds_cache[sound_key]


class GenericPartInspector:
    """Generic part inspection engine that uses YAML specifications"""
    
    def __init__(self, part_id: str):
        """Initialize inspector with part specification"""
        self.part_id = part_id
        self.spec = PartSpecificationLoader.get_part_spec(part_id)
        self.global_settings = PartSpecificationLoader.get_global_settings()
        
        # Extract common settings
        self.pitchSpec = self.spec.get('pitchSpec', [])
        self.idSpec = self.spec.get('idSpec', [])
        self.tolerance_pitch = self.spec.get('tolerance_pitch', [])
        self.pixelMultiplier = self.spec.get('pixelMultiplier', 1.0)
        self.color_ok = tuple(self.spec.get('color_ok', [0, 255, 0]))
        self.color_ng = tuple(self.spec.get('color_ng', [200, 30, 50]))
        self.linecolor = tuple(self.spec.get('linecolor', [20, 120, 120]))
        self.text_offset = self.spec.get('text_offset', 40)
        self.bbox_offset = self.spec.get('bbox_offset', 1)
        
        # Font setup
        font_path = self.global_settings.get('font_path', 'aikensa/gui/resources/font/NotoSansJP-ExtraBold.ttf')
        self.font_path = font_path
        self.font_sizes = {
            'small': self.global_settings.get('text_size_small', 50.0),
            'normal': self.global_settings.get('text_size_normal', 100.0),
            'large': self.global_settings.get('text_size_large', 130.0),
        }
    
    def inspect_keypoint_based(self, image, keypoint_detections, **kwargs) -> Tuple:
        """
        Generic keypoint-based inspection
        
        Args:
            image: Input image
            keypoint_detections: Keypoint detection results
            **kwargs: Additional detection data (segmentation, etc.)
        
        Returns:
            Tuple: (result_image, measuredPitch, resultPitch, resultid, status)
        """
        # Initialize result containers
        measuredPitch = [0] * len(self.pitchSpec)
        resultPitch = [0] * len(self.pitchSpec)
        resultid = [0] * len(self.idSpec)
        status = "OK"
        print_status = ""
        
        # TODO: Implement generic keypoint inspection logic
        # This would iterate through keypoint_detections and:
        # 1. Calculate distances between consecutive keypoints
        # 2. Compare against tolerance_pitch using check_tolerance
        # 3. Build the result arrays
        
        return image, measuredPitch, resultPitch, resultid, status
    
    def inspect_segmentation_based(self, image, segmentation_results, **kwargs) -> Tuple:
        """
        Generic segmentation-based inspection
        
        Args:
            image: Input image
            segmentation_results: Segmentation detection results
            **kwargs: Additional detection data
        
        Returns:
            Tuple: (result_image, measuredPitch, resultPitch, resultid, status)
        """
        # Initialize result containers
        measuredPitch = [0] * len(self.pitchSpec)
        resultPitch = [0] * len(self.pitchSpec)
        resultid = [0] * len(self.idSpec)
        status = "OK"
        print_status = ""
        
        # TODO: Implement generic segmentation inspection logic
        # This would analyze segmentation masks and:
        # 1. Extract feature points from segmentation
        # 2. Calculate measurements
        # 3. Compare against tolerances
        
        return image, measuredPitch, resultPitch, resultid, status
    
    def draw_results(self, image, measuredPitch, resultPitch, status, print_status) -> np.ndarray:
        """Draw inspection results on image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)
        
        font_size = self.font_sizes.get('normal', 100.0)
        try:
            font = ImageFont.truetype(self.font_path, int(font_size))
        except Exception as e:
            print(f"Font loading error: {e}")
            font = ImageFont.load_default()
        
        color = self.color_ok if status == "OK" else self.color_ng
        
        draw.text((120, 5), status, font=font, fill=color)
        draw.text((120, 100), print_status, font=font, fill=color)
        
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return image
    
    def play_result_sound(self, status):
        """Play sound based on inspection result"""
        try:
            if status == "OK":
                sound = PartSpecificationLoader.get_sound('ok_v2')
            else:
                sound = PartSpecificationLoader.get_sound('ng_v2')
            sound.play()
        except Exception as e:
            print(f"Sound playback error: {e}")
    
    def get_spec_value(self, key: str, default=None):
        """Safely get a specification value"""
        return self.spec.get(key, default)


# Convenience function for backward compatibility
def get_part_inspector(part_id: str) -> GenericPartInspector:
    """Factory function to create part inspector"""
    return GenericPartInspector(part_id)


def load_part_config(part_id: str) -> Dict:
    """Load part configuration from YAML"""
    return PartSpecificationLoader.get_part_spec(part_id)
