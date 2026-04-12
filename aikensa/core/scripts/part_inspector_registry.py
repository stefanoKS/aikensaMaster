"""
Part Inspector Registry
=======================

This module provides a centralized registry system for part inspection modules.
It eliminates the need for hardcoded imports and widget conditionals in the 
inspection thread, making it trivial to add new parts.

Usage:
    # Initialize registry
    registry = PartInspectorRegistry("aikensa/core/config/part_registry.yaml")
    
    # Get part configuration
    part_config = registry.get_part_by_widget(5)
    
    # Execute inspection dynamically
    result = registry.execute_inspection(widget_id=5, image=img, ...)
    
    # Get all registered parts
    all_parts = registry.get_all_parts()

Adding a new part:
    1. Add entry to part_registry.yaml with module_path and widget_id
    2. Create the part inspection module with a `partcheck` function
    3. That's it! The system will auto-discover and register it
"""

import yaml
import importlib
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from PyQt5.QtCore import pyqtSignal

logger = logging.getLogger(__name__)


@dataclass
class PartConfig:
    """Configuration for a single part inspection module"""
    widget_id: int
    part_number: str
    part_name: str
    module_path: str
    partcheck_function: str
    signal_name: str
    directory_name: str
    inspection_type: str
    has_katabu: bool = False
    katabu_side: Optional[str] = None
    keypoint_crop_px: Optional[int] = None
    keypoint_model_imgsz: Optional[int] = None
    inference_crop: Optional[List[int]] = None
    katabu_crop: Optional[List[int]] = None
    narrow_height: bool = False
    supports_clip_picking: bool = False
    
    # Lazy-loaded attributes
    _module: Optional[Any] = None
    _partcheck_func: Optional[Callable] = None
    
    def get_partcheck_function(self) -> Callable:
        """Lazy-load and return the partcheck function from the module"""
        if self._partcheck_func is None:
            if self._module is None:
                try:
                    self._module = importlib.import_module(self.module_path)
                    logger.info(f"Loaded module: {self.module_path}")
                except ImportError as e:
                    logger.error(f"Failed to import module {self.module_path}: {e}")
                    raise
            
            self._partcheck_func = getattr(self._module, self.partcheck_function, None)
            if self._partcheck_func is None:
                raise AttributeError(
                    f"Module {self.module_path} does not have function '{self.partcheck_function}'"
                )
        
        return self._partcheck_func


class PartInspectorRegistry:
    """
    Central registry for all part inspection modules.
    
    This class manages the mapping between widget IDs and their corresponding
    inspection modules, eliminating hardcoded imports and conditionals.
    """
    
    def __init__(self, registry_yaml_path: str = "aikensa/core/config/part_registry.yaml"):
        self.registry_path = registry_yaml_path
        self.parts: Dict[int, PartConfig] = {}  # widget_id -> PartConfig
        self.widget_groups: Dict[str, List[int]] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """Load part configurations from YAML file"""
        try:
            with open(self.registry_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load individual part configurations
            for part_data in config.get('parts', []):
                part_config = PartConfig(
                    widget_id=part_data['widget_id'],
                    part_number=part_data['part_number'],
                    part_name=part_data['part_name'],
                    module_path=part_data['module_path'],
                    partcheck_function=part_data['partcheck_function'],
                    signal_name=part_data['signal_name'],
                    directory_name=part_data['directory_name'],
                    inspection_type=part_data['inspection_type'],
                    has_katabu=part_data.get('has_katabu', False),
                    katabu_side=part_data.get('katabu_side'),
                    keypoint_crop_px=part_data.get('keypoint_crop_px'),
                    keypoint_model_imgsz=part_data.get('keypoint_model_imgsz'),
                    inference_crop=part_data.get('inference_crop'),
                    katabu_crop=part_data.get('katabu_crop'),
                    narrow_height=part_data.get('narrow_height', False),
                    supports_clip_picking=part_data.get('supports_clip_picking', False)
                )
                
                self.parts[part_config.widget_id] = part_config
            
            # Load widget groups
            self.widget_groups = config.get('widget_groups', {})
            
            logger.info(f"Loaded {len(self.parts)} parts from registry")
            
        except FileNotFoundError:
            logger.warning(f"Registry file not found: {self.registry_path}. Using empty registry.")
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            raise
    
    def get_part_by_widget(self, widget_id: int) -> Optional[PartConfig]:
        """Get part configuration by widget ID"""
        return self.parts.get(widget_id)
    
    def get_all_parts(self) -> Dict[int, PartConfig]:
        """Get all registered parts"""
        return self.parts
    
    def get_widget_group(self, group_name: str) -> List[int]:
        """Get list of widget IDs for a specific group"""
        return self.widget_groups.get(group_name, [])
    
    def widget_has_feature(self, widget_id: int, feature: str) -> bool:
        """Check if a widget has a specific feature"""
        part = self.get_part_by_widget(widget_id)
        if not part:
            return False
        
        feature_map = {
            'katabu': part.has_katabu,
            'narrow_height': part.narrow_height,
            'clip_picking': part.supports_clip_picking
        }
        
        return feature_map.get(feature, False)
    
    def get_directory_name(self, widget_id: int) -> str:
        """Get directory name for a widget (for saving results)"""
        part = self.get_part_by_widget(widget_id)
        return part.directory_name if part else f"unknown_{widget_id}"
    
    def get_part_name(self, widget_id: int) -> str:
        """Get human-readable part name"""
        part = self.get_part_by_widget(widget_id)
        return part.part_name if part else f"Unknown Part {widget_id}"
    
    def execute_inspection(
        self,
        widget_id: int,
        image,
        clip_detection_result=None,
        segmentation_result=None,
        keypoint_result=None,
        part_name=None,
        **kwargs
    ) -> tuple:
        """
        Dynamically execute the inspection function for a given widget.
        
        Args:
            widget_id: The widget ID to inspect
            image: The image to inspect
            clip_detection_result: Optional clip detection results
            segmentation_result: Optional segmentation results
            keypoint_result: Optional keypoint detection results
            part_name: Optional part name override
            **kwargs: Additional arguments to pass to the inspection function
        
        Returns:
            Tuple containing inspection results (varies by part type)
        """
        part_config = self.get_part_by_widget(widget_id)
        
        if not part_config:
            raise ValueError(f"No part registered for widget_id {widget_id}")
        
        # Get the partcheck function
        partcheck_func = part_config.get_partcheck_function()
        
        # Use part name from config if not provided
        if part_name is None:
            part_name = f"P{part_config.part_number}"
        
        # Execute the inspection function
        try:
            result = partcheck_func(
                image,
                clip_detection_result,
                segmentation_result,
                keypoint_result,
                part_name,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Error executing inspection for widget {widget_id}: {e}")
            raise
    
    def get_widget_to_directory_map(self) -> Dict[int, str]:
        """Get mapping of widget_id -> directory_name (for backward compatibility)"""
        return {wid: part.directory_name for wid, part in self.parts.items()}
    
    def get_widget_to_name_map(self) -> Dict[int, str]:
        """Get mapping of widget_id -> part_number with 'P' prefix"""
        return {wid: f"P{part.part_number}" for wid, part in self.parts.items()}
    
    def is_inspection_widget(self, widget_id: int) -> bool:
        """Check if widget is an inspection widget"""
        return widget_id in self.get_widget_group('inspection_widgets')
    
    def needs_katabu_emission(self, widget_id: int) -> bool:
        """Check if widget needs katabu (kata-bu) image emission"""
        return widget_id in self.get_widget_group('katabu_widgets')
    
    def is_narrow_height(self, widget_id: int) -> bool:
        """Check if widget uses narrow height display"""
        return widget_id in self.get_widget_group('narrow_height')
    
    def supports_clip_picking(self, widget_id: int) -> bool:
        """Check if widget supports clip picking order logic"""
        return widget_id in self.get_widget_group('clip_picking_supported')
    
    def generate_signals_for_thread(self) -> Dict[str, type]:
        """
        Generate PyQt signal definitions for all registered parts.
        
        Returns:
            Dict mapping signal names to signal types (for dynamic class creation)
        """
        signals = {}
        for part in self.parts.values():
            # Create signal: pyqtSignal(list, list)
            signals[part.signal_name] = pyqtSignal(list, list)
        
        return signals
    
    def reload_registry(self):
        """Reload the registry from YAML (useful for development/testing)"""
        self.parts.clear()
        self.widget_groups.clear()
        self._load_registry()


# Singleton instance for global access
_registry_instance: Optional[PartInspectorRegistry] = None


def get_registry(registry_path: str = "aikensa/core/config/part_registry.yaml") -> PartInspectorRegistry:
    """
    Get the global registry instance (singleton pattern).
    
    Args:
        registry_path: Path to the registry YAML file
    
    Returns:
        PartInspectorRegistry instance
    """
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance = PartInspectorRegistry(registry_path)
    
    return _registry_instance


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize registry
    registry = get_registry()
    
    # Print all registered parts
    print(f"\n{'='*60}")
    print("REGISTERED PARTS")
    print(f"{'='*60}")
    for widget_id, part in registry.get_all_parts().items():
        print(f"Widget {widget_id:2d}: {part.part_name:30s} ({part.inspection_type})")
    
    # Print widget groups
    print(f"\n{'='*60}")
    print("WIDGET GROUPS")
    print(f"{'='*60}")
    for group_name, widget_ids in registry.widget_groups.items():
        print(f"{group_name:30s}: {widget_ids}")
    
    # Test feature checking
    print(f"\n{'='*60}")
    print("FEATURE CHECKS")
    print(f"{'='*60}")
    for widget_id in [5, 6, 17, 18, 31]:
        part = registry.get_part_by_widget(widget_id)
        if part:
            print(f"Widget {widget_id}: {part.part_name}")
            print(f"  - Has Katabu: {part.has_katabu}")
            print(f"  - Narrow Height: {part.narrow_height}")
            print(f"  - Supports Clip Picking: {part.supports_clip_picking}")
            print()
