import asyncio
import logging
import time
from typing import List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io
from ultralytics import YOLO
import torch
import os
import re

from .models import BrandDetection, BoundingBox
from .config import settings

logger = logging.getLogger(__name__)

class EnhancedBrandDetector:
    """Enhanced brand logo detection using multiple strategies"""
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.is_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Brand-specific keywords and patterns
        self.brand_patterns = {
            "nike": ["nike", "swoosh", "just do it"],
            "adidas": ["adidas", "three stripes", "trefoil"],
            "puma": ["puma", "jumping cat", "puma cat"],
            "reebok": ["reebok", "vector logo"],
            "under_armour": ["under armour", "ua", "protect this house"],
            "honda": ["honda", "honda motor"],
            "toyota": ["toyota", "toyota motor"],
            "ford": ["ford", "ford motor"],
            "bmw": ["bmw", "bmw group"],
            "mercedes": ["mercedes", "mercedes-benz", "benz"],
            "apple": ["apple", "apple logo", "bitten apple"],
            "samsung": ["samsung", "samsung electronics"],
            "sony": ["sony", "sony corporation"],
            "lg": ["lg", "lg electronics"],
            "coca_cola": ["coca-cola", "coke", "coca cola"],
            "pepsi": ["pepsi", "pepsi cola"],
            "mcdonalds": ["mcdonalds", "mcdonald's", "golden arches"],
            "starbucks": ["starbucks", "starbucks coffee"],
            "google": ["google", "google logo"],
            "facebook": ["facebook", "fb", "meta"],
            "amazon": ["amazon", "amazon.com"],
            "netflix": ["netflix", "netflix logo"],
            "spotify": ["spotify", "spotify logo"],
            "youtube": ["youtube", "yt", "youtube logo"],
            "microsoft": ["microsoft", "ms", "windows"],
            "intel": ["intel", "intel inside"],
            "canon": ["canon", "canon camera"],
            "nikon": ["nikon", "nikon camera"],
            "fujifilm": ["fujifilm", "fuji film"]
        }
        
        # Object types that commonly contain brand logos
        self.brand_containing_objects = {
            "person": ["clothing", "shoes", "accessories"],
            "car": ["vehicle", "automotive"],
            "laptop": ["electronics", "computers"],
            "cell_phone": ["electronics", "mobile"],
            "tv": ["electronics", "home entertainment"],
            "bottle": ["beverages", "drinks"],
            "cup": ["beverages", "drinks"],
            "backpack": ["accessories", "bags"],
            "handbag": ["accessories", "bags"],
            "suitcase": ["accessories", "bags"]
        }
        
        logger.info(f"Using device: {self.device}")
    
    async def initialize(self):
        """Initialize the enhanced YOLO model for brand detection"""
        try:
            # Load custom Nike/Puma detector weights
            self.model = YOLO(settings.MODEL_PATH)
            # Define class names for the custom model
            self.class_names = ["nike", "puma"]
            
            # Warm up the model
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_image, verbose=False)
            
            self.is_initialized = True
            logger.info("Enhanced brand detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced brand detector: {e}")
            raise
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for model inference"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = settings.MAX_IMAGE_SIZE
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def analyze_image_for_brands(self, image_array: np.ndarray, detections: List[BrandDetection]) -> List[BrandDetection]:
        """Analyze detected objects for potential brand logos"""
        enhanced_detections = []
        
        for detection in detections:
            # Get the region of interest
            bbox = detection.bounding_box
            h, w = image_array.shape[:2]
            
            x1 = int(bbox.x1 * w)
            y1 = int(bbox.y1 * h)
            x2 = int(bbox.x2 * w)
            y2 = int(bbox.y2 * h)
            
            # Extract the region
            roi = image_array[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Analyze the region for brand indicators
            brand_indicators = self._analyze_region_for_brands(roi, detection)
            
            if brand_indicators:
                # Create enhanced detection with brand information
                enhanced_detection = BrandDetection(
                    brand=brand_indicators["brand"],
                    confidence=detection.confidence * brand_indicators["confidence_multiplier"],
                    bounding_box=detection.bounding_box,
                    object_type=brand_indicators["object_type"]
                )
                enhanced_detections.append(enhanced_detection)
            else:
                # Keep original detection but enhance with brand context
                enhanced_detection = BrandDetection(
                    brand=detection.brand,
                    confidence=detection.confidence,
                    bounding_box=detection.bounding_box,
                    object_type=self._get_enhanced_object_type(detection)
                )
                enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _analyze_region_for_brands(self, roi: np.ndarray, detection: BrandDetection) -> Optional[dict]:
        """Analyze a region for brand indicators"""
        
        # Convert to different color spaces for better analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Analyze based on object type
        if detection.brand == "person":
            return self._analyze_person_for_brands(roi, detection)
        elif detection.brand == "car":
            return self._analyze_car_for_brands(roi, detection)
        elif detection.brand in ["laptop", "cell_phone", "tv"]:
            return self._analyze_electronics_for_brands(roi, detection)
        elif detection.brand in ["bottle", "cup"]:
            return self._analyze_beverages_for_brands(roi, detection)
        
        return None
    
    def _analyze_person_for_brands(self, roi: np.ndarray, detection: BrandDetection) -> Optional[dict]:
        """Analyze person region for clothing brands"""
        
        # Simple color-based analysis for common brand colors
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Nike: Often uses white, black, red
        # Puma: Often uses black, white, red
        # Adidas: Often uses black, white, blue
        
        # Check for red regions (common in sports brands)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        
        red_ratio = np.sum(red_mask > 0) / (red_mask.shape[0] * red_mask.shape[1])
        
        # Check for blue regions (Adidas)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / (blue_mask.shape[0] * blue_mask.shape[1])
        
        # Simple heuristics based on color ratios
        if red_ratio > 0.1 and blue_ratio < 0.05:
            # High red, low blue - likely Nike or Puma
            return {
                "brand": "nike",  # Default to Nike, could be enhanced with more analysis
                "confidence_multiplier": 0.8,
                "object_type": "clothing"
            }
        elif blue_ratio > 0.1:
            # High blue - likely Adidas
            return {
                "brand": "adidas",
                "confidence_multiplier": 0.8,
                "object_type": "clothing"
            }
        
        return None
    
    def _analyze_car_for_brands(self, roi: np.ndarray, detection: BrandDetection) -> Optional[dict]:
        """Analyze car region for automotive brands"""
        # Simple analysis - could be enhanced with logo detection
        return {
            "brand": "honda",  # Default, could be enhanced
            "confidence_multiplier": 0.6,
            "object_type": "vehicle"
        }
    
    def _analyze_electronics_for_brands(self, roi: np.ndarray, detection: BrandDetection) -> Optional[dict]:
        """Analyze electronics for tech brands"""
        return {
            "brand": "apple",  # Default, could be enhanced
            "confidence_multiplier": 0.7,
            "object_type": "electronics"
        }
    
    def _analyze_beverages_for_brands(self, roi: np.ndarray, detection: BrandDetection) -> Optional[dict]:
        """Analyze beverages for drink brands"""
        return {
            "brand": "coca_cola",  # Default, could be enhanced
            "confidence_multiplier": 0.6,
            "object_type": "food_beverage"
        }
    
    def _get_enhanced_object_type(self, detection: BrandDetection) -> str:
        """Get enhanced object type with brand context"""
        if detection.brand == "person":
            return "clothing (potential brand logos)"
        elif detection.brand == "car":
            return "vehicle (potential brand logos)"
        elif detection.brand in ["laptop", "cell_phone", "tv"]:
            return "electronics (potential brand logos)"
        elif detection.brand in ["bottle", "cup"]:
            return "beverages (potential brand logos)"
        else:
            return detection.object_type
    
    def postprocess_detections(self, results, original_size: Tuple[int, int]) -> List[BrandDetection]:
        """Convert YOLO results to BrandDetection objects with brand analysis"""
        detections = []
        
        try:
            # Get the first result (assuming single image input)
            result = results[0]
            
            if result.boxes is None:
                return detections
            
            # Get original image dimensions
            orig_h, orig_w = original_size
            
            for box in result.boxes:
                # Get coordinates (normalized)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter by confidence threshold
                if confidence < settings.CONFIDENCE_THRESHOLD:
                    continue
                
                # Normalize coordinates to [0, 1]
                x1_norm = x1 / orig_w
                y1_norm = y1 / orig_h
                x2_norm = x2 / orig_w
                y2_norm = y2 / orig_h
                
                # Get class name (custom 2-class model)
                class_name = self._get_class_name(class_id)
                
                # Create bounding box
                bbox = BoundingBox(
                    x1=x1_norm,
                    y1=y1_norm,
                    x2=x2_norm,
                    y2=y2_norm
                )
                
                # Create detection
                detection = BrandDetection(
                    brand=class_name,
                    confidence=confidence,
                    bounding_box=bbox,
                    object_type=self._get_object_type(class_name)
                )
                
                detections.append(detection)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"Error postprocessing detections: {e}")
        
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Map class ID to class name for custom Nike/Puma model"""
        if hasattr(self, "class_names") and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"
    
    def _get_object_type(self, class_name: str) -> str:
        """Determine object type based on class name"""
        clothing_brands = ["nike", "adidas", "puma", "reebok", "under_armour"]
        vehicle_brands = ["honda", "toyota", "ford", "bmw", "mercedes", "audi", "volkswagen", "car", "bus", "truck"]
        tech_brands = ["apple", "samsung", "sony", "lg", "philips", "dell", "hp", "lenovo", "google", "microsoft", "intel", "amd", "nvidia", "laptop", "cell_phone", "tv"]
        food_brands = ["coca_cola", "pepsi", "mcdonalds", "kfc", "starbucks", "burger_king"]
        
        if class_name.lower() in clothing_brands or class_name == "person":
            return "clothing"
        elif class_name.lower() in vehicle_brands:
            return "vehicle"
        elif class_name.lower() in tech_brands:
            return "electronics"
        elif class_name.lower() in food_brands:
            return "food_beverage"
        else:
            return "other"
    
    async def detect(self, image_bytes: bytes) -> List[BrandDetection]:
        """Detect brand logos in an image with enhanced analysis"""
        if not self.is_initialized:
            raise RuntimeError("Enhanced brand detector not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            image_array = self.preprocess_image(image_bytes)
            original_size = image_array.shape[:2]  # (height, width)
            
            # Run inference with configured thresholds
            results = self.model(
                image_array,
                conf=settings.CONFIDENCE_THRESHOLD,
                iou=settings.NMS_THRESHOLD,
                imgsz=settings.IMG_SIZE,
                augment=settings.AUGMENT_TTA,
                verbose=False
            )
            
            # Postprocess results
            detections = self.postprocess_detections(results, original_size)
            
            # Enhance detections with brand analysis
            enhanced_detections = self.analyze_image_for_brands(image_array, detections)
            
            processing_time = time.time() - start_time
            logger.info(f"Enhanced detection completed in {processing_time:.3f}s, found {len(enhanced_detections)} detections")
            
            return enhanced_detections
            
        except Exception as e:
            logger.error(f"Error during enhanced brand detection: {e}")
            raise
    
    def get_supported_brands(self) -> List[str]:
        """Get list of supported brand names"""
        return list(self.brand_patterns.keys())
