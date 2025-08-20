#!/usr/bin/env python3
"""
Comprehensive Brand Logo Detection Test Script
Tests the trained Nike/Puma detection model with various options
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BrandDetectorTester:
    def __init__(self, model_path="models/brand_detector.pt"):
        """Initialize the brand detector tester"""
        self.model_path = model_path
        self.class_names = ["nike", "puma"]
        self.colors = [(255, 0, 0), (0, 255, 0)]  # Blue for Nike, Green for Puma
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def test_single_image(self, image_path, confidence_threshold=0.5, save_result=True):
        """Test a single image and return results"""
        print(f"\n📸 Testing: {image_path}")
        
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Run inference
        results = self.model(image_path, conf=confidence_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    brand_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'brand': brand_name,
                        'confidence': confidence,
                        'bounding_box': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        }
                    }
                    detections.append(detection)
                    
                    print(f"  ✅ Detected: {brand_name}")
                    print(f"     Confidence: {confidence:.3f}")
                    print(f"     Bounding Box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
        
        if not detections:
            print(f"  ❌ No brands detected (confidence threshold: {confidence_threshold})")
        
        # Save result image if requested
        if save_result and detections:
            self.save_result_image(image_path, detections)
        
        return {
            'image_path': image_path,
            'total_detections': len(detections),
            'detections': detections,
            'confidence_threshold': confidence_threshold
        }
    
    def save_result_image(self, image_path, detections):
        """Save the result image with bounding boxes"""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return
        
        # Draw bounding boxes
        for detection in detections:
            brand = detection['brand']
            confidence = detection['confidence']
            bbox = detection['bounding_box']
            
            # Get color based on brand
            color_idx = 0 if brand == "nike" else 1
            color = self.colors[color_idx]
            
            # Draw rectangle
            cv2.rectangle(image, 
                         (int(bbox['x1']), int(bbox['y1'])), 
                         (int(bbox['x2']), int(bbox['y2'])), 
                         color, 2)
            
            # Add label
            label = f"{brand.upper()} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(image, 
                         (int(bbox['x1']), int(bbox['y1'] - label_size[1] - 10)), 
                         (int(bbox['x1'] + label_size[0]), int(bbox['y1'])), 
                         color, -1)
            
            # Draw label text
            cv2.putText(image, label, 
                       (int(bbox['x1']), int(bbox['y1'] - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save result
        output_path = f"test_results/{Path(image_path).stem}_result.jpg"
        Path("test_results").mkdir(exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"  💾 Result saved to: {output_path}")
    
    def test_batch(self, image_dir, confidence_threshold=0.5):
        """Test a batch of images"""
        print(f"\n📁 Testing batch from: {image_dir}")
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            return
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.JPG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images to test")
        
        # Test each image
        results = []
        for image_path in image_files:
            result = self.test_single_image(str(image_path), confidence_threshold, save_result=True)
            if result:
                results.append(result)
        
        # Generate summary
        self.generate_summary(results)
        
        return results
    
    def generate_summary(self, results):
        """Generate a summary of test results"""
        if not results:
            return
        
        print(f"\n📊 Test Summary")
        print("=" * 50)
        
        total_images = len(results)
        total_detections = sum(r['total_detections'] for r in results)
        images_with_detections = sum(1 for r in results if r['total_detections'] > 0)
        
        print(f"Total images tested: {total_images}")
        print(f"Images with detections: {images_with_detections}")
        print(f"Total detections: {total_detections}")
        print(f"Detection rate: {images_with_detections/total_images*100:.1f}%")
        
        # Brand breakdown
        brand_counts = {}
        for result in results:
            for detection in result['detections']:
                brand = detection['brand']
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        print(f"\nBrand breakdown:")
        for brand, count in brand_counts.items():
            print(f"  {brand.upper()}: {count} detections")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results/test_results_{timestamp}.json"
        Path("test_results").mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
    
    def test_confidence_thresholds(self, image_path, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """Test different confidence thresholds on a single image"""
        print(f"\n🎯 Testing confidence thresholds on: {image_path}")
        print("=" * 60)
        
        results = {}
        for threshold in thresholds:
            print(f"\nConfidence threshold: {threshold}")
            result = self.test_single_image(image_path, threshold, save_result=False)
            results[threshold] = result
        
        # Display comparison
        print(f"\n📈 Confidence Threshold Comparison")
        print("-" * 40)
        for threshold, result in results.items():
            if result:
                print(f"Threshold {threshold}: {result['total_detections']} detections")
            else:
                print(f"Threshold {threshold}: No detections")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Test Brand Logo Detection Model")
    parser.add_argument("--model", type=str, default="models/brand_detector.pt", 
                       help="Path to trained model")
    parser.add_argument("--image", type=str, help="Single image to test")
    parser.add_argument("--batch", type=str, default="test_images", 
                       help="Directory containing images to test")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--test-thresholds", action="store_true", 
                       help="Test different confidence thresholds")
    parser.add_argument("--no-save", action="store_true", 
                       help="Don't save result images")
    
    args = parser.parse_args()
    
    print("🎯 Brand Logo Detection Model Tester")
    print("=" * 50)
    
    try:
        # Initialize tester
        tester = BrandDetectorTester(args.model)
        
        if args.image:
            # Test single image
            if args.test_thresholds:
                tester.test_confidence_thresholds(args.image)
            else:
                tester.test_single_image(args.image, args.confidence, not args.no_save)
        
        elif args.batch:
            # Test batch of images
            tester.test_batch(args.batch, args.confidence)
        
        else:
            # Default: test all images in test_images directory
            tester.test_batch("test_images", args.confidence)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("\n✅ Testing completed!")
    return 0

if __name__ == "__main__":
    exit(main())
