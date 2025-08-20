#!/usr/bin/env python3
"""
Quick Brand Detection Test
Simple script to test the trained Nike/Puma model with any image
"""

from ultralytics import YOLO
from pathlib import Path
import sys

def quick_test(image_path, confidence=0.01):
    """Quick test of the model with a single image"""
    print(f"🎯 Quick Brand Detection Test")
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {confidence}")
    print("=" * 50)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load model
    try:
        model = YOLO("models/brand_detector.pt")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run inference
    print(f"\n📸 Running detection...")
    results = model(image_path, conf=confidence)
    
    # Process results
    class_names = ["nike", "puma"]
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"\n✅ Found {len(boxes)} detection(s):")
            for i, box in enumerate(boxes):
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                brand_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                print(f"  {i+1}. {brand_name.upper()}")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Bounding Box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                
                detections.append({
                    'brand': brand_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
        else:
            print(f"\n❌ No brands detected (try lowering confidence threshold)")
    
    return detections

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 quick_test.py <image_path> [confidence_threshold]")
        print("Example: python3 quick_test.py test_images/puma_anot2.webp 0.01")
        return
    
    image_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    
    quick_test(image_path, confidence)

if __name__ == "__main__":
    main()
