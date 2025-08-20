#!/usr/bin/env python3
"""
Direct test of trained Nike/Puma model
"""

from ultralytics import YOLO
from pathlib import Path

def test_trained_model():
    """Test the trained model directly"""
    print("🧪 Testing trained Nike/Puma model directly...")
    
    # Load the trained model
    model = YOLO("models/brand_detector.pt")
    
    # Test images
    test_images = [
        "test_images/nike_blk_anot2.jpg",
        "test_images/nike_child_anot1.webp", 
        "test_images/puma_anot2.webp",
        "test_images/puma_my_tee_anot1.JPG"
    ]
    
    # Class names
    class_names = ["nike", "puma"]
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\n📸 Testing: {image_path}")
            
            # Run inference
            results = model(image_path)
            
            # Process results
            found_brands = False
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Get class name
                        brand_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                        
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        print(f"  ✅ Detected: {brand_name}")
                        print(f"     Confidence: {confidence:.3f}")
                        print(f"     Bounding Box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                        found_brands = True
                
                if not found_brands:
                    print(f"  ❌ No brands detected")

def main():
    """Main function"""
    print("🎯 Direct Trained Model Test")
    print("=" * 40)
    test_trained_model()
    
    print("\n📋 Expected Results:")
    print("- nike_blk_anot2.jpg: Should detect Nike")
    print("- nike_child_anot1.webp: Should detect Nike") 
    print("- puma_anot2.webp: Should detect Puma")
    print("- puma_my_tee_anot1.JPG: Should detect Puma")

if __name__ == "__main__":
    main()
