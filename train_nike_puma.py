#!/usr/bin/env python3
"""
Simple Training Script for Nike and Puma Brand Detection
"""

import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

def setup_training_environment():
    """Setup the training environment"""
    print("🔧 Setting up training environment...")
    
    # Create necessary directories
    directories = [
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/labels/train",
        "dataset/labels/val",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def copy_annotation_images():
    """Copy images from annotation_images to dataset"""
    print("\n📸 Copying annotation images to dataset...")
    
    annotation_dir = Path("annotation_images")
    train_images_dir = Path("dataset/images/train")
    val_images_dir = Path("dataset/images/val")
    
    if not annotation_dir.exists():
        print("❌ annotation_images folder not found!")
        print("Please add your annotated images to the annotation_images folder first.")
        return False
    
    # Copy images to training set
    for image_file in annotation_dir.glob("*"):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.JPG']:
            # Copy to training set
            shutil.copy2(image_file, train_images_dir / image_file.name)
            print(f"✅ Copied {image_file.name} to training set")
            
            # Also copy to validation set
            shutil.copy2(image_file, val_images_dir / image_file.name)
            print(f"✅ Copied {image_file.name} to validation set")
    
    return True

def copy_annotation_labels():
    """Copy annotation labels to dataset"""
    print("\n🏷️ Copying annotation labels to dataset...")
    
    annotation_dir = Path("annotation_images")
    train_labels_dir = Path("dataset/labels/train")
    val_labels_dir = Path("dataset/labels/val")
    
    # Copy .txt annotation files
    for label_file in annotation_dir.glob("*.txt"):
        # Copy to training labels
        shutil.copy2(label_file, train_labels_dir / label_file.name)
        print(f"✅ Copied {label_file.name} to training labels")
        
        # Also copy to validation labels
        shutil.copy2(label_file, val_labels_dir / label_file.name)
        print(f"✅ Copied {label_file.name} to validation labels")

def create_dataset_config():
    """Create dataset.yaml configuration file"""
    print("\n⚙️ Creating dataset configuration...")
    
    # Get absolute path to dataset directory
    dataset_path = os.path.abspath("dataset")
    
    yaml_content = f"""# Nike and Puma Brand Detection Dataset
path: {dataset_path}  # dataset root directory
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes - Nike and Puma only
nc: 2  # number of classes
names:
  0: nike
  1: puma
"""
    
    with open("dataset/dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print("✅ Created dataset.yaml configuration")

def train_model():
    """Train the Nike and Puma brand detection model"""
    print("\n🚀 Starting model training...")
    
    try:
        # Load a pre-trained YOLOv8 model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data='dataset/dataset.yaml',
            epochs=100,
            imgsz=640,
            batch=8,
            name='nike_puma_detector',
            project='training',
            patience=20,
            save=True,
            save_period=10,
            device='cpu',
            workers=2,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=False,
            fraction=1.0,
        )
        
        print("✅ Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def copy_trained_model():
    """Copy the trained model to the models directory"""
    print("\n📁 Copying trained model...")
    
    source_path = "training/nike_puma_detector/weights/best.pt"
    target_path = "models/brand_detector.pt"
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
        print(f"✅ Copied trained model to {target_path}")
        return True
    else:
        print(f"❌ Trained model not found at {source_path}")
        return False

def main():
    """Main function to run the training pipeline"""
    print("🎯 Nike and Puma Brand Detection Training")
    print("=" * 50)
    
    # Step 1: Setup environment
    setup_training_environment()
    
    # Step 2: Copy annotation images
    if not copy_annotation_images():
        return
    
    # Step 3: Copy annotation labels
    copy_annotation_labels()
    
    # Step 4: Create dataset config
    create_dataset_config()
    
    # Step 5: Train model
    if train_model():
        # Step 6: Copy trained model
        if copy_trained_model():
            print("\n🎉 Training completed!")
            print("\n📋 Summary:")
            print("- Model trained for Nike and Puma detection")
            print("- Model saved to: models/brand_detector.pt")
            print("- You can now test the model with: python3 test_api.py")
        else:
            print("\n❌ Failed to copy trained model.")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
