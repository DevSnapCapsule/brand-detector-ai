#!/usr/bin/env python3
"""
Simple Training Script for Nike and Puma Brand Detection
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import random
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

def _gather_pairs() -> List[Tuple[Path, Path]]:
    """Gather image/label pairs by basename present in annotations and images."""
    images_dir = Path("annotation_images/images")
    labels_dir = Path("annotation_images/annotations")
    if not images_dir.exists() or not labels_dir.exists():
        return []
    pairs: List[Tuple[Path, Path]] = []
    image_map = {p.stem: p for p in images_dir.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.JPG']}
    for lbl in labels_dir.glob("*.txt"):
        stem = lbl.stem
        if stem in image_map:
            pairs.append((image_map[stem], lbl))
    return pairs


def _read_label_class(lbl_path: Path) -> int:
    """Read first class id from YOLO label file (assumes one object per image here)."""
    try:
        with open(lbl_path, "r") as f:
            line = f.readline().strip()
            if not line:
                return -1
            cls = int(line.split()[0])
            return cls
    except Exception:
        return -1


def split_and_copy_dataset(train_ratio: float = 0.8) -> bool:
    """Stratified split of pairs into train/val and copy into dataset dirs."""
    print("\n📦 Preparing stratified train/val split...")
    pairs = _gather_pairs()
    if not pairs:
        print("❌ No image/label pairs found. Ensure images and matching .txt are present.")
        return False

    class_to_pairs = {0: [], 1: []}
    for img, lbl in pairs:
        c = _read_label_class(lbl)
        if c in class_to_pairs:
            class_to_pairs[c].append((img, lbl))

    # Shuffle per class for randomness
    rng = random.Random(42)
    for c in class_to_pairs:
        rng.shuffle(class_to_pairs[c])

    train_pairs: List[Tuple[Path, Path]] = []
    val_pairs: List[Tuple[Path, Path]] = []
    for c, plist in class_to_pairs.items():
        n = len(plist)
        k = max(1, int(n * train_ratio)) if n > 0 else 0
        train_pairs.extend(plist[:k])
        val_pairs.extend(plist[k:])

    # Ensure dirs
    train_images_dir = Path("dataset/images/train"); train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir = Path("dataset/images/val"); val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir = Path("dataset/labels/train"); train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir = Path("dataset/labels/val"); val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing content to avoid leakage
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        for p in d.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass

    # Copy
    for img, lbl in train_pairs:
        shutil.copy2(img, train_images_dir / img.name)
        shutil.copy2(lbl, train_labels_dir / lbl.name)
    for img, lbl in val_pairs:
        shutil.copy2(img, val_images_dir / img.name)
        shutil.copy2(lbl, val_labels_dir / lbl.name)

    print(f"✅ Train images: {len(train_pairs)} | Val images: {len(val_pairs)}")
    return True

def copy_annotation_labels():
    # Deprecated by split_and_copy_dataset
    return True

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
            epochs=200,
            imgsz=960,
            batch=4,
            name='nike_puma_detector',
            project='training',
            patience=50,
            save=True,
            save_period=10,
            device='cpu',
            workers=0,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=0,
            degrees=10.0,
            translate=0.1,
            scale=0.9,
            shear=2.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            flipud=0.0,
            mixup=0.0,
            mosaic=0.7,
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
    
    # Step 2-3: Prepare stratified split and copy files
    if not split_and_copy_dataset(train_ratio=0.8):
        return
    
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
