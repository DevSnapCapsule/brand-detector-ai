#!/usr/bin/env python3
"""
Brand Logo Detection API Test Script
Tests the API with sample images
"""

import requests
import base64
import json
import time
from pathlib import Path
import argparse

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_base64_upload(api_url: str, image_path: str):
    """Test base64 upload endpoint"""
    print(f"\n=== Testing: {image_path} ===")
    
    try:
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        
        # Prepare request data
        data = {"image": image_base64}
        
        # Make request
        response = requests.post(f"{api_url}/detect-base64", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Total brands detected: {result['total_brands']}")
            
            for i, detection in enumerate(result['detections']):
                print(f"  {i+1}. {detection['brand']} (confidence: {detection['confidence']:.3f})")
                print(f"     Object type: {detection['object_type']}")
                print(f"     Bounding box: ({detection['bounding_box']['x1']:.3f}, {detection['bounding_box']['y1']:.3f}) to ({detection['bounding_box']['x2']:.3f}, {detection['bounding_box']['y2']:.3f})")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_health_endpoint(api_url: str):
    """Test health check endpoint"""
    print("\n=== Testing Health Endpoint ===")
    
    try:
        response = requests.get(f"{api_url}/health")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API is healthy!")
            print(f"Model loaded: {result['model_loaded']}")
            print(f"Supported brands: {len(result['supported_brands'])}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_brands_endpoint(api_url: str):
    """Test brands endpoint"""
    print("\n=== Testing Brands Endpoint ===")
    
    try:
        response = requests.get(f"{api_url}/brands")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Total supported brands: {result['total_brands']}")
            print("Supported brands:")
            for brand in result['supported_brands'][:10]:  # Show first 10
                print(f"  - {brand}")
            if len(result['supported_brands']) > 10:
                print(f"  ... and {len(result['supported_brands']) - 10} more")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test Brand Logo Detection API")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image-dir", type=str, default="test_images", help="Directory containing test images")
    
    args = parser.parse_args()
    
    print("🚀 Brand Logo Detection API Test")
    print(f"API URL: {args.api_url}")
    print(f"Test images directory: {args.image_dir}")
    
    # Test health endpoint first
    test_health_endpoint(args.api_url)
    
    # Test brands endpoint
    test_brands_endpoint(args.api_url)
    
    # Test with images
    image_dir = Path(args.image_dir)
    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.webp")) + list(image_dir.glob("*.JPG"))
        
        if not image_files:
            print(f"\n⚠️  No image files found in {args.image_dir}")
            print("Please add some test images (JPG, JPEG, PNG, WEBP) to test the detection functionality.")
            return
        
        print(f"\n📸 Found {len(image_files)} test images")
        
        for image_path in image_files:
            test_base64_upload(args.api_url, str(image_path))
            time.sleep(1)  # Small delay between requests
    else:
        print(f"\n⚠️  Test images directory {args.image_dir} not found")
        print("Please create the directory and add some test images to test the detection functionality.")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()
