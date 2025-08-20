from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
from typing import List, Dict, Any
import base64
from io import BytesIO

from src.enhanced_brand_detector import EnhancedBrandDetector as BrandDetector
from src.models import DetectionResponse, BrandDetection
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brand Logo Detection API",
    description="AI-powered brand logo detection from images",
    version="1.0.0"
)

# Add CORS middleware for mobile app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize brand detector
brand_detector = BrandDetector()

@app.on_event("startup")
async def startup_event():
    """Initialize the brand detector on startup"""
    try:
        await brand_detector.initialize()
        logger.info("Brand detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize brand detector: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Brand Logo Detection API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": brand_detector.is_initialized,
        "supported_brands": brand_detector.get_supported_brands()
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_brands(file: UploadFile = File(...)):
    """
    Detect brand logos in an uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        DetectionResponse: List of detected brands with confidence scores and bounding boxes
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Detect brands
        detections = await brand_detector.detect(image_data)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            total_brands=len(detections)
        )
        
    except Exception as e:
        logger.error(f"Error in brand detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-base64", response_model=DetectionResponse)
async def detect_brands_base64(image_data: Dict[str, str]):
    """
    Detect brand logos from base64 encoded image
    
    Args:
        image_data: Dictionary containing base64 encoded image
    
    Returns:
        DetectionResponse: List of detected brands with confidence scores and bounding boxes
    """
    try:
        if "image" not in image_data:
            raise HTTPException(status_code=400, detail="Image data not provided")
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data["image"])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Detect brands
        detections = await brand_detector.detect(image_bytes)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            total_brands=len(detections)
        )
        
    except Exception as e:
        logger.error(f"Error in brand detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/brands")
async def get_supported_brands():
    """Get list of supported brand logos"""
    return {
        "supported_brands": brand_detector.get_supported_brands(),
        "total_brands": len(brand_detector.get_supported_brands())
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
