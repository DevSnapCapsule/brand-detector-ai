from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., description="Top-left x coordinate (normalized)")
    y1: float = Field(..., description="Top-left y coordinate (normalized)")
    x2: float = Field(..., description="Bottom-right x coordinate (normalized)")
    y2: float = Field(..., description="Bottom-right y coordinate (normalized)")
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

class BrandDetection(BaseModel):
    """Individual brand detection result"""
    brand: str = Field(..., description="Detected brand name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    object_type: Optional[str] = Field(None, description="Type of object (e.g., 'clothing', 'vehicle', 'electronics')")
    
    class Config:
        schema_extra = {
            "example": {
                "brand": "nike",
                "confidence": 0.95,
                "bounding_box": {
                    "x1": 0.1,
                    "y1": 0.2,
                    "x2": 0.3,
                    "y2": 0.4
                },
                "object_type": "clothing"
            }
        }

class DetectionResponse(BaseModel):
    """API response for brand detection"""
    success: bool = Field(..., description="Whether the detection was successful")
    detections: List[BrandDetection] = Field(..., description="List of detected brands")
    total_brands: int = Field(..., description="Total number of brands detected")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    image_size: Optional[tuple[int, int]] = Field(None, description="Original image dimensions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "detections": [
                    {
                        "brand": "nike",
                        "confidence": 0.95,
                        "bounding_box": {
                            "x1": 0.1,
                            "y1": 0.2,
                            "x2": 0.3,
                            "y2": 0.4
                        },
                        "object_type": "clothing"
                    }
                ],
                "total_brands": 1,
                "processing_time": 0.15,
                "image_size": [1920, 1080],
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    supported_brands: List[str] = Field(..., description="List of supported brands")
    version: str = Field(..., description="API version")
