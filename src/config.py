import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Model settings
    MODEL_PATH: str = "models/brand_detector.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    NMS_THRESHOLD: float = 0.4
    IMG_SIZE: int = 960
    AUGMENT_TTA: bool = False
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Supported brands (Nike and Puma only)
    SUPPORTED_BRANDS: List[str] = [
        "nike", "puma"
    ]
    
    # Image processing settings
    MAX_IMAGE_SIZE: int = 1920  # Maximum image dimension
    BATCH_SIZE: int = 1
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

# Create settings instance
settings = Settings()
