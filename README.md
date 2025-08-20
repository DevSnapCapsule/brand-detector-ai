# Brand Logo Detection API

A production-ready AI-powered brand logo detection API built with FastAPI and YOLOv8. This API can detect brand logos from images across various categories including clothing, vehicles, electronics, and more.

## Features

- 🎯 **Multi-brand Detection**: Detect multiple brands in a single image
- 🚀 **High Performance**: Built with YOLOv8 for fast and accurate detection
- 📱 **Mobile App Ready**: Designed for integration with iOS/Android apps
- 🔧 **Production Ready**: Comprehensive error handling and logging
- 📊 **Detailed Results**: Returns confidence scores, bounding boxes, and object types
- 🔄 **Flexible Input**: Support for both file uploads and base64 encoded images
- 🎓 **Training Ready**: Complete setup for custom model training

## Supported Brands

The API currently supports detection of logos from these categories:

### Clothing & Sports
- Nike, Adidas, Puma, Reebok, Under Armour

### Automotive
- Honda, Toyota, Ford, BMW, Mercedes, Audi, Volkswagen

### Technology
- Apple, Samsung, Sony, LG, Philips, Dell, HP, Lenovo
- Google, Microsoft, Intel, AMD, NVIDIA

### Food & Beverage
- Coca-Cola, Pepsi, McDonald's, KFC, Starbucks, Burger King

### Media & Entertainment
- Netflix, Spotify, YouTube, Facebook, Amazon

### Photography
- Canon, Nikon, Fujifilm

## Quick Start

### Prerequisites

- Python 3.8+
- Training data (for custom model)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Brand_Detector_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   python app.py
   ```

4. **Test the API**
   ```bash
   python test_api.py
   ```

## API Endpoints

### Health Check
```http
GET /health
```
Returns the health status of the API and model.

### Get Supported Brands
```http
GET /brands
```
Returns a list of all supported brand names.

### Detect Brands (File Upload)
```http
POST /detect
Content-Type: multipart/form-data

file: <image_file>
```

### Detect Brands (Base64)
```http
POST /detect-base64
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```

## API Response Format

```json
{
  "success": true,
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
```

## Training Custom Model

To train a custom model for better brand detection:

### 1. Setup Training Environment
```bash
python create_training_data.py
```

### 2. Collect Training Data
- Download images with brand logos (100+ per brand)
- Use sources like Unsplash, Pexels, or stock photo sites
- Organize in `brand_dataset/images/train/` and `brand_dataset/images/val/`

### 3. Annotate Images
```bash
# Install LabelImg annotation tool
pip install labelImg

# Run LabelImg
labelImg
```

**Annotation Process:**
- Load images in LabelImg
- Draw bounding boxes around brand logos
- Label with correct brand name (nike, puma, adidas, etc.)
- Export in YOLO format
- Save annotations in `brand_dataset/labels/train/` and `brand_dataset/labels/val/`

### 4. Train the Model
```bash
python train_brand_model.py
```

### 5. Use Trained Model
The trained model will be saved as `models/brand_detector.pt` and automatically used by the API.

## Testing

### Test with Sample Images

1. **Add your test images** to the `test_images/` directory

2. **Run tests**:
   ```bash
   python test_api.py
   ```

### Test Specific Endpoints

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test brands endpoint
curl http://localhost:8000/brands

# Test base64 upload
python test_api.py
```

## Configuration

The API can be configured through environment variables or the `src/config.py` file:

- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.5)
- `MAX_IMAGE_SIZE`: Maximum image dimension (default: 1920)
- `MODEL_PATH`: Path to custom trained model (default: "models/brand_detector.pt")

## Mobile App Integration

The API is designed for mobile app integration:

### iOS (Swift)
```swift
import Foundation

class BrandDetectorAPI {
    private let baseURL = "https://your-api-domain.com"
    
    func detectBrands(imageData: Data, completion: @escaping (Result<[BrandDetection], Error>) -> Void) {
        let url = URL(string: "\(baseURL)/detect")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle response
        }.resume()
    }
}
```

### Android (Kotlin)
```kotlin
class BrandDetectorAPI {
    private val baseURL = "https://your-api-domain.com"
    
    suspend fun detectBrands(imageFile: File): Result<List<BrandDetection>> {
        return try {
            val client = OkHttpClient()
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", "image.jpg", 
                    RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile))
                .build()
            
            val request = Request.Builder()
                .url("$baseURL/detect")
                .post(requestBody)
                .build()
            
            val response = client.newCall(request).execute()
            // Parse response
            Result.success(emptyList()) // Replace with actual parsing
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

## Project Structure

```
Brand_Detector_AI/
├── app.py                      # Main FastAPI application
├── test_api.py                 # API testing script
├── create_training_data.py     # Training setup script
├── train_brand_model.py        # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── models.py               # Pydantic models
│   └── enhanced_brand_detector.py  # Brand detection logic
├── test_images/                # Test images directory
├── brand_dataset/              # Training dataset (created by setup)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   ├── dataset.yaml            # Dataset configuration
│   └── classes.txt             # Brand classes
└── models/                     # Trained models directory
```

## Performance Optimization

### Model Optimization
- Use TensorRT for GPU acceleration
- Quantize models for faster inference
- Use smaller model variants (YOLOv8n, YOLOv8s) for mobile deployment

### API Optimization
- Implement response caching
- Use async processing for batch requests
- Optimize image preprocessing pipeline

## Monitoring and Logging

The API includes comprehensive logging and monitoring:
- Request/response logging
- Performance metrics
- Error tracking
- Health check endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## Roadmap

- [x] Basic brand detection API
- [x] Training setup and scripts
- [x] Mobile app integration
- [ ] Custom model training pipeline
- [ ] Real-time video processing
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Cloud deployment templates
