# 📚 API Reference

## Overview

The Plankton Species Classifier provides both programmatic APIs and command-line interfaces for integration into research workflows and applications.

## 🔗 REST API Endpoints (Flask)

### Base URL
```
http://localhost:5000  # Development
https://your-domain.com  # Production
```

### Authentication
Currently no authentication required. For production use, implement:
- API keys for rate limiting
- JWT tokens for user management
- OAuth for third-party integrations

---

## 📤 Prediction Endpoints

### Single Image Prediction

**Endpoint:** `POST /api/predict`

**Description:** Classify a single plankton image

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `top_k` (optional): Number of top predictions (default: 5)
- `enhance` (optional): Apply image enhancement (default: true)

**Supported Formats:** JPG, PNG, WEBP, TIFF

**Example Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@plankton_image.jpg" \
  -F "top_k=3"
```

**Response Format:**
```json
{
  "status": "success",
  "predictions": [
    {
      "species": "copepod_calanoid",
      "confidence": 85.34,
      "index": 12
    },
    {
      "species": "diatom_chain",
      "confidence": 8.76,
      "index": 23
    },
    {
      "species": "protist_other",
      "confidence": 3.21,
      "index": 45
    }
  ],
  "processing_time": 0.156,
  "image_info": {
    "width": 224,
    "height": 224,
    "format": "JPEG",
    "enhanced": true
  }
}
```

**Error Responses:**
```json
{
  "status": "error",
  "message": "Invalid image format",
  "code": "INVALID_FORMAT"
}
```

### URL-based Prediction

**Endpoint:** `POST /api/predict-url`

**Parameters:**
```json
{
  "url": "https://example.com/plankton.jpg",
  "top_k": 5,
  "enhance": true
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/plankton.jpg"}'
```

### Batch Prediction

**Endpoint:** `POST /api/predict-batch`

**Description:** Process multiple images in a single request

**Parameters:**
- `files[]`: Multiple image files
- `top_k`: Number of predictions per image

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "filename": "image1.jpg",
      "predictions": [...],
      "processing_time": 0.145
    },
    {
      "filename": "image2.jpg",
      "predictions": [...],
      "processing_time": 0.152
    }
  ],
  "total_processing_time": 0.297
}
```

---

## 📊 Model Information Endpoints

### Model Status

**Endpoint:** `GET /api/model-info`

**Response:**
```json
{
  "status": "loaded",
  "model_info": {
    "name": "EfficientNetV2-S",
    "architecture": "CNN",
    "num_classes": 54,
    "num_parameters": 21458408,
    "input_size": [224, 224, 3],
    "model_path": "outputs/models/cnn_final_colab.pth"
  },
  "performance": {
    "top1_accuracy": 75.5,
    "top3_accuracy": 92.3,
    "top5_accuracy": 96.4,
    "inference_time_ms": 156
  },
  "dataset_info": {
    "name": "WHOI Plankton 2014",
    "num_species": 54,
    "total_images": 20644,
    "train_test_split": "70/15/15"
  }
}
```

### Species Information

**Endpoint:** `GET /api/species`

**Response:**
```json
{
  "species": [
    {
      "index": 0,
      "name": "acantharia_protist",
      "display_name": "Acantharia Protist",
      "description": "Marine protist with radial symmetry",
      "sample_count": 123
    },
    ...
  ],
  "total_species": 54
}
```

### Individual Species Details

**Endpoint:** `GET /api/species/{species_name}`

**Response:**
```json
{
  "name": "copepod_calanoid",
  "display_name": "Calanoid Copepod",
  "index": 12,
  "description": "Small marine crustaceans, key members of zooplankton",
  "characteristics": [
    "Elongated antennae",
    "Segmented body",
    "Swimming appendages"
  ],
  "habitat": "Marine pelagic zones",
  "size_range": "1-5mm",
  "sample_images": [
    "/static/samples/copepod_calanoid_1.jpg",
    "/static/samples/copepod_calanoid_2.jpg"
  ]
}
```

---

## 💻 Python API

### Core Prediction Interface

```python
from src.utils.predict import PlanktonPredictor

# Initialize predictor
predictor = PlanktonPredictor(
    model_path="outputs/models/cnn_final.pth",
    config_path="src/utils/config.py"
)

# Predict single image
result = predictor.predict_image("path/to/image.jpg", top_k=5)

# Predict from URL
result = predictor.predict_url("https://example.com/plankton.jpg")

# Batch prediction
results = predictor.predict_batch([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
])
```

### Model Management

```python
from src.models.cnn_model import PlanktonCNN
from src.utils.config import CNN_CONFIG

# Load model
model = PlanktonCNN(
    num_classes=CNN_CONFIG['num_classes'],
    model_name=CNN_CONFIG['model_name'],
    pretrained=False
)

# Load checkpoint
checkpoint = torch.load("outputs/models/cnn_final.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Set evaluation mode
model.eval()
```

### Data Processing

```python
from src.data.dataset import PlanktonDataset, get_transforms

# Create dataset
dataset = PlanktonDataset(
    data_dir="2014_clean",
    transform=get_transforms(train=False),
    train=False
)

# Get data loader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Process images
for batch_idx, (images, labels) in enumerate(loader):
    # Your processing code here
    pass
```

---

## 🖥️ Command Line Interface

### Prediction Commands

```bash
# Single image prediction
python src/utils/predict.py --image path/to/image.jpg

# Multiple images
python src/utils/predict.py --image img1.jpg img2.jpg img3.jpg

# URL prediction
python src/utils/predict.py --url "https://example.com/plankton.jpg"

# Batch processing with custom output
python src/utils/predict.py \
  --batch-dir /path/to/images/ \
  --output results.json \
  --top-k 3
```

### Model Evaluation

```bash
# Evaluate model performance
python src/utils/evaluate.py --model cnn

# Generate detailed report
python src/utils/evaluate.py \
  --model cnn \
  --output-dir evaluation_results/ \
  --visualize

# Cross-validation
python src/utils/evaluate.py --cross-validate --folds 5
```

### Training Commands

```bash
# Train CNN model
python src/training/train_cnn.py

# Custom training configuration
python src/training/train_cnn.py \
  --config custom_config.py \
  --epochs 50 \
  --batch-size 16

# Resume training
python src/training/train_cnn.py \
  --resume outputs/models/checkpoint_epoch_25.pth

# Traditional ML training
python src/training/train_traditional.py --model svm
```

---

## ⚙️ Configuration API

### Runtime Configuration

```python
from src.utils.config import CNN_CONFIG, update_config

# Modify configuration at runtime
new_config = CNN_CONFIG.copy()
new_config['batch_size'] = 16
new_config['learning_rate'] = 0.001

# Apply configuration
update_config(new_config)
```

### Environment Variables

```bash
# Model configuration
export MODEL_PATH="outputs/models/custom_model.pth"
export BATCH_SIZE=32
export DEVICE="cuda:0"

# Web interface configuration
export FLASK_PORT=5000
export STREAMLIT_PORT=8501
export MAX_UPLOAD_SIZE="10MB"

# Dataset configuration
export DATA_DIR="2014_clean"
export CACHE_DIR="/tmp/plankton_cache"
```

---

## 📋 Error Codes & Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `MODEL_NOT_LOADED` | Model file not found or failed to load | Check model path and file integrity |
| `INVALID_IMAGE` | Image format not supported or corrupted | Use JPG, PNG, WEBP, or TIFF format |
| `PROCESSING_ERROR` | Error during image processing | Check image content and try again |
| `MEMORY_ERROR` | Out of memory during prediction | Reduce batch size or image resolution |
| `NETWORK_ERROR` | Failed to download image from URL | Check URL and network connectivity |

### Error Handling Example

```python
try:
    result = predictor.predict_image("image.jpg")
except ModelNotLoadedError as e:
    print(f"Model loading failed: {e}")
except InvalidImageError as e:
    print(f"Invalid image: {e}")
except ProcessingError as e:
    print(f"Processing failed: {e}")
```

---

## 📈 Performance & Rate Limits

### Response Times
- **Single Prediction**: 100-300ms (GPU), 300-1000ms (CPU)
- **Batch Processing**: ~150ms per image (GPU)
- **Model Loading**: 2-5 seconds (first request)

### Resource Usage
- **Memory**: 2-4GB for model + processing
- **GPU Memory**: 1-2GB for inference
- **CPU Usage**: 1-4 cores depending on batch size

### Rate Limiting (Production)
```python
# Recommended limits
RATE_LIMITS = {
    'requests_per_minute': 60,
    'requests_per_hour': 1000,
    'concurrent_requests': 5,
    'max_file_size': '10MB'
}
```

---

## 🔧 Integration Examples

### Web Application Integration

```javascript
// JavaScript client example
async function classifyPlankton(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    return result;
}
```

### Python Client

```python
import requests

def classify_image(image_path, api_url="http://localhost:5000"):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{api_url}/api/predict", files=files)
        return response.json()

# Usage
result = classify_image("plankton.jpg")
print(f"Predicted species: {result['predictions'][0]['species']}")
```

### Research Pipeline Integration

```python
import pandas as pd
from src.utils.predict import PlanktonPredictor

# Initialize predictor
predictor = PlanktonPredictor()

# Process research dataset
results = []
for image_path in image_list:
    prediction = predictor.predict_image(image_path)
    results.append({
        'image': image_path,
        'species': prediction['predictions'][0]['species'],
        'confidence': prediction['predictions'][0]['confidence']
    })

# Save results
df = pd.DataFrame(results)
df.to_csv('research_results.csv', index=False)
```

---

## 🛡️ Security & Best Practices

### Input Validation
- Validate file types and sizes
- Sanitize filenames and paths
- Check image dimensions and content
- Implement timeout for URL downloads

### Production Deployment
- Use HTTPS in production
- Implement authentication and authorization
- Add rate limiting and monitoring
- Log all requests and responses
- Regular security updates

### Error Logging
```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

## 📞 Support & Troubleshooting

For API-related issues:
1. Check error codes and messages
2. Verify input format and parameters
3. Test with provided examples
4. Check server logs for detailed error information
5. Report issues with complete request/response details