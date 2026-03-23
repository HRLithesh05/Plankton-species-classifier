"""
Flask Application for Plankton Species Classification
Integrating the beautiful HTML frontend with PyTorch ML models
"""

import os
import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import requests
import numpy as np

# Optional advanced processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, using basic processing")

try:
    from scipy import ndimage, stats
    from skimage import exposure, filters, morphology, segmentation, feature
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy/Scikit-image not available, using basic processing")

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import the custom CNN model
try:
    from models.cnn_model import PlanktonCNN
    model_available = True
except ImportError:
    model_available = False
    print("Warning: PlanktonCNN not available, using fallback model")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
idx_to_class = None
class_to_idx = None
transform = None

def load_model():
    """Load the trained CNN model with proper state dict handling."""
    global model, idx_to_class, class_to_idx, transform

    model_path = Path("outputs/models/cnn_final_colab.pth")
    mapping_path = Path("outputs/models/class_mapping_colab.json")

    if not model_path.exists():
        model_path = Path("outputs/models/cnn_final.pth")
        mapping_path = Path("outputs/models/class_mapping.json")
    if not model_path.exists():
        model_path = Path("outputs/models/best_model_finetune.pth")

    if not model_path.exists():
        print("Error: No model file found")
        return False

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Load class mappings
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
                class_to_idx = mapping['class_to_idx']
                idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
                num_classes = len(class_to_idx)
        else:
            class_to_idx = checkpoint.get('class_to_idx', {})
            idx_to_class = checkpoint.get('idx_to_class', {})
            if isinstance(list(idx_to_class.keys())[0], str):
                idx_to_class = {int(k): v for k, v in idx_to_class.items()}
            num_classes = checkpoint.get('num_classes', len(class_to_idx))

        config = checkpoint.get('config', {'model_name': 'efficientnet_v2_s', 'dropout': 0.25})
        state_dict = checkpoint['model_state_dict']
        has_backbone = any(k.startswith('backbone.classifier') for k in state_dict.keys())

        if has_backbone:
            # EfficientNetV2-S with backbone structure (Colab model)
            model = models.efficientnet_v2_s(weights=None)
            in_features = model.classifier[1].in_features
            dropout = config.get('dropout', 0.25)
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout * 0.5),
                nn.Linear(512, num_classes)
            )
            new_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval()

            class Wrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.model = m
                def forward(self, x):
                    return self.model(x)

            model = Wrapper(model)
        else:
            # Custom CNN model (local model)
            if model_available:
                model = PlanktonCNN(
                    num_classes=num_classes,
                    model_name=config['model_name'],
                    pretrained=False,
                    dropout=config.get('dropout', 0.25),
                    freeze_backbone=False
                )
                model.load_state_dict(state_dict)
                model.eval()
            else:
                print("Error: PlanktonCNN not available and no fallback option")
                return False

        # Set up robust transforms for better web image handling
        transform = create_robust_transforms()

        print(f"Model loaded successfully with {num_classes} classes")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def normalize_microscopy_image(image):
    """
    Advanced preprocessing to normalize web images to training data standards.
    This creates uniformity between web images and dataset images.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array for advanced processing
    img_array = np.array(image)

    # Step 1: Noise reduction and smoothing (for compression artifacts)
    if CV2_AVAILABLE:
        # Use bilateral filter to reduce noise while preserving edges
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
    else:
        # Fallback: use PIL gaussian blur
        temp_img = Image.fromarray(img_array)
        temp_img = temp_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img_array = np.array(temp_img)

    # Step 2: Contrast and brightness normalization
    if SCIPY_AVAILABLE:
        # Histogram equalization on each channel
        for i in range(3):  # RGB channels
            img_array[:,:,i] = exposure.equalize_hist(img_array[:,:,i]) * 255
    else:
        # Fallback: basic contrast enhancement
        img = Image.fromarray(img_array.astype(np.uint8))
        img = ImageOps.autocontrast(img, cutoff=3)
        img_array = np.array(img)

    # Step 3: Color normalization (match training data color distribution)
    # Normalize each channel to have similar statistics as training data
    target_mean = [120, 130, 125]  # Typical microscopy image means
    target_std = [45, 50, 48]      # Typical microscopy image std

    for i in range(3):
        channel = img_array[:,:,i].astype(np.float32)
        current_mean = np.mean(channel)
        current_std = np.std(channel)

        if current_std > 0:
            # Normalize to target statistics
            normalized_channel = (channel - current_mean) / current_std
            normalized_channel = normalized_channel * target_std[i] + target_mean[i]

            # Clip to valid range
            normalized_channel = np.clip(normalized_channel, 0, 255)
            img_array[:,:,i] = normalized_channel.astype(np.uint8)

    # Step 4: Background normalization
    if SCIPY_AVAILABLE:
        # Try to detect and normalize background
        try:
            # Convert to grayscale for background detection
            gray = np.mean(img_array, axis=2)

            # Use Otsu's method to separate foreground from background
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(gray)

            # Create mask for background pixels
            background_mask = gray > thresh

            # Normalize background to be more uniform
            if np.sum(background_mask) > 0:
                for i in range(3):
                    channel = img_array[:,:,i].astype(np.float32)
                    bg_mean = np.mean(channel[background_mask])

                    # Adjust background to typical microscopy background (light gray)
                    target_bg = 240
                    adjustment = target_bg - bg_mean
                    channel[background_mask] += adjustment

                    # Clip values
                    channel = np.clip(channel, 0, 255)
                    img_array[:,:,i] = channel.astype(np.uint8)
        except Exception as e:
            print(f"Background normalization failed: {e}")

    # Step 5: Edge enhancement (important for microscopic features)
    if CV2_AVAILABLE:
        # Subtle edge enhancement
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])

        for i in range(3):
            enhanced = cv2.filter2D(img_array[:,:,i], -1, kernel)
            enhanced = np.clip(enhanced, 0, 255)
            img_array[:,:,i] = enhanced.astype(np.uint8)
    else:
        # Fallback: use PIL unsharp mask
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=20, threshold=3))
        img_array = np.array(img)

    # Step 6: Final gamma correction (microscopy images often need this)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.power(img_array, 0.9)  # Slight gamma correction
    img_array = (img_array * 255).astype(np.uint8)

    return Image.fromarray(img_array)

def analyze_image_quality(image):
    """Analyze image quality and suggest preprocessing adjustments."""
    img_array = np.array(image)

    # Calculate image statistics
    mean_brightness = np.mean(img_array)
    contrast = np.std(img_array)

    # Detect if image is too dark, too bright, or low contrast
    quality_issues = []

    if mean_brightness < 80:
        quality_issues.append("too_dark")
    elif mean_brightness > 180:
        quality_issues.append("too_bright")

    if contrast < 30:
        quality_issues.append("low_contrast")
    elif contrast > 80:
        quality_issues.append("high_contrast")

    # Check for color cast
    r_mean, g_mean, b_mean = np.mean(img_array[:,:,0]), np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
    if max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean) > 20:
        quality_issues.append("color_cast")

    return quality_issues

def enhance_image_advanced(image):
    """
    Enhanced image processing with quality-aware adjustments.
    This is the main function that creates uniformity with dataset images.
    """
    # Analyze image quality first
    quality_issues = analyze_image_quality(image)

    # Apply targeted fixes based on detected issues
    if "too_dark" in quality_issues:
        image = ImageEnhance.Brightness(image).enhance(1.3)
    elif "too_bright" in quality_issues:
        image = ImageEnhance.Brightness(image).enhance(0.8)

    if "low_contrast" in quality_issues:
        image = ImageEnhance.Contrast(image).enhance(1.4)
    elif "high_contrast" in quality_issues:
        image = ImageEnhance.Contrast(image).enhance(0.8)

    if "color_cast" in quality_issues:
        # Apply color balancing
        image = ImageOps.autocontrast(image, cutoff=2)

    # Apply comprehensive normalization
    image = normalize_microscopy_image(image)

    return image

def create_robust_transforms():
    """Create more robust transforms that better handle web images."""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Larger initial size
        transforms.CenterCrop(224),     # Center crop like validation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image_with_tta(image, top_k=5):
    """Predict with Test-Time Augmentation and advanced preprocessing for uniformity."""
    global model, idx_to_class

    if model is None or idx_to_class is None:
        return None

    try:
        print("Applying advanced image normalization for uniformity...")

        # Apply comprehensive preprocessing for uniformity
        enhanced_image = enhance_image_advanced(image)

        # Create multiple augmented versions for TTA (now with normalized base)
        tta_transforms = [
            # Original normalized
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Center crop (matches training validation exactly)
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Multiple scale approach (robust to size variations)
            transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Slight brightness variation (handles remaining lighting differences)
            transforms.Compose([
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        ]

        # Collect predictions from all augmentations
        all_predictions = []

        with torch.no_grad():
            for i, transform in enumerate(tta_transforms):
                img_tensor = transform(enhanced_image).unsqueeze(0)
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                all_predictions.append(probabilities)

        # Weighted average predictions (give more weight to center crop)
        weights = [1.0, 2.0, 1.5, 1.0]  # Center crop gets highest weight
        weighted_predictions = []

        for i, pred in enumerate(all_predictions):
            weighted_predictions.append(pred * weights[i])

        # Calculate weighted average
        total_weight = sum(weights)
        avg_predictions = torch.sum(torch.stack(weighted_predictions), dim=0) / total_weight

        top_probs, top_indices = torch.topk(avg_predictions, top_k)

        # Format predictions with confidence calibration
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            # Apply confidence calibration for web images
            calibrated_confidence = calibrate_confidence(prob.item())

            predictions.append({
                'species': idx_to_class[idx.item()],
                'confidence': float(calibrated_confidence * 100)
            })

        print(f"Prediction completed using advanced normalization. Top confidence: {predictions[0]['confidence']:.1f}%")
        return predictions

    except Exception as e:
        print(f"Error during advanced TTA prediction: {e}")
        # Fallback to simple prediction
        return predict_image_simple(image, top_k)

def calibrate_confidence(raw_confidence):
    """
    Calibrate confidence scores for web images to be more realistic.
    Web images typically should have slightly lower confidence than dataset images.
    """
    # Apply sigmoid-like calibration
    calibrated = raw_confidence * 0.9  # Slight reduction for web images

    # Ensure minimum confidence threshold
    if calibrated < 0.1:
        calibrated = 0.1

    return calibrated

def predict_image_simple(image, top_k=5):
    """Fallback simple prediction method with advanced preprocessing."""
    global model, idx_to_class, transform

    if model is None or idx_to_class is None:
        return None

    try:
        # Apply advanced preprocessing for uniformity
        image = enhance_image_advanced(image)
        img_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)

        # Format predictions with confidence calibration
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            calibrated_confidence = calibrate_confidence(prob.item())
            predictions.append({
                'species': idx_to_class[idx.item()],
                'confidence': float(calibrated_confidence * 100)
            })

        return predictions

    except Exception as e:
        print(f"Error during simple prediction: {e}")
        return None

def load_image_from_url(url):
    """Load image from URL with detailed error handling."""
    try:
        # Check for problematic domains
        problematic_domains = ['shutterstock.com', 'gettyimages.com', 'adobe.com', 'istockphoto.com']
        if any(domain in url.lower() for domain in problematic_domains):
            return None, "Protected content: This image is from a stock photo service and cannot be accessed directly. Please download and upload the image file instead."

        # Add better headers to avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/*, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        response = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()

        # Check if content is actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return None, f"Invalid content type: Expected image, got {content_type}"

        # Check file size (limit to 50MB)
        if len(response.content) > 50 * 1024 * 1024:
            return None, "Image too large: Maximum size is 50MB"

        return Image.open(BytesIO(response.content)), None

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 403:
            return None, "Access forbidden: This image is protected and cannot be accessed directly. Try downloading and uploading the image file instead."
        elif status_code == 404:
            return None, "Image not found: The URL does not point to a valid image."
        elif status_code == 429:
            return None, "Rate limited: Too many requests to this server. Please try again later."
        else:
            return None, f"HTTP Error {status_code}: Unable to access the image."

    except requests.exceptions.ConnectionError:
        return None, "Connection failed: Unable to reach the image server. Check your internet connection."

    except requests.exceptions.Timeout:
        return None, "Request timeout: The image server took too long to respond. Try again or use a different image."

    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"

    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None, f"Image processing error: {str(e)}"

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index_enhanced.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for image prediction."""
    try:
        image = None

        # Check for uploaded file
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            image = Image.open(file.stream)

        # Check for URL (only if request has JSON data)
        elif request.json and 'url' in request.json:
            url = request.json['url']
            image, error_msg = load_image_from_url(url)
            if image is None:
                return jsonify({'error': error_msg or 'Could not load image from URL'}), 400

        # Check for base64 image (only if request has JSON data)
        elif request.json and 'image_data' in request.json:
            image_data = request.json['image_data']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))

        # Validate image was loaded
        if image is None:
            return jsonify({'error': 'No valid image provided. Please upload a file, provide a URL, or send base64 image data.'}), 400

        # Make prediction with Test-Time Augmentation for better web image handling
        predictions = predict_image_with_tta(image)

        if predictions is None:
            return jsonify({'error': 'Failed to make prediction. Model may not be loaded correctly.'}), 500

        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None
        })

    except Exception as e:
        print(f"Error in prediction API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model-info')
def model_info():
    """Get model information."""
    return jsonify({
        'available': model is not None,
        'classes': len(idx_to_class) if idx_to_class else 0,
        'architecture': 'EfficientNetV2-S',
        'accuracy': {
            'top1': 75.5,
            'top3': 92.3,
            'top5': 96.4
        }
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

# Initialize the model when the app starts
print("Loading model...")
model_loaded = load_model()
if not model_loaded:
    print("Warning: Model failed to load. Predictions will not work.")
else:
    print("Model loaded successfully!")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)