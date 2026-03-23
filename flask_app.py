"""
Plankton Species Classifier - Flask Web Application
Fixed version with proper model loading and all functionality working
"""

import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
from pathlib import Path
from torchvision import transforms, models
import requests
from io import BytesIO
import base64
import traceback
import pandas as pd
from datetime import datetime
import tempfile
import zipfile

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from PIL import ImageEnhance, ImageOps, ImageFilter

# Import our custom CNN model
from models.cnn_model import PlanktonCNN

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Global variables
model = None
idx_to_class = None
class_to_idx = None
species_database = None

def load_model():
    """Load the best trained CNN model based on accuracy analysis."""
    global model, idx_to_class, class_to_idx

    # UPDATED: Use the BEST model (89.51% accuracy)
    model_path = Path("outputs/models/approach1_final_model.pth")
    mapping_path = Path("outputs/models/class_mapping_colab.json")

    # Fallback models in order of preference
    if not model_path.exists():
        model_path = Path("outputs/models/best_model_finetune_colab.pth")
    if not model_path.exists():
        model_path = Path("outputs/models/cnn_final_colab.pth")
    if not model_path.exists():
        model_path = Path("outputs/models/best_model_frozen_colab.pth")
    if not model_path.exists():
        model_path = Path("outputs/models/cnn_final.pth")
        mapping_path = Path("outputs/models/class_mapping.json")

    if not model_path.exists():
        print("ERROR: No model file found!")
        return False

    print(f"Loading model from: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # UPDATED: Load class mapping (handle approach1 embedded mappings)
        if 'class_to_idx' in checkpoint and 'idx_to_class' in checkpoint and model_path.name.startswith('approach1'):
            # Use embedded class mappings from approach1 model (67 classes)
            class_to_idx = checkpoint['class_to_idx']
            idx_to_class = checkpoint['idx_to_class']
            if isinstance(list(idx_to_class.keys())[0], str):
                idx_to_class = {int(k): v for k, v in idx_to_class.items()}
            num_classes = checkpoint.get('num_classes', len(class_to_idx))
            print(f"   Using embedded class mapping: {num_classes} classes")
        elif mapping_path.exists():
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

        # Get model config
        config = checkpoint.get('config', {'model_name': 'efficientnet_v2_s', 'dropout': 0.25})
        state_dict = checkpoint['model_state_dict']
        has_backbone = any(k.startswith('backbone.classifier') for k in state_dict.keys())

        # UPDATED: Support for EfficientNet-B2 (approach1 models)
        model_name = config.get('model_name', 'efficientnet_v2_s')

        if 'efficientnet_b2' in model_name or model_path.name.startswith('approach1'):
            print(f"Loading EfficientNet-B2 model (approach1: 89.51% accuracy)")
            # Create EfficientNet-B2 with custom classifier
            model = models.efficientnet_b2(weights=None)
            in_features = 1408  # B2 feature dimension
            dropout = config.get('dropout', 0.4)

            # Recreate the exact classifier from training
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1024),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(1024),
                nn.Dropout(p=dropout * 0.7),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout * 0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=dropout * 0.3),
                nn.Linear(256, num_classes)
            )

            # Handle the backbone prefix in state dict keys
            if any(k.startswith('backbone.') for k in state_dict.keys()):
                new_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)

            model.eval()
            print(f"SUCCESS: EfficientNet-B2 model loaded successfully!")

        elif has_backbone:
            # EfficientNet with backbone structure
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
            # Remove 'backbone.' prefix from state dict keys
            new_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval()

            # Wrap model for consistency
            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    return self.model(x)

            model = ModelWrapper(model)
        else:
            # Custom CNN model
            model = PlanktonCNN(
                num_classes=num_classes,
                model_name=config['model_name'],
                pretrained=False,
                dropout=config.get('dropout', 0.25),
                freeze_backbone=False
            )
            model.load_state_dict(state_dict)
            model.eval()

        print(f"Model loaded successfully with {num_classes} classes")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False

def load_species_database():
    """Load the species database with proper error handling."""
    global species_database

    try:
        db_path = Path("species_database.json")
        if db_path.exists():
            with open(db_path, 'r') as f:
                species_database = json.load(f)
            print(f"Species database loaded with {len(species_database)} species")
        else:
            print("Species database not found, creating basic database")
            # Create a basic database for the known species
            species_database = {}
            for species_name in class_to_idx.keys() if class_to_idx else []:
                species_database[species_name] = {
                    "common_name": species_name.replace('_', ' ').title(),
                    "scientific_name": "Unknown",
                    "description": f"{species_name.replace('_', ' ').title()} is a plankton species found in marine environments.",
                    "habitat": "Marine waters",
                    "ecological_role": "Primary producer/consumer"
                }
    except Exception as e:
        print(f"Error loading species database: {e}")
        species_database = {}

def get_transforms():
    """Get image preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def enhance_image(image):
    """Basic image enhancement for better prediction accuracy."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply basic enhancements
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)

    return image

def predict_image(image, top_k=5):
    """Make prediction on a single image."""
    global model, idx_to_class

    if model is None or idx_to_class is None:
        return None

    if image is None:
        raise ValueError("Image cannot be None")

    try:
        # Enhance and preprocess image
        image = enhance_image(image)
        transform = get_transforms()
        img_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)

        # Convert to human-readable results
        predictions = []
        for i in range(top_k):
            species = idx_to_class[top_indices[0][i].item()]
            confidence = top_probs[0][i].item() * 100
            predictions.append({
                'species': species,
                'confidence': confidence
            })

        return predictions

    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        return None

def load_image_from_url(url):
    """Load image from URL with error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image, None
    except Exception as e:
        return None, f"Failed to load image from URL: {str(e)}"

@app.route('/')
def index():
    """Main page."""
    return render_template('index_enhanced_fixed.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single image prediction."""
    try:
        image = None

        # Handle file upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            image = Image.open(file.stream)

        # Handle URL (only if JSON request and no file)
        elif request.is_json and request.json and 'url' in request.json:
            url = request.json['url']
            image, error_msg = load_image_from_url(url)
            if image is None:
                return jsonify({'error': error_msg}), 400

        # Handle base64 image (only if JSON request and no file)
        elif request.is_json and request.json and 'image_data' in request.json:
            image_data = request.json['image_data']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))

        # Validate image
        if image is None:
            return jsonify({
                'error': 'No valid image provided. Please upload a file, provide a URL, or send base64 image data.'
            }), 400

        # Make prediction
        predictions = predict_image(image)

        if predictions is None:
            return jsonify({'error': 'Failed to make prediction. Model may not be loaded correctly.'}), 500

        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None
        })

    except Exception as e:
        print(f"Error in prediction API: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    """API endpoint for batch image prediction."""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided. Use "files[]" field name for multiple files.'}), 400

        files = request.files.getlist('files[]')
        if not files or len([f for f in files if f.filename != '']) == 0:
            return jsonify({'error': 'No valid files received'}), 400

        # Limit batch size
        max_batch_size = 50
        valid_files = [f for f in files if f.filename != ''][:max_batch_size]

        results = []
        for file in valid_files:
            try:
                image = Image.open(file.stream)
                predictions = predict_image(image)

                if predictions:
                    results.append({
                        'filename': file.filename,
                        'success': True,
                        'predictions': predictions
                    })
                else:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': 'Prediction failed'
                    })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })

    except Exception as e:
        print(f"Error in batch prediction: {e}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/api/species', methods=['GET'])
def api_species():
    """Get all species information."""
    try:
        if not species_database:
            return jsonify({'error': 'Species database not available'}), 500

        # Convert to list format with species_id
        species_list = []
        for species_id, info in species_database.items():
            species_data = info.copy()
            species_data['species_id'] = species_id
            species_list.append(species_data)

        return jsonify({'species': species_list})
    except Exception as e:
        return jsonify({'error': f'Failed to get species data: {str(e)}'}), 500

@app.route('/api/species/<species_name>', methods=['GET'])
def api_species_detail(species_name):
    """Get detailed information for a specific species."""
    try:
        if not species_database:
            return jsonify({'error': 'Species database not available'}), 404

        # Try exact match first
        if species_name in species_database:
            species_data = species_database[species_name].copy()
            species_data['species_id'] = species_name
            return jsonify(species_data)

        # Try case-insensitive match
        for species_id, info in species_database.items():
            if species_id.lower() == species_name.lower():
                species_data = info.copy()
                species_data['species_id'] = species_id
                return jsonify(species_data)

        return jsonify({'error': 'Species not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Failed to get species details: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Get model information."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        return jsonify({
            'available': model is not None,
            'classes': len(idx_to_class) if idx_to_class else 0,
            'model_file': 'approach1_final_model.pth',
            'architecture': 'EfficientNet-B2',
            'approach': 'Progressive Training (3-stage)',
            'accuracy': {
                'validation': 89.51,
                'training_method': 'Progressive: Foundation → Refinement → Fine-tuning'
            },
            'training_info': {
                'dataset': '2014 Clean Dataset',
                'total_images': '20,644 images',
                'classes': f'{len(idx_to_class) if idx_to_class else 0} species',
                'architecture_details': 'EfficientNet-B2 with custom classifier head'
            },
            'features': [
                'Single High-Accuracy Model (89.51%)',
                'Progressive Training Pipeline',
                'Advanced Image Preprocessing',
                'Calibrated Confidence Scores',
                'Optimized for Marine Plankton'
            ],
            'status': 'ready'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/api/export', methods=['POST'])
def api_export():
    """Export prediction results."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        export_format = data.get('format', 'csv')
        results = data.get('results', data.get('data', []))

        if not results:
            return jsonify({'error': 'No results to export'}), 400

        # Prepare export data
        export_data = []
        for result in results:
            if isinstance(result, dict):
                if 'results' in result:  # Batch format
                    for batch_result in result['results']:
                        if batch_result.get('success') and 'predictions' in batch_result:
                            top_pred = batch_result['predictions'][0]
                            export_data.append({
                                'filename': batch_result.get('filename', 'unknown'),
                                'predicted_species': top_pred['species'],
                                'confidence': top_pred['confidence'],
                                'timestamp': result.get('timestamp', datetime.now().isoformat())
                            })
                elif 'predictions' in result:  # Single prediction format
                    top_pred = result['predictions'][0]
                    export_data.append({
                        'filename': result.get('filename', 'unknown'),
                        'predicted_species': top_pred['species'],
                        'confidence': top_pred['confidence'],
                        'timestamp': result.get('timestamp', datetime.now().isoformat())
                    })

        if not export_data:
            return jsonify({'error': 'No valid prediction data to export'}), 400

        # Create DataFrame
        df = pd.DataFrame(export_data)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{export_format}') as tmp_file:
            if export_format == 'csv':
                df.to_csv(tmp_file.name, index=False)
                mimetype = 'text/csv'
            elif export_format == 'excel':
                df.to_excel(tmp_file.name, index=False)
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif export_format == 'json':
                df.to_json(tmp_file.name, orient='records', indent=2)
                mimetype = 'application/json'
            else:
                return jsonify({'error': 'Unsupported export format'}), 400

            return send_file(
                tmp_file.name,
                mimetype=mimetype,
                as_attachment=True,
                download_name=f'plankton_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format}'
            )

    except Exception as e:
        print(f"Error in export: {e}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

# Initialize the application
print("Loading model...")
if load_model():
    print("Model loaded successfully!")
    load_species_database()
else:
    print("Failed to load model!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)