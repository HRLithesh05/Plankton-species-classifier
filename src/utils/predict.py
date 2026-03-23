"""
Inference script for predicting plankton species from images.

Usage:
    python predict.py path/to/image.png                    # Single image
    python predict.py path/to/folder --batch               # Batch prediction
    python predict.py image.png --model cnn                # Use CNN model
    python predict.py image.png --model svm                # Use SVM model
    python predict.py image.png --top-k 5                  # Show top 5 predictions
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config
from dataset import get_transforms, extract_traditional_features
from models.cnn_model import PlanktonCNN


def load_cnn_model(device: torch.device) -> Tuple[torch.nn.Module, Dict[int, str]]:
    """Load trained CNN model."""
    import torch.nn as nn
    from torchvision import models as tv_models

    # Try Colab-trained model first (best accuracy)
    model_path = config.MODEL_DIR / "cnn_final_colab.pth"
    mapping_path = config.MODEL_DIR / "class_mapping_colab.json"

    # Fallback to other models
    if not model_path.exists():
        model_path = config.MODEL_DIR / "cnn_final.pth"
        mapping_path = config.MODEL_DIR / "class_mapping.json"
    if not model_path.exists():
        model_path = config.MODEL_DIR / "best_model_finetune.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"No CNN model found in {config.MODEL_DIR}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load class mapping from JSON file
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
            num_classes = len(mapping['class_to_idx'])
    else:
        raise FileNotFoundError(f"Class mapping not found: {mapping_path}")

    model_config = checkpoint.get('config', {'model_name': 'efficientnet_v2_s', 'dropout': 0.25})

    # Load weights
    state_dict = checkpoint['model_state_dict']

    # Check if this is a Colab model (has backbone.classifier keys)
    is_colab_model = any(k.startswith('backbone.classifier') for k in state_dict.keys())

    if is_colab_model:
        # Colab model - create EfficientNetV2-S directly
        model = tv_models.efficientnet_v2_s(weights=None)

        in_features = model.classifier[1].in_features
        dropout = model_config.get('dropout', 0.25)

        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes)
        )

        # Remap keys: backbone.X -> X
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)
    else:
        # Local model structure
        model = PlanktonCNN(
            num_classes=num_classes,
            model_name=model_config['model_name'],
            pretrained=False,
            dropout=model_config.get('dropout', 0.25),
            freeze_backbone=False
        )
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"Loaded model: {model_path.name}")

    return model, idx_to_class


def load_traditional_model(model_type: str = 'svm'):
    """Load trained traditional ML model."""
    import joblib
    from models.traditional_model import TraditionalMLClassifier

    model_path = config.MODEL_DIR / f"traditional_ml_{model_type}.joblib"
    scaler_path = config.MODEL_DIR / "traditional_ml_scaler.joblib"
    mapping_path = config.MODEL_DIR / "traditional_ml_class_mapping.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = TraditionalMLClassifier.load(model_path)

    scaler = None
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    idx_to_class = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
            idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}

    return model, scaler, idx_to_class


def predict_cnn(
    image_path: str,
    model: PlanktonCNN,
    device: torch.device,
    idx_to_class: Dict[int, str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Predict species using CNN model."""
    transform = get_transforms(config.IMAGE_SIZE, is_training=False)

    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

    predictions = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        class_name = idx_to_class.get(int(idx), f"Class_{idx}")
        predictions.append((class_name, float(prob)))

    return predictions


def predict_traditional(
    image_path: str,
    model,
    scaler,
    idx_to_class: Dict[int, str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Predict species using traditional ML model."""
    # Extract features
    features = extract_traditional_features(image_path, config.TRADITIONAL_ML_CONFIG)
    features = features.reshape(1, -1)

    # Scale features
    if scaler is not None:
        features = scaler.transform(features)

    # Predict
    probs = model.predict_proba(features)[0]

    # Get top-k predictions
    top_indices = np.argsort(probs)[-top_k:][::-1]

    predictions = []
    for idx in top_indices:
        class_name = idx_to_class.get(int(idx), f"Class_{idx}")
        predictions.append((class_name, float(probs[idx])))

    return predictions


def main(args):
    """Main prediction function."""
    print("Plankton Species Prediction")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if args.model == 'cnn':
        print(f"Loading CNN model...")
        model, idx_to_class = load_cnn_model(device)
        predict_fn = lambda path: predict_cnn(path, model, device, idx_to_class, args.top_k)
    else:
        print(f"Loading {args.model.upper()} model...")
        model, scaler, idx_to_class = load_traditional_model(args.model)
        predict_fn = lambda path: predict_traditional(path, model, scaler, idx_to_class, args.top_k)

    # Get image paths
    input_path = Path(args.input)

    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = list(input_path.glob("*.png")) + \
                     list(input_path.glob("*.jpg")) + \
                     list(input_path.glob("*.jpeg"))
    else:
        print(f"Error: {input_path} not found")
        return

    print(f"Processing {len(image_paths)} image(s)...\n")

    # Make predictions
    results = []

    for img_path in tqdm(image_paths, desc="Predicting"):
        try:
            predictions = predict_fn(str(img_path))

            result = {
                'image': str(img_path),
                'predictions': predictions
            }
            results.append(result)

            # Print results
            print(f"\n{img_path.name}:")
            for i, (class_name, prob) in enumerate(predictions, 1):
                print(f"  {i}. {class_name}: {prob*100:.2f}%")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Save results if batch mode
    if args.batch and len(results) > 1:
        output_path = Path(args.output) if args.output else config.RESULTS_DIR / "predictions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plankton species from images")

    parser.add_argument('input', type=str,
                        help='Image file or directory path')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'svm', 'rf'],
                        help='Model to use for prediction')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode for directory input')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (for batch mode)')

    args = parser.parse_args()
    main(args)
