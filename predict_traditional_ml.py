"""
Traditional ML Model Inference Script
=====================================
Test the trained traditional ML model on new images
"""

import numpy as np
import pickle
from skimage import io, color, feature, transform
from skimage.measure import label, regionprops
import sys
import os

# Load the trained model
MODEL_PATH = "outputs/models/traditional_ml_model.pkl"

print("Loading traditional ML model...")
with open(MODEL_PATH, 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
scaler = model_package['scaler']
label_encoder = model_package['label_encoder']
IMG_SIZE = model_package['img_size']

print("Model loaded successfully")
print(f"  Validation Accuracy: {model_package['validation_accuracy']*100:.2f}%")
print(f"  Number of classes: {model_package['num_classes']}")


def extract_hog_features(image):
    """Extract HOG features"""
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    features = feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features


def extract_color_histogram(image):
    """Extract color histogram features"""
    if len(image.shape) == 2:
        hist = np.histogram(image, bins=32, range=(0, 1))[0]
        return hist.flatten()

    hist_features = []
    for channel in range(3):
        hist = np.histogram(image[:, :, channel], bins=8, range=(0, 1))[0]
        hist_features.extend(hist)

    return np.array(hist_features)


def extract_shape_features(image):
    """Extract basic shape features"""
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    thresh = gray > 0.3
    labeled = label(thresh)
    regions = regionprops(labeled)

    if len(regions) == 0:
        return np.array([0, 0, 0, 0, 0])

    region = max(regions, key=lambda r: r.area)
    area = region.area / (IMG_SIZE[0] * IMG_SIZE[1])
    perimeter = region.perimeter / (2 * (IMG_SIZE[0] + IMG_SIZE[1]))
    eccentricity = region.eccentricity
    solidity = region.solidity
    extent = region.extent

    return np.array([area, perimeter, eccentricity, solidity, extent])


def extract_all_features(image):
    """Extract all features"""
    hog_features = extract_hog_features(image)
    color_features = extract_color_histogram(image)
    shape_features = extract_shape_features(image)
    return np.concatenate([hog_features, color_features, shape_features])


def predict_image(image_path):
    """Predict the species of a plankton image"""
    # Load and preprocess
    image = io.imread(image_path)

    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    image = transform.resize(image, IMG_SIZE, anti_aliasing=True)

    if image.max() > 1.0:
        image = image / 255.0

    # Extract features
    features = extract_all_features(image)
    features = features.reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    species_name = label_encoder.inverse_transform([prediction])[0]

    # Get decision function for confidence
    decision = model.decision_function(features_scaled)[0]
    confidence = np.max(decision)

    return species_name, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python predict_traditional_ml.py <image_path>")
        print("\nExample:")
        print("  python predict_traditional_ml.py test_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print(f"\nPredicting species for: {image_path}")
    print("-" * 50)

    species, confidence = predict_image(image_path)

    print(f"Predicted Species: {species}")
    print(f"Confidence Score: {confidence:.4f}")
    print("-" * 50)
