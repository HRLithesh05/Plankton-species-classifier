"""
Traditional ML Plankton Classifier Training Script (GPU-Accelerated)
===================================================================
This script trains a traditional machine learning model using:
- HOG (Histogram of Oriented Gradients) features
- Color Histogram features
- Basic shape features
- GPU-accelerated SVM classifier with RBF kernel (using cuML/RAPIDS)

Expected Accuracy: 50-65% (baseline for comparison with CNN approach)
Training Time: ~5-10 minutes on GPU (4060 or better)

Requirements: NVIDIA GPU with CUDA support
"""

import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated ML libraries (cuML from RAPIDS)
try:
    from cuml.svm import SVC as cuSVC
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.model_selection import GridSearchCV as cuGridSearchCV
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU libraries loaded successfully (cuML/RAPIDS)")
except ImportError:
    print("⚠ cuML not found. Falling back to CPU (scikit-learn)")
    print("  Install cuML: conda install -c rapidsai -c conda-forge -c nvidia cuml")
    from sklearn.svm import SVC as cuSVC
    from sklearn.preprocessing import StandardScaler as cuStandardScaler
    from sklearn.model_selection import GridSearchCV as cuGridSearchCV
    GPU_AVAILABLE = False

# Standard libraries for feature extraction and evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from skimage import io, color, feature, transform, exposure
from skimage.measure import label, regionprops
from PIL import Image

# Configuration
DATASET_PATH = "2014_clean"
MODEL_OUTPUT_PATH = "outputs/models/traditional_ml_model.pkl"
IMG_SIZE = (128, 128)  # Smaller than CNN for faster feature extraction
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42
MAX_IMAGES_PER_CLASS = 1000  # Increased limit for GPU (faster training)

# Create output directory
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

print("=" * 70)
print("Traditional ML Plankton Classifier - GPU-Accelerated Training")
print("=" * 70)
if GPU_AVAILABLE:
    print("🚀 GPU Mode: Enabled (cuML/RAPIDS)")
    print("   NVIDIA GPU detected - Training will be much faster!")
else:
    print("💻 CPU Mode: Enabled (scikit-learn)")
    print("   Consider installing cuML for GPU acceleration")

# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_hog_features(image):
    """Extract HOG (Histogram of Oriented Gradients) features"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # HOG parameters (standard, not over-tuned)
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
    """Extract color histogram features from RGB channels"""
    if len(image.shape) == 2:
        # Grayscale image - create single channel histogram
        hist = np.histogram(image, bins=32, range=(0, 1))[0]
        return hist.flatten()

    # RGB histograms (8 bins per channel for simplicity)
    hist_features = []
    for channel in range(3):
        hist = np.histogram(image[:, :, channel], bins=8, range=(0, 1))[0]
        hist_features.extend(hist)

    return np.array(hist_features)


def extract_shape_features(image):
    """Extract basic shape features"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    # Threshold to get binary image
    thresh = gray > 0.3

    # Get basic shape properties
    labeled = label(thresh)
    regions = regionprops(labeled)

    if len(regions) == 0:
        # No regions found, return default features
        return np.array([0, 0, 0, 0, 0])

    # Use the largest region
    region = max(regions, key=lambda r: r.area)

    # Extract shape features
    area = region.area / (IMG_SIZE[0] * IMG_SIZE[1])  # Normalized
    perimeter = region.perimeter / (2 * (IMG_SIZE[0] + IMG_SIZE[1]))  # Normalized
    eccentricity = region.eccentricity
    solidity = region.solidity
    extent = region.extent

    return np.array([area, perimeter, eccentricity, solidity, extent])


def extract_all_features(image):
    """Extract all features and concatenate"""
    hog_features = extract_hog_features(image)
    color_features = extract_color_histogram(image)
    shape_features = extract_shape_features(image)

    # Concatenate all features
    all_features = np.concatenate([hog_features, color_features, shape_features])
    return all_features


def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    try:
        # Load image
        image = io.imread(image_path)

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = color.gray2rgb(image)

        # Remove alpha channel if present
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Resize
        image = transform.resize(image, IMG_SIZE, anti_aliasing=True)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        return image
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


# ============================================================================
# Data Loading
# ============================================================================

print("\n[1/5] Loading dataset from:", DATASET_PATH)

# Get all species directories
species_dirs = [d for d in os.listdir(DATASET_PATH)
                if os.path.isdir(os.path.join(DATASET_PATH, d))]
species_dirs = sorted(species_dirs)

print(f"Found {len(species_dirs)} species classes")

# Load images and labels
X_features = []
y_labels = []
species_count = {}

for idx, species_name in enumerate(species_dirs):
    species_path = os.path.join(DATASET_PATH, species_name)
    image_files = [f for f in os.listdir(species_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # Limit images per class for faster training
    image_files = image_files[:MAX_IMAGES_PER_CLASS]

    species_count[species_name] = len(image_files)
    print(f"Processing [{idx+1}/{len(species_dirs)}] {species_name}: {len(image_files)} images", end='\r')

    for img_file in image_files:
        img_path = os.path.join(species_path, img_file)

        # Load and preprocess
        image = load_and_preprocess_image(img_path)
        if image is None:
            continue

        # Extract features
        try:
            features = extract_all_features(image)
            X_features.append(features)
            y_labels.append(species_name)
        except Exception as e:
            print(f"\nError extracting features from {img_path}: {e}")
            continue

print(f"\n\nTotal images loaded: {len(X_features)}")
print(f"Feature vector size: {len(X_features[0])}")

# ============================================================================
# Data Preparation
# ============================================================================

print("\n[2/5] Preparing data...")

# Convert to numpy arrays
X = np.array(X_features)
y = np.array(y_labels)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded,
    test_size=1-TRAIN_SPLIT,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Feature scaling (important for SVM)
print("\n[3/5] Scaling features...")
if GPU_AVAILABLE:
    print("Using GPU-accelerated StandardScaler (cuML)")
scaler = cuStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ============================================================================
# Model Training
# ============================================================================

print("\n[4/5] Training SVM classifier...")
if GPU_AVAILABLE:
    print("🚀 Using GPU-accelerated SVM (cuML) - This will be MUCH faster!")
    print("Training on NVIDIA GPU with CUDA acceleration...")
    print("Estimated time: 5-10 minutes")
else:
    print("Training on CPU - This may take 15-30 minutes...")

# Define SVM with limited hyperparameter search
# We're using a baseline approach, not over-optimizing
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
}

# Create SVM classifier
svm = cuSVC(
    kernel='rbf',
    random_state=RANDOM_STATE,
    verbose=False,
    class_weight='balanced'  # Handle class imbalance
)

# Grid search with cross-validation
grid_search = cuGridSearchCV(
    svm,
    param_grid,
    cv=3,  # 3-fold CV for speed
    scoring='accuracy',
    n_jobs=-1,  # Use all available resources
    verbose=2
)

# Train
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_

# ============================================================================
# Evaluation
# ============================================================================

print("\n[5/5] Evaluating model...")

# Predictions
y_train_pred = best_model.predict(X_train_scaled)
y_val_pred = best_model.predict(X_val_scaled)

# Convert to numpy if using cuML (GPU arrays need conversion for sklearn metrics)
if GPU_AVAILABLE:
    try:
        y_train_pred = cp.asnumpy(y_train_pred) if hasattr(y_train_pred, '__cuda_array_interface__') else y_train_pred
        y_val_pred = cp.asnumpy(y_val_pred) if hasattr(y_val_pred, '__cuda_array_interface__') else y_val_pred
    except:
        pass  # Already numpy arrays

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Detailed classification report
print("\n" + "="*70)
print("Classification Report (Validation Set)")
print("="*70)
print(classification_report(
    y_val,
    y_val_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

# ============================================================================
# Save Model
# ============================================================================

print("\nSaving model to:", MODEL_OUTPUT_PATH)

# Package everything needed for inference
model_package = {
    'model': best_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'img_size': IMG_SIZE,
    'feature_vector_size': X.shape[1],
    'training_accuracy': train_accuracy,
    'validation_accuracy': val_accuracy,
    'num_classes': len(label_encoder.classes_),
    'class_names': label_encoder.classes_.tolist(),
    'hyperparameters': grid_search.best_params_,
    'gpu_trained': GPU_AVAILABLE
}

with open(MODEL_OUTPUT_PATH, 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n✓ Model saved successfully!")
print(f"  Model file size: {os.path.getsize(MODEL_OUTPUT_PATH) / (1024*1024):.2f} MB")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Training Summary")
print("="*70)
print(f"Approach: Traditional ML (HOG + Color Histograms + Shape Features + SVM)")
print(f"Acceleration: {'GPU (cuML/RAPIDS)' if GPU_AVAILABLE else 'CPU (scikit-learn)'}")
print(f"Dataset: {len(X)} images from {len(species_dirs)} species")
print(f"Feature Vector Size: {X.shape[1]} dimensions")
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Model Saved: {MODEL_OUTPUT_PATH}")
print("="*70)

print("\n✓ Training complete! Model ready for comparison with CNN approach.")
print("\nExpected performance: Traditional ML (~60%) vs CNN (89.51%)")
print("This demonstrates the superiority of deep learning for complex image classification.")
