"""
Configuration settings for Plankton Species Classifier.
Optimized for RTX 4060 Ti (8.59 GB VRAM).
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "2014"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA SETTINGS
# =============================================================================
# Image settings
IMAGE_SIZE = 224  # EfficientNetV2 default
CHANNELS = 3

# Dataset split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Class balancing strategy
# Options: 'undersample', 'oversample', 'weighted', 'sqrt_weighted'
CLASS_BALANCE_STRATEGY = 'sqrt_weighted'

# CRITICAL: Better filtering to handle extreme imbalance
MIN_SAMPLES_PER_CLASS = 20  # Increased from 5 - remove tiny classes
MAX_SAMPLES_PER_CLASS = 2000  # Reduced from 5000 - cap giant classes!

# Exclude problematic classes (too generic/noisy)
EXCLUDE_CLASSES = ['mix', 'detritus', 'mix_elongated']  # Remove mega-classes

# =============================================================================
# CNN TRAINING SETTINGS (EfficientNetV2-B0)
# Optimized for RTX 4060 Ti (8.59 GB VRAM)
# =============================================================================
CNN_CONFIG = {
    # Model - Proven working model
    'model_name': 'efficientnet_v2_s',
    'pretrained': True,
    'freeze_backbone': True,

    # Training Phase 1 (frozen backbone)
    'batch_size': 32,
    'epochs_frozen': 35,  # More epochs for cleaner dataset
    'learning_rate_frozen': 2e-3,  # Slightly higher LR

    # Training Phase 2 (fine-tuning)
    'epochs_finetune': 35,  # More epochs
    'learning_rate_finetune': 3e-5,  # Higher than 1e-5
    'unfreeze_layers': 60,  # Moderate unfreezing

    # Optimizer
    'optimizer': 'adamw',
    'weight_decay': 0.01,  # Moderate regularization

    # Scheduler
    'scheduler': 'cosine',
    'warmup_epochs': 3,

    # Regularization - BALANCED for cleaner data
    'dropout': 0.3,  # Lower since data is cleaner
    'label_smoothing': 0.1,  # Moderate

    # Early stopping
    'patience': 12,
    'min_delta': 0.001,

    # Data augmentation
    'augmentation': True,

    # Mixed precision training
    'mixed_precision': True,

    # Gradient accumulation
    'gradient_accumulation_steps': 2,

    # Number of workers
    'num_workers': 4,
}

# =============================================================================
# TRADITIONAL ML SETTINGS
# =============================================================================
TRADITIONAL_ML_CONFIG = {
    # Feature extraction
    'image_size': 64,  # Smaller for faster feature extraction

    # HOG parameters
    'hog_orientations': 9,
    'hog_pixels_per_cell': (8, 8),
    'hog_cells_per_block': (2, 2),

    # LBP parameters
    'lbp_radius': 1,
    'lbp_n_points': 8,

    # SVM parameters
    'svm_kernel': 'rbf',
    'svm_C': 10,
    'svm_gamma': 'scale',
    'svm_cache_size': 4000,  # Use 4GB RAM cache for faster SVM

    # Random Forest parameters
    'rf_n_estimators': 300,
    'rf_max_depth': None,
    'rf_min_samples_split': 2,

    # Max samples for traditional ML (REDUCED for speed!)
    # With 88 classes, even 500/class = 44K total samples
    'max_train_samples': 500,  # Was 50000 - TOO MANY!

    # Number of parallel jobs (-1 = use ALL CPU cores)
    'n_jobs': -1,
}

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================
EVAL_CONFIG = {
    'top_k_accuracy': [1, 3, 5],
    'confusion_matrix': True,
    'per_class_metrics': True,
    'save_predictions': True,
}

# =============================================================================
# LOGGING
# =============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_to_file': True,
}
