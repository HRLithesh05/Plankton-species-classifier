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

# Minimum samples per class to include
MIN_SAMPLES_PER_CLASS = 5

# Maximum samples per class for training (to handle extreme imbalance)
MAX_SAMPLES_PER_CLASS = 5000

# =============================================================================
# CNN TRAINING SETTINGS (EfficientNetV2-B0)
# Optimized for RTX 4060 Ti (8.59 GB VRAM)
# =============================================================================
CNN_CONFIG = {
    # Model
    'model_name': 'efficientnet_v2_s',  # 's' fits well in 8GB VRAM
    'pretrained': True,
    'freeze_backbone': True,  # Phase 1: only train classifier head

    # Training Phase 1 (frozen backbone)
    'batch_size': 32,  # Safe for 8GB VRAM
    'epochs_frozen': 15,
    'learning_rate_frozen': 1e-3,

    # Training Phase 2 (fine-tuning)
    'epochs_finetune': 15,
    'learning_rate_finetune': 1e-5,
    'unfreeze_layers': 50,  # Number of layers to unfreeze from end

    # Optimizer
    'optimizer': 'adamw',
    'weight_decay': 0.01,

    # Scheduler
    'scheduler': 'cosine',
    'warmup_epochs': 2,

    # Regularization
    'dropout': 0.3,
    'label_smoothing': 0.1,

    # Early stopping
    'patience': 7,
    'min_delta': 0.001,

    # Data augmentation (handled in dataset.py)
    'augmentation': True,

    # Mixed precision training (for faster training on RTX 4060 Ti)
    'mixed_precision': True,

    # Gradient accumulation (effective larger batch size)
    'gradient_accumulation_steps': 2,

    # Number of workers for data loading
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

    # Random Forest parameters
    'rf_n_estimators': 300,
    'rf_max_depth': None,
    'rf_min_samples_split': 2,

    # Max samples for traditional ML (memory constraint)
    'max_train_samples': 50000,

    # Number of parallel jobs
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
