"""
Unified Training Pipeline for Plankton Species Classifier
========================================================

A configurable training script that supports multiple training strategies:
- Fast training (1-2 hours, good accuracy)
- Optimized training (4-6 hours, best accuracy)
- Colab training (Google Colab optimized)
- Traditional ML (SVM, Random Forest)

Usage:
    python train.py --profile fast
    python train.py --profile optimized
    python train.py --profile colab
    python train.py --profile traditional
    python train.py --config custom_config.py

Author: Plankton Classifier Team
"""

import os
import sys
import json
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import CNN_CONFIG, TRADITIONAL_ML_CONFIG

# =============================================================================
# TRAINING PROFILES
# =============================================================================

TRAINING_PROFILES = {
    'fast': {
        'description': 'Fast training with OneCycleLR scheduler (1-2 hours)',
        'model_type': 'cnn',
        'model_name': 'efficientnet_v2_s',
        'epochs_frozen': 25,
        'epochs_finetune': 20,
        'lr_frozen': 3e-3,
        'lr_finetune': 1e-4,
        'scheduler': 'onecycle',
        'batch_size': 32,
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'early_stopping_patience': 10,
        'target_accuracy': 75.0,
        'estimated_time': '1.5-2 hours'
    },

    'optimized': {
        'description': 'Optimized training for maximum accuracy (4-6 hours)',
        'model_type': 'cnn',
        'model_name': 'efficientnet_v2_s',
        'epochs_frozen': 80,
        'epochs_finetune': 50,
        'lr_frozen': 2e-3,
        'lr_finetune': 3e-5,
        'scheduler': 'cosine',
        'batch_size': 28,
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'early_stopping_patience': 20,
        'target_accuracy': 80.0,
        'estimated_time': '4-6 hours'
    },

    'colab': {
        'description': 'Google Colab optimized training',
        'model_type': 'cnn',
        'model_name': 'efficientnet_v2_s',
        'epochs_frozen': 35,
        'epochs_finetune': 35,
        'lr_frozen': 2e-3,
        'lr_finetune': 3e-5,
        'scheduler': 'cosine',
        'batch_size': 24,  # Conservative for Colab memory limits
        'gradient_accumulation_steps': 3,
        'mixed_precision': True,
        'early_stopping_patience': 15,
        'target_accuracy': 77.0,
        'estimated_time': '2.5-3.5 hours'
    },

    'traditional': {
        'description': 'Traditional ML with SVM and Random Forest',
        'model_type': 'traditional',
        'models': ['svm', 'rf'],
        'feature_extraction': 'hog_lbp_stats',
        'max_samples_per_class': 500,
        'n_jobs': -1,
        'target_accuracy': 65.0,
        'estimated_time': '2-4 hours'
    },

    'debug': {
        'description': 'Quick debug training (minimal epochs for testing)',
        'model_type': 'cnn',
        'model_name': 'efficientnet_v2_s',
        'epochs_frozen': 2,
        'epochs_finetune': 2,
        'lr_frozen': 1e-3,
        'lr_finetune': 1e-4,
        'scheduler': 'step',
        'batch_size': 8,
        'gradient_accumulation_steps': 1,
        'mixed_precision': False,
        'early_stopping_patience': 5,
        'target_accuracy': 30.0,
        'estimated_time': '5-10 minutes'
    }
}

# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified Training Pipeline for Plankton Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Training Profiles Available:
{chr(10).join([f'  {name}: {profile["description"]} ({profile["estimated_time"]})' for name, profile in TRAINING_PROFILES.items()])}

Examples:
  python train.py --profile fast
  python train.py --profile optimized --epochs-frozen 60
  python train.py --profile traditional --model svm
  python train.py --config my_config.py
        """
    )

    # Profile selection
    parser.add_argument(
        '--profile', '-p',
        choices=list(TRAINING_PROFILES.keys()),
        default='fast',
        help='Training profile to use (default: fast)'
    )

    # Custom configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom configuration file'
    )

    # Override options
    parser.add_argument('--epochs-frozen', type=int, help='Override frozen epochs')
    parser.add_argument('--epochs-finetune', type=int, help='Override finetune epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--model-name', type=str, help='Override model architecture')

    # Traditional ML options
    parser.add_argument(
        '--model',
        choices=['svm', 'rf', 'both'],
        help='Traditional ML model to train (for traditional profile)'
    )

    # General options
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--dry-run', action='store_true', help='Show configuration and exit')

    return parser.parse_args()

# =============================================================================
# CONFIGURATION BUILDER
# =============================================================================

def build_config(args):
    """Build configuration from arguments and profile."""
    # Start with base profile
    config = TRAINING_PROFILES[args.profile].copy()

    # Load custom config file if provided
    if args.config:
        custom_config = {}
        exec(open(args.config).read(), {}, custom_config)
        config.update(custom_config)

    # Apply command line overrides
    overrides = {
        'epochs_frozen': args.epochs_frozen,
        'epochs_finetune': args.epochs_finetune,
        'batch_size': args.batch_size,
        'lr_frozen': args.learning_rate,
        'lr_finetune': args.learning_rate,
        'model_name': args.model_name,
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    # Set paths
    config['data_dir'] = Path(args.data_dir) if args.data_dir else Path('2014_clean')
    config['output_dir'] = Path(args.output_dir) if args.output_dir else Path('outputs/models')

    # Set device
    if args.device:
        config['device'] = args.device
    else:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Traditional ML specific
    if config['model_type'] == 'traditional' and args.model:
        if args.model == 'both':
            config['models'] = ['svm', 'rf']
        else:
            config['models'] = [args.model]

    # Add metadata
    config.update({
        'profile_name': args.profile,
        'seed': args.seed,
        'verbose': args.verbose,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'command_line': ' '.join(sys.argv)
    })

    return config

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function."""
    args = parse_arguments()
    config = build_config(args)

    print("=" * 80)
    print("PLANKTON SPECIES CLASSIFIER - UNIFIED TRAINING PIPELINE")
    print("=" * 80)
    print(f"Profile: {config['profile_name']} ({config['description']})")
    print(f"Target Accuracy: {config['target_accuracy']}%")
    print(f"Estimated Time: {config['estimated_time']}")
    print(f"Device: {config['device']}")
    print(f"Data Directory: {config['data_dir']}")
    print(f"Output Directory: {config['output_dir']}")

    if args.dry_run:
        print("\\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("\\nDry run completed. Remove --dry-run flag to start training.")
        return

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = config['output_dir'] / f"config_{config['timestamp']}.json"
    with open(config_path, 'w') as f:
        # Convert non-serializable objects to strings
        serializable_config = {}
        for k, v in config.items():
            if isinstance(v, Path):
                serializable_config[k] = str(v)
            else:
                try:
                    json.dumps(v)  # Test if serializable
                    serializable_config[k] = v
                except:
                    serializable_config[k] = str(v)

        json.dump(serializable_config, f, indent=2)

    print(f"Configuration saved to: {config_path}")

    # Route to appropriate training function
    if config['model_type'] == 'cnn':
        from train_cnn_unified import train_cnn
        results = train_cnn(config)
    elif config['model_type'] == 'traditional':
        from train_traditional_unified import train_traditional
        results = train_traditional(config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    print("\\nTraining completed successfully!")
    print(f"Best Accuracy: {results.get('best_accuracy', 'N/A'):.2f}%")
    print(f"Model saved to: {results.get('model_path', 'N/A')}")
    print("=" * 80)

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\nTraining failed: {e}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        sys.exit(1)