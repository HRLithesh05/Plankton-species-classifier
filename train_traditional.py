"""
Traditional ML Training Script for Plankton Species Classification.
Uses SVM and Random Forest with hand-crafted features.

Usage:
    python train_traditional.py                      # Train both SVM and RF
    python train_traditional.py --model svm          # Train only SVM
    python train_traditional.py --model rf           # Train only Random Forest
    python train_traditional.py --max-samples 10000  # Limit training samples
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    top_k_accuracy_score
)
from tqdm import tqdm

import config
from dataset import (
    load_image_paths, split_dataset, set_seed,
    extract_traditional_features, get_feature_size
)
from models.traditional_model import (
    TraditionalMLClassifier, create_svm, create_random_forest,
    EnsembleClassifier
)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_traditional_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

    return logger


def extract_all_features(
    image_paths: List[str],
    cfg: Dict,
    desc: str = "Extracting features"
) -> np.ndarray:
    """Extract features from all images."""
    features = []

    for path in tqdm(image_paths, desc=desc):
        feat = extract_traditional_features(path, cfg)
        features.append(feat)

    return np.array(features)


def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 5
) -> float:
    """Compute top-k accuracy."""
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)


def evaluate_model(
    model: TraditionalMLClassifier,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    logger: logging.Logger,
    name: str = "Test"
) -> Dict:
    """Evaluate a model and log results."""
    logger.info(f"\nEvaluating on {name} set...")

    # Predictions
    start_time = time.time()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    inference_time = time.time() - start_time

    # Metrics
    accuracy = accuracy_score(y, predictions)
    top3_acc = compute_top_k_accuracy(y, probabilities, k=3)
    top5_acc = compute_top_k_accuracy(y, probabilities, k=5)

    logger.info(f"  {name} Accuracy (Top-1): {accuracy*100:.2f}%")
    logger.info(f"  {name} Accuracy (Top-3): {top3_acc*100:.2f}%")
    logger.info(f"  {name} Accuracy (Top-5): {top5_acc*100:.2f}%")
    logger.info(f"  Inference time: {inference_time:.2f}s ({len(y)/inference_time:.1f} samples/s)")

    # Per-class report (summary)
    report = classification_report(
        y, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Log worst performing classes
    class_f1_scores = [(name, report[name]['f1-score'])
                       for name in class_names if name in report]
    class_f1_scores.sort(key=lambda x: x[1])

    logger.info(f"\n  Worst 10 classes by F1-score:")
    for name, f1 in class_f1_scores[:10]:
        logger.info(f"    {name}: {f1:.3f}")

    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'inference_time': inference_time,
        'classification_report': report
    }


def train_model(
    model: TraditionalMLClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: List[str],
    logger: logging.Logger
) -> Dict:
    """Train a single model."""
    model_name = model.classifier_type.upper()
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training time: {train_time:.1f}s")

    # Evaluate on training set
    train_results = evaluate_model(
        model, X_train, y_train, class_names, logger, "Train"
    )

    # Evaluate on validation set
    val_results = evaluate_model(
        model, X_val, y_val, class_names, logger, "Validation"
    )

    return {
        'train_time': train_time,
        'train_results': train_results,
        'val_results': val_results
    }


def main(args):
    """Main training function."""
    set_seed(config.RANDOM_SEED)
    logger = setup_logging(config.LOG_DIR)

    logger.info("Traditional ML Training for Plankton Classification")
    logger.info("=" * 60)

    cfg = config.TRADITIONAL_ML_CONFIG.copy()

    # Override max samples if specified
    if args.max_samples:
        cfg['max_train_samples'] = args.max_samples

    logger.info(f"\nConfiguration:")
    for key, value in cfg.items():
        logger.info(f"  {key}: {value}")

    # Load image paths
    logger.info("\nLoading image paths...")
    image_paths, labels, class_to_idx, idx_to_class = load_image_paths(
        config.DATA_DIR,
        min_samples=config.MIN_SAMPLES_PER_CLASS,
        max_samples=cfg['max_train_samples']
    )

    num_classes = len(class_to_idx)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Total samples: {len(image_paths):,}")

    # Split dataset
    logger.info("\nSplitting dataset...")
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = split_dataset(
        image_paths, labels,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.RANDOM_SEED
    )

    # Extract features
    logger.info("\nExtracting features...")
    feature_size = get_feature_size(cfg)
    logger.info(f"Feature vector size: {feature_size}")

    X_train = extract_all_features(X_train_paths, cfg, "Training features")
    X_val = extract_all_features(X_val_paths, cfg, "Validation features")
    X_test = extract_all_features(X_test_paths, cfg, "Test features")

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Scale features
    logger.info("\nScaling features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler
    scaler_path = config.MODEL_DIR / "traditional_ml_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save class mapping
    mapping_path = config.MODEL_DIR / "traditional_ml_class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(k): v for k, v in idx_to_class.items()}
        }, f, indent=2)
    logger.info(f"Class mapping saved to {mapping_path}")

    # Train models
    results = {}
    models_to_train = []

    if args.model in ['svm', 'both']:
        models_to_train.append(('svm', create_svm(cfg)))
    if args.model in ['rf', 'both']:
        models_to_train.append(('random_forest', create_random_forest(cfg)))

    for model_name, model in models_to_train:
        # Train
        train_results = train_model(
            model, X_train, y_train, X_val, y_val, class_names, logger
        )

        # Evaluate on test set
        test_results = evaluate_model(
            model, X_test, y_test, class_names, logger, "Test"
        )

        # Save model
        model_path = config.MODEL_DIR / f"traditional_ml_{model_name}.joblib"
        model.save(model_path)

        results[model_name] = {
            **train_results,
            'test_results': test_results
        }

    # Train ensemble if both models were trained
    if len(models_to_train) == 2:
        logger.info(f"\n{'='*60}")
        logger.info("Training ENSEMBLE (SVM + Random Forest)")
        logger.info(f"{'='*60}")

        svm_model = results['svm']['model'] if 'model' in results['svm'] else \
            TraditionalMLClassifier.load(config.MODEL_DIR / "traditional_ml_svm.joblib")
        rf_model = results['random_forest']['model'] if 'model' in results['random_forest'] else \
            TraditionalMLClassifier.load(config.MODEL_DIR / "traditional_ml_random_forest.joblib")

        ensemble = EnsembleClassifier([svm_model, rf_model])

        # Note: ensemble doesn't need to be fit separately since base models are already fit
        ensemble_test_results = ensemble.evaluate(X_test, y_test, class_names)

        logger.info(f"\nEnsemble Test Results:")
        logger.info(f"  Accuracy (Top-1): {ensemble_test_results['accuracy']*100:.2f}%")
        logger.info(f"  Accuracy (Top-3): {ensemble_test_results['top_3_accuracy']*100:.2f}%")
        logger.info(f"  Accuracy (Top-5): {ensemble_test_results['top_5_accuracy']*100:.2f}%")

        results['ensemble'] = {'test_results': ensemble_test_results}

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")

    for model_name, model_results in results.items():
        test_res = model_results['test_results']
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Test Accuracy (Top-1): {test_res['accuracy']*100:.2f}%")
        if 'top3_accuracy' in test_res:
            logger.info(f"  Test Accuracy (Top-3): {test_res['top3_accuracy']*100:.2f}%")
            logger.info(f"  Test Accuracy (Top-5): {test_res['top5_accuracy']*100:.2f}%")
        elif 'top_3_accuracy' in test_res:
            logger.info(f"  Test Accuracy (Top-3): {test_res['top_3_accuracy']*100:.2f}%")
            logger.info(f"  Test Accuracy (Top-5): {test_res['top_5_accuracy']*100:.2f}%")

    # Save results
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results_path = config.RESULTS_DIR / "traditional_ml_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("\nTraining completed!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traditional ML for plankton classification")

    parser.add_argument('--model', type=str, default='both',
                        choices=['svm', 'rf', 'both'],
                        help='Which model to train (default: both)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if args.seed:
        config.RANDOM_SEED = args.seed

    main(args)
