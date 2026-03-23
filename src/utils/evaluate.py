"""
Evaluation and Comparison Script for Plankton Species Classification.
Compares CNN vs Traditional ML performance.

FIXED: Uses saved class mappings from model checkpoints instead of regenerating.

Usage:
    python evaluate.py                        # Evaluate all trained models
    python evaluate.py --model cnn            # Evaluate only CNN
    python evaluate.py --visualize            # Generate visualizations
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import config
from dataset import (
    set_seed, get_transforms, PlanktonDataset,
    extract_traditional_features
)
from models.cnn_model import PlanktonCNN
from models.traditional_model import TraditionalMLClassifier


def load_cnn_model(model_path: Path, device: torch.device) -> Tuple[PlanktonCNN, Dict]:
    """Load trained CNN model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = PlanktonCNN(
        num_classes=checkpoint['num_classes'],
        model_name=checkpoint['config']['model_name'],
        pretrained=False,
        dropout=checkpoint['config']['dropout'],
        freeze_backbone=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def create_test_loader_from_saved_mapping(
    class_to_idx: Dict[str, int],
    batch_size: int = 32,
    num_workers: int = 4,
    test_ratio: float = 0.15
) -> Tuple[DataLoader, List[str], List[int]]:
    """
    Create test loader using SAVED class mapping (not regenerated).
    Only includes images from classes the model was trained on.
    """
    from sklearn.model_selection import train_test_split

    data_dir = config.DATA_DIR
    image_paths = []
    labels = []

    print(f"Loading test data for {len(class_to_idx)} trained classes...")

    for class_name, class_idx in tqdm(class_to_idx.items(), desc="Scanning classes"):
        class_dir = data_dir / class_name

        if not class_dir.exists():
            print(f"  Warning: Class directory not found: {class_name}")
            continue

        # Find all images
        class_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
            class_images.extend(list(class_dir.glob(ext)))

        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(class_idx)

    print(f"Found {len(image_paths)} total images across {len(class_to_idx)} classes")

    # Split to get test set (use same seed as training for consistency)
    # We do 70/15/15 split, so test is last 15%
    _, X_test, _, y_test = train_test_split(
        image_paths, labels,
        test_size=0.15,
        stratify=labels,
        random_state=config.RANDOM_SEED
    )

    print(f"Test set: {len(X_test)} samples")

    # Create dataset and loader
    transform = get_transforms(config.IMAGE_SIZE, is_training=False)
    test_dataset = PlanktonDataset(X_test, y_test, transform=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return test_loader, X_test, y_test


def evaluate_cnn(
    model: PlanktonCNN,
    test_loader: DataLoader,
    device: torch.device,
    idx_to_class: Dict[int, str]
) -> Dict:
    """Evaluate CNN on test set."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating CNN"):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # Top-k accuracy
    top3_correct = 0
    top5_correct = 0
    for i, label in enumerate(all_labels):
        top5_preds = np.argsort(all_probs[i])[-5:]
        if label in top5_preds[-3:]:
            top3_correct += 1
        if label in top5_preds:
            top5_correct += 1

    top3_acc = top3_correct / len(all_labels)
    top5_acc = top5_correct / len(all_labels)

    # Per-class metrics
    num_classes = len(idx_to_class)
    class_names = [idx_to_class.get(i, f"class_{i}") for i in range(num_classes)]

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        'model_type': 'CNN (EfficientNetV2)',
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report,
        'num_classes': num_classes
    }


def generate_comparison_report(results: Dict[str, Dict]) -> str:
    """Generate a comparison report between models."""
    report = []
    report.append("=" * 70)
    report.append("PLANKTON CLASSIFICATION - MODEL COMPARISON REPORT")
    report.append("=" * 70)
    report.append("")

    # Summary table
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 60)
    report.append(f"{'Model':<35} {'Top-1':>10} {'Top-3':>10} {'Top-5':>10}")
    report.append("-" * 60)

    for model_name, model_results in results.items():
        acc = model_results['accuracy'] * 100
        top3 = model_results['top3_accuracy'] * 100
        top5 = model_results['top5_accuracy'] * 100
        model_type = model_results['model_type']
        report.append(f"{model_type:<35} {acc:>9.2f}% {top3:>9.2f}% {top5:>9.2f}%")

    report.append("-" * 60)
    report.append("")

    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        report.append(f"BEST MODEL: {best_model[1]['model_type']}")
        report.append(f"  Accuracy: {best_model[1]['accuracy']*100:.2f}%")
        report.append(f"  Classes: {best_model[1].get('num_classes', 'N/A')}")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


def main(args):
    """Main evaluation function."""
    set_seed(config.RANDOM_SEED)

    print("=" * 60)
    print("PLANKTON CLASSIFICATION - MODEL EVALUATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    results = {}

    # Evaluate CNN
    if args.model in ['cnn', 'all']:
        cnn_model_path = config.MODEL_DIR / "cnn_final.pth"

        if cnn_model_path.exists():
            print("\n" + "=" * 50)
            print("EVALUATING CNN MODEL")
            print("=" * 50)

            # Load model and get SAVED class mapping
            print("\nLoading CNN model...")
            model, checkpoint = load_cnn_model(cnn_model_path, device)

            # Use class mapping from checkpoint (CRITICAL!)
            class_to_idx = checkpoint.get('class_to_idx', {})
            idx_to_class = checkpoint.get('idx_to_class', {})

            # Convert keys to proper types
            if isinstance(list(idx_to_class.keys())[0], str):
                idx_to_class = {int(k): v for k, v in idx_to_class.items()}

            print(f"Model trained on {len(class_to_idx)} classes")

            # Create test loader using SAVED mapping
            print("\nCreating test data loader using saved class mapping...")
            test_loader, test_paths, test_labels = create_test_loader_from_saved_mapping(
                class_to_idx,
                batch_size=32,
                num_workers=0 if device.type == 'cpu' else 4
            )

            # Evaluate
            print("\nEvaluating...")
            cnn_results = evaluate_cnn(model, test_loader, device, idx_to_class)
            results['cnn'] = cnn_results

            print(f"\n{'='*50}")
            print("CNN RESULTS")
            print(f"{'='*50}")
            print(f"  Classes evaluated: {cnn_results['num_classes']}")
            print(f"  Top-1 Accuracy: {cnn_results['accuracy']*100:.2f}%")
            print(f"  Top-3 Accuracy: {cnn_results['top3_accuracy']*100:.2f}%")
            print(f"  Top-5 Accuracy: {cnn_results['top5_accuracy']*100:.2f}%")
        else:
            print(f"\nCNN model not found at {cnn_model_path}")

    # Traditional ML - Skip due to sklearn version incompatibility
    if args.model in ['traditional', 'all']:
        print("\n" + "=" * 50)
        print("TRADITIONAL ML EVALUATION")
        print("=" * 50)
        print("\nNOTE: Traditional ML models have sklearn version incompatibility.")
        print("The models were trained with sklearn 1.3.2 but current version is 1.7.2")
        print("Please retrain traditional ML models with: python train_traditional.py")
        print("\nSkipping Traditional ML evaluation.")

    if not results:
        print("\nNo models could be evaluated. Please train models first.")
        return

    # Generate comparison report
    print("\n" + "=" * 60)
    report = generate_comparison_report(results)
    print(report)

    # Save report
    report_path = config.RESULTS_DIR / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save detailed results
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

    results_for_json = {}
    for model_name, model_results in results.items():
        results_for_json[model_name] = {
            'model_type': model_results['model_type'],
            'accuracy': model_results['accuracy'],
            'top3_accuracy': model_results['top3_accuracy'],
            'top5_accuracy': model_results['top5_accuracy'],
            'num_classes': model_results.get('num_classes', 0),
        }

    results_path = config.RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results_for_json), f, indent=2)
    print(f"Results saved to {results_path}")

    print("\nEvaluation completed!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate plankton classification models")

    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'traditional', 'all'],
                        help='Which models to evaluate (default: cnn)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')

    args = parser.parse_args()
    main(args)
