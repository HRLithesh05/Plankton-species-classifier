"""
Evaluation and Comparison Script for Plankton Species Classification.
Compares CNN vs Traditional ML performance.

Usage:
    python evaluate.py                        # Evaluate all trained models
    python evaluate.py --model cnn            # Evaluate only CNN
    python evaluate.py --visualize            # Generate visualizations
    python evaluate.py --confusion-matrix     # Generate confusion matrices
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from tqdm import tqdm

import config
from dataset import (
    create_data_loaders, set_seed, load_image_paths, split_dataset,
    extract_traditional_features, get_transforms
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


def evaluate_cnn(
    model: PlanktonCNN,
    test_loader: torch.utils.data.DataLoader,
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
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
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
        'classification_report': report
    }


def evaluate_traditional_ml(
    model_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx_to_class: Dict[int, str]
) -> Dict:
    """Evaluate traditional ML model on test set."""
    model = TraditionalMLClassifier.load(model_path)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Top-k accuracy
    top3_correct = 0
    top5_correct = 0
    for i, label in enumerate(y_test):
        top5_preds = np.argsort(probabilities[i])[-5:]
        if label in top5_preds[-3:]:
            top3_correct += 1
        if label in top5_preds:
            top5_correct += 1

    top3_acc = top3_correct / len(y_test)
    top5_acc = top5_correct / len(y_test)

    # Per-class metrics
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(
        y_test, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        'model_type': f'Traditional ML ({model.classifier_type.upper()})',
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'predictions': predictions,
        'labels': y_test,
        'probabilities': probabilities,
        'classification_report': report
    }


def generate_comparison_report(results: Dict[str, Dict]) -> str:
    """Generate a comparison report between models."""
    report = []
    report.append("=" * 80)
    report.append("PLANKTON CLASSIFICATION - MODEL COMPARISON REPORT")
    report.append("=" * 80)
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
        report.append(f"{model_results['model_type']:<35} {acc:>9.2f}% {top3:>9.2f}% {top5:>9.2f}%")

    report.append("-" * 60)
    report.append("")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    report.append(f"BEST MODEL: {best_model[1]['model_type']}")
    report.append(f"  Accuracy: {best_model[1]['accuracy']*100:.2f}%")
    report.append("")

    # Per-class analysis (best and worst)
    for model_name, model_results in results.items():
        report.append(f"\n{model_results['model_type']} - Per-Class Analysis")
        report.append("-" * 50)

        class_report = model_results['classification_report']

        # Sort classes by F1 score
        class_scores = []
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                class_scores.append((class_name, metrics['f1-score'], metrics['support']))

        class_scores.sort(key=lambda x: x[1], reverse=True)

        report.append("\nBest 5 classes (by F1-score):")
        for name, f1, support in class_scores[:5]:
            report.append(f"  {name}: F1={f1:.3f} (n={int(support)})")

        report.append("\nWorst 5 classes (by F1-score):")
        for name, f1, support in class_scores[-5:]:
            report.append(f"  {name}: F1={f1:.3f} (n={int(support)})")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def generate_visualizations(results: Dict[str, Dict], save_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Skipping visualizations.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    metrics = ['accuracy', 'top3_accuracy', 'top5_accuracy']
    x = np.arange(len(models))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [results[m][metric] * 100 for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison - Accuracy Metrics')
    ax.set_xticks(x + width)
    model_labels = [results[m]['model_type'] for m in models]
    ax.set_xticklabels(model_labels, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()

    # 2. Confusion matrices for each model
    for model_name, model_results in results.items():
        predictions = model_results['predictions']
        labels = model_results['labels']

        # Get unique classes
        unique_classes = np.unique(np.concatenate([predictions, labels]))
        n_classes = len(unique_classes)

        # Only plot if manageable number of classes
        if n_classes <= 30:
            cm = confusion_matrix(labels, predictions)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - {model_results["model_type"]}')

            plt.tight_layout()
            plt.savefig(save_dir / f'confusion_matrix_{model_name}.png', dpi=150)
            plt.close()

    # 3. Per-class F1 score comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    for model_name, model_results in results.items():
        class_report = model_results['classification_report']

        class_scores = []
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                class_scores.append((class_name, metrics['f1-score']))

        class_scores.sort(key=lambda x: x[1], reverse=True)

        # Plot top 20 classes
        names = [x[0] for x in class_scores[:20]]
        scores = [x[1] for x in class_scores[:20]]

        ax.barh(names, scores, alpha=0.7, label=model_results['model_type'])

    ax.set_xlabel('F1 Score')
    ax.set_title('Top 20 Classes by F1 Score')
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_f1_comparison.png', dpi=150)
    plt.close()

    print(f"Visualizations saved to {save_dir}")


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
            print("\nLoading CNN model...")
            model, checkpoint = load_cnn_model(cnn_model_path, device)

            # Get class mapping
            idx_to_class = checkpoint.get('idx_to_class', {})
            if not idx_to_class:
                mapping_path = config.MODEL_DIR / "class_mapping.json"
                if mapping_path.exists():
                    with open(mapping_path) as f:
                        mapping = json.load(f)
                        idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}

            # Create test loader
            print("Creating test data loader...")
            _, _, test_loader, class_to_idx, idx_to_class = create_data_loaders(
                batch_size=32,
                num_workers=4
            )

            print("Evaluating CNN...")
            cnn_results = evaluate_cnn(model, test_loader, device, idx_to_class)
            results['cnn'] = cnn_results

            print(f"\nCNN Results:")
            print(f"  Top-1 Accuracy: {cnn_results['accuracy']*100:.2f}%")
            print(f"  Top-3 Accuracy: {cnn_results['top3_accuracy']*100:.2f}%")
            print(f"  Top-5 Accuracy: {cnn_results['top5_accuracy']*100:.2f}%")
        else:
            print(f"CNN model not found at {cnn_model_path}")

    # Evaluate Traditional ML
    if args.model in ['traditional', 'all']:
        # Prepare test features
        print("\nPreparing test data for traditional ML...")

        # Load paths and class mapping
        image_paths, labels, class_to_idx, idx_to_class = load_image_paths(
            config.DATA_DIR,
            min_samples=config.MIN_SAMPLES_PER_CLASS,
            max_samples=config.TRADITIONAL_ML_CONFIG['max_train_samples']
        )

        # Split
        _, _, X_test_paths, _, _, y_test = split_dataset(
            image_paths, labels,
            seed=config.RANDOM_SEED
        )

        # Extract features
        print("Extracting features for test set...")
        cfg = config.TRADITIONAL_ML_CONFIG
        X_test = np.array([
            extract_traditional_features(p, cfg)
            for p in tqdm(X_test_paths, desc="Extracting features")
        ])
        y_test = np.array(y_test)

        # Scale features
        import joblib
        scaler_path = config.MODEL_DIR / "traditional_ml_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_test = scaler.transform(X_test)

        # Evaluate SVM
        svm_path = config.MODEL_DIR / "traditional_ml_svm.joblib"
        if svm_path.exists():
            print("\nEvaluating SVM...")
            svm_results = evaluate_traditional_ml(svm_path, X_test, y_test, idx_to_class)
            results['svm'] = svm_results

            print(f"\nSVM Results:")
            print(f"  Top-1 Accuracy: {svm_results['accuracy']*100:.2f}%")
            print(f"  Top-3 Accuracy: {svm_results['top3_accuracy']*100:.2f}%")
            print(f"  Top-5 Accuracy: {svm_results['top5_accuracy']*100:.2f}%")

        # Evaluate Random Forest
        rf_path = config.MODEL_DIR / "traditional_ml_random_forest.joblib"
        if rf_path.exists():
            print("\nEvaluating Random Forest...")
            rf_results = evaluate_traditional_ml(rf_path, X_test, y_test, idx_to_class)
            results['random_forest'] = rf_results

            print(f"\nRandom Forest Results:")
            print(f"  Top-1 Accuracy: {rf_results['accuracy']*100:.2f}%")
            print(f"  Top-3 Accuracy: {rf_results['top3_accuracy']*100:.2f}%")
            print(f"  Top-5 Accuracy: {rf_results['top5_accuracy']*100:.2f}%")

    if not results:
        print("\nNo trained models found. Please train models first.")
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

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        generate_visualizations(results, config.RESULTS_DIR / "plots")

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

    # Remove large arrays for JSON serialization
    results_for_json = {}
    for model_name, model_results in results.items():
        results_for_json[model_name] = {
            'model_type': model_results['model_type'],
            'accuracy': model_results['accuracy'],
            'top3_accuracy': model_results['top3_accuracy'],
            'top5_accuracy': model_results['top5_accuracy'],
            'classification_report': model_results['classification_report']
        }

    results_path = config.RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results_for_json), f, indent=2)
    print(f"Detailed results saved to {results_path}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate plankton classification models")

    parser.add_argument('--model', type=str, default='all',
                        choices=['cnn', 'traditional', 'all'],
                        help='Which models to evaluate')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--confusion-matrix', action='store_true',
                        help='Generate confusion matrices')

    args = parser.parse_args()

    if args.confusion_matrix:
        args.visualize = True

    main(args)
