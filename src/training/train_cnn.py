"""
CNN Training Script for Plankton Species Classification.
Uses EfficientNetV2 with transfer learning.

Usage:
    python train_cnn.py                    # Full training with default settings
    python train_cnn.py --epochs 20        # Custom epochs
    python train_cnn.py --batch-size 16    # Smaller batch for less VRAM
    python train_cnn.py --no-finetune      # Skip fine-tuning phase
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import config
from dataset import create_data_loaders, set_seed, compute_class_weights
from models.cnn_model import (
    create_model, create_optimizer, create_scheduler, PlanktonCNN
)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_cnn_{timestamp}.log"

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


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

        return self.early_stop

    def load_best_model(self, model: nn.Module, device: torch.device):
        """Load the best model state."""
        model.load_state_dict({k: v.to(device) for k, v in self.best_model_state.items()})


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True,
    accumulation_steps: int = 1
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with mixed precision
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            # Step scheduler (per step)
            if scheduler is not None:
                scheduler.step()

        # Calculate metrics
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, float, float, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-3 and Top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1)
            for i, label in enumerate(labels):
                if label in top5_pred[i, :3]:
                    correct_top3 += 1
                if label in top5_pred[i]:
                    correct_top5 += 1

    val_loss = total_loss / len(val_loader)
    val_acc = correct / total
    val_acc_top3 = correct_top3 / total
    val_acc_top5 = correct_top5 / total

    return val_loss, val_acc, val_acc_top3, val_acc_top5


def train_phase(
    model: PlanktonCNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Dict,
    phase: str,
    logger: logging.Logger,
    save_dir: Path
) -> Dict:
    """Train a single phase (frozen or fine-tuning)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {phase.upper()} phase")
    logger.info(f"{'='*60}")

    # Setup
    if phase == 'frozen':
        epochs = cfg['epochs_frozen']
    else:
        epochs = cfg['epochs_finetune']
        # Unfreeze backbone layers for fine-tuning
        model.unfreeze_backbone(num_layers=cfg['unfreeze_layers'])

    optimizer = create_optimizer(model, cfg, phase=phase)
    scheduler = create_scheduler(
        optimizer, cfg, epochs,
        steps_per_epoch=len(train_loader)
    )

    use_amp = cfg['mixed_precision'] and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    early_stopping = EarlyStopping(
        patience=cfg['patience'],
        min_delta=cfg['min_delta']
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_acc_top3': [], 'val_acc_top5': [],
        'lr': []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, use_amp, cfg['gradient_accumulation_steps']
        )

        # Validate
        val_loss, val_acc, val_acc_top3, val_acc_top5 = validate(
            model, val_loader, criterion, device, use_amp
        )

        epoch_time = time.time() - start_time

        # Log metrics
        logger.info(
            f"[{phase}] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Top-3: {val_acc_top3*100:.2f}% | Top-5: {val_acc_top5*100:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_acc_top3'].append(val_acc_top3)
        history['val_acc_top5'].append(val_acc_top5)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = save_dir / f"best_model_{phase}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_acc_top5': val_acc_top5
            }, checkpoint_path)
            logger.info(f"  -> New best model saved: {val_acc*100:.2f}%")

        # Early stopping
        if early_stopping(val_acc, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            early_stopping.load_best_model(model, device)
            break

    return history


def main(args):
    """Main training function."""
    # Setup
    set_seed(config.RANDOM_SEED)
    logger = setup_logging(config.LOG_DIR)

    logger.info("Plankton CNN Training")
    logger.info("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Configuration
    cfg = config.CNN_CONFIG.copy()

    # Override with command line arguments
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.epochs:
        cfg['epochs_frozen'] = args.epochs
        cfg['epochs_finetune'] = args.epochs // 2
    if args.lr:
        cfg['learning_rate_frozen'] = args.lr

    logger.info(f"\nConfiguration:")
    for key, value in cfg.items():
        logger.info(f"  {key}: {value}")

    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = create_data_loaders(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        image_size=config.IMAGE_SIZE,
        augmentation=cfg['augmentation'],
        use_weighted_sampler=True
    )

    num_classes = len(class_to_idx)
    logger.info(f"\nNumber of classes: {num_classes}")

    # Save class mapping
    mapping_path = config.MODEL_DIR / "class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(k): v for k, v in idx_to_class.items()}
        }, f, indent=2)
    logger.info(f"Class mapping saved to {mapping_path}")

    # Create model
    logger.info("\nCreating model...")
    model, criterion = create_model(num_classes, cfg, device)

    # Get class weights for loss function
    train_labels = train_loader.dataset.labels
    class_weights = compute_class_weights(
        train_labels,
        strategy=config.CLASS_BALANCE_STRATEGY,
        num_classes=num_classes
    ).to(device)

    # Update criterion with weights
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.get('label_smoothing', 0.0)
    )

    # Training Phase 1: Frozen backbone
    history_frozen = train_phase(
        model, train_loader, val_loader, criterion,
        device, cfg, 'frozen', logger, config.MODEL_DIR
    )

    # Training Phase 2: Fine-tuning (optional)
    history_finetune = {}
    if not args.no_finetune:
        history_finetune = train_phase(
            model, train_loader, val_loader, criterion,
            device, cfg, 'finetune', logger, config.MODEL_DIR
        )

    # Save final model
    final_model_path = config.MODEL_DIR / "cnn_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'config': cfg,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class
    }, final_model_path)
    logger.info(f"\nFinal model saved to {final_model_path}")

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc, test_acc_top3, test_acc_top5 = validate(
        model, test_loader, criterion, device, cfg['mixed_precision']
    )

    logger.info(f"\n{'='*60}")
    logger.info("FINAL TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Test Accuracy (Top-1): {test_acc*100:.2f}%")
    logger.info(f"Test Accuracy (Top-3): {test_acc_top3*100:.2f}%")
    logger.info(f"Test Accuracy (Top-5): {test_acc_top5*100:.2f}%")

    # Save training history
    history = {
        'frozen': history_frozen,
        'finetune': history_finetune,
        'test_results': {
            'accuracy': test_acc,
            'top3_accuracy': test_acc_top3,
            'top5_accuracy': test_acc_top5
        }
    }

    history_path = config.RESULTS_DIR / "cnn_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    logger.info("\nTraining completed!")

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for plankton classification")

    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs for frozen phase')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--no-finetune', action='store_true',
                        help='Skip fine-tuning phase')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if args.seed:
        config.RANDOM_SEED = args.seed

    main(args)
