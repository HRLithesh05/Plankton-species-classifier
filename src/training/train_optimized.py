"""
=============================================================================
PLANKTON SPECIES CLASSIFIER - OPTIMIZED TRAINING SCRIPT
=============================================================================

This script trains an EfficientNetV2 model for plankton classification.
Optimized for RTX 4060 Ti (8GB VRAM).

INSTRUCTIONS FOR RUNNING:
1. Make sure you have the dataset in the '2014' folder
2. Install requirements: pip install torch torchvision scikit-learn scikit-image tqdm pillow
3. Run: python train_optimized.py

The script will:
- Train for 80 epochs (frozen) + 50 epochs (fine-tuning)
- Save the best model to 'outputs/models/'
- Expected accuracy: 70-80%
- Training time: ~4-6 hours on RTX 4060 Ti

Author: Auto-generated for Plankton Classification Project
"""

import os
import sys
import json
import random
import logging
import time
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR MAXIMUM ACCURACY
# =============================================================================

CONFIG = {
    # Paths
    'data_dir': Path('2014'),
    'output_dir': Path('outputs'),
    'model_dir': Path('outputs/models'),
    'log_dir': Path('outputs/logs'),

    # Data settings
    'image_size': 224,
    'min_samples_per_class': 15,  # Lower threshold to keep more classes
    'max_samples_per_class': 3000,  # Higher cap
    'exclude_classes': ['mix', 'detritus', 'mix_elongated'],  # Remove noisy classes

    # Split ratios
    'train_ratio': 0.75,
    'val_ratio': 0.15,
    'test_ratio': 0.10,
    'random_seed': 42,

    # Model
    'model_name': 'efficientnet_v2_s',
    'pretrained': True,

    # Training Phase 1: Frozen backbone
    'batch_size': 32,  # Safe for 8GB VRAM
    'epochs_frozen': 80,
    'lr_frozen': 2e-3,

    # Training Phase 2: Fine-tuning
    'epochs_finetune': 50,
    'lr_finetune': 5e-5,
    'unfreeze_layers': 80,

    # Optimizer
    'weight_decay': 0.01,
    'warmup_epochs': 5,

    # Regularization - REDUCED for better learning
    'dropout': 0.2,
    'label_smoothing': 0.05,

    # Early stopping
    'patience': 20,

    # Performance
    'mixed_precision': True,
    'num_workers': 4,
    'gradient_accumulation': 2,
}

# =============================================================================
# SETUP
# =============================================================================

def setup_logging():
    """Setup logging configuration."""
    CONFIG['log_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['model_dir'].mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = CONFIG['log_dir'] / f"train_optimized_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# DATA LOADING
# =============================================================================

def load_image_paths(
    data_dir: Path,
    min_samples: int,
    max_samples: int,
    exclude_classes: List[str]
) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    """Load all image paths and labels."""

    # Get all class directories
    all_classes = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    # Filter out excluded classes
    valid_classes = [c for c in all_classes if c not in exclude_classes]

    image_paths = []
    labels = []
    class_counts = Counter()
    skipped = []

    print(f"\nLoading images from {data_dir}...")
    print(f"Excluding: {exclude_classes}")

    for class_name in tqdm(valid_classes, desc="Scanning classes"):
        class_dir = data_dir / class_name

        # Find all images
        class_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
            class_images.extend(list(class_dir.glob(ext)))

        # Skip if too few samples
        if len(class_images) < min_samples:
            skipped.append((class_name, len(class_images)))
            continue

        # Cap samples
        if len(class_images) > max_samples:
            class_images = random.sample(class_images, max_samples)

        class_counts[class_name] = len(class_images)

        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(class_name)

    # Create class mappings
    class_names = sorted(class_counts.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # Convert string labels to indices
    labels = [class_to_idx[label] for label in labels]

    print(f"\n{'='*50}")
    print(f"Dataset Summary:")
    print(f"  Total images: {len(image_paths):,}")
    print(f"  Number of classes: {len(class_to_idx)}")
    print(f"  Min samples/class: {min(class_counts.values()):,}")
    print(f"  Max samples/class: {max(class_counts.values()):,}")
    print(f"  Skipped {len(skipped)} classes with < {min_samples} samples")
    print(f"{'='*50}\n")

    return image_paths, labels, class_to_idx, idx_to_class


class PlanktonDataset(Dataset):
    """PyTorch Dataset for plankton images."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            # Return a gray image on error
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int, is_training: bool = True):
    """Get image transforms with strong augmentation for training."""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),  # Plankton can be any orientation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Compute class weights using square root balancing."""
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)  # Avoid division by zero

    total = counts.sum()
    weights = np.sqrt(total / counts)
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights)


def create_data_loaders(config: dict):
    """Create train, val, test data loaders."""

    # Load data
    image_paths, labels, class_to_idx, idx_to_class = load_image_paths(
        config['data_dir'],
        config['min_samples_per_class'],
        config['max_samples_per_class'],
        config['exclude_classes']
    )

    # Split: train -> (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(config['val_ratio'] + config['test_ratio']),
        stratify=labels,
        random_state=config['random_seed']
    )

    # Split: val -> test
    rel_test = config['test_ratio'] / (config['val_ratio'] + config['test_ratio'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=rel_test,
        stratify=y_temp,
        random_state=config['random_seed']
    )

    print(f"Data splits:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")

    # Transforms
    train_transform = get_transforms(config['image_size'], is_training=True)
    eval_transform = get_transforms(config['image_size'], is_training=False)

    # Datasets
    train_dataset = PlanktonDataset(X_train, y_train, train_transform)
    val_dataset = PlanktonDataset(X_val, y_val, eval_transform)
    test_dataset = PlanktonDataset(X_test, y_test, eval_transform)

    # Weighted sampler for class imbalance
    class_weights = compute_class_weights(y_train, len(class_to_idx))
    sample_weights = [class_weights[label].item() for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_weights


# =============================================================================
# MODEL
# =============================================================================

class PlanktonCNN(nn.Module):
    """EfficientNetV2-based CNN for plankton classification."""

    def __init__(self, num_classes: int, model_name: str = 'efficientnet_v2_s',
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        self.num_classes = num_classes

        # Load pretrained backbone
        weights = 'DEFAULT' if pretrained else None

        if model_name == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(weights=weights)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        elif model_name == 'efficientnet_v2_m':
            self.backbone = models.efficientnet_v2_m(weights=weights)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = -1):
        # Unfreeze all
        for param in self.backbone.parameters():
            param.requires_grad = True

        if num_layers > 0:
            # Freeze all except last num_layers
            modules = list(self.backbone.modules())
            for module in modules[:-num_layers]:
                for param in module.parameters():
                    param.requires_grad = False


# =============================================================================
# TRAINING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_state = None

    def __call__(self, val_acc: float, model: nn.Module) -> bool:
        if self.best_score is None or val_acc > self.best_score:
            self.best_score = val_acc
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def load_best(self, model: nn.Module, device: torch.device):
        model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})


def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, accum_steps):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
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

            # Top-5
            _, top5 = outputs.topk(5, dim=1)
            for i, label in enumerate(labels):
                if label in top5[i]:
                    correct_top5 += 1

    return total_loss / len(loader), correct / total, correct_top5 / total


def train_phase(model, train_loader, val_loader, criterion, device, config,
                phase: str, logger, class_weights):
    """Train one phase (frozen or finetune)."""

    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE: {phase.upper()}")
    logger.info(f"{'='*60}")

    if phase == 'frozen':
        epochs = config['epochs_frozen']
        lr = config['lr_frozen']
        model.freeze_backbone()
    else:
        epochs = config['epochs_finetune']
        lr = config['lr_finetune']
        model.unfreeze_backbone(config['unfreeze_layers'])

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config['weight_decay']
    )

    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        progress = (epoch - config['warmup_epochs']) / (epochs - config['warmup_epochs'])
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    use_amp = config['mixed_precision'] and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_acc_top5': []}

    for epoch in range(epochs):
        start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, use_amp, config['gradient_accumulation']
        )

        # Validate
        val_loss, val_acc, val_acc_top5 = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step()

        elapsed = time.time() - start

        # Log
        logger.info(
            f"[{phase}] Epoch {epoch+1}/{epochs} | "
            f"Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}% | "
            f"Top-5: {val_acc_top5*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_acc_top5'].append(val_acc_top5)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_acc_top5': val_acc_top5
            }
            torch.save(checkpoint, config['model_dir'] / f'best_model_{phase}.pth')
            logger.info(f"  -> New best model saved: {val_acc*100:.2f}%")

        # Early stopping
        if early_stopping(val_acc, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            early_stopping.load_best(model, device)
            break

    return history, best_val_acc


def main():
    """Main training function."""

    # Setup
    logger = setup_logging()
    set_seed(CONFIG['random_seed'])

    logger.info("="*60)
    logger.info("PLANKTON SPECIES CLASSIFIER - OPTIMIZED TRAINING")
    logger.info("="*60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_weights = \
        create_data_loaders(CONFIG)

    num_classes = len(class_to_idx)
    logger.info(f"Number of classes: {num_classes}")

    # Save class mapping
    mapping_path = CONFIG['model_dir'] / 'class_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(k): v for k, v in idx_to_class.items()}
        }, f, indent=2)
    logger.info(f"Class mapping saved to {mapping_path}")

    # Create model
    logger.info("\nCreating model...")
    model = PlanktonCNN(
        num_classes=num_classes,
        model_name=CONFIG['model_name'],
        pretrained=CONFIG['pretrained'],
        dropout=CONFIG['dropout']
    )
    model = model.to(device)

    # Loss function with class weights and label smoothing
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CONFIG['label_smoothing']
    )

    # Phase 1: Frozen backbone
    history_frozen, best_frozen = train_phase(
        model, train_loader, val_loader, criterion, device,
        CONFIG, 'frozen', logger, class_weights
    )

    # Phase 2: Fine-tuning
    history_finetune, best_finetune = train_phase(
        model, train_loader, val_loader, criterion, device,
        CONFIG, 'finetune', logger, class_weights
    )

    # Load best model for final evaluation
    best_path = CONFIG['model_dir'] / 'best_model_finetune.pth'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Final test evaluation
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("="*60)

    test_loss, test_acc, test_acc_top5 = validate(model, test_loader, criterion, device, CONFIG['mixed_precision'])

    logger.info(f"\nTEST RESULTS:")
    logger.info(f"  Top-1 Accuracy: {test_acc*100:.2f}%")
    logger.info(f"  Top-5 Accuracy: {test_acc_top5*100:.2f}%")

    # Save final model with all metadata
    final_model_path = CONFIG['model_dir'] / 'cnn_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'config': {
            'model_name': CONFIG['model_name'],
            'dropout': CONFIG['dropout'],
            'image_size': CONFIG['image_size']
        },
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'test_accuracy': test_acc,
        'test_accuracy_top5': test_acc_top5
    }, final_model_path)
    logger.info(f"\nFinal model saved to {final_model_path}")

    # Save training history
    history = {
        'frozen': history_frozen,
        'finetune': history_finetune,
        'test_results': {
            'accuracy': test_acc,
            'top5_accuracy': test_acc_top5
        }
    }
    history_path = CONFIG['output_dir'] / 'results' / 'cnn_training_history.json'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nFiles to send back:")
    logger.info(f"  1. {CONFIG['model_dir'] / 'best_model_finetune.pth'}")
    logger.info(f"  2. {CONFIG['model_dir'] / 'cnn_final.pth'}")
    logger.info(f"  3. {CONFIG['model_dir'] / 'class_mapping.json'}")
    logger.info(f"\nExpected accuracy: {test_acc*100:.2f}%")

    return test_acc


if __name__ == "__main__":
    main()
