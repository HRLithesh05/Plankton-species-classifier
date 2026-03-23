"""
=============================================================================
PLANKTON SPECIES CLASSIFIER - FAST TRAINING SCRIPT
=============================================================================

FASTER training with OneCycleLR scheduler + optimized settings.
Target: 75-82% accuracy in ~1.5-2 hours on RTX 4060 Ti

INSTRUCTIONS:
1. Dataset in '2014' folder
2. pip install torch torchvision scikit-learn scikit-image tqdm pillow
3. python train_fast.py

Author: Auto-generated for Plankton Classification Project
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

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
# CONFIGURATION - FAST + HIGH ACCURACY
# =============================================================================

CONFIG = {
    # Paths
    'data_dir': Path('2014'),
    'output_dir': Path('outputs/models'),

    # Model
    'model_name': 'efficientnet_v2_s',
    'image_size': 224,

    # FAST Training - Phase 1 (frozen backbone)
    'epochs_frozen': 25,        # Reduced from 80
    'lr_frozen': 3e-3,          # Higher LR with OneCycleLR
    'batch_size': 48,           # Larger batch = faster

    # FAST Training - Phase 2 (fine-tuning)
    'epochs_finetune': 20,      # Reduced from 50
    'lr_finetune': 1e-4,        # Higher for OneCycleLR
    'unfreeze_layers': 80,      # Unfreeze more layers

    # Regularization - lighter for speed
    'dropout': 0.25,
    'label_smoothing': 0.1,
    'weight_decay': 0.01,

    # Early stopping
    'patience': 8,              # Stop faster if no improvement

    # Data filtering
    'min_samples_per_class': 15,
    'max_samples_per_class': 2500,
    'exclude_classes': ['mix', 'detritus', 'mix_elongated'],

    # Hardware
    'num_workers': 4,
    'pin_memory': True,
    'mixed_precision': True,

    # Seed
    'seed': 42,
}

# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU found, using CPU (will be very slow)")
    return device

# =============================================================================
# DATA AUGMENTATION - STRONG BUT FAST
# =============================================================================

def get_train_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

def get_val_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# =============================================================================
# DATASET
# =============================================================================

class PlanktonDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def load_dataset(config: dict) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    """Load and filter dataset."""
    data_dir = config['data_dir']

    print(f"\nLoading dataset from {data_dir}...")

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # Collect all images
    class_counts = {}
    all_images = []
    all_labels = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # Skip excluded classes
        if class_name in config['exclude_classes']:
            continue

        # Find images
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
            images.extend(list(class_dir.glob(ext)))

        if len(images) < config['min_samples_per_class']:
            continue

        # Cap samples per class
        if len(images) > config['max_samples_per_class']:
            images = random.sample(images, config['max_samples_per_class'])

        class_counts[class_name] = len(images)

        for img_path in images:
            all_images.append(str(img_path))
            all_labels.append(class_name)

    # Create class mappings
    class_names = sorted(class_counts.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # Convert labels to indices
    labels = [class_to_idx[label] for label in all_labels]

    print(f"Loaded {len(all_images)} images from {len(class_names)} classes")
    print(f"Samples per class: {min(class_counts.values())} - {max(class_counts.values())}")

    return all_images, labels, class_to_idx, idx_to_class

def create_data_loaders(
    images: List[str],
    labels: List[int],
    config: dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders with weighted sampling."""

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.15, stratify=labels, random_state=config['seed']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=config['seed']
    )

    print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Transforms
    train_transform = get_train_transforms(config['image_size'])
    val_transform = get_val_transforms(config['image_size'])

    # Datasets
    train_dataset = PlanktonDataset(X_train, y_train, transform=train_transform)
    val_dataset = PlanktonDataset(X_val, y_val, transform=val_transform)
    test_dataset = PlanktonDataset(X_test, y_test, transform=val_transform)

    # Weighted sampler for class imbalance
    class_counts = Counter(y_train)
    weights = [1.0 / np.sqrt(class_counts[label]) for label in y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    return train_loader, val_loader, test_loader

# =============================================================================
# MODEL
# =============================================================================

class PlanktonCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25):
        super().__init__()

        # Load pretrained EfficientNetV2-S
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = 80):
        # Unfreeze last N layers
        all_params = list(self.backbone.features.parameters())
        for param in all_params[-num_layers:]:
            param.requires_grad = True

# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def evaluate_topk(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 3, 5]
) -> Dict[int, float]:
    """Evaluate top-k accuracy."""
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    results = {}
    for k in k_values:
        _, top_k_preds = all_probs.topk(k, dim=1)
        correct = sum(all_labels[i] in top_k_preds[i] for i in range(len(all_labels)))
        results[k] = correct / len(all_labels)

    return results

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_model(config: dict):
    """Main training function."""
    set_seed(config['seed'])
    device = get_device()

    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # Load data
    images, labels, class_to_idx, idx_to_class = load_dataset(config)
    num_classes = len(class_to_idx)

    print(f"\nNumber of classes: {num_classes}")

    # Save class mapping
    mapping_path = config['output_dir'] / 'class_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(k): v for k, v in idx_to_class.items()}
        }, f, indent=2)
    print(f"Saved class mapping to {mapping_path}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(images, labels, config)

    # Create model
    print(f"\nCreating {config['model_name']} model...")
    model = PlanktonCNN(num_classes=num_classes, dropout=config['dropout'])
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    # Mixed precision
    scaler = GradScaler() if config['mixed_precision'] else None
    use_amp = config['mixed_precision'] and device.type == 'cuda'

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # ==========================================================================
    # PHASE 1: FROZEN BACKBONE
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Training with frozen backbone")
    print("="*60)

    model.freeze_backbone()

    # OneCycleLR for fast convergence
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr_frozen'],
        weight_decay=config['weight_decay']
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr_frozen'],
        epochs=config['epochs_frozen'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    patience_counter = 0

    for epoch in range(config['epochs_frozen']):
        print(f"\nEpoch {epoch+1}/{config['epochs_frozen']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, use_amp
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'phase': 'frozen'
            }, config['output_dir'] / 'best_model_frozen.pth')
            print(f">> New best: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ==========================================================================
    # PHASE 2: FINE-TUNING
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning backbone")
    print("="*60)

    # Load best frozen model
    checkpoint = torch.load(config['output_dir'] / 'best_model_frozen.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best frozen model (acc: {checkpoint['val_acc']*100:.2f}%)")

    # Unfreeze backbone
    model.unfreeze_backbone(config['unfreeze_layers'])

    # New optimizer with lower LR
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr_finetune'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr_finetune'],
        epochs=config['epochs_finetune'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    patience_counter = 0

    for epoch in range(config['epochs_finetune']):
        print(f"\nEpoch {epoch+1}/{config['epochs_finetune']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, use_amp
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'phase': 'finetune'
            }, config['output_dir'] / 'best_model_finetune.pth')
            print(f">> New best: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ==========================================================================
    # FINAL EVALUATION
    # ==========================================================================
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)

    # Load best model
    best_path = config['output_dir'] / 'best_model_finetune.pth'
    if not best_path.exists():
        best_path = config['output_dir'] / 'best_model_frozen.pth'

    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    topk_acc = evaluate_topk(model, test_loader, device, [1, 3, 5])

    print(f"\nTest Results:")
    print(f"  Top-1 Accuracy: {topk_acc[1]*100:.2f}%")
    print(f"  Top-3 Accuracy: {topk_acc[3]*100:.2f}%")
    print(f"  Top-5 Accuracy: {topk_acc[5]*100:.2f}%")

    # Save final model with all info
    final_path = config['output_dir'] / 'cnn_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': {str(k): v for k, v in idx_to_class.items()},
        'config': {
            'model_name': config['model_name'],
            'dropout': config['dropout'],
            'image_size': config['image_size']
        },
        'test_accuracy': topk_acc,
        'best_val_acc': best_val_acc
    }, final_path)

    print(f"\nSaved final model to {final_path}")

    # Save history
    history_path = config['output_dir'].parent / 'results'
    history_path.mkdir(parents=True, exist_ok=True)

    with open(history_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Test Top-1 Accuracy: {topk_acc[1]*100:.2f}%")
    print(f"Test Top-5 Accuracy: {topk_acc[5]*100:.2f}%")
    print(f"\nFiles to send back:")
    print(f"  1. {config['output_dir'] / 'best_model_finetune.pth'}")
    print(f"  2. {config['output_dir'] / 'cnn_final.pth'}")
    print(f"  3. {config['output_dir'] / 'class_mapping.json'}")

    return model, history, topk_acc

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PLANKTON CLASSIFIER - FAST TRAINING")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = datetime.now()

    try:
        model, history, results = train_model(CONFIG)

        elapsed = datetime.now() - start_time
        print(f"\nTotal training time: {elapsed}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
