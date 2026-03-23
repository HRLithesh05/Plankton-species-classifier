#!/usr/bin/env python3
"""
🚀 APPROACH 1 IMPROVED: Advanced EfficientNet for Plankton Classification (Colab Optimized)
- Fixed deprecation warnings
- Better progress tracking
- Improved error handling
- Optimized for Colab environment
- Target: 82-85% accuracy

Changes from original:
✅ Fixed PyTorch AMP deprecation warnings
✅ Better logging and progress bars
✅ Automatic model size adjustment based on GPU memory
✅ Improved data augmentation
✅ Better checkpoint saving
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
import json
import time
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import zipfile
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_gpu_memory_gb():
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0

def choose_optimal_config():
    """Choose optimal configuration based on available GPU memory."""
    gpu_memory = get_gpu_memory_gb()

    if gpu_memory >= 15:  # T4, V100, etc.
        return {
            'model_size': 'b4',
            'batch_sizes': [32, 24, 16],
            'max_image_size': 384
        }
    elif gpu_memory >= 12:  # RTX 3080, etc.
        return {
            'model_size': 'b3',
            'batch_sizes': [24, 20, 12],
            'max_image_size': 320
        }
    else:  # Smaller GPUs
        return {
            'model_size': 'b2',
            'batch_sizes': [16, 12, 8],
            'max_image_size': 288
        }

class AdvancedPlanktonDataset(Dataset):
    """Advanced dataset with progressive resizing and heavy augmentation."""

    def __init__(self, image_paths, labels, transform=None, phase='train'):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            label = self.labels[idx]

            # Load image with error handling
            image = Image.open(image_path).convert('RGB')

            # Basic quality check
            if image.size[0] < 32 or image.size[1] < 32:
                # Skip very small images
                return self.__getitem__((idx + 1) % len(self.image_paths))

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Warning: Error loading image {self.image_paths[idx]}: {e}")
            # Return next image on error
            return self.__getitem__((idx + 1) % len(self.image_paths))

class AdvancedEfficientNet(nn.Module):
    """Advanced EfficientNet with custom head for plankton classification."""

    def __init__(self, num_classes=67, model_size='b4', dropout=0.4):
        super().__init__()

        print(f"🏗️  Building EfficientNet-{model_size.upper()} model...")

        # Choose EfficientNet variant based on size
        if model_size == 'b4':
            self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1')
            in_features = 1792
        elif model_size == 'b3':
            self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
            in_features = 1536
        elif model_size == 'b2':
            self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
            in_features = 1408
        else:  # b1 fallback
            self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1')
            in_features = 1280

        # Advanced classifier head optimized for microscopic images
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout * 0.3),
            nn.Linear(256, num_classes)
        )

        self.model_size = model_size

    def forward(self, x):
        return self.backbone(x)

class ImprovedProgressiveTrainer:
    """Improved progressive training with better error handling and logging."""

    def __init__(self, num_classes=67):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get optimal configuration
        self.config = choose_optimal_config()

        print(f"🚀 Initializing Advanced EfficientNet Trainer")
        print(f"   Device: {self.device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = get_gpu_memory_gb()
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"   Optimal Config: EfficientNet-{self.config['model_size'].upper()}")

        # Progressive training stages (adjusted based on GPU memory)
        max_size = self.config['max_image_size']
        batch_sizes = self.config['batch_sizes']

        self.stages = [
            {
                'name': 'Foundation',
                'size': 224,
                'epochs': 12,
                'lr': 3e-4,
                'batch_size': batch_sizes[0]
            },
            {
                'name': 'Refinement',
                'size': min(288, max_size),
                'epochs': 18,
                'lr': 1e-4,
                'batch_size': batch_sizes[1]
            },
            {
                'name': 'Fine-tuning',
                'size': max_size,
                'epochs': 12,
                'lr': 5e-5,
                'batch_size': batch_sizes[2]
            },
        ]

    def get_transforms(self, size, phase):
        """Get transforms for current stage and phase."""
        if phase == 'train':
            return transforms.Compose([
                transforms.Resize((int(size * 1.15), int(size * 1.15))),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.85, 1.15),
                    shear=10
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.25))
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def extract_and_prepare_data(self):
        """Extract dataset and prepare file lists with better error handling."""
        print("📦 Extracting and preparing dataset...")

        # Extract dataset if needed
        if not os.path.exists('2014_clean'):
            if os.path.exists('2014_clean.zip'):
                print("   Extracting 2014_clean.zip...")
                with zipfile.ZipFile('2014_clean.zip', 'r') as zip_ref:
                    zip_ref.extractall('.')
            else:
                raise FileNotFoundError("Dataset file '2014_clean.zip' not found!")

        # Collect all image paths and labels
        image_paths = []
        labels = []
        class_names = []
        class_counts = {}

        data_dir = Path('2014_clean')
        print("   Scanning dataset directories...")

        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir():
                class_names.append(class_dir.name)
                class_idx = len(class_names) - 1

                # Find images with multiple extensions
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
                        list(class_dir.glob('*.png')) + list(class_dir.glob('*.bmp'))

                valid_images = 0
                for img_path in images:
                    try:
                        # Quick validation
                        with Image.open(img_path) as img:
                            if img.size[0] >= 32 and img.size[1] >= 32:
                                image_paths.append(str(img_path))
                                labels.append(class_idx)
                                valid_images += 1
                    except Exception:
                        # Skip corrupted images
                        continue

                class_counts[class_dir.name] = valid_images

        print(f"📊 Dataset Statistics:")
        print(f"   Classes: {len(class_names)}")
        print(f"   Total valid images: {len(image_paths)}")
        print(f"   Average per class: {len(image_paths)/len(class_names):.1f}")

        # Show class distribution
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top classes: {sorted_classes[:3]}")
        print(f"   Smallest classes: {sorted_classes[-3:]}")

        # Create class mappings
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}

        # Stratified train-test split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )

        print(f"   Training images: {len(train_paths)}")
        print(f"   Validation images: {len(val_paths)}")

        return {
            'train_paths': train_paths,
            'val_paths': val_paths,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'num_classes': len(class_names)
        }

    def train_stage(self, model, stage_config, data_info, stage_num):
        """Train one progressive stage with improved logging."""
        stage_name = stage_config['name']
        print(f"\n🎯 Stage {stage_num}: {stage_name}")
        print(f"   Image Size: {stage_config['size']}px")
        print(f"   Epochs: {stage_config['epochs']}")
        print(f"   Batch Size: {stage_config['batch_size']}")
        print(f"   Learning Rate: {stage_config['lr']:.0e}")

        # Create data loaders for this stage
        train_transform = self.get_transforms(stage_config['size'], 'train')
        val_transform = self.get_transforms(stage_config['size'], 'val')

        train_dataset = AdvancedPlanktonDataset(
            data_info['train_paths'],
            data_info['train_labels'],
            train_transform,
            'train'
        )
        val_dataset = AdvancedPlanktonDataset(
            data_info['val_paths'],
            data_info['val_labels'],
            val_transform,
            'val'
        )

        # Adjust batch size if memory issues
        current_batch_size = stage_config['batch_size']

        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=current_batch_size * 2,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                current_batch_size = max(4, current_batch_size // 2)
                print(f"   ⚠️  Reducing batch size to {current_batch_size} due to memory constraints")

                train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=2)
            else:
                raise e

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=stage_config['lr'],
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=stage_config['epochs']//3, T_mult=2, eta_min=stage_config['lr']/100
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Use newer AMP API to avoid deprecation warnings
        scaler = torch.amp.GradScaler('cuda')

        best_val_acc = 0.0
        patience = 8
        patience_counter = 0

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(stage_config['epochs']):
            start_time = time.time()

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{stage_config['epochs']}")
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                # Mixed precision forward pass (using newer API)
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Statistics
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%',
                    'LR': f'{current_lr:.2e}'
                })

            scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    with torch.amp.autocast('cuda'):
                        output = model(data)
                        loss = criterion(output, target)

                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()

            # Calculate metrics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            epoch_time = time.time() - start_time

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1:2d}: Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | Time: {epoch_time:4.1f}s")

            # Save best model and early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch,
                    'stage': stage_num,
                    'stage_name': stage_name,
                    'config': {
                        'model_size': self.config['model_size'],
                        'image_size': stage_config['size'],
                        'num_classes': data_info['num_classes']
                    },
                    'class_to_idx': data_info['class_to_idx'],
                    'idx_to_class': data_info['idx_to_class']
                }
                torch.save(checkpoint, f'approach1_stage{stage_num}_{stage_name.lower()}_best.pth')

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

        print(f"✅ Stage {stage_num} ({stage_name}) completed!")
        print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
        return best_val_acc, history

    def train_progressive(self):
        """Run progressive training through all stages."""
        print("🚀 Starting Advanced Progressive EfficientNet Training")
        print("=" * 60)

        # Prepare data
        data_info = self.extract_and_prepare_data()

        if data_info['num_classes'] != self.num_classes:
            print(f"📝 Adjusting model for {data_info['num_classes']} classes")
            self.num_classes = data_info['num_classes']

        # Initialize model with optimal configuration
        try:
            model = AdvancedEfficientNet(
                num_classes=data_info['num_classes'],
                model_size=self.config['model_size']
            ).to(self.device)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️  GPU memory insufficient for planned model, reducing size...")
                # Fallback to smaller model
                if self.config['model_size'] == 'b4':
                    self.config['model_size'] = 'b3'
                elif self.config['model_size'] == 'b3':
                    self.config['model_size'] = 'b2'
                else:
                    self.config['model_size'] = 'b1'

                model = AdvancedEfficientNet(
                    num_classes=data_info['num_classes'],
                    model_size=self.config['model_size']
                ).to(self.device)
            else:
                raise e

        overall_history = {}
        stage_accuracies = []

        # Progressive training through stages
        for stage_num, stage_config in enumerate(self.stages, 1):
            try:
                best_acc, history = self.train_stage(
                    model, stage_config, data_info, stage_num
                )

                stage_accuracies.append(best_acc)
                overall_history[f'stage_{stage_num}_{stage_config["name"]}'] = {
                    'best_accuracy': best_acc,
                    'history': history,
                    'config': stage_config
                }

                # Memory cleanup
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️  Stage {stage_num} OOM, reducing batch size and retrying...")
                    # Reduce batch size significantly
                    stage_config['batch_size'] = max(4, stage_config['batch_size'] // 3)
                    best_acc, history = self.train_stage(
                        model, stage_config, data_info, stage_num
                    )
                    stage_accuracies.append(best_acc)
                    overall_history[f'stage_{stage_num}_{stage_config["name"]}'] = {
                        'best_accuracy': best_acc,
                        'history': history,
                        'config': stage_config
                    }
                else:
                    raise e

        # Save final comprehensive model
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_to_idx': data_info['class_to_idx'],
            'idx_to_class': data_info['idx_to_class'],
            'num_classes': data_info['num_classes'],
            'config': {
                'model_name': f'efficientnet_{self.config["model_size"]}',
                'approach': 'progressive_training_v2',
                'final_accuracy': max(stage_accuracies),
                'all_stage_accuracies': stage_accuracies
            },
            'training_history': overall_history,
            'gpu_info': {
                'name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'memory_gb': get_gpu_memory_gb()
            }
        }

        torch.save(final_checkpoint, 'approach1_final_model.pth')

        # Display results
        print("\n🎉 Progressive Training Completed!")
        print("=" * 50)
        print("📄 Models Saved:")
        for i, stage in enumerate(self.stages, 1):
            stage_name = stage['name'].lower()
            print(f"  - approach1_stage{i}_{stage_name}_best.pth ({stage_accuracies[i-1]:.2f}%)")
        print(f"  - approach1_final_model.pth")

        print(f"\n📊 Training Summary:")
        print(f"  Best Stage Accuracy: {max(stage_accuracies):.2f}%")
        print(f"  Model Architecture: EfficientNet-{self.config['model_size'].upper()}")
        print(f"  Total Classes: {data_info['num_classes']}")
        print(f"  Training Images: {len(data_info['train_paths'])}")

        return overall_history

def main():
    """Main training function with better error handling."""
    print("🚀 APPROACH 1 IMPROVED: Advanced EfficientNet Training")
    print("=" * 60)

    # System checks
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = get_gpu_memory_gb()
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")

        if gpu_memory < 8:
            print("⚠️  Warning: Limited GPU memory detected. Training may be slower.")
    else:
        print("⚠️  No GPU available, using CPU (will be very slow)")
        response = input("Continue with CPU training? (y/N): ")
        if response.lower() != 'y':
            return

    # Check dataset
    if not (os.path.exists('2014_clean') or os.path.exists('2014_clean.zip')):
        print("❌ Dataset not found!")
        print("   Please upload '2014_clean.zip' to this directory")
        return

    try:
        # Start training
        trainer = ImprovedProgressiveTrainer()
        history = trainer.train_progressive()

        # Summary
        print("\n🏆 FINAL TRAINING SUMMARY")
        print("=" * 40)
        for stage_name, stage_data in history.items():
            print(f"{stage_name}: {stage_data['best_accuracy']:.2f}%")

        print(f"\n🎯 Ready to integrate into Flask app!")
        print("   Run: python integrate_models.py")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()