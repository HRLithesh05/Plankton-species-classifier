"""
Dataset module for Plankton Species Classifier.
Handles data loading, augmentation, and class imbalance.
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_mapping(data_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create class name to index mapping.
    Returns both forward and reverse mappings.
    """
    class_names = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    return class_to_idx, idx_to_class


def load_image_paths(
    data_dir: Path,
    min_samples: int = 5,
    max_samples: Optional[int] = None,
    exclude_classes: List[str] = None
) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    """
    Load all image paths and labels from the dataset directory.

    Args:
        data_dir: Path to dataset directory
        min_samples: Minimum samples required per class
        max_samples: Maximum samples per class (None for no limit)
        exclude_classes: List of class names to exclude

    Returns:
        image_paths, labels, class_to_idx, idx_to_class
    """
    if exclude_classes is None:
        exclude_classes = []

    class_to_idx, idx_to_class = get_class_mapping(data_dir)

    image_paths = []
    labels = []
    class_counts = Counter()
    skipped_classes = []
    excluded_classes_found = []

    print(f"Loading images from {data_dir}...")
    if exclude_classes:
        print(f"Excluding classes: {exclude_classes}")

    for class_name, class_idx in tqdm(class_to_idx.items(), desc="Scanning classes"):
        # Skip excluded classes
        if class_name in exclude_classes:
            excluded_classes_found.append(class_name)
            continue

        class_dir = data_dir / class_name

        # Find all images in this class
        class_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            class_images.extend(list(class_dir.glob(ext)))

        # Skip classes with too few samples
        if len(class_images) < min_samples:
            skipped_classes.append((class_name, len(class_images)))
            continue

        # Limit samples if max_samples is set
        if max_samples and len(class_images) > max_samples:
            class_images = random.sample(class_images, max_samples)

        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(class_idx)
            class_counts[class_name] += 1

    if excluded_classes_found:
        print(f"\nExcluded {len(excluded_classes_found)} mega-classes:")
        for name in excluded_classes_found:
            print(f"  - {name}")

    if skipped_classes:
        print(f"\nSkipped {len(skipped_classes)} classes with < {min_samples} samples:")
        for name, count in skipped_classes[:10]:  # Show first 10
            print(f"  - {name}: {count} samples")
        if len(skipped_classes) > 10:
            print(f"  ... and {len(skipped_classes)-10} more")

    # Update mappings to only include classes with enough samples
    valid_classes = set(class_counts.keys())
    new_class_to_idx = {
        name: idx for idx, name in enumerate(sorted(valid_classes))
    }
    new_idx_to_class = {idx: name for name, idx in new_class_to_idx.items()}

    # Remap labels
    old_to_new = {
        class_to_idx[name]: new_class_to_idx[name]
        for name in valid_classes
    }
    labels = [old_to_new[label] for label in labels]

    print(f"\nDataset summary:")
    print(f"  Total images: {len(image_paths):,}")
    print(f"  Number of classes: {len(new_class_to_idx)}")
    print(f"  Min samples/class: {min(class_counts.values()):,}")
    print(f"  Max samples/class: {max(class_counts.values()):,}")

    return image_paths, labels, new_class_to_idx, new_idx_to_class


def split_dataset(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List, List, List, List, List, List]:
    """
    Split dataset into train, validation, and test sets.
    Uses stratified split to maintain class distribution.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_ratio,
        stratify=y_temp,
        random_state=seed
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(
    labels: List[int],
    strategy: str = 'sqrt_weighted',
    num_classes: int = None
) -> torch.Tensor:
    """
    Compute class weights to handle imbalance.

    Strategies:
        - 'inverse': weight = 1 / count
        - 'sqrt_inverse': weight = 1 / sqrt(count)
        - 'effective': effective number of samples (CB focal loss paper)
        - 'sqrt_weighted': balanced sqrt weighting
    """
    class_counts = Counter(labels)

    if num_classes is None:
        num_classes = len(class_counts)

    counts = np.array([class_counts.get(i, 1) for i in range(num_classes)])

    if strategy == 'inverse':
        weights = 1.0 / counts
    elif strategy == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(counts)
    elif strategy == 'effective':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    elif strategy == 'sqrt_weighted':
        total = counts.sum()
        weights = np.sqrt(total / counts)
    else:
        weights = np.ones(num_classes)

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights)


def get_sample_weights(labels: List[int], class_weights: torch.Tensor) -> List[float]:
    """Get per-sample weights for WeightedRandomSampler."""
    return [class_weights[label].item() for label in labels]


class PlanktonDataset(Dataset):
    """
    PyTorch Dataset for plankton images.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        cache_images: bool = False
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}

        if cache_images:
            print("Caching images in memory...")
            for i, path in enumerate(tqdm(image_paths)):
                self.image_cache[i] = self._load_image(path)

    def _load_image(self, path: str) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            img = Image.open(path)
            # Convert grayscale or RGBA to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), (128, 128, 128))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx].copy()
        else:
            image = self._load_image(self.image_paths[idx])

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    augmentation: bool = True
) -> transforms.Compose:
    """
    Get image transforms for training or evaluation.
    Balanced augmentation - not too aggressive.
    """
    # ImageNet normalization (for pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training and augmentation:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            # Moderate color augmentation
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.15,
                hue=0.08
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        # Evaluation transforms
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def create_data_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augmentation: bool = True,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Dict]:
    """
    Create train, validation, and test data loaders.

    Returns:
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class
    """
    set_seed(config.RANDOM_SEED)

    # Get exclude list from config if it exists
    exclude_classes = getattr(config, 'EXCLUDE_CLASSES', [])

    # Load image paths and labels
    image_paths, labels, class_to_idx, idx_to_class = load_image_paths(
        config.DATA_DIR,
        min_samples=config.MIN_SAMPLES_PER_CLASS,
        max_samples=config.MAX_SAMPLES_PER_CLASS,
        exclude_classes=exclude_classes
    )

    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        image_paths, labels,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.RANDOM_SEED
    )

    # Get transforms
    train_transform = get_transforms(image_size, is_training=True, augmentation=augmentation)
    eval_transform = get_transforms(image_size, is_training=False)

    # Create datasets
    train_dataset = PlanktonDataset(X_train, y_train, transform=train_transform)
    val_dataset = PlanktonDataset(X_val, y_val, transform=eval_transform)
    test_dataset = PlanktonDataset(X_test, y_test, transform=eval_transform)

    # Create sampler for handling class imbalance
    if use_weighted_sampler:
        class_weights = compute_class_weights(
            y_train,
            strategy=config.CLASS_BALANCE_STRATEGY,
            num_classes=len(class_to_idx)
        )
        sample_weights = get_sample_weights(y_train, class_weights)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class


# =============================================================================
# Traditional ML Feature Extraction
# =============================================================================

def extract_traditional_features(image_path: str, config_dict: dict) -> np.ndarray:
    """
    Extract hand-crafted features for traditional ML.

    Features extracted:
        - HOG (Histogram of Oriented Gradients)
        - LBP (Local Binary Patterns)
        - Shape descriptors (contour-based)
        - Basic statistics
    """
    import cv2
    from skimage.feature import hog, local_binary_pattern

    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Return zeros if image can't be loaded
        return np.zeros(get_feature_size(config_dict))

    img_size = config_dict['image_size']
    img = cv2.resize(img, (img_size, img_size))

    features = []

    # 1. HOG features
    hog_features = hog(
        img,
        orientations=config_dict['hog_orientations'],
        pixels_per_cell=config_dict['hog_pixels_per_cell'],
        cells_per_block=config_dict['hog_cells_per_block'],
        feature_vector=True,
        block_norm='L2-Hys'
    )
    features.append(hog_features)

    # 2. LBP features (histogram)
    lbp = local_binary_pattern(
        img,
        P=config_dict['lbp_n_points'],
        R=config_dict['lbp_radius'],
        method='uniform'
    )
    n_bins = config_dict['lbp_n_points'] + 2
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )
    features.append(lbp_hist)

    # 3. Shape features from contours
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

        # Bounding rect features
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / (h + 1e-6)
        extent = area / (w * h + 1e-6)

        # Convex hull
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)

        # Equivalent diameter
        equiv_diameter = np.sqrt(4 * area / np.pi)

        shape_features = np.array([
            area / (img_size ** 2),  # Normalized area
            perimeter / (4 * img_size),  # Normalized perimeter
            circularity,
            aspect_ratio,
            extent,
            solidity,
            equiv_diameter / img_size
        ])
    else:
        shape_features = np.zeros(7)

    features.append(shape_features)

    # 4. Basic image statistics
    stats = np.array([
        img.mean() / 255.0,
        img.std() / 255.0,
        np.median(img) / 255.0,
        img.min() / 255.0,
        img.max() / 255.0
    ])
    features.append(stats)

    return np.concatenate(features)


def get_feature_size(config_dict: dict) -> int:
    """Calculate total feature vector size."""
    img_size = config_dict['image_size']
    hog_pixels = config_dict['hog_pixels_per_cell'][0]
    hog_cells = config_dict['hog_cells_per_block'][0]
    hog_orient = config_dict['hog_orientations']

    # HOG size calculation
    cells_per_img = img_size // hog_pixels
    blocks_per_img = cells_per_img - hog_cells + 1
    hog_size = blocks_per_img * blocks_per_img * hog_cells * hog_cells * hog_orient

    # LBP histogram size
    lbp_size = config_dict['lbp_n_points'] + 2

    # Shape features
    shape_size = 7

    # Statistics
    stats_size = 5

    return hog_size + lbp_size + shape_size + stats_size


def prepare_traditional_ml_data(
    max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
    """
    Prepare feature matrices for traditional ML training.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx, idx_to_class
    """
    from sklearn.preprocessing import StandardScaler
    import joblib

    set_seed(config.RANDOM_SEED)

    # Get exclude list from config if it exists
    exclude_classes = getattr(config, 'EXCLUDE_CLASSES', [])

    # Load paths
    image_paths, labels, class_to_idx, idx_to_class = load_image_paths(
        config.DATA_DIR,
        min_samples=config.MIN_SAMPLES_PER_CLASS,
        max_samples=max_samples or config.TRADITIONAL_ML_CONFIG['max_train_samples'],
        exclude_classes=exclude_classes
    )

    # Split
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = split_dataset(
        image_paths, labels,
        seed=config.RANDOM_SEED
    )

    cfg = config.TRADITIONAL_ML_CONFIG

    print("\nExtracting features...")

    # Extract features
    print("  Training set...")
    X_train = np.array([
        extract_traditional_features(p, cfg)
        for p in tqdm(X_train_paths, desc="  Train features")
    ])

    print("  Validation set...")
    X_val = np.array([
        extract_traditional_features(p, cfg)
        for p in tqdm(X_val_paths, desc="  Val features")
    ])

    print("  Test set...")
    X_test = np.array([
        extract_traditional_features(p, cfg)
        for p in tqdm(X_test_paths, desc="  Test features")
    ])

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler
    scaler_path = config.MODEL_DIR / 'feature_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    return (
        X_train, X_val, X_test,
        np.array(y_train), np.array(y_val), np.array(y_test),
        class_to_idx, idx_to_class
    )


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = create_data_loaders(
        batch_size=16,
        num_workers=0  # Use 0 for testing
    )

    print(f"\nClass mapping (first 10):")
    for i in range(min(10, len(idx_to_class))):
        print(f"  {i}: {idx_to_class[i]}")

    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} - {labels.max()}")
