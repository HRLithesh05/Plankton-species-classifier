"""
Data Preprocessing Script for Plankton Dataset
Moves unbalanced classes to separate folder and analyzes remaining data.
"""

import os
import shutil
from pathlib import Path
from collections import Counter
import json
import numpy as np
from tqdm import tqdm


def analyze_dataset(data_dir: Path):
    """Analyze class distribution in dataset."""
    print("=" * 70)
    print("ANALYZING DATASET")
    print("=" * 70)

    class_counts = {}

    for class_dir in data_dir.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            imgs = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            class_counts[class_dir.name] = len(imgs)

    # Sort by count
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])

    total_images = sum(class_counts.values())
    total_classes = len(class_counts)

    counts_array = np.array(list(class_counts.values()))

    print(f"\n📊 Dataset Statistics:")
    print(f"  Total classes: {total_classes}")
    print(f"  Total images: {total_images:,}")
    print(f"  Mean images/class: {counts_array.mean():.0f}")
    print(f"  Median images/class: {np.median(counts_array):.0f}")
    print(f"  Std deviation: {counts_array.std():.0f}")
    print(f"  Min images/class: {counts_array.min()}")
    print(f"  Max images/class: {counts_array.max()}")

    # Percentiles
    print(f"\n📈 Percentiles:")
    print(f"  25th percentile: {np.percentile(counts_array, 25):.0f} images/class")
    print(f"  50th percentile: {np.percentile(counts_array, 50):.0f} images/class")
    print(f"  75th percentile: {np.percentile(counts_array, 75):.0f} images/class")
    print(f"  90th percentile: {np.percentile(counts_array, 90):.0f} images/class")

    # Show extremes
    print(f"\n🔻 Bottom 15 classes (smallest):")
    for name, count in sorted_counts[:15]:
        print(f"  {name:40s} {count:6,} images")

    print(f"\n🔺 Top 15 classes (largest):")
    for name, count in sorted_counts[-15:]:
        print(f"  {name:40s} {count:6,} images")

    return class_counts, sorted_counts


def move_unbalanced_classes(
    data_dir: Path,
    unbalanced_dir: Path,
    min_threshold: int = 20,
    max_threshold: int = 2000,
    exclude_classes: list = None
):
    """
    Move unbalanced classes to separate folder.

    Args:
        data_dir: Original data directory
        unbalanced_dir: Directory to move unbalanced classes to
        min_threshold: Minimum images required (classes below this are moved)
        max_threshold: Maximum images allowed (classes above this are capped or moved)
        exclude_classes: List of class names to completely exclude (e.g., ['mix', 'detritus'])
    """
    if exclude_classes is None:
        exclude_classes = []

    print("\n" + "=" * 70)
    print("MOVING UNBALANCED CLASSES")
    print("=" * 70)
    print(f"Min threshold: {min_threshold} images")
    print(f"Max threshold: {max_threshold} images")
    print(f"Excluded classes: {exclude_classes}")

    # Create unbalanced directory
    unbalanced_dir.mkdir(exist_ok=True)

    moved_too_small = []
    moved_too_large = []
    moved_excluded = []
    kept_classes = []
    capped_classes = []

    for class_dir in tqdm(list(data_dir.iterdir()), desc="Processing classes"):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue

        class_name = class_dir.name
        imgs = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        img_count = len(imgs)

        # Check if excluded
        if class_name in exclude_classes:
            dest_dir = unbalanced_dir / class_name
            if not dest_dir.exists():
                shutil.move(str(class_dir), str(dest_dir))
                moved_excluded.append((class_name, img_count))
            continue

        # Check if too small
        if img_count < min_threshold:
            dest_dir = unbalanced_dir / class_name
            if not dest_dir.exists():
                shutil.move(str(class_dir), str(dest_dir))
                moved_too_small.append((class_name, img_count))

        # Check if too large
        elif img_count > max_threshold:
            # Cap the class by moving excess images to unbalanced folder
            excess_dir = unbalanced_dir / f"{class_name}_excess"
            excess_dir.mkdir(exist_ok=True)

            # Keep only max_threshold images, move rest
            excess_count = img_count - max_threshold
            imgs_to_move = imgs[max_threshold:]

            for img in imgs_to_move:
                shutil.move(str(img), str(excess_dir / img.name))

            moved_too_large.append((class_name, img_count, max_threshold))
            capped_classes.append((class_name, img_count, max_threshold))

        else:
            kept_classes.append((class_name, img_count))

    # Report
    print(f"\n📤 Moved {len(moved_too_small)} classes (too small < {min_threshold}):")
    for name, count in moved_too_small[:10]:
        print(f"  {name:40s} {count:6,} images")
    if len(moved_too_small) > 10:
        print(f"  ... and {len(moved_too_small) - 10} more")

    if moved_excluded:
        print(f"\n🚫 Moved {len(moved_excluded)} excluded classes:")
        for name, count in moved_excluded:
            print(f"  {name:40s} {count:6,} images")

    if capped_classes:
        print(f"\n✂️ Capped {len(capped_classes)} classes (too large > {max_threshold}):")
        for name, original, kept in capped_classes:
            print(f"  {name:40s} {original:6,} → {kept:6,} images (moved {original-kept:,} excess)")

    print(f"\n✅ Kept {len(kept_classes)} balanced classes")

    return kept_classes, moved_too_small, moved_too_large, moved_excluded


def analyze_cleaned_dataset(data_dir: Path):
    """Analyze the cleaned, balanced dataset."""
    print("\n" + "=" * 70)
    print("ANALYZING CLEANED DATASET")
    print("=" * 70)

    class_counts = {}

    for class_dir in data_dir.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            imgs = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            class_counts[class_dir.name] = len(imgs)

    if not class_counts:
        print("❌ ERROR: No classes remaining after filtering!")
        return None

    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    total_images = sum(class_counts.values())
    total_classes = len(class_counts)
    counts_array = np.array(list(class_counts.values()))

    print(f"\n📊 Cleaned Dataset Statistics:")
    print(f"  Total classes: {total_classes}")
    print(f"  Total images: {total_images:,}")
    print(f"  Mean images/class: {counts_array.mean():.0f}")
    print(f"  Median images/class: {np.median(counts_array):.0f}")
    print(f"  Std deviation: {counts_array.std():.0f}")
    print(f"  Min images/class: {counts_array.min()}")
    print(f"  Max images/class: {counts_array.max()}")

    # Calculate balance ratio (lower is better)
    balance_ratio = counts_array.max() / counts_array.min()
    print(f"\n⚖️ Balance Ratio: {balance_ratio:.2f}x (max/min)")
    if balance_ratio < 5:
        print("  ✅ EXCELLENT - Very balanced dataset!")
    elif balance_ratio < 10:
        print("  ✅ GOOD - Well balanced dataset")
    elif balance_ratio < 50:
        print("  ⚠️ MODERATE - Some imbalance remains")
    else:
        print("  ❌ POOR - Still highly imbalanced")

    # Check if enough data
    print(f"\n🎯 Training Readiness:")

    # Estimate samples after 70/15/15 split
    avg_train = counts_array.mean() * 0.7
    avg_val = counts_array.mean() * 0.15
    avg_test = counts_array.mean() * 0.15

    print(f"  Average samples per split (70/15/15):")
    print(f"    Train: {avg_train:.0f} samples/class")
    print(f"    Val:   {avg_val:.0f} samples/class")
    print(f"    Test:  {avg_test:.0f} samples/class")

    # Recommendations
    print(f"\n💡 Recommendations:")
    if total_classes < 10:
        print("  ⚠️ Very few classes - might want to adjust thresholds")
    elif total_classes < 30:
        print("  ✅ Moderate number of classes - good for focused classification")
    else:
        print("  ✅ Good number of classes for robust classification")

    if avg_train < 50:
        print("  ⚠️ Low training samples per class - consider data augmentation")
    elif avg_train < 200:
        print("  ✅ Moderate training samples - data augmentation recommended")
    else:
        print("  ✅ Good training samples per class")

    if total_images < 5000:
        print("  ⚠️ Small dataset - use strong augmentation and transfer learning")
    elif total_images < 20000:
        print("  ✅ Medium dataset - good for CNN with transfer learning")
    else:
        print("  ✅ Large dataset - excellent for deep learning")

    # Save analysis
    analysis = {
        'total_classes': total_classes,
        'total_images': total_images,
        'mean_images_per_class': float(counts_array.mean()),
        'median_images_per_class': float(np.median(counts_array)),
        'std_images_per_class': float(counts_array.std()),
        'min_images_per_class': int(counts_array.min()),
        'max_images_per_class': int(counts_array.max()),
        'balance_ratio': float(balance_ratio),
        'class_distribution': {name: count for name, count in sorted_counts}
    }

    return analysis


def calculate_optimal_thresholds(class_counts: dict, exclude_mega_classes: list = None):
    """
    Calculate statistically optimal thresholds for class balancing.

    Uses a two-phase approach:
    1. Remove mega-outliers (classes that dominate the dataset)
    2. Calculate thresholds based on remaining distribution
    """
    if exclude_mega_classes is None:
        exclude_mega_classes = []

    print("\n📊 CALCULATING OPTIMAL THRESHOLDS")
    print("=" * 70)

    # Get counts array
    all_counts = np.array(list(class_counts.values()))
    all_names = list(class_counts.keys())

    # Calculate overall statistics
    total_images = all_counts.sum()
    mean_count = all_counts.mean()
    median_count = np.median(all_counts)
    std_count = all_counts.std()

    print(f"\n📈 Overall Statistics (before filtering):")
    print(f"  Total images: {total_images:,}")
    print(f"  Mean: {mean_count:.0f}")
    print(f"  Median: {median_count:.0f}")
    print(f"  Std Dev: {std_count:.0f}")

    # Identify mega-outliers (classes that are > 99th percentile)
    p99 = np.percentile(all_counts, 99)
    mega_outliers = [(name, class_counts[name]) for name in all_names
                     if class_counts[name] > p99]

    print(f"\n🚨 Mega-outliers (>99th percentile = {p99:.0f} images):")
    for name, count in sorted(mega_outliers, key=lambda x: x[1], reverse=True):
        pct = (count / total_images) * 100
        print(f"  {name:40s} {count:7,} images ({pct:5.1f}% of dataset)")

    # Suggest excluding these mega-outliers
    suggested_excludes = [name for name, count in mega_outliers]
    if not exclude_mega_classes:
        exclude_mega_classes = suggested_excludes

    print(f"\n💡 Suggested exclusions: {exclude_mega_classes}")

    # Filter out mega-outliers for threshold calculation
    filtered_counts = np.array([count for name, count in class_counts.items()
                                if name not in exclude_mega_classes])

    if len(filtered_counts) == 0:
        print("❌ ERROR: All classes would be excluded!")
        return None

    # Calculate statistics on filtered data
    filtered_mean = filtered_counts.mean()
    filtered_median = np.median(filtered_counts)
    filtered_std = filtered_counts.std()

    print(f"\n📊 Statistics (after removing mega-outliers):")
    print(f"  Remaining classes: {len(filtered_counts)}")
    print(f"  Mean: {filtered_mean:.0f}")
    print(f"  Median: {filtered_median:.0f}")
    print(f"  Std Dev: {filtered_std:.0f}")

    # Calculate percentiles
    p25 = np.percentile(filtered_counts, 25)
    p50 = np.percentile(filtered_counts, 50)
    p75 = np.percentile(filtered_counts, 75)
    p90 = np.percentile(filtered_counts, 90)
    p95 = np.percentile(filtered_counts, 95)

    print(f"\n📊 Percentiles (filtered data):")
    print(f"  25th: {p25:.0f}")
    print(f"  50th (median): {p50:.0f}")
    print(f"  75th: {p75:.0f}")
    print(f"  90th: {p90:.0f}")
    print(f"  95th: {p95:.0f}")

    # Calculate IQR (Interquartile Range) for outlier detection
    q1 = np.percentile(filtered_counts, 25)
    q3 = np.percentile(filtered_counts, 75)
    iqr = q3 - q1

    print(f"\n📐 IQR Analysis:")
    print(f"  Q1 (25th): {q1:.0f}")
    print(f"  Q3 (75th): {q3:.0f}")
    print(f"  IQR: {iqr:.0f}")

    # Recommend thresholds based on different strategies
    print(f"\n🎯 THRESHOLD RECOMMENDATIONS:")

    # Strategy 1: Conservative (good balance, smaller dataset)
    min_conservative = max(20, p25)  # At least 20, or 25th percentile
    max_conservative = p75
    print(f"\n  1️⃣ CONSERVATIVE (Best Balance):")
    print(f"     Min: {min_conservative:.0f} (25th percentile)")
    print(f"     Max: {max_conservative:.0f} (75th percentile)")
    print(f"     → Good for: Maximum balance, smaller but high-quality dataset")

    # Strategy 2: Moderate (balanced approach)
    min_moderate = max(15, filtered_median * 0.5)
    max_moderate = p90
    print(f"\n  2️⃣ MODERATE (Recommended ⭐):")
    print(f"     Min: {min_moderate:.0f} (50% of median)")
    print(f"     Max: {max_moderate:.0f} (90th percentile)")
    print(f"     → Good for: Balance between data quantity and quality")

    # Strategy 3: Liberal (more data, some imbalance OK)
    min_liberal = max(10, filtered_median * 0.3)
    max_liberal = p95
    print(f"\n  3️⃣ LIBERAL (More Data):")
    print(f"     Min: {min_liberal:.0f} (30% of median)")
    print(f"     Max: {max_liberal:.0f} (95th percentile)")
    print(f"     → Good for: Keeping more data, accepting some imbalance")

    # Strategy 4: Target Balance Ratio (10x-20x max/min ratio)
    target_ratio_10x = filtered_median * 10
    target_ratio_20x = filtered_median * 20
    print(f"\n  4️⃣ TARGET BALANCE RATIO:")
    print(f"     Min: {filtered_median:.0f} (median)")
    print(f"     Max (10x): {target_ratio_10x:.0f} (10× median)")
    print(f"     Max (20x): {target_ratio_20x:.0f} (20× median)")
    print(f"     → Good for: Achieving specific balance ratio")

    # Return the MODERATE strategy as default
    return {
        'min_threshold': int(min_moderate),
        'max_threshold': int(max_moderate),
        'exclude_classes': exclude_mega_classes,
        'alternatives': {
            'conservative': {'min': int(min_conservative), 'max': int(max_conservative)},
            'moderate': {'min': int(min_moderate), 'max': int(max_moderate)},
            'liberal': {'min': int(min_liberal), 'max': int(max_liberal)},
            'balanced_10x': {'min': int(filtered_median), 'max': int(target_ratio_10x)},
            'balanced_20x': {'min': int(filtered_median), 'max': int(target_ratio_20x)}
        }
    }


def main():
    """Main preprocessing pipeline."""
    # Paths
    data_dir = Path("2014")
    unbalanced_dir = Path("unbalanced")

    if not data_dir.exists():
        print(f"❌ ERROR: Data directory {data_dir} not found!")
        return

    # Step 1: Analyze original dataset
    print("\n🔍 STEP 1: Analyzing Original Dataset")
    original_counts, sorted_counts = analyze_dataset(data_dir)

    # Step 2: Calculate optimal thresholds statistically
    print("\n🎯 STEP 2: Calculating Optimal Thresholds")

    thresholds = calculate_optimal_thresholds(original_counts)

    if thresholds is None:
        print("❌ Failed to calculate thresholds")
        return

    # Use the moderate strategy by default
    MIN_THRESHOLD = thresholds['min_threshold']
    MAX_THRESHOLD = thresholds['max_threshold']
    EXCLUDE_CLASSES = thresholds['exclude_classes']

    print(f"\n✅ SELECTED STRATEGY: MODERATE")
    print(f"  Min threshold: {MIN_THRESHOLD}")
    print(f"  Max threshold: {MAX_THRESHOLD}")
    print(f"  Excluded classes: {EXCLUDE_CLASSES}")

    # Option to choose different strategy
    print(f"\n💡 Want a different strategy?")
    print(f"   Edit the script and choose: conservative, moderate, liberal, balanced_10x, or balanced_20x")

    # Ask for confirmation
    print("\n⚠️ WARNING: This will move files! Make sure you have a backup.")
    response = input("Continue? (yes/no): ").strip().lower()

    if response != 'yes':
        print("❌ Aborted by user")
        return

    # Step 3: Move unbalanced classes
    print("\n🔄 STEP 3: Moving Unbalanced Classes")
    kept, moved_small, moved_large, moved_excluded = move_unbalanced_classes(
        data_dir,
        unbalanced_dir,
        MIN_THRESHOLD,
        MAX_THRESHOLD,
        EXCLUDE_CLASSES
    )

    # Step 4: Analyze cleaned dataset
    print("\n✨ STEP 4: Analyzing Cleaned Dataset")
    analysis = analyze_cleaned_dataset(data_dir)

    if analysis:
        # Save analysis report
        report_path = Path("data_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Analysis report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Directories:")
    print(f"  Balanced data:   {data_dir}")
    print(f"  Unbalanced data: {unbalanced_dir}")
    print(f"\nNext step: Train your model using the balanced dataset!")


if __name__ == "__main__":
    main()
