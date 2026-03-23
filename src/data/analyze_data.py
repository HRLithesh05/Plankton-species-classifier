"""
Quick Data Analysis Script - See statistics WITHOUT moving files.
Run this first to see threshold recommendations.
"""

import numpy as np
from pathlib import Path
from collections import Counter


def analyze_dataset(data_dir: Path):
    """Analyze class distribution."""
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    class_counts = {}

    for class_dir in data_dir.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            imgs = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            class_counts[class_dir.name] = len(imgs)

    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    total_images = sum(class_counts.values())
    counts_array = np.array(list(class_counts.values()))

    print(f"\n📊 Overall Statistics:")
    print(f"  Total classes: {len(class_counts)}")
    print(f"  Total images: {total_images:,}")
    print(f"  Mean: {counts_array.mean():.0f}")
    print(f"  Median: {np.median(counts_array):.0f}")
    print(f"  Std Dev: {counts_array.std():.0f}")
    print(f"  Min: {counts_array.min()}")
    print(f"  Max: {counts_array.max()}")

    # Percentiles
    print(f"\n📈 Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(counts_array, p):.0f}")

    # Show extremes
    print(f"\n🔻 Bottom 10 classes:")
    for name, count in sorted_counts[:10]:
        print(f"  {name:40s} {count:6,}")

    print(f"\n🔺 Top 10 classes:")
    for name, count in sorted_counts[-10:]:
        pct = (count / total_images) * 100
        print(f"  {name:40s} {count:7,} ({pct:5.1f}%)")

    return class_counts


def calculate_recommendations(class_counts: dict):
    """Calculate threshold recommendations."""
    all_counts = np.array(list(class_counts.values()))
    all_names = list(class_counts.keys())
    total_images = all_counts.sum()

    # Find mega-outliers (>99th percentile)
    p99 = np.percentile(all_counts, 99)
    mega_outliers = [(name, class_counts[name]) for name in all_names
                     if class_counts[name] > p99]

    print(f"\n🚨 Mega-outliers (>99th percentile = {p99:.0f}):")
    for name, count in sorted(mega_outliers, key=lambda x: x[1], reverse=True):
        pct = (count / total_images) * 100
        print(f"  {name:40s} {count:7,} ({pct:5.1f}% of dataset)")

    # Filter out mega-outliers
    exclude_names = [name for name, _ in mega_outliers]
    filtered_counts = np.array([count for name, count in class_counts.items()
                                if name not in exclude_names])

    if len(filtered_counts) == 0:
        print("❌ All classes are outliers!")
        return

    # Calculate stats on filtered data
    median = np.median(filtered_counts)
    p25 = np.percentile(filtered_counts, 25)
    p75 = np.percentile(filtered_counts, 75)
    p90 = np.percentile(filtered_counts, 90)
    p95 = np.percentile(filtered_counts, 95)

    print(f"\n📊 Statistics (after removing mega-outliers):")
    print(f"  Remaining classes: {len(filtered_counts)}")
    print(f"  Mean: {filtered_counts.mean():.0f}")
    print(f"  Median: {median:.0f}")

    print(f"\n🎯 RECOMMENDED THRESHOLDS:")

    print(f"\n  1️⃣ CONSERVATIVE (Max Balance):")
    print(f"     Min: {max(20, p25):.0f}")
    print(f"     Max: {p75:.0f}")
    print(f"     Exclude: {exclude_names}")

    print(f"\n  2️⃣ MODERATE (Recommended ⭐):")
    print(f"     Min: {max(15, median * 0.5):.0f}")
    print(f"     Max: {p90:.0f}")
    print(f"     Exclude: {exclude_names}")

    print(f"\n  3️⃣ LIBERAL (More Data):")
    print(f"     Min: {max(10, median * 0.3):.0f}")
    print(f"     Max: {p95:.0f}")
    print(f"     Exclude: {exclude_names}")

    # Estimate resulting dataset size
    print(f"\n📦 Estimated Dataset Sizes:")
    for strategy, (min_t, max_t) in [
        ("Conservative", (max(20, p25), p75)),
        ("Moderate", (max(15, median * 0.5), p90)),
        ("Liberal", (max(10, median * 0.3), p95))
    ]:
        kept_classes = sum(1 for c in filtered_counts if min_t <= c <= max_t)
        est_images = sum(min(c, max_t) for c in filtered_counts if c >= min_t)
        print(f"  {strategy:15s} ~{kept_classes:2d} classes, ~{est_images:6,.0f} images")


if __name__ == "__main__":
    data_dir = Path("2014")

    if not data_dir.exists():
        print("❌ Data directory '2014' not found!")
    else:
        class_counts = analyze_dataset(data_dir)
        calculate_recommendations(class_counts)

        print("\n" + "=" * 70)
        print("✅ Analysis complete!")
        print("   Run 'python preprocess_data.py' to apply preprocessing")
        print("=" * 70)
