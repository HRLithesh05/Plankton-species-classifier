"""
=============================================================================
DATASET PREPROCESSING & CLEANING SCRIPT
=============================================================================

Run this LOCALLY before uploading to Colab to:
1. Remove corrupted/unreadable images
2. Filter out classes with too few samples
3. Resize images to 256x256 (faster upload, still good quality)
4. Remove duplicate images
5. Create a clean, smaller dataset ready for training

Usage:
    python preprocess_dataset.py

Output:
    Creates '2014_clean' folder with preprocessed dataset
    Zip it and upload to Colab!

"""

import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Input/Output
    'input_dir': Path('2014'),
    'output_dir': Path('2014_clean'),

    # Image settings
    'target_size': 256,  # Resize to 256x256 (saves space, good for 224 training)
    'quality': 95,       # JPEG quality (95 = high quality, smaller files)
    'output_format': 'JPEG',  # JPEG is smaller than PNG

    # Class filtering
    'min_samples_per_class': 15,   # Remove classes with fewer samples
    'max_samples_per_class': 2500, # Cap classes with too many samples

    # Classes to exclude (too noisy/generic)
    'exclude_classes': [
        'mix',
        'detritus',
        'mix_elongated',
        'bad',
        'badfocus',
        'unknown',
    ],

    # Processing
    'num_workers': 8,  # Parallel processing threads
    'remove_duplicates': True,
}

# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================

def get_image_hash(image_path: str) -> str:
    """Get hash of image content for duplicate detection."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB and resize for consistent hashing
            img = img.convert('RGB').resize((64, 64))
            return hashlib.md5(img.tobytes()).hexdigest()
    except:
        return None

def validate_image(image_path: str) -> bool:
    """Check if image is valid and readable."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open to actually load the image data
        with Image.open(image_path) as img:
            img.load()
            # Check minimum size
            if img.size[0] < 32 or img.size[1] < 32:
                return False
        return True
    except:
        return False

def process_image(args) -> dict:
    """Process a single image: validate, resize, save."""
    input_path, output_path, target_size, quality, output_format = args

    result = {
        'input': input_path,
        'output': output_path,
        'status': 'success',
        'error': None,
        'original_size': 0,
        'new_size': 0,
    }

    try:
        # Validate
        if not validate_image(input_path):
            result['status'] = 'invalid'
            return result

        # Load image
        with Image.open(input_path) as img:
            # Convert to RGB (removes alpha channel, ensures compatibility)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            result['original_size'] = os.path.getsize(input_path)

            # Resize maintaining aspect ratio, then center crop
            width, height = img.size
            scale = target_size / min(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Center crop to exact size
            left = (new_width - target_size) // 2
            top = (new_height - target_size) // 2
            img = img.crop((left, top, left + target_size, top + target_size))

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format == 'JPEG':
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
            else:
                img.save(output_path, output_format, optimize=True)

            result['new_size'] = os.path.getsize(output_path)

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result

# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def preprocess_dataset(config: dict):
    """Main preprocessing function."""

    input_dir = config['input_dir']
    output_dir = config['output_dir']

    print("=" * 60)
    print("DATASET PREPROCESSING & CLEANING")
    print("=" * 60)

    # Check input directory
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # Clean output directory
    if output_dir.exists():
        print(f"\nRemoving existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # ==========================================================================
    # STEP 1: Scan all classes and images
    # ==========================================================================
    print(f"\n[STEP 1/5] Scanning dataset...")

    class_data = {}
    total_images = 0

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # Skip excluded classes
        if class_name.lower() in [c.lower() for c in config['exclude_classes']]:
            print(f"  Excluding class: {class_name}")
            continue

        # Find all images
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp', '*.BMP']:
            images.extend(list(class_dir.glob(ext)))

        if len(images) < config['min_samples_per_class']:
            print(f"  Skipping class (too few samples): {class_name} ({len(images)} images)")
            continue

        class_data[class_name] = images
        total_images += len(images)

    print(f"\nFound {total_images} images in {len(class_data)} classes")

    # ==========================================================================
    # STEP 2: Remove duplicates (optional)
    # ==========================================================================
    if config['remove_duplicates']:
        print(f"\n[STEP 2/5] Removing duplicate images...")

        duplicates_removed = 0

        for class_name, images in tqdm(class_data.items(), desc="Scanning for duplicates"):
            # Hash all images in this class
            hashes = {}
            unique_images = []

            for img_path in images:
                img_hash = get_image_hash(str(img_path))
                if img_hash and img_hash not in hashes:
                    hashes[img_hash] = img_path
                    unique_images.append(img_path)
                elif img_hash:
                    duplicates_removed += 1

            class_data[class_name] = unique_images

        print(f"Removed {duplicates_removed} duplicate images")
    else:
        print(f"\n[STEP 2/5] Skipping duplicate removal")

    # ==========================================================================
    # STEP 3: Balance classes (cap max samples)
    # ==========================================================================
    print(f"\n[STEP 3/5] Balancing classes...")

    import random
    random.seed(42)

    capped_count = 0
    for class_name, images in class_data.items():
        if len(images) > config['max_samples_per_class']:
            original = len(images)
            class_data[class_name] = random.sample(images, config['max_samples_per_class'])
            capped_count += original - config['max_samples_per_class']
            print(f"  Capped {class_name}: {original} → {config['max_samples_per_class']}")

    if capped_count > 0:
        print(f"Removed {capped_count} images from oversized classes")

    # ==========================================================================
    # STEP 4: Process and save images
    # ==========================================================================
    print(f"\n[STEP 4/5] Processing and resizing images...")

    # Prepare processing tasks
    tasks = []
    for class_name, images in class_data.items():
        for img_path in images:
            # Determine output path
            if config['output_format'] == 'JPEG':
                output_filename = img_path.stem + '.jpg'
            else:
                output_filename = img_path.name

            output_path = output_dir / class_name / output_filename

            tasks.append((
                str(img_path),
                output_path,
                config['target_size'],
                config['quality'],
                config['output_format']
            ))

    # Process in parallel
    results = {
        'success': 0,
        'invalid': 0,
        'error': 0,
        'original_size': 0,
        'new_size': 0,
    }

    with ThreadPoolExecutor(max_workers=config['num_workers']) as executor:
        futures = [executor.submit(process_image, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results[result['status']] += 1
            results['original_size'] += result.get('original_size', 0)
            results['new_size'] += result.get('new_size', 0)

    print(f"\nProcessing results:")
    print(f"  Success: {results['success']}")
    print(f"  Invalid/Corrupted: {results['invalid']}")
    print(f"  Errors: {results['error']}")

    # Size reduction
    if results['original_size'] > 0:
        reduction = (1 - results['new_size'] / results['original_size']) * 100
        original_mb = results['original_size'] / (1024 * 1024)
        new_mb = results['new_size'] / (1024 * 1024)
        print(f"\nSize: {original_mb:.1f} MB → {new_mb:.1f} MB ({reduction:.1f}% reduction)")

    # ==========================================================================
    # STEP 5: Generate summary
    # ==========================================================================
    print(f"\n[STEP 5/5] Generating summary...")

    # Count final dataset
    final_stats = {}
    total_final = 0

    for class_dir in sorted(output_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.*')))
            final_stats[class_dir.name] = count
            total_final += count

    # Save stats
    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'total_images': total_final,
            'num_classes': len(final_stats),
            'class_counts': final_stats,
            'config': {
                'target_size': config['target_size'],
                'min_samples': config['min_samples_per_class'],
                'max_samples': config['max_samples_per_class'],
            }
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total images: {total_final}")
    print(f"Total classes: {len(final_stats)}")
    print(f"Images per class: {min(final_stats.values())} - {max(final_stats.values())}")

    # Calculate folder size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    print(f"Total size: {total_size / (1024*1024):.1f} MB")

    print(f"\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"1. Zip the folder:  Right-click '{output_dir}' → Send to → Compressed folder")
    print(f"2. Upload to Colab: Use the zip upload method")
    print(f"3. In Colab, unzip: !unzip -q 2014_clean.zip")
    print(f"4. Run training:    !python train_colab.py --data_path '2014_clean'")

    return final_stats

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\nPlankton Dataset Preprocessing Tool")
    print("-" * 40)

    # Check for PIL
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed. Run: pip install pillow")
        sys.exit(1)

    # Run preprocessing
    stats = preprocess_dataset(CONFIG)

    print("\nDone!")
