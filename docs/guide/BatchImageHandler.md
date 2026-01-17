# BatchImageHandler Usage Guide

## Table of Contents

- [Overview](#overview)
  - [Key Features](#key-features)
  - [Design Philosophy](#design-philosophy)
- [Quick Start](#quick-start)
- [Common Patterns](#common-patterns)
  - [Pattern 1: Loading Images from Directories](#pattern-1-loading-images-from-directories)
  - [Pattern 2: Filtering and Cleaning Datasets](#pattern-2-filtering-and-cleaning-datasets)
  - [Pattern 3: Parallel Processing with Custom Functions](#pattern-3-parallel-processing-with-custom-functions)
  - [Pattern 4: Memory-Efficient Chunk Processing](#pattern-4-memory-efficient-chunk-processing)
  - [Pattern 5: Duplicate Detection and Removal](#pattern-5-duplicate-detection-and-removal)
- [Real-World Workflows](#real-world-workflows)
  - [Workflow 1: Preparing ML Training Dataset](#workflow-1-preparing-ml-training-dataset)
  - [Workflow 2: Cleaning and Deduplicating Photo Library](#workflow-2-cleaning-and-deduplicating-photo-library)
  - [Workflow 3: Batch Thumbnail Generation](#workflow-3-batch-thumbnail-generation)
  - [Workflow 4: Dataset Quality Control and Uniformity Verification](#workflow-4-dataset-quality-control-and-uniformity-verification)
  - [Workflow 5: Creating ML Dataset Splits](#workflow-5-creating-ml-dataset-splits)
  - [Workflow 6: Processing Large Image Collections in Chunks](#workflow-6-processing-large-image-collections-in-chunks)
- [Best Practices](#best-practices)
  - [Performance](#performance)
  - [Memory Management](#memory-management)
  - [Common Pitfalls](#common-pitfalls)
  - [Limitations](#limitations)
- [Integration Examples](#integration-examples)
  - [PyTorch Integration](#pytorch-integration)
  - [NumPy Integration](#numpy-integration)
- [When to Use This Class](#when-to-use-this-class)
  - [Ideal Use Cases](#ideal-use-cases)
  - [When to Look Elsewhere](#when-to-look-elsewhere)
  - [Comparison with Alternatives](#comparison-with-alternatives)
- [See Also](#see-also)

---

## Overview

`BatchImageHandler` is a high-performance batch processing class designed for efficiently handling collections of images. It provides parallel execution, intelligent filtering, transformations, and dataset preparation capabilities for machine learning workflows.

The class manages collections of `ImageContext` instances and provides chainable methods for filtering, transforming, analyzing, and exporting image datasets. It's optimized for real-world scenarios like preparing ML training data, cleaning photo libraries, and processing large image collections.

**Key Capabilities:**
- Load thousands of images from directories with pattern matching
- Filter by dimensions, aspect ratio, file size, validity, and duplicates
- Apply transformations in parallel (resize, adjust, grayscale, etc.)
- Analyze batch statistics and detect outliers
- Split datasets for ML training (train/val/test)
- Export to PyTorch, NumPy, or other ML frameworks
- Process images in memory-efficient chunks

This class is essential for anyone working with large image collections, whether for machine learning dataset preparation, photo management, or automated image processing pipelines.

### Key Features

- **Parallel Processing**: Automatic multi-threaded execution for CPU-intensive operations with configurable worker counts
- **Intelligent Filtering**: Remove invalid images, filter by size/aspect ratio/file size, detect and remove duplicates using perceptual hashing
- **Chainable Transformations**: Fluent API for applying sequences of operations (resize, adjust, crop, etc.) with method chaining
- **ML Dataset Conversion**: Direct export to PyTorch tensors, NumPy arrays with train/val/test splitting and uniformity verification
- **Memory Efficiency**: Chunk-based processing for handling datasets larger than RAM, with automatic unloading
- **Error Resilience**: Comprehensive error tracking and recovery—failed images don't stop batch processing

### Design Philosophy

`BatchImageHandler` follows a **fluent, chainable API** design where most methods return `self`, allowing you to build complex processing pipelines in a readable, sequential manner. This design mirrors popular data processing libraries like Pandas.

**Core Principles:**

1. **Lazy Loading by Default**: Images are not loaded into memory until needed, minimizing memory footprint
2. **Fail-Safe Processing**: Individual image failures are logged but don't halt batch operations
3. **Parallel by Default**: Most operations use parallel processing automatically, with sequential fallbacks available
4. **Immutable Sources**: Original files are never modified unless explicitly saved
5. **Method Chaining**: All transformation methods return `self` for fluent composition

**Memory Management Strategy:**
- Images are stored as `ImageContext` instances that can be loaded/unloaded independently
- Parallel operations load only what's needed per worker
- Chunk processing allows handling datasets larger than available RAM
- Context manager support ensures automatic cleanup

**Understanding these principles will help you:**
- Design efficient processing pipelines that scale to thousands of images
- Avoid memory exhaustion when working with large datasets
- Debug failures without losing entire batch results
- Write cleaner, more maintainable image processing code

---

## Quick Start

### Installation

```bash
# Install the image_toolkit package
pip install image_toolkit

# For duplicate detection (optional)
pip install imagehash

# For PyTorch integration (optional)
pip install torch
```

### Minimal Example

```python
from image_toolkit.batch_handler import BatchImageHandler

# Load all JPGs from a directory
batch = BatchImageHandler.from_directory("photos/", "*.jpg")
print(f"Loaded {len(batch)} images")
```

### Common Workflow

```python
# Load, filter, resize, and save
batch = (BatchImageHandler.from_directory("photos/", "*.jpg")
         .filter_valid()
         .filter_by_size(min_width=800)
         .resize(width=1024)
         .save("output/", prefix="processed_"))

print(f"Processed {len(batch)} images")
```

### Getting Results

```python
# Get batch statistics
stats = batch.get_batch_stats()
print(f"Average size: {stats['width']['mean']}x{stats['height']['mean']}")

# Convert to PyTorch dataset
batch.resize(width=224, height=224)
dataset = batch.to_dataset(format='torch', normalized=True)
print(f"Dataset shape: {dataset.shape}")  # [N, 3, 224, 224]
```

---

## Common Patterns

### Pattern 1: Loading Images from Directories

**When to use:** You have images organized in directories and need to load them based on file patterns.

```python
from image_toolkit.batch_handler import BatchImageHandler

# Load from single directory with pattern
batch = BatchImageHandler.from_directory("photos/", "*.jpg")

# Auto-expand pattern (both are equivalent)
batch = BatchImageHandler.from_directory("photos/", "jpg")

# Recursive glob pattern
batch = BatchImageHandler.from_glob("photos/**/*.jpg")

# Load specific file list
from pathlib import Path
paths = [Path("img1.jpg"), Path("img2.jpg")]
batch = BatchImageHandler(paths)
```

**Key points:**
- `from_directory()` automatically filters for common image extensions
- Pattern matching supports wildcards (`*.jpg`, `*.png`, etc.)
- `from_glob()` enables recursive directory traversal with `**`
- Empty results print a warning but don't raise exceptions

---

### Pattern 2: Filtering and Cleaning Datasets

**When to use:** You need to remove corrupted, invalid, or unwanted images from a batch.

```python
# Remove corrupted and invalid images
batch = (BatchImageHandler.from_directory("raw_photos/")
         .filter_valid(parallel=True))

# Filter by image dimensions
batch.filter_by_size(min_width=800, max_width=4000, min_height=600)

# Filter by aspect ratio (landscape images only)
batch.filter_by_aspect_ratio(min_ratio=1.3, max_ratio=2.0)

# Filter by file size on disk
batch.filter_by_file_size(min_size='100KB', max_size='10MB')

# Chain multiple filters
batch = (BatchImageHandler.from_directory("photos/")
         .filter_valid()
         .filter_by_size(min_width=1024)
         .filter_by_file_size(min_size='50KB')
         .filter_by_aspect_ratio(min_ratio=0.75, max_ratio=1.5))
```

**Key points:**
- Filtering is non-destructive (original files unchanged)
- Filters can be chained for complex criteria
- Failed images are logged in `batch.get_errors()`
- File size filtering doesn't load images into memory (fast)

---

### Pattern 3: Parallel Processing with Custom Functions

**When to use:** You need to apply custom transformations that aren't built-in methods.

```python
from image_toolkit.handler import ImageHandler

def custom_pipeline(img: ImageHandler) -> ImageHandler:
    """Custom processing function"""
    img.resize_aspect(width=800)
    img.adjust(brightness=1.15, contrast=1.05)
    if img.is_grayscale_content():
        img.to_grayscale()
    return img

# Apply custom function to all images in parallel
batch = BatchImageHandler.from_directory("photos/")
batch.map(custom_pipeline, parallel=True, workers=8)

# Process and save
batch.save("output/")
```

**Key points:**
- Custom functions receive `ImageHandler` instances
- Must return `ImageHandler` for chaining to work
- `workers` parameter controls thread pool size
- Set `parallel=False` for debugging or sequential dependencies

---

### Pattern 4: Memory-Efficient Chunk Processing

**When to use:** Processing datasets larger than available RAM.

```python
def heavy_transform(img):
    """Memory-intensive transformation"""
    img.resize_aspect(width=2048)
    img.adjust(brightness=1.2)
    return img

# Process 10,000 images in chunks of 100
batch = BatchImageHandler.from_directory("huge_dataset/")
batch.process_in_chunks(
    chunk_size=100,
    func=heavy_transform,
    parallel=True
)

# Each chunk is unloaded after processing
batch.save("output/", prefix="processed_")
```

**Key points:**
- Chunks are processed sequentially, images within chunks in parallel
- Each chunk is automatically unloaded after processing
- Ideal for datasets with 1000+ high-resolution images
- Errors are accumulated across all chunks

---

### Pattern 5: Duplicate Detection and Removal

**When to use:** Removing duplicate or near-duplicate images from a collection.

```python
# Remove duplicates using perceptual hashing
batch = (BatchImageHandler.from_directory("photos/")
         .filter_duplicates(
             hash_method='dhash',
             threshold=5,
             keep='largest',
             parallel=True
         ))

# Analyze duplicates without removing
analysis = batch.analyze_duplicates(hash_method='dhash', threshold=5)
print(f"Found {analysis['stats']['num_duplicate_groups']} duplicate groups")

for group in analysis['duplicate_groups']:
    print(f"Duplicate group with {len(group)} images:")
    for ctx in group:
        print(f"  - {ctx.path.name}")
```

**Key points:**
- Supports multiple hashing algorithms (dhash, phash, ahash, whash)
- `threshold` controls matching strictness (lower = stricter)
- `keep` strategies: 'first', 'last', 'largest', 'smallest'
- Parallel processing significantly speeds up hash computation

---

## Real-World Workflows

### Workflow 1: Preparing ML Training Dataset

**Scenario:** You have a folder of raw images and need to prepare a clean, uniform dataset for training a neural network.

**Solution:**

```python
from image_toolkit.batch_handler import BatchImageHandler
from pathlib import Path

# Step 1: Load and filter valid images
batch = (BatchImageHandler.from_directory("raw_dataset/", "*.jpg")
         .filter_valid(parallel=True)
         .filter_by_size(min_width=512, min_height=512))

print(f"After filtering: {len(batch)} images")

# Step 2: Remove duplicates to avoid data leakage
batch.filter_duplicates(hash_method='dhash', threshold=5, keep='largest')
print(f"After deduplication: {len(batch)} images")

# Step 3: Verify dataset uniformity
report = batch.verify_uniformity(
    check_size=False,  # Sizes will vary
    check_format=True,
    check_mode=True
)

if not report['uniform']:
    print("⚠️ Dataset has non-uniform properties:")
    print(f"  Formats: {report['formats']}")
    print(f"  Modes: {report['modes']}")
    # Convert all to RGB if needed
    if len(report['modes']) > 1:
        batch.apply_transform('to_rgba', parallel=True)

# Step 4: Resize to uniform dimensions
batch.resize(width=224, height=224)

# Step 5: Split into train/val/test sets
splits = batch.split_dataset(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True,
    random_seed=42
)

# Step 6: Save splits to separate directories
splits['train'].save("dataset/train/", quality=95)
splits['val'].save("dataset/val/", quality=95)
splits['test'].save("dataset/test/", quality=95)

# Step 7: Convert to PyTorch tensors
train_tensor = splits['train'].to_dataset(format='torch', normalized=True)
val_tensor = splits['val'].to_dataset(format='torch', normalized=True)

print(f"Training set: {train_tensor.shape}")
print(f"Validation set: {val_tensor.shape}")
```

**Explanation:**
- Filtering removes corrupted and undersized images that would hurt training
- Duplicate removal prevents data leakage between train/val/test sets
- Uniform dimensions are required for batching in neural networks
- Reproducible splitting (via `random_seed`) ensures consistent experiments
- Normalized tensors are ready for direct use in PyTorch dataloaders

---

### Workflow 2: Cleaning and Deduplicating Photo Library

**Scenario:** You have thousands of photos accumulated over years with duplicates, thumbnails, and corrupted files that need cleaning.

**Solution:**

```python
from image_toolkit.batch_handler import BatchImageHandler

# Load all images recursively
batch = BatchImageHandler.from_glob("~/Photos/**/*.{jpg,png,heic}")
print(f"Found {len(batch)} total images")

# Remove corrupted images
original_count = len(batch)
batch.filter_valid(parallel=True)
print(f"Removed {original_count - len(batch)} corrupted images")

# Remove tiny files (likely thumbnails or corrupted)
batch.filter_by_file_size(min_size='50KB')
print(f"Remaining: {len(batch)} images")

# Detect and analyze duplicates
analysis = batch.analyze_duplicates(
    hash_method='dhash',
    threshold=8,  # Higher threshold for edited versions
    parallel=True
)

print(f"\nDuplicate Analysis:")
print(f"  Total duplicate groups: {analysis['stats']['num_duplicate_groups']}")
print(f"  Total duplicates: {analysis['stats']['total_duplicates']}")
print(f"  Unique images: {analysis['stats']['num_unique_images']}")

# Review first duplicate group
if analysis['duplicate_groups']:
    print("\nFirst duplicate group:")
    for ctx in analysis['duplicate_groups'][0]:
        size_mb = ctx.path.stat().st_size / (1024*1024)
        print(f"  {ctx.path.name} - {size_mb:.2f}MB")

# Remove duplicates, keeping largest file
batch.filter_duplicates(
    hash_method='dhash',
    threshold=8,
    keep='largest',  # Keep highest quality version
    parallel=True
)

print(f"\nFinal clean library: {len(batch)} images")

# Check for any errors during processing
errors = batch.get_errors()
if errors:
    print(f"\n⚠️ {len(errors)} images failed processing:")
    for path, error in errors[:5]:  # Show first 5
        print(f"  {path.name}: {error}")
```

**Explanation:**
- Recursive glob pattern finds images in all subdirectories
- Small file size filter removes thumbnails and corrupted files
- Higher threshold (8) catches edited versions and resized copies
- Keeping 'largest' preserves highest quality among duplicates
- Error tracking ensures you don't lose track of problematic files

---

### Workflow 3: Batch Thumbnail Generation

**Scenario:** Generate consistent thumbnails for a web gallery with watermarks and metadata.

**Solution:**

```python
from image_toolkit.batch_handler import BatchImageHandler
from image_toolkit.handler import ImageHandler

def create_thumbnail(img: ImageHandler) -> ImageHandler:
    """Create web-optimized thumbnail with watermark"""
    # Resize to 300px width, maintaining aspect ratio
    img.resize_aspect(width=300)

    # Add subtle watermark
    img.draw_text(
        text="© 2024 MyGallery",
        position=(10, img.img.height - 20),
        font_size=12,
        color=(255, 255, 255, 128)  # Semi-transparent white
    )

    # Slight sharpening for web display
    img.adjust(contrast=1.05)

    return img

# Process all images
batch = (BatchImageHandler.from_directory("originals/", "*.jpg")
         .filter_valid()
         .filter_by_size(min_width=300))  # Skip images already smaller

# Apply thumbnail transformation in parallel
batch.map(create_thumbnail, parallel=True, workers=8)

# Save as JPEGs with web-optimized quality
batch.save(
    output_dir="thumbnails/",
    prefix="thumb_",
    quality=85,  # Balance quality vs file size
    overwrite=False
)

# Create grid visualization
grid = batch.create_grid(
    rows=4,
    cols=6,
    cell_size=(300, 300),
    padding=10
)
grid.save("gallery_preview.jpg")

print(f"Generated {len(batch)} thumbnails")
```

**Explanation:**
- Custom function allows complex per-image operations
- Watermarking protects images on public galleries
- Quality=85 is optimal for web (good quality, small files)
- Grid visualization creates a contact sheet preview
- `overwrite=False` prevents accidentally replacing existing thumbnails

---

### Workflow 4: Dataset Quality Control and Uniformity Verification

**Scenario:** Before training a machine learning model, verify that all images meet quality and uniformity requirements.

**Solution:**

```python
from image_toolkit.batch_handler import BatchImageHandler

# Load dataset
batch = BatchImageHandler.from_directory("ml_dataset/train/")
print(f"Loaded {len(batch)} images")

# Step 1: Basic quality checks
batch.filter_valid(parallel=True)
batch.filter_by_size(min_width=224, min_height=224)
batch.filter_by_file_size(min_size='10KB')  # Remove corrupted/empty files

# Step 2: Verify uniformity
report = batch.verify_uniformity(
    check_size=True,
    check_format=True,
    check_mode=True,
    check_aspect_ratio=False
)

print("\n=== Uniformity Report ===")
print(f"Uniform: {report['uniform']}")
print(f"Total images: {report['total_images']}")

if report['sizes']:
    print(f"\nSize distribution:")
    for size, count in list(report['sizes'].items())[:5]:
        print(f"  {size[0]}x{size[1]}: {count} images")

if report['formats']:
    print(f"\nFormat distribution:")
    for fmt, count in report['formats'].items():
        print(f"  {fmt}: {count} images")

if report['modes']:
    print(f"\nColor mode distribution:")
    for mode, count in report['modes'].items():
        print(f"  {mode}: {count} images")

# Step 3: Fix non-uniformity issues
if not report['uniform']:
    print("\n⚠️ Fixing uniformity issues...")

    # Convert all to RGB if mixed modes
    if len(report['modes']) > 1:
        print("Converting all images to RGB...")
        batch.apply_transform('to_rgba', parallel=True)

    # Resize all to most common size
    if len(report['sizes']) > 1 and report['checks']['size']['expected']:
        target_width, target_height = report['checks']['size']['expected']
        print(f"Resizing all to {target_width}x{target_height}...")
        batch.resize(width=target_width, height=target_height)

# Step 4: Statistical outlier detection
outliers = batch.detect_outliers(metric='size', threshold=3.0)
if outliers:
    print(f"\n⚠️ Found {len(outliers)} statistical outliers:")
    for path in outliers[:5]:
        print(f"  {path.name}")

# Step 5: Get comprehensive statistics
stats = batch.get_batch_stats()
print(f"\n=== Batch Statistics ===")
print(f"Width:  {stats['width']['mean']:.0f} ± {stats['width']['std']:.0f} px")
print(f"Height: {stats['height']['mean']:.0f} ± {stats['height']['std']:.0f} px")
print(f"Aspect: {stats['aspect_ratio']['mean']:.2f} ± {stats['aspect_ratio']['std']:.2f}")

# Step 6: Save cleaned dataset
batch.save("ml_dataset/train_cleaned/", quality=95)

print(f"\n✓ Quality control complete. {len(batch)} images ready for training.")
```

**Explanation:**
- Uniformity verification catches common dataset issues before training
- Automatic fixing converts mixed formats/modes to consistent state
- Statistical outlier detection finds unusual images that might be errors
- Comprehensive stats help understand dataset characteristics
- Cleaned dataset is saved separately to preserve originals

---

### Workflow 5: Creating ML Dataset Splits

**Scenario:** Split a single folder of labeled images into train/validation/test sets with reproducible randomization.

**Solution:**

```python
from image_toolkit.batch_handler import BatchImageHandler
import json

# Load all images
batch = (BatchImageHandler.from_directory("labeled_images/", "*.jpg")
         .filter_valid()
         .filter_duplicates(hash_method='dhash', threshold=5))

print(f"Total images: {len(batch)}")

# Create reproducible 70/15/15 split
splits = batch.split_dataset(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True,
    random_seed=42  # Ensures reproducibility
)

train_batch = splits['train']
val_batch = splits['val']
test_batch = splits['test']

print(f"Train: {len(train_batch)} images")
print(f"Val:   {len(val_batch)} images")
print(f"Test:  {len(test_batch)} images")

# Resize all splits to uniform size
for split_name, split_batch in splits.items():
    split_batch.resize(width=224, height=224)

# Save splits to separate directories
train_batch.save("dataset_splits/train/", quality=95)
val_batch.save("dataset_splits/val/", quality=95)
test_batch.save("dataset_splits/test/", quality=95)

# Export metadata for reproducibility
metadata = {
    'random_seed': 42,
    'train_count': len(train_batch),
    'val_count': len(val_batch),
    'test_count': len(test_batch),
    'total_count': len(batch),
    'ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15}
}

with open('dataset_splits/split_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Convert to PyTorch tensors for immediate use
train_tensor = train_batch.to_dataset(format='torch', normalized=True)
val_tensor = val_batch.to_dataset(format='torch', normalized=True)
test_tensor = test_batch.to_dataset(format='torch', normalized=True)

print(f"\nTensor shapes:")
print(f"  Train: {train_tensor.shape}")
print(f"  Val:   {val_tensor.shape}")
print(f"  Test:  {test_tensor.shape}")
```

**Explanation:**
- Deduplication before splitting prevents data leakage
- `random_seed=42` ensures identical splits across runs
- Each split is an independent `BatchImageHandler` instance
- Saving metadata allows reproducing exact splits later
- Direct tensor conversion is ready for PyTorch DataLoader

---

### Workflow 6: Processing Large Image Collections in Chunks

**Scenario:** Process 50,000 high-resolution images on a machine with limited RAM.

**Solution:**

```python
from image_toolkit.batch_handler import BatchImageHandler
from image_toolkit.handler import ImageHandler

def standardize_image(img: ImageHandler) -> ImageHandler:
    """Standardize images for archival"""
    # Resize large images
    width, height = img.img.size
    if width > 2048 or height > 2048:
        img.resize_aspect(width=2048)

    # Convert to standard RGB
    if img.img.mode != 'RGB':
        img.to_rgba()

    # Normalize brightness/contrast
    img.adjust(brightness=1.0, contrast=1.05)

    return img

# Load all images (paths only, not loaded into memory)
batch = BatchImageHandler.from_glob("archive/**/*.{jpg,png,tiff}")
print(f"Found {len(batch)} images to process")

# Set up progress tracking
def progress_callback(current, total):
    percent = (current / total) * 100
    print(f"Progress: {current}/{total} ({percent:.1f}%)")

batch.set_progress_callback(progress_callback)

# Process in chunks to avoid memory exhaustion
batch.process_in_chunks(
    chunk_size=200,  # Process 200 images at a time
    func=standardize_image,
    parallel=True
)

# Save processed images
batch.save(
    output_dir="archive_processed/",
    prefix="std_",
    quality=92,
    overwrite=False
)

# Check for errors
errors = batch.get_errors()
if errors:
    print(f"\n⚠️ {len(errors)} images failed:")
    with open('processing_errors.log', 'w') as f:
        for path, error in errors:
            f.write(f"{path}: {error}\n")
            print(f"  {path.name}: {error}")
else:
    print("\n✓ All images processed successfully!")

# Generate final statistics
stats = batch.get_batch_stats()
print(f"\nFinal Statistics:")
print(f"  Avg dimensions: {stats['width']['mean']:.0f}x{stats['height']['mean']:.0f}")
print(f"  Total processed: {len(batch)}")
```

**Explanation:**
- Loading paths only (without images) uses minimal memory
- Chunk size of 200 balances memory usage vs processing overhead
- Each chunk is automatically unloaded after processing
- Progress callback provides real-time feedback for long operations
- Error logging ensures failed images don't get lost
- This approach can handle datasets 10x larger than available RAM

---

## Best Practices

### Performance

**Optimization 1: Use parallel processing for CPU-bound operations**

```python
# SLOW: Sequential processing (takes minutes for 1000 images)
batch = BatchImageHandler.from_directory("photos/")
batch.resize(width=800, parallel=False)

# FAST: Parallel processing (takes seconds)
batch.resize(width=800, parallel=True, workers=8)
```

**Why it matters:** Parallel processing leverages multiple CPU cores. On an 8-core machine, you can expect 4-6x speedup for operations like resizing, filtering, and transformations.

---

**Optimization 2: Filter before transforming**

```python
# SLOW: Transform then filter (wastes CPU on images you'll discard)
batch = (BatchImageHandler.from_directory("photos/")
         .resize(width=1024, parallel=True)
         .filter_by_size(min_width=800))

# FAST: Filter then transform (only process images you'll keep)
batch = (BatchImageHandler.from_directory("photos/")
         .filter_by_size(min_width=800)
         .resize(width=1024, parallel=True))
```

**Why it matters:** Filtering is much faster than transformation. Removing unwanted images first avoids expensive resize operations on images you'll discard anyway.

---

**Optimization 3: Use chunk processing for large datasets**

```python
# SLOW: Load all 10,000 images into memory (may crash with OOM)
batch = BatchImageHandler.from_directory("huge_dataset/")
batch.map(heavy_transform, parallel=True)

# FAST: Process in chunks (uses constant memory)
batch.process_in_chunks(chunk_size=100, func=heavy_transform, parallel=True)
```

**Why it matters:** Chunk processing keeps memory usage constant regardless of dataset size. Essential for processing 1000+ high-resolution images.

---

### Memory Management

**⚠️ Important: Always unload images after batch operations**

```python
# BAD: Images stay loaded in memory after processing
batch = BatchImageHandler.from_directory("photos/")
batch.resize(width=800)
batch.save("output/")
# Memory still occupied here!

# GOOD: Explicitly unload to free memory
batch.unload()

# BETTER: Use context manager for automatic cleanup
with BatchImageHandler.from_directory("photos/") as batch:
    batch.resize(width=800).save("output/")
# Automatically unloaded after with block
```

---

**⚠️ Important: Use deep vs shallow copy appropriately**

```python
# BAD: Deep copy wastes memory when you just need to filter
original = BatchImageHandler.from_directory("photos/").resize(width=800)
copy1 = original.copy(deep=True)  # Duplicates all pixel data!
copy1.filter_by_size(min_width=600)

# GOOD: Shallow copy for filtering operations
copy2 = original.copy(deep=False)
copy2.filter_by_size(min_width=600)  # Fast, minimal memory
```

---

### Common Pitfalls

**1. Not checking batch size after filtering**

**Problem:** Filters can remove more images than expected, leading to empty batches.

```python
# ERROR: Might process zero images without realizing
batch = (BatchImageHandler.from_directory("photos/")
         .filter_by_size(min_width=5000)  # Too strict!
         .resize(width=800))

# CORRECT: Check batch size after filtering
batch.filter_by_size(min_width=5000)
if len(batch) == 0:
    print("⚠️ No images match filter criteria!")
else:
    batch.resize(width=800)
```

**Explanation:** Always verify `len(batch)` after aggressive filtering to avoid processing empty batches.

---

**2. Forgetting to handle errors**

**Problem:** Silent failures during batch processing can corrupt results.

```python
# ERROR: Ignores processing errors
batch = BatchImageHandler.from_directory("photos/")
batch.map(custom_function, parallel=True)
batch.save("output/")

# CORRECT: Check and log errors
batch.map(custom_function, parallel=True)
errors = batch.get_errors()
if errors:
    print(f"⚠️ {len(errors)} images failed:")
    for path, error in errors[:10]:
        print(f"  {path.name}: {error}")
batch.save("output/")
```

**Explanation:** Always check `get_errors()` after batch operations to ensure data integrity.

---

**3. Using wrong duplicate threshold**

**Problem:** Incorrect threshold removes too many or too few duplicates.

```python
# ERROR: Threshold too high (removes similar but different images)
batch.filter_duplicates(hash_method='dhash', threshold=25)

# ERROR: Threshold too low (misses near-duplicates)
batch.filter_duplicates(hash_method='dhash', threshold=1)

# CORRECT: Use recommended thresholds based on hash method
batch.filter_duplicates(hash_method='dhash', threshold=5)  # Default
batch.filter_duplicates(hash_method='phash', threshold=8)  # For edited versions
```

**Explanation:** Use recommended defaults: dhash=5, phash=8, ahash=3, whash=10. Analyze first with `analyze_duplicates()` before removing.

---

**4. Not verifying uniformity before ML dataset conversion**

**Problem:** Non-uniform images cause batch processing failures in ML frameworks.

```python
# ERROR: Convert without checking uniformity
batch = BatchImageHandler.from_directory("dataset/")
dataset = batch.to_dataset(format='torch')  # May raise ValueError!

# CORRECT: Verify and fix uniformity first
report = batch.verify_uniformity(check_size=True)
if not report['uniform']:
    target_size = max(report['sizes'].items(), key=lambda x: x[1])[0]
    batch.resize(width=target_size[0], height=target_size[1])
dataset = batch.to_dataset(format='torch')
```

**Explanation:** `to_dataset()` requires all images to have identical dimensions. Always verify uniformity first.

---

**5. Modifying batch during iteration**

**Problem:** Changing batch size while iterating causes unexpected behavior.

```python
# ERROR: Modifying batch during iteration
batch = BatchImageHandler.from_directory("photos/")
for img in batch:
    if img.img.size[0] < 800:
        batch.filter_by_size(min_width=800)  # Don't do this!

# CORRECT: Collect modifications and apply after iteration
to_remove = []
for i, img in enumerate(batch):
    if img.img.size[0] < 800:
        to_remove.append(i)

# Or use built-in filtering instead
batch.filter_by_size(min_width=800)
```

**Explanation:** Use built-in filtering methods instead of manually modifying during iteration.

---

### Limitations

**What this class does NOT do:**

- **Class-balanced dataset splitting**: Assumes unlabeled images in a flat directory
  - *Alternative:* Manually group by class, split each class separately, then combine

- **Video processing**: Only handles static images (JPEG, PNG, TIFF, etc.)
  - *Alternative:* Extract frames with `opencv-python`, then process with BatchImageHandler

- **Real-time streaming**: Designed for batch processing, not live camera feeds
  - *Alternative:* Use `ImageHandler` for single-image processing in a loop

- **GPU-accelerated transformations**: Uses CPU-based PIL operations
  - *Alternative:* Use `torchvision.transforms` or `tf.image` for GPU acceleration

**Edge cases to be aware of:**
- Extremely large images (>100MP) may be slow even with parallel processing
- Perceptual hashing with `hash_size=16` can be 4x slower than `hash_size=8`
- Parallel processing overhead becomes significant for very small images (<100KB)
- Glob patterns are case-sensitive on Linux/Mac, case-insensitive on Windows

---

## Integration Examples

### PyTorch Integration

**Use case:** Creating a PyTorch DataLoader from a directory of images.

```python
from image_toolkit.batch_handler import BatchImageHandler
import torch
from torch.utils.data import TensorDataset, DataLoader

# Prepare uniform dataset
batch = (BatchImageHandler.from_directory("train_images/", "*.jpg")
         .filter_valid()
         .resize(width=224, height=224))

# Convert to PyTorch tensors (N, C, H, W) normalized to [0, 1]
images = batch.to_dataset(
    format='torch',
    normalized=True,
    channels_first=True
)

print(f"Tensor shape: {images.shape}")  # torch.Size([1000, 3, 224, 224])

# Create mock labels (replace with real labels)
labels = torch.randint(0, 10, (len(images),))

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training loop
for batch_images, batch_labels in dataloader:
    # batch_images shape: [32, 3, 224, 224]
    # batch_labels shape: [32]
    pass
```

**Key points:**
- `channels_first=True` produces (C, H, W) format required by PyTorch
- `normalized=True` scales pixels to [0.0, 1.0] range
- All images must be same size before conversion
- Tensor is on CPU by default; use `.to(device)` to move to GPU

---

### NumPy Integration

**Use case:** Analyzing image data with NumPy and scikit-learn.

```python
from image_toolkit.batch_handler import BatchImageHandler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and prepare images
batch = (BatchImageHandler.from_directory("dataset/", "*.png")
         .filter_valid()
         .resize(width=64, height=64)
         .to_grayscale())

# Convert to NumPy array (N, H, W, C) or (N, H, W) for grayscale
images = batch.to_dataset(
    format='numpy',
    normalized=True,
    channels_first=False
)

print(f"Array shape: {images.shape}")  # (500, 64, 64) for grayscale

# Flatten images for PCA
n_images = images.shape[0]
images_flat = images.reshape(n_images, -1)  # (500, 4096)

# Apply PCA dimensionality reduction
pca = PCA(n_components=50)
images_reduced = pca.fit_transform(images_flat)

print(f"Reduced shape: {images_reduced.shape}")  # (500, 50)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize first principal component
plt.plot(pca.explained_variance_ratio_[:10])
plt.title("PCA Explained Variance")
plt.xlabel("Component")
plt.ylabel("Variance Ratio")
plt.show()
```

**Key points:**
- `channels_first=False` produces (H, W, C) format compatible with scikit-learn
- NumPy format is ideal for statistical analysis and classical ML algorithms
- Grayscale conversion reduces dimensionality for faster processing
- Flattening is required for most scikit-learn algorithms

---

## When to Use This Class

### Ideal Use Cases

**✅ Use BatchImageHandler when:**

- **Preparing ML datasets**: You need to filter, resize, and split thousands of images for training neural networks
- **Batch image processing**: You have 100+ images that need identical transformations (resize, adjust, annotate)
- **Photo library management**: You need to deduplicate, organize, or clean large photo collections
- **Dataset quality control**: You need to verify uniformity, detect outliers, or validate image quality across a dataset
- **Thumbnail generation**: You need to create consistent thumbnails or preview grids for galleries
- **Image pipeline automation**: You're building automated workflows for image ingestion, validation, and processing

**Example scenario:** You're training a CNN classifier and have 10,000 raw images in various formats and sizes. You need to remove duplicates, resize to 224x224, split into train/val/test, and export to PyTorch tensors. BatchImageHandler handles this entire workflow in a few lines of code with parallel processing.

---

### When to Look Elsewhere

**❌ Avoid BatchImageHandler when:**

- **Processing single images**: For one-off image operations, use `ImageHandler` instead
  - *Instead use:* `ImageHandler` provides the same transformations without batch overhead

- **Real-time video processing**: For live camera feeds or video frame-by-frame processing
  - *Instead use:* OpenCV (`cv2.VideoCapture`) for real-time streaming and video I/O

- **GPU-accelerated transformations**: For maximum speed on large batches with GPU
  - *Instead use:* `torchvision.transforms` or `tf.image` for GPU-accelerated operations

- **Advanced computer vision tasks**: For object detection, segmentation, or feature extraction
  - *Instead use:* Specialized libraries like `detectron2`, `mmdetection`, or `ultralytics`

- **Web-scale image processing**: For processing millions of images in distributed systems
  - *Instead use:* Cloud-based services (AWS Rekognition, Google Vision) or Spark with image processing libraries

---

### Comparison with Alternatives

| Feature | BatchImageHandler | ImageHandler | torchvision | PIL/Pillow |
|---------|-------------------|--------------|-------------|------------|
| **Batch operations** | ✅ Optimized | ❌ Single only | ✅ Yes | ❌ Single only |
| **Parallel processing** | ✅ Built-in | ❌ No | ❌ Manual | ❌ Manual |
| **Memory efficiency** | ✅ Chunk processing | ✅ Lazy loading | ⚠️ Manual | ⚠️ Manual |
| **ML integration** | ✅ Direct export | ⚠️ Manual | ✅ Native | ❌ Manual |
| **Duplicate detection** | ✅ Built-in | ❌ No | ❌ No | ❌ No |
| **Error resilience** | ✅ Automatic | ⚠️ Manual | ❌ Raises | ❌ Raises |
| **Best for** | Batch datasets | Single images | PyTorch pipelines | Low-level control |
| **Learning curve** | Medium | Low | Medium | Low |
| **Performance** | Fast (parallel) | Fast | Very fast (GPU) | Medium |

**When to choose BatchImageHandler:**
- Processing 100+ images with consistent operations
- Need automatic error handling across many files
- Want built-in duplicate detection and filtering
- Preparing datasets for ML frameworks

**When to choose alternatives:**
- ImageHandler: Single image operations
- torchvision: Pure PyTorch workflows with GPU acceleration
- PIL/Pillow: Maximum control over low-level image operations

---

## See Also

### Documentation
- **[BatchImageHandler API Reference](../api/BatchImageHandler.md)** - Complete method signatures and parameters
- **[ImageHandler Usage Guide](../guide/ImageHandler.md)** - Single image processing workflows
- **[ImageHandler API Reference](../api/ImageHandler.md)** - ImageHandler method reference

### Related Classes
- **`ImageHandler`** - Use for single image operations when you don't need batch processing
- **`ParallelExecutor`** - Internal class handling parallel execution (advanced users only)
- **`ImageContext`** - Internal class representing image state (advanced users only)

### External Resources
- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html) - Integrating with PyTorch training
- [imagehash Library](https://github.com/JohannesBuchner/imagehash) - Perceptual hashing algorithms used for duplicate detection
- [Pillow Documentation](https://pillow.readthedocs.io/) - Underlying image processing library
