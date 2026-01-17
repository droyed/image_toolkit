# ImageHandler Usage Guide

## Table of Contents

- [Overview](#overview)
  - [Key Features](#key-features)
  - [Design Philosophy](#design-philosophy)
- [Quick Start](#quick-start)
- [Common Patterns](#common-patterns)
  - [Pattern 1: Method Chaining for Image Pipelines](#pattern-1-method-chaining-for-image-pipelines)
  - [Pattern 2: Context Manager for Memory Safety](#pattern-2-context-manager-for-memory-safety)
  - [Pattern 3: Lazy Loading for Performance](#pattern-3-lazy-loading-for-performance)
  - [Pattern 4: EXIF-Aware Processing](#pattern-4-exif-aware-processing)
  - [Pattern 5: Format Conversion Workflows](#pattern-5-format-conversion-workflows)
- [Real-World Workflows](#real-world-workflows)
  - [Workflow 1: Preparing Images for Neural Networks](#workflow-1-preparing-images-for-neural-networks)
  - [Workflow 2: Creating Social Media Thumbnails](#workflow-2-creating-social-media-thumbnails)
  - [Workflow 3: Building Training Datasets](#workflow-3-building-training-datasets)
  - [Workflow 4: Photo Enhancement Pipeline](#workflow-4-photo-enhancement-pipeline)
  - [Workflow 5: Data Augmentation for ML](#workflow-5-data-augmentation-for-ml)
  - [Workflow 6: Scientific Image Analysis](#workflow-6-scientific-image-analysis)
- [Best Practices](#best-practices)
  - [Performance](#performance)
  - [Memory Management](#memory-management)
  - [Common Pitfalls](#common-pitfalls)
  - [Limitations](#limitations)
- [Integration Examples](#integration-examples)
  - [PyTorch Integration](#pytorch-integration)
  - [NumPy Integration](#numpy-integration)
  - [Matplotlib Integration](#matplotlib-integration)
- [When to Use This Class](#when-to-use-this-class)
  - [Ideal Use Cases](#ideal-use-cases)
  - [When to Look Elsewhere](#when-to-look-elsewhere)
  - [Comparison with Alternatives](#comparison-with-alternatives)
- [See Also](#see-also)

---

## Overview

`ImageHandler` is a high-level, user-friendly interface for image processing that combines loading, transformation, annotation, and analysis operations into a single, chainable API. Built on top of PIL/Pillow, it provides a streamlined workflow for common image tasks while maintaining full control over memory and performance.

**Who should use it:**
- Data scientists preparing image datasets for machine learning
- Developers building image processing pipelines
- Researchers working with scientific image analysis
- Anyone needing efficient, chainable image operations

**What makes it different:**
Unlike raw PIL or OpenCV, `ImageHandler` emphasizes method chaining, automatic memory management, and a consistent API across all operations. It's designed for workflows where you need to apply multiple transformations sequentially without writing verbose boilerplate code.

**Key Capabilities:**
- Fluent method chaining for multi-step transformations
- Lazy loading for memory efficiency
- Context manager support for automatic cleanup
- Direct conversion to PyTorch/NumPy formats
- EXIF metadata preservation and manipulation

The class follows a "load once, transform many, save when ready" philosophy that minimizes I/O overhead while keeping your code clean and expressive.

### Key Features

- **Chainable Operations**: All transformation methods return `self`, enabling clean pipeline syntax
- **Memory Efficiency**: Lazy loading and explicit unload control prevent unnecessary memory consumption
- **ML-Ready Output**: Direct conversion to PyTorch tensors, NumPy arrays with normalization
- **EXIF Preservation**: Read, modify, and preserve metadata across transformations
- **Type Safety**: Consistent return types and clear error messages for debugging
- **Context Manager**: Automatic resource cleanup with Python's `with` statement

### Design Philosophy

`ImageHandler` treats images as mutable contexts that flow through transformation pipelines. Each method modifies the internal image state and returns the handler itself, allowing natural chaining without intermediate variables.

**Key design decisions:**

1. **Mutable State with Chaining**: Unlike functional approaches, transformations modify the handler in-place. This reduces memory overhead for long pipelines while maintaining readable syntax.

2. **Lazy Loading by Default**: The constructor accepts a path but doesn't load pixels until needed. Use `open()` for immediate loading or rely on automatic loading when transformations are called.

3. **Explicit Memory Control**: Methods like `unload()` and context managers give you fine-grained control over when memory is freed, critical for processing large image collections.

4. **Uniform Method Signatures**: All transformations follow `method(params) -> self` pattern, making the API predictable and easy to learn.

**Understanding these principles will help you:**
- Write more efficient image processing pipelines
- Avoid common memory issues in large-scale processing
- Integrate seamlessly with ML frameworks like PyTorch

---

## Quick Start

### Minimal Example

```python
# Import
from image_toolkit.handler import ImageHandler

# Simplest possible usage
handler = ImageHandler.open("photo.jpg")
handler.resize_aspect(width=800).save("output.jpg")
```

### Common Workflow

```python
# Typical usage showing method chaining
result = (ImageHandler.open("photo.jpg")
          .resize_aspect(width=800)
          .adjust(brightness=1.2, contrast=1.1)
          .save("enhanced.jpg"))
```

### Getting Results

```python
# Convert to different formats for downstream use
handler = ImageHandler.open("photo.jpg")

# NumPy array for analysis
array = handler.to_array(normalize=True)  # Shape: (H, W, C), range [0, 1]

# PyTorch tensor for deep learning
tensor = handler.to_tensor(normalize=True, device="cuda")  # Shape: (C, H, W)

# Statistics dictionary
stats = handler.get_stats()  # {'width': 1920, 'height': 1080, ...}
```

---

## Common Patterns

### Pattern 1: Method Chaining for Image Pipelines

**When to use:** You need to apply multiple transformations sequentially and want clean, readable code.

```python
from image_toolkit.handler import ImageHandler

# Chain multiple operations without intermediate variables
result = (ImageHandler.open("raw_photo.jpg")
          .apply_exif_orientation()  # Fix rotation from camera
          .resize_aspect(width=1024, height=768, padding_color=(255, 255, 255))
          .adjust(brightness=1.15, contrast=1.05)
          .to_rgba()  # Convert to RGBA for transparency
          .save("processed.png", quality=95))

# The handler can still be used after saving
print(result.get_stats())
```

**Key points:**
- Every transformation returns `self` for continuous chaining
- Operations are applied in the order they're called
- Save doesn't terminate the chain; you can continue processing

---

### Pattern 2: Context Manager for Memory Safety

**When to use:** Processing many images in a loop where memory leaks could accumulate.

```python
from pathlib import Path
from image_toolkit.handler import ImageHandler

# Process a directory of images with automatic cleanup
image_paths = Path("photos/").glob("*.jpg")

for path in image_paths:
    with ImageHandler.open(path) as handler:
        (handler
         .resize_aspect(width=800)
         .adjust(brightness=1.1)
         .save(f"output/{path.stem}_processed.jpg"))
    # Image memory automatically freed here
```

**Key points:**
- `with` statement ensures `unload()` is called automatically
- Critical for batch processing to prevent memory buildup
- Works even if exceptions occur during processing

---

### Pattern 3: Lazy Loading for Performance

**When to use:** You need to validate or inspect images before committing to full pixel loading.

```python
from image_toolkit.handler import ImageHandler

# Create handler without loading pixels
handler = ImageHandler("large_photo.tif")

# Fast metadata checks (no pixel data loaded yet)
if not handler.is_valid():
    print("Corrupted file!")
    exit(1)

# Read EXIF without loading image
exif = handler.read_exif()
if exif.get('Orientation') != 1:
    print("Image needs rotation")

# Now load and process only if needed
handler.load()
handler.resize_aspect(width=512).save("thumbnail.jpg")
handler.unload()  # Explicit cleanup
```

**Key points:**
- Constructor doesn't load pixels; use `load()` explicitly or let transformations auto-load
- Metadata operations work without full image loading
- Useful for filtering large image collections

---

### Pattern 4: EXIF-Aware Processing

**When to use:** Working with photos from cameras/phones that embed orientation and metadata.

```python
from image_toolkit.handler import ImageHandler

# Preserve metadata through transformations
handler = ImageHandler.open("camera_photo.jpg")

# Read and inspect EXIF
exif = handler.read_exif(prefer_exiftool=True)
print(f"Camera: {exif.get('Make')} {exif.get('Model')}")
print(f"ISO: {exif.get('ISOSpeedRatings')}")

# Auto-rotate based on EXIF orientation tag
handler.apply_exif_orientation()

# Process while keeping metadata
(handler
 .resize_aspect(width=2048)
 .save_with_metadata("processed.jpg"))  # Preserves EXIF
```

**Key points:**
- Use `prefer_exiftool=True` for comprehensive metadata extraction
- `apply_exif_orientation()` fixes camera rotation automatically
- `save_with_metadata()` preserves EXIF across transformations

---

### Pattern 5: Format Conversion Workflows

**When to use:** Converting images between formats with specific requirements.

```python
from image_toolkit.handler import ImageHandler

# PNG to JPEG with quality control
(ImageHandler.open("transparent.png")
 .to_rgba()  # Ensure alpha channel handling
 .format_convert("RGB")  # JPEG doesn't support transparency
 .save("output.jpg", quality=90))

# JPEG to WebP for web optimization
(ImageHandler.open("photo.jpg")
 .resize_aspect(width=1920)
 .format_convert("RGB")
 .save("photo.webp", quality=85))

# Any format to grayscale PNG
(ImageHandler.open("color_image.bmp")
 .to_grayscale(keep_2d=False)  # Keep as 3-channel for compatibility
 .save("grayscale.png"))
```

**Key points:**
- Always consider color mode when converting formats
- JPEG doesn't support transparency; convert to RGB first
- Use `format_convert()` for explicit mode changes

---

## Real-World Workflows

### Workflow 1: Preparing Images for Neural Networks

**Scenario:** You have a directory of images with varying sizes and need to prepare them for a CNN that requires 224×224 RGB inputs with ImageNet normalization.

**Solution:**

```python
from image_toolkit.handler import ImageHandler
from pathlib import Path
import torch

# Directory containing raw training images
image_dir = Path("raw_dataset/")
output_dir = Path("processed_dataset/")
output_dir.mkdir(exist_ok=True)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Process each image
for img_path in image_dir.glob("*.jpg"):
    try:
        # Load and prepare for model
        handler = ImageHandler.open(img_path)

        # Resize to fixed size with aspect ratio preservation
        handler.resize_aspect(width=224, height=224, padding_color=(114, 114, 114))

        # Convert to RGB (some images might be grayscale)
        if handler.is_grayscale_mode():
            handler.to_rgba().format_convert("RGB")

        # Apply ImageNet normalization
        handler.normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Save processed image
        handler.save(output_dir / img_path.name, quality=95)

        # Optional: convert to tensor for immediate use
        tensor = handler.to_tensor(normalize=False, device="cuda")  # Already normalized

        # Clean up memory
        handler.unload()

    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

print(f"Processed {len(list(output_dir.glob('*.jpg')))} images")
```

**Explanation:**
- `resize_aspect` maintains proportions and pads to exact dimensions (critical for batching)
- Gray padding (114, 114, 114) is commonly used in object detection
- Normalization follows PyTorch's ImageNet pretrained model convention
- Memory cleanup prevents accumulation when processing thousands of images

**Alternative approaches:**
- For very large datasets, use `BatchImageHandler` for parallel processing
- For on-the-fly loading during training, use PyTorch's `Dataset` with `ImageHandler` in `__getitem__`

---

### Workflow 2: Creating Social Media Thumbnails

**Scenario:** Generate square thumbnails for Instagram/Facebook from landscape and portrait photos, with branding overlay.

**Solution:**

```python
from image_toolkit.handler import ImageHandler
from pathlib import Path

def create_social_thumbnail(input_path, output_path, size=1080):
    """
    Create square thumbnail with centered content and branded border.
    """
    # Load and process
    handler = ImageHandler.open(input_path)

    # Auto-rotate based on EXIF (phones often save rotated)
    handler.apply_exif_orientation()

    # Create square with padding
    handler.square_pad(size=size, fill_color=(245, 245, 245))  # Light gray background

    # Enhance for social media (slightly boost colors)
    handler.adjust(brightness=1.05, contrast=1.08)

    # Add subtle sharpening (optional)
    # handler.filter_blur(radius=1)  # Slight blur can reduce compression artifacts

    # Add watermark text (if needed)
    handler.draw_text(
        text="@mycompany",
        position=(size - 200, size - 50),
        font_size=24,
        color=(150, 150, 150)
    )

    # Save with high quality for upload
    handler.save(output_path, quality=95)
    handler.unload()

    return output_path

# Batch process
input_dir = Path("raw_photos/")
output_dir = Path("social_thumbnails/")
output_dir.mkdir(exist_ok=True)

for photo in input_dir.glob("*.jpg"):
    output_path = output_dir / f"{photo.stem}_thumb.jpg"
    create_social_thumbnail(photo, output_path, size=1080)
    print(f"Created: {output_path}")
```

**Explanation:**
- `square_pad` is ideal for social media where square crops are required
- EXIF orientation fixes phone photo rotation issues
- Slight brightness/contrast boost compensates for compression
- Gray background is less jarring than black for portrait images

---

### Workflow 3: Building Training Datasets

**Scenario:** Convert a raw image collection into a PyTorch-ready dataset with train/val splits and data augmentation.

**Solution:**

```python
from image_toolkit.handler import ImageHandler
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import random

class CustomImageDataset(Dataset):
    """PyTorch Dataset using ImageHandler for loading."""

    def __init__(self, image_paths, transform=True, target_size=(256, 256)):
        self.image_paths = image_paths
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # Load with ImageHandler
        handler = ImageHandler.open(path)

        # Base preprocessing
        handler.resize_aspect(
            width=self.target_size[0],
            height=self.target_size[1],
            padding_color=(0, 0, 0)
        )

        # Data augmentation (only during training)
        if self.transform:
            # Random horizontal flip
            if random.random() > 0.5:
                handler.flip_horizontal()

            # Random brightness/contrast
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            handler.adjust(brightness=brightness, contrast=contrast)

            # Random rotation
            angle = random.uniform(-10, 10)
            handler.rotate(angle, expand=False, fillcolor=(0, 0, 0))

        # Convert to tensor
        tensor = handler.to_tensor(normalize=True, device="cpu")

        # Extract label from filename (e.g., "cat_001.jpg" -> 0, "dog_001.jpg" -> 1)
        label = 0 if "cat" in path.stem else 1

        # Clean up
        handler.unload()

        return tensor, label

# Prepare dataset
all_images = list(Path("dataset/").glob("*.jpg"))
random.shuffle(all_images)

# 80/20 train/val split
split_idx = int(len(all_images) * 0.8)
train_paths = all_images[:split_idx]
val_paths = all_images[split_idx:]

# Create datasets
train_dataset = CustomImageDataset(train_paths, transform=True)
val_dataset = CustomImageDataset(val_paths, transform=False)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Use in training loop
for batch_idx, (images, labels) in enumerate(train_loader):
    # images shape: [32, 3, 256, 256]
    # labels shape: [32]
    print(f"Batch {batch_idx}: {images.shape}")
    # ... forward pass, loss, backward ...
```

**Explanation:**
- `ImageHandler` integrates cleanly with PyTorch's `Dataset` interface
- Augmentation is applied on-the-fly during loading (more memory efficient than pre-computing)
- `unload()` prevents memory leaks across epochs
- Each `__getitem__` call creates a fresh handler to avoid state sharing

---

### Workflow 4: Photo Enhancement Pipeline

**Scenario:** Automatically enhance a collection of underexposed or low-contrast photos.

**Solution:**

```python
from image_toolkit.handler import ImageHandler
from pathlib import Path
import numpy as np

def auto_enhance_photo(input_path, output_path):
    """
    Automatically enhance photo based on histogram analysis.
    """
    handler = ImageHandler.open(input_path)

    # Get image statistics
    stats = handler.get_stats()
    channel_stats = handler.get_channel_stats()

    # Calculate average brightness (0-255)
    avg_brightness = np.mean([
        channel_stats['red']['mean'],
        channel_stats['green']['mean'],
        channel_stats['blue']['mean']
    ])

    # Determine brightness adjustment
    target_brightness = 128  # Mid-tone
    if avg_brightness < 100:
        # Underexposed
        brightness_factor = 1.0 + (target_brightness - avg_brightness) / 255
        contrast_factor = 1.15  # Boost contrast for flat images
    elif avg_brightness > 160:
        # Overexposed
        brightness_factor = 1.0 - (avg_brightness - target_brightness) / 255
        contrast_factor = 1.05
    else:
        # Well-exposed, minimal adjustment
        brightness_factor = 1.0
        contrast_factor = 1.08

    # Apply enhancements
    handler.adjust(brightness=brightness_factor, contrast=contrast_factor)

    # Check if image is grayscale content (common issue with scans)
    if handler.is_grayscale_content(tolerance=0.02):
        print(f"Converting {input_path.name} to grayscale (detected as B&W)")
        handler.to_grayscale(keep_2d=False)

    # Save enhanced version
    handler.save(output_path, quality=95)

    # Report adjustments
    print(f"Enhanced {input_path.name}: "
          f"brightness={brightness_factor:.2f}, contrast={contrast_factor:.2f}")

    handler.unload()

# Process directory
input_dir = Path("raw_photos/")
output_dir = Path("enhanced_photos/")
output_dir.mkdir(exist_ok=True)

for photo in input_dir.glob("*.jpg"):
    output_path = output_dir / f"{photo.stem}_enhanced.jpg"
    auto_enhance_photo(photo, output_path)
```

**Explanation:**
- Uses `get_channel_stats()` to analyze image brightness programmatically
- Adaptive adjustment based on histogram (avoids one-size-fits-all enhancement)
- Detects grayscale content to optimize file size
- Suitable for batch processing photo collections

---

### Workflow 5: Data Augmentation for ML

**Scenario:** Generate augmented versions of a limited training dataset to improve model generalization.

**Solution:**

```python
from image_toolkit.handler import ImageHandler
from pathlib import Path
import random

def augment_image(handler, augmentation_type):
    """Apply specific augmentation to handler."""
    if augmentation_type == "flip_h":
        return handler.flip_horizontal()
    elif augmentation_type == "flip_v":
        return handler.flip_vertical()
    elif augmentation_type == "rotate_90":
        return handler.rotate(90)
    elif augmentation_type == "rotate_180":
        return handler.rotate(180)
    elif augmentation_type == "rotate_270":
        return handler.rotate(270)
    elif augmentation_type == "brighten":
        return handler.adjust(brightness=1.2, contrast=1.0)
    elif augmentation_type == "darken":
        return handler.adjust(brightness=0.8, contrast=1.0)
    elif augmentation_type == "high_contrast":
        return handler.adjust(brightness=1.0, contrast=1.3)
    elif augmentation_type == "blur":
        return handler.filter_blur(radius=2)
    else:
        return handler

def generate_augmentations(input_path, output_dir, num_augmentations=5):
    """Generate multiple augmented versions of a single image."""

    augmentation_types = [
        "flip_h", "flip_v", "rotate_90", "rotate_270",
        "brighten", "darken", "high_contrast", "blur"
    ]

    # Randomly select augmentations
    selected_augs = random.sample(augmentation_types, num_augmentations)

    for idx, aug_type in enumerate(selected_augs):
        # Load fresh copy for each augmentation
        handler = ImageHandler.open(input_path)

        # Apply augmentation
        augment_image(handler, aug_type)

        # Save with descriptive name
        output_name = f"{input_path.stem}_aug_{aug_type}_{idx}.jpg"
        handler.save(output_dir / output_name, quality=95)
        handler.unload()

# Augment dataset
input_dir = Path("original_dataset/")
output_dir = Path("augmented_dataset/")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    # Keep original
    handler = ImageHandler.open(img_path)
    handler.save(output_dir / img_path.name)
    handler.unload()

    # Generate 5 augmented versions
    generate_augmentations(img_path, output_dir, num_augmentations=5)

print(f"Augmented dataset from {len(list(input_dir.glob('*.jpg')))} to "
      f"{len(list(output_dir.glob('*.jpg')))} images")
```

**Explanation:**
- Creates multiple variations from each source image
- Each augmentation is saved as a separate file (useful for offline augmentation)
- Can be combined with `BatchImageHandler` for parallel processing
- Alternative to on-the-fly augmentation when training on limited hardware

---

### Workflow 6: Scientific Image Analysis

**Scenario:** Analyze microscopy images to extract quantitative measurements and dominant colors.

**Solution:**

```python
from image_toolkit.handler import ImageHandler
from pathlib import Path
import numpy as np
import csv

def analyze_microscopy_image(image_path):
    """Extract quantitative measurements from microscopy image."""

    handler = ImageHandler.open(image_path)

    # Get basic statistics
    stats = handler.get_stats()
    channel_stats = handler.get_channel_stats()

    # Compute histogram for distribution analysis
    histogram = handler.compute_histogram(bins=256)

    # Detect dominant colors (useful for stain analysis)
    dominant_colors = handler.detect_dominant_colors(n_colors=3)

    # Check if image is effectively grayscale
    is_grayscale = handler.is_grayscale_content(tolerance=0.01)

    # Convert to array for custom analysis
    img_array = handler.to_array(normalize=False)  # Keep 0-255 range

    # Custom analysis: calculate mean intensity in center region
    h, w = img_array.shape[:2]
    center_region = img_array[h//4:3*h//4, w//4:3*w//4]
    center_mean_intensity = np.mean(center_region)

    # Prepare results
    results = {
        'filename': image_path.name,
        'width': stats['width'],
        'height': stats['height'],
        'mode': stats['mode'],
        'is_grayscale': is_grayscale,
        'red_mean': channel_stats['red']['mean'],
        'green_mean': channel_stats['green']['mean'],
        'blue_mean': channel_stats['blue']['mean'],
        'red_std': channel_stats['red']['std'],
        'green_std': channel_stats['green']['std'],
        'blue_std': channel_stats['blue']['std'],
        'center_intensity': center_mean_intensity,
        'dominant_color_1': dominant_colors[0] if dominant_colors else None,
        'dominant_color_2': dominant_colors[1] if len(dominant_colors) > 1 else None,
        'dominant_color_3': dominant_colors[2] if len(dominant_colors) > 2 else None,
    }

    handler.unload()
    return results

# Analyze all images in directory
input_dir = Path("microscopy_images/")
results = []

for img_path in input_dir.glob("*.tif"):
    try:
        result = analyze_microscopy_image(img_path)
        results.append(result)
        print(f"Analyzed: {img_path.name}")
    except Exception as e:
        print(f"Failed: {img_path.name} - {e}")

# Export to CSV
output_csv = "analysis_results.csv"
if results:
    keys = results[0].keys()
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Exported {len(results)} results to {output_csv}")
```

**Explanation:**
- Combines built-in statistics methods with custom NumPy analysis
- `compute_histogram()` provides distribution data for quality control
- Dominant color detection useful for identifying staining artifacts
- Results exported to CSV for further analysis in Excel/R/Python

---

## Best Practices

### Performance

**Optimization 1: Batch Processing with Explicit Unloading**

```python
# SLOW: Keeps all images in memory
handlers = []
for path in image_paths:
    h = ImageHandler.open(path)
    h.resize_aspect(width=800)
    handlers.append(h)  # Memory keeps growing!

# FAST: Process and release immediately
for path in image_paths:
    with ImageHandler.open(path) as h:
        h.resize_aspect(width=800).save(f"output/{path.name}")
    # Memory freed after each iteration
```

**Why it matters:** Processing 1000 high-res images can consume 20+ GB RAM in the slow version, but only ~50 MB in the fast version. Always unload images after processing, especially in loops.

---

**Optimization 2: Lazy Loading for Filtering**

```python
# SLOW: Loads full pixel data unnecessarily
valid_images = []
for path in image_paths:
    h = ImageHandler.open(path)  # Loads full image
    if h.get_stats()['width'] >= 800:
        valid_images.append(h)

# FAST: Check metadata without loading pixels
from PIL import Image
valid_images = []
for path in image_paths:
    with Image.open(path) as img:
        if img.size[0] >= 800:  # Check width
            valid_images.append(ImageHandler.open(path))
```

**Why it matters:** Metadata checks are 100-1000× faster than full image loading. For filtering large collections, this difference is substantial.

---

### Memory Management

**⚠️ Important: Always Unload After Converting to Tensors**

```python
# BAD: Image stays in memory after tensor conversion
handler = ImageHandler.open("large_image.tif")
tensor = handler.to_tensor()
# Handler still holds a copy of the image data!

# GOOD: Explicit cleanup
handler = ImageHandler.open("large_image.tif")
tensor = handler.to_tensor()
handler.unload()  # Free the PIL image from memory

# BETTER: Use context manager
with ImageHandler.open("large_image.tif") as handler:
    tensor = handler.to_tensor()
# Automatically unloaded
```

**Why it matters:** Tensor conversion creates a copy; the original PIL image remains in memory unless explicitly freed. This can double memory usage for large batches.

---

### Common Pitfalls

**1. Forgetting EXIF Orientation**

**Problem:** Photos from smartphones appear rotated even after processing.

```python
# ERROR: Image appears sideways after resize
handler = ImageHandler.open("phone_photo.jpg")
handler.resize_aspect(width=800).save("output.jpg")
# Output is rotated 90 degrees!

# CORRECT: Apply EXIF orientation first
handler = ImageHandler.open("phone_photo.jpg")
handler.apply_exif_orientation()  # Fix rotation
handler.resize_aspect(width=800).save("output.jpg")
```

**Explanation:** Cameras save images in sensor orientation and embed a rotation tag in EXIF. Always call `apply_exif_orientation()` before transformations when working with camera photos.

---

**2. Padding Color Mismatches**

**Problem:** Black borders appear around resized images intended for white backgrounds.

```python
# ERROR: Black padding on white website
handler.resize_aspect(width=800, height=600, padding_color=(0, 0, 0))
# Looks bad on white backgrounds

# CORRECT: Match padding to background
handler.resize_aspect(width=800, height=600, padding_color=(255, 255, 255))
# Seamless integration with white backgrounds
```

**Explanation:** Default padding is black (0, 0, 0). For web thumbnails, social media, or documents, use white (255, 255, 255) or the target background color.

---

**3. Format Conversion Without Mode Check**

**Problem:** Saving PNG with transparency as JPEG creates artifacts or errors.

```python
# ERROR: JPEG doesn't support transparency
handler = ImageHandler.open("logo.png")  # RGBA mode
handler.save("logo.jpg")  # May show black background or error

# CORRECT: Convert mode for target format
handler = ImageHandler.open("logo.png")
if handler.img.mode == 'RGBA':
    handler.format_convert("RGB")  # Remove alpha channel
handler.save("logo.jpg")
```

**Explanation:** JPEG format doesn't support transparency. Convert RGBA to RGB first, or use PNG/WebP for images with transparency.

---

**4. Normalizing Already-Normalized Data**

**Problem:** Double normalization produces incorrect value ranges.

```python
# ERROR: Normalizing twice
handler = ImageHandler.open("photo.jpg")
handler.normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
tensor = handler.to_tensor(normalize=True)  # Normalizes again!
# Values are now in wrong range

# CORRECT: Normalize once
handler = ImageHandler.open("photo.jpg")
handler.normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
tensor = handler.to_tensor(normalize=False)  # Don't normalize again
```

**Explanation:** `to_tensor(normalize=True)` applies its own normalization. If you've already called `normalize()`, set `normalize=False` in conversion methods.

---

### Limitations

**What this class does NOT do:**

- **Advanced Annotations**: Complex drawing operations like curves, filled shapes, or custom fonts
  - *Alternative:* Use PIL's `ImageDraw` directly or integrate with `opencv-python`

- **Video Processing**: Handles single images only, not video frames or sequences
  - *Alternative:* Use `opencv-python` or `moviepy` for video

- **RAW Image Formats**: Cannot directly open CR2, NEF, ARW, etc.
  - *Alternative:* Use `rawpy` to convert to TIFF/JPEG first, then process with `ImageHandler`

- **Lossless Rotations**: Rotation uses interpolation (slight quality loss)
  - *Alternative:* For JPEG, use `jpegtran` or `pillow-jpls` for lossless 90° rotations

**Edge cases to be aware of:**
- Very large images (>10000×10000 pixels) may cause memory issues on some systems
- EXIF orientation may not work correctly for all camera manufacturers
- Normalization assumes RGB order; BGR formats require manual handling

---

## Integration Examples

### PyTorch Integration

**Use case:** Loading images directly into PyTorch training pipelines with custom preprocessing.

```python
from image_toolkit.handler import ImageHandler
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load with ImageHandler
        handler = ImageHandler.open(self.image_paths[idx])

        # Apply custom transforms if provided
        if self.transform:
            handler = self.transform(handler)

        # Convert to tensor (shape: C, H, W)
        tensor = handler.to_tensor(
            normalize=True,
            device="cuda"  # Load directly to GPU
        )

        handler.unload()
        return tensor

# Define transform pipeline
def my_transform(handler):
    return (handler
            .resize_aspect(width=224, height=224)
            .normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]))

# Create dataset
dataset = ImageDataset(image_paths, transform=my_transform)
```

**Key points:**
- `to_tensor()` returns PyTorch-compatible tensors with channels-first layout
- Can load directly to CUDA device for GPU training
- Integrates seamlessly with PyTorch's DataLoader

---

### NumPy Integration

**Use case:** Scientific analysis requiring NumPy array operations.

```python
from image_toolkit.handler import ImageHandler
import numpy as np
import matplotlib.pyplot as plt

# Load image as NumPy array
handler = ImageHandler.open("microscopy.tif")
img_array = handler.to_array(normalize=False)  # Shape: (H, W, C), dtype: uint8

# Perform NumPy operations
grayscale = np.mean(img_array, axis=2)  # Convert to grayscale manually
threshold = grayscale > 128  # Binary threshold
contours = np.where(threshold)  # Find bright regions

# Modify array and convert back
img_array[:, :, 0] = 0  # Remove red channel
modified_handler = ImageHandler.open("microscopy.tif")
modified_handler._ctx.img = Image.fromarray(img_array)
modified_handler.save("modified.tif")
```

**Key points:**
- `to_array()` returns standard NumPy arrays (H, W, C) format
- Full compatibility with NumPy, SciPy, scikit-image
- Can modify arrays and reload into handler

---

### Matplotlib Integration

**Use case:** Visualizing images and analysis results in notebooks.

```python
from image_toolkit.handler import ImageHandler
import matplotlib.pyplot as plt

# Load and display image
handler = ImageHandler.open("photo.jpg")

# Quick display (opens system viewer)
handler.show(title="Original Image")

# Matplotlib display (inline in notebooks)
handler.inspect(title="Analysis View", block=False)

# Custom matplotlib figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
img_array = handler.to_array(normalize=False)
axes[0].imshow(img_array)
axes[0].set_title("Original")
axes[0].axis('off')

# Grayscale
handler_gray = handler.copy()
handler_gray.to_grayscale()
axes[1].imshow(handler_gray.to_array(normalize=False), cmap='gray')
axes[1].set_title("Grayscale")
axes[1].axis('off')

# Enhanced
handler_enhanced = handler.copy()
handler_enhanced.adjust(brightness=1.3, contrast=1.2)
axes[2].imshow(handler_enhanced.to_array(normalize=False))
axes[2].set_title("Enhanced")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Clean up
handler.unload()
handler_gray.unload()
handler_enhanced.unload()
```

**Key points:**
- `inspect()` provides quick Matplotlib visualization
- `to_array()` works directly with `imshow()`
- Use `copy()` to create independent handlers for comparisons

---

## When to Use This Class

### Ideal Use Cases

**✅ Use ImageHandler when:**

- **Preparing ML training datasets**: Need to resize, normalize, and convert thousands of images to tensors with consistent preprocessing pipelines.

- **Building image processing pipelines**: Want clean, chainable syntax for multi-step transformations without managing intermediate variables.

- **Prototyping in notebooks**: Need quick image loading, visualization, and analysis during exploratory data work.

- **Processing camera photos**: Working with JPEG images from phones/cameras that require EXIF orientation handling and metadata preservation.

- **Memory-constrained environments**: Need explicit control over when images are loaded/unloaded (batch processing on limited RAM).

**Example scenario:** You're training a ResNet classifier and have 10,000 raw images of varying sizes. You need to resize them to 224×224, apply normalization, convert to tensors, and split into train/val sets. `ImageHandler` provides a clean API for this entire workflow with minimal code.

---

### When to Look Elsewhere

**❌ Avoid ImageHandler when:**

- **Real-time video processing**: Need to process video frames at 30+ FPS
  - *Instead use:* OpenCV (`cv2.VideoCapture`) for direct video stream handling

- **Advanced computer vision algorithms**: Need feature detection, object tracking, stereo vision, etc.
  - *Instead use:* OpenCV or scikit-image for specialized CV algorithms

- **Massive parallel processing**: Processing millions of images and need distributed computing
  - *Instead use:* Apache Spark with image libraries or cloud-based solutions (AWS Lambda, GCP Cloud Functions)

- **Pixel-perfect lossless operations**: Need guaranteed no quality loss for medical/scientific imaging
  - *Instead use:* `rawpy` + `imageio` for RAW formats or specialized medical imaging libraries like `pydicom`

---

### Comparison with Alternatives

| Feature | ImageHandler | PIL/Pillow | OpenCV | scikit-image |
|---------|-------------|-----------|--------|--------------|
| **Ease of Use** | Chainable API | Verbose | Moderate | Moderate |
| **Memory Control** | Explicit (unload) | Manual | Manual | Manual |
| **ML Integration** | Built-in (tensor/array) | Manual conversion | NumPy only | NumPy only |
| **EXIF Handling** | Automatic | Basic | Limited | None |
| **Performance** | Good (PIL-based) | Good | Excellent | Good |
| **Best For** | ML preprocessing, pipelines | General image I/O | Real-time CV | Scientific analysis |

---

## See Also

### Documentation
- **[ImageHandler API Reference](../api/ImageHandler.md)** - Complete method signatures and parameters
- **[BatchImageHandler Usage Guide](../guide/BatchImageHandler.md)** - Parallel processing for image collections
- **[BatchImageHandler API Reference](../api/BatchImageHandler.md)** - Batch processing method reference

### Related Classes
- **`BatchImageHandler`** - When you need to process hundreds/thousands of images in parallel with filtering and duplicate detection
- **`ImageContext`** - Lower-level context object if you need direct access to internal state

### External Resources
- [Pillow Documentation](https://pillow.readthedocs.io/) - Underlying image library
- [PyTorch Vision Transforms](https://pytorch.org/vision/stable/transforms.html) - Alternative augmentation approach
- [ImageNet Preprocessing Guide](https://pytorch.org/vision/stable/models.html) - Standard normalization for pretrained models
