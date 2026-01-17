# Image Toolkit

**High-Performance Python Image Processing with ML-First Design**

A comprehensive image processing library built on fluent API design, featuring TurboJPEG acceleration, parallel batch processing, and direct PyTorch/NumPy integration.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](pyproject.toml)

## Features

- **Fluent API Design** - Clean method chaining for readable image processing pipelines
- **TurboJPEG Acceleration** - 3-5x faster JPEG I/O compared to standard PIL
- **Parallel Batch Processing** - Multi-threaded operations with configurable worker pools
- **ML Integration** - Direct export to PyTorch tensors and NumPy arrays with built-in normalization
- **Perceptual Hashing** - Duplicate detection with dhash, phash, ahash, and whash algorithms
- **Memory Efficient** - Chunk processing for datasets larger than RAM

## Installation

```bash
pip install image_toolkit
```

Optional dependencies:
```bash
pip install "image_toolkit[ml]"    # PyTorch support
pip install "image_toolkit[exif]"  # Advanced EXIF handling
pip install "image_toolkit[dev]"   # Development tools
pip install "image_toolkit[all]"   # Everything
```

## Quick Start

### Single Image Processing

```python
from image_toolkit.handler import ImageHandler

# Load, transform, and save with method chaining
(ImageHandler.open("photo.jpg")
 .resize_aspect(width=800)
 .adjust(brightness=1.2, contrast=1.1)
 .save("enhanced.jpg"))
```

### Batch Processing

```python
from image_toolkit.batch_handler import BatchImageHandler

# Process entire directory in parallel
batch = (BatchImageHandler.from_directory("photos/", "*.jpg")
         .filter_valid()
         .resize(width=1024)
         .save("output/", prefix="processed_"))

print(f"Processed {len(batch)} images")
```

### ML Dataset Preparation

```python
# Complete ML preprocessing pipeline
dataset = (BatchImageHandler.from_directory("raw_photos/", "*.jpg")
           .filter_valid().filter_duplicates(hash_method='dhash')
           .resize(width=224, height=224)
           .to_dataset(format='torch', normalized=True))
# Ready for training: dataset.shape → torch.Size([1000, 3, 224, 224])
```

## Key Capabilities

**Image Transformations:**
- Resize with aspect ratio preservation
- Crop, rotate, flip operations
- Brightness/contrast adjustment
- Format conversion and color mode changes

**Batch Operations:**
- Parallel processing with configurable workers
- Filter by size, aspect ratio, file size
- Duplicate detection and removal
- Quality control and outlier detection

**ML Features:**
- PyTorch tensor conversion with GPU support
- ImageNet normalization built-in
- Train/val/test dataset splitting
- Chunk processing for large datasets

**Analysis Tools:**
- EXIF metadata extraction
- Statistical analysis and visualization
- Perceptual hashing for duplicates
- Dominant color detection

## Performance

- **TurboJPEG**: 3-5x faster JPEG I/O
- **Parallel Processing**: Up to 6x speedup with 8 workers
- **Memory Efficient**: Process datasets 10x larger than RAM via chunking

## Comparison

### `ImageHandler` vs alternatives

| Feature | ImageHandler | PIL/Pillow | OpenCV | scikit-image |
|---------|-------------|-----------|--------|--------------|
| **Ease of Use** | Chainable API | Verbose | Moderate | Moderate |
| **Memory Control** | Explicit (unload) | Manual | Manual | Manual |
| **ML Integration** | Built-in (tensor/array) | Manual conversion | NumPy only | NumPy only |
| **EXIF Handling** | Automatic | Basic | Limited | None |
| **Performance** | Good (PIL-based) | Good | Excellent | Good |
| **Best For** | ML preprocessing, pipelines | General image I/O | Real-time CV | Scientific analysis |


### `BatchImageHandler` vs alternatives

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


## Documentation Index

### User Guides
- [ImageHandler Guide](docs/guide/ImageHandler.md) - Comprehensive manual for single-image processing, including chaining, transformations, and I/O.
- [BatchImageHandler Guide](docs/guide/BatchImageHandler.md) - Complete guide for parallel batch processing, filtering, and workflow orchestration.

### API Reference
- [ImageHandler API](docs/api/ImageHandler.md) - Detailed technical reference for all `ImageHandler` methods and arguments.
- [BatchImageHandler API](docs/api/BatchImageHandler.md) - Detailed technical reference for all `BatchImageHandler` methods and arguments.

### Advanced Topics
- [Batch Indexing](docs/advanced/batchimagehandler_indexing.md) - Deep dive into advanced indexing, slicing, and selecting images within a batch.

### Other
- [Architecture](docs/ARCHITECTURE.md) - Overview of the library's design philosophy, including the Context Pattern and separation of concerns.
- [Test Suite](tests/README.md) - Documentation for the test suite, explaining how to run and extend tests.

## Testing

```bash
# Run all tests
python3 run_all_tests.py

# Run specific test
python3 tests/test_basic_operations.py

# Performance benchmarks
python3 tests/benchmark_parallel.py
```

## Attributions

Images used in this project are from the Oxford-IIIT Pet Dataset by Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar, available at https://www.robots.ox.ac.uk/~vgg/data/pets/ under the Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/by-sa/4.0/). Copyright remains with the original image owners.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
