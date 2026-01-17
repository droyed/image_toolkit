# Changelog

All notable changes to image_toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-17

### Added

#### Core Image Processing
- Fluent API design for intuitive image manipulation workflows
- `ImageHandler` class for single-image operations with method chaining
- `BatchImageHandler` class for parallel batch processing of multiple images
- Context manager support (`with` statement) for automatic resource cleanup
- Automatic format detection and validation

#### Image Transformations
- `resize()` - Resize images with custom dimensions
- `resize_aspect()` - Aspect-ratio preserving resize with intelligent cropping
- `thumbnail()` - Generate thumbnails with multiple sizing strategies (fit, cover, stretch)
- `crop()` - Extract image regions with coordinate-based cropping
- `rotate()` - Rotate images by arbitrary angles with automatic dimension adjustment
- `flip()` - Horizontal and vertical image flipping
- `adjust_brightness()` - Brightness adjustment with clamping
- `adjust_contrast()` - Contrast enhancement/reduction
- `adjust_saturation()` - Color saturation control
- `grayscale()` - Color to grayscale conversion
- `blur()` - Gaussian blur with configurable radius

#### Format Conversion & Compression
- Support for JPEG, PNG, WebP, TIFF, BMP formats
- `convert()` - Format conversion with quality control
- `optimize()` - File size optimization with configurable quality/compression
- Progressive JPEG encoding support
- WebP format support for modern web applications

#### Analysis & Information
- `get_info()` - Extract comprehensive image metadata (dimensions, format, mode, size)
- `analyze_duplicates()` - Perceptual hash-based duplicate detection across image sets
- Statistical analysis capabilities for image properties

#### Machine Learning Integration
- `to_tensor()` - Convert PIL Images to PyTorch tensors with normalization
- `from_tensor()` - Convert PyTorch tensors back to PIL Images
- Support for ImageNet normalization presets
- Custom normalization parameter support
- Automatic channel reordering (CHW ↔ HWC)

#### Performance Features
- TurboJPEG acceleration for JPEG operations (3-5x speedup over PIL)
- Parallel batch processing with configurable worker pools
- Memory-efficient streaming for large image sets
- Lazy evaluation support for transformation pipelines
- Intelligent fallback mechanisms when TurboJPEG unavailable

#### Error Handling & Validation
- Comprehensive error handling with descriptive messages
- Input validation for all transformation parameters
- Graceful degradation when optional dependencies missing
- Type hints throughout codebase for better IDE support

### Performance

- **JPEG Operations:** 3-5x speedup using TurboJPEG vs PIL for decode/encode operations
- **Batch Processing:** Near-linear scaling with CPU cores for parallel image processing
- **Memory Efficiency:** Streaming batch processing prevents memory exhaustion on large datasets
- **Perceptual Hashing:** Fast duplicate detection using pHash algorithm (average hashing)

### Documentation

- Comprehensive README.md with quick start guide and feature overview
- Detailed README_detailed.md with extensive examples and use cases
- Architecture documentation (docs/ARCHITECTURE.md) explaining design philosophy
- API reference documentation for all public methods
- Installation instructions for core and optional dependencies
- Performance comparison benchmarks
- Machine learning integration examples
- Batch processing usage patterns

### Testing

- 15+ test modules covering all major functionality:
  - test_basic.py - Core functionality tests
  - test_transformations.py - Image transformation operations
  - test_batch.py - Batch processing and parallelization
  - test_formats.py - Format conversion and compatibility
  - test_ml_integration.py - PyTorch tensor operations
  - test_analysis.py - Image analysis and duplicate detection
  - test_error_handling.py - Error conditions and edge cases
  - test_context_manager.py - Resource management
  - test_turbojpeg.py - TurboJPEG acceleration
  - test_optimization.py - Image optimization
  - Additional specialized test suites
- High test coverage across all modules
- Integration tests for real-world workflows
- Performance benchmarks

### Dependencies

#### Core Dependencies
- Python 3.8+
- Pillow (PIL) 10.0.0+ - Image processing foundation
- NumPy - Numerical operations and array handling

#### Optional Dependencies
- PyTurboJPEG - Hardware-accelerated JPEG operations (3-5x faster)
- PyTorch - Machine learning tensor integration
- imagehash - Perceptual hashing for duplicate detection

[Unreleased]: https://github.com/yourusername/image_toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/image_toolkit/releases/tag/v0.1.0
