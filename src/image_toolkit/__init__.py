"""
Image Toolkit - Modular image processing library.

Main components:
- ImageHandler: Unified interface for single image operations
- BatchImageHandler: Efficient batch processing with parallel execution
- Core components: ImageLoader, ImageTransformer, ImageAnnotator, ImageAnalyzer

Example (single image):
    >>> from image_toolkit import ImageHandler
    >>> handler = ImageHandler.open("photo.jpg")
    >>> handler.resize_aspect(width=800).adjust(brightness=1.2).save("output.jpg")

Example (batch processing):
    >>> from image_toolkit import BatchImageHandler
    >>> batch = BatchImageHandler.from_directory("photos/", "*.jpg")
    >>> batch.filter_valid().resize(width=800).save("output/")

"""

# Main convenience imports
from .handler import ImageHandler
from .batch_handler import BatchImageHandler

# Optional: expose core components for advanced users
from .core import ImageLoader, ImageTransformer, ImageAnnotator, ImageAnalyzer

__version__ = '0.1.0'

__all__ = [
    'ImageHandler',
    'BatchImageHandler',
    'ImageLoader',
    'ImageTransformer',
    'ImageAnnotator',
    'ImageAnalyzer'
]
