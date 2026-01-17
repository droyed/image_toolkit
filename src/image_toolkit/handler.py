"""
ImageHandler - Unified interface using ImageContext.
"""
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Any
from PIL import ExifTags
import numpy as np

from .core import ImageContext, ImageLoader, ImageTransformer, ImageAnnotator, ImageAnalyzer


class ImageHandler:
    """
    Unified interface for image operations using ImageContext.

    All operations work on a shared ImageContext.

    Example:
        >>> handler = ImageHandler.open("photo.jpg")
        >>> handler.resize_aspect(width=800).adjust(brightness=1.2).save("output.jpg")
        >>> stats = handler.get_stats()
    """

    def __init__(self, path: Union[str, Path]):
        """
        Initialize ImageHandler with a file path.

        Args:
            path: Path to the image file
        """
        self._ctx = ImageContext.from_path(path)

    @classmethod
    def open(cls, path: Union[str, Path]) -> 'ImageHandler':
        """
        Create instance and load image.

        Args:
            path: Path to the image file

        Returns:
            Loaded ImageHandler instance

        Example:
            >>> handler = ImageHandler.open("photo.jpg")
        """
        handler = cls(path)
        return handler.load()

    @property
    def img(self) -> Optional[Image.Image]:
        """
        Get current PIL Image object.

        Returns:
            PIL Image or None if not loaded
        """
        return self._ctx.img

    @property
    def path(self) -> Path:
        """Get image file path."""
        return self._ctx.path

    @property
    def metadata(self) -> Dict:
        """Get image metadata."""
        return self._ctx.metadata

    def _ensure_loaded(self):
        """Lazy load if not already loaded."""
        if not self._ctx.is_loaded():
            ImageLoader.load(self._ctx)

    # ========================================================================
    # I/O OPERATIONS
    # ========================================================================

    def is_valid(self) -> bool:
        """Check if image file is valid."""
        return ImageLoader.is_valid(self._ctx)

    def load(self, force: bool = False) -> 'ImageHandler':
        """Load image."""
        ImageLoader.load(self._ctx, force=force)
        return self

    def unload(self) -> 'ImageHandler':
        """Free memory."""
        ImageLoader.unload(self._ctx)
        return self

    def save(self, output_path: Union[str, Path], quality: int = 95) -> 'ImageHandler':
        """
        Save image.

        Args:
            output_path: Where to save
            quality: JPEG quality (1-100)

        Returns:
            self for method chaining
        """
        self._ensure_loaded()
        ImageLoader.save(self._ctx, output_path, quality=quality)
        return self

    # ========================================================================
    # TRANSFORMATIONS
    # ========================================================================

    def resize_aspect(self, width: Optional[int] = None, height: Optional[int] = None,
                      padding_color: Tuple[int, int, int] = (0, 0, 0)) -> 'ImageHandler':
        """Resize maintaining aspect ratio."""
        self._ensure_loaded()
        ImageTransformer.resize_aspect(self._ctx, width, height, padding_color)
        return self

    def square_pad(self, size: int, fill_color: Tuple[int, int, int] = (0, 0, 0)) -> 'ImageHandler':
        """Resize and pad to square."""
        self._ensure_loaded()
        ImageTransformer.square_pad(self._ctx, size, fill_color)
        return self

    def add_margin(self, top: int = 0, right: int = 0, bottom: int = 0, left: int = 0,
                   color: Tuple[int, int, int] = (0, 0, 0)) -> 'ImageHandler':
        """Add colored border."""
        self._ensure_loaded()
        ImageTransformer.add_margin(self._ctx, top, right, bottom, left, color)
        return self

    def pad_to_size(self, target_w: int, target_h: int,
                    color: Tuple[int, int, int] = (0, 0, 0)) -> 'ImageHandler':
        """Pad to exact dimensions."""
        self._ensure_loaded()
        ImageTransformer.pad_to_size(self._ctx, target_w, target_h, color)
        return self

    def adjust(self, brightness: float = 1.0, contrast: float = 1.0) -> 'ImageHandler':
        """Adjust brightness and contrast."""
        self._ensure_loaded()
        ImageTransformer.adjust(self._ctx, brightness, contrast)
        return self

    def filter_blur(self, radius: int = 2) -> 'ImageHandler':
        """Apply Gaussian blur."""
        self._ensure_loaded()
        ImageTransformer.filter_blur(self._ctx, radius)
        return self

    def to_grayscale(self, keep_2d: bool = False) -> 'ImageHandler':
        """Convert to grayscale."""
        self._ensure_loaded()
        ImageTransformer.to_grayscale(self._ctx, keep_2d)
        return self

    def crop(self, box: Tuple[int, int, int, int], normalized: bool = False) -> 'ImageHandler':
        """Crop to region."""
        self._ensure_loaded()
        ImageTransformer.crop(self._ctx, box, normalized)
        return self

    def extract_crops(self, boxes: List[Union[Tuple, Dict]], normalized: bool = False) -> List[Image.Image]:
        """Extract multiple crops."""
        self._ensure_loaded()
        return ImageTransformer.extract_crops(self._ctx, boxes, normalized)

    def flip_horizontal(self) -> 'ImageHandler':
        """Flip horizontally."""
        self._ensure_loaded()
        ImageTransformer.flip_horizontal(self._ctx)
        return self

    def flip_vertical(self) -> 'ImageHandler':
        """Flip vertically."""
        self._ensure_loaded()
        ImageTransformer.flip_vertical(self._ctx)
        return self

    def rotate(self, angle: float, **kwargs) -> 'ImageHandler':
        """Rotate by angle."""
        self._ensure_loaded()
        ImageTransformer.rotate(self._ctx, angle, **kwargs)
        return self

    def to_rgba(self) -> 'ImageHandler':
        """Convert to RGBA."""
        self._ensure_loaded()
        ImageTransformer.to_rgba(self._ctx)
        return self

    # ========================================================================
    # ANNOTATIONS
    # ========================================================================

    def draw_bbox(self, box: Tuple[float, float, float, float],
                  label: Optional[str] = None, **kwargs) -> 'ImageHandler':
        """Draw bounding box."""
        self._ensure_loaded()
        ImageAnnotator.draw_bbox(self._ctx, box, label, **kwargs)
        return self

    def draw_bboxes(self, boxes: List[Union[Tuple, Dict]], **kwargs) -> 'ImageHandler':
        """Draw multiple bounding boxes."""
        self._ensure_loaded()
        ImageAnnotator.draw_bboxes(self._ctx, boxes, **kwargs)
        return self

    def draw_polygon(self, points: List[Tuple[int, int]], **kwargs) -> 'ImageHandler':
        """Draw polygon."""
        self._ensure_loaded()
        ImageAnnotator.draw_polygon(self._ctx, points, **kwargs)
        return self

    def draw_mask(self, mask: np.ndarray, **kwargs) -> 'ImageHandler':
        """Draw mask overlay."""
        self._ensure_loaded()
        ImageAnnotator.draw_mask(self._ctx, mask, **kwargs)
        return self

    def draw_keypoints(self, keypoints: List[Tuple[int, int]], **kwargs) -> 'ImageHandler':
        """Draw keypoints."""
        self._ensure_loaded()
        ImageAnnotator.draw_keypoints(self._ctx, keypoints, **kwargs)
        return self

    def draw_text(self, text: str, position: Tuple[int, int], **kwargs) -> 'ImageHandler':
        """Draw text."""
        self._ensure_loaded()
        ImageAnnotator.draw_text(self._ctx, text, position, **kwargs)
        return self

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    def get_stats(self) -> dict:
        """Get image statistics."""
        self._ensure_loaded()
        return ImageAnalyzer.get_stats(self._ctx)

    def to_array(self, normalize: bool = True) -> np.ndarray:
        """Convert to NumPy array."""
        self._ensure_loaded()
        return ImageAnalyzer.to_numpy(self._ctx, normalized=normalize)

    def to_tensor(self, normalize: bool = True, device: str = "cpu"):
        """Convert to PyTorch tensor."""
        self._ensure_loaded()
        return ImageAnalyzer.to_tensor(self._ctx, normalized=normalize, device=device, channels_first=True)

    def show(self, title: Optional[str] = None) -> 'ImageHandler':
        """Display using system viewer."""
        self._ensure_loaded()
        ImageAnalyzer.show(self._ctx, title=title)
        return self

    def inspect(self, title: Optional[str] = None, block: bool = True):
        """Display using Matplotlib."""
        self._ensure_loaded()
        result = ImageAnalyzer.inspect(self._ctx, title=title, block=block)
        return result if block else self

    def normalize(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None, **kwargs) -> 'ImageHandler':
        """Apply normalization."""
        self._ensure_loaded()
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        ImageAnalyzer.normalize(self._ctx, mean, std, **kwargs)
        return self

    def denormalize(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> 'ImageHandler':
        """Reverse normalization."""
        self._ensure_loaded()
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        ImageAnalyzer.denormalize(self._ctx, mean, std)
        return self

    def get_channel_stats(self) -> dict:
        """Get per-channel statistics."""
        self._ensure_loaded()
        return ImageAnalyzer.get_channel_stats(self._ctx)

    def is_grayscale_mode(self) -> bool:
        """Check if image mode is grayscale."""
        self._ensure_loaded()
        return ImageTransformer.is_grayscale_mode(self._ctx)

    def is_grayscale_content(self, tolerance: float = 0.01) -> bool:
        """Analyze if image content is grayscale."""
        self._ensure_loaded()
        return ImageAnalyzer.is_grayscale_content(self._ctx, tolerance)

    def compute_histogram(self, bins: int = 256) -> dict:
        """Compute histogram."""
        self._ensure_loaded()
        return ImageAnalyzer.compute_histogram(self._ctx, bins)

    def detect_dominant_colors(self, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors."""
        self._ensure_loaded()
        return ImageAnalyzer.detect_dominant_colors(self._ctx, n_colors)

    # ========================================================================
    # EXIF & METADATA
    # ========================================================================

    def read_exif(self, prefer_exiftool: bool = True) -> Dict[str, Any]:
        """Read EXIF metadata."""
        exif_data = {}
        source = "none"

        if prefer_exiftool:
            try:
                import exiftool
                with exiftool.ExifToolHelper() as et:
                    metadata = et.get_metadata([str(self.path)])
                    if metadata and len(metadata) > 0:
                        entry = metadata[0]
                        for tag, value in entry.items():
                            if tag == "SourceFile":
                                continue
                            if isinstance(value, bytes):
                                exif_data[tag] = value.decode('utf-8', errors='ignore')
                            elif isinstance(value, (np.integer, np.floating)):
                                exif_data[tag] = value.item()
                            elif isinstance(value, np.ndarray):
                                exif_data[tag] = value.tolist()
                            else:
                                exif_data[tag] = value
                        source = "exiftool"
                        print(f"✓ Using ExifTool for EXIF metadata extraction")
            except (ImportError, Exception):
                pass

        if source == "none":
            if prefer_exiftool:
                print(f"⚠️ ExifTool not available, falling back to Pillow EXIF reader")
            try:
                with Image.open(self.path) as im:
                    exif_dict = im.getexif()
                    if exif_dict:
                        for tag_id, value in exif_dict.items():
                            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                            if tag_id == 34853:  # GPSInfo
                                gps_data = {}
                                if isinstance(value, dict):
                                    for gps_tag_id, gps_value in value.items():
                                        gps_tag_name = ExifTags.GPSTAGS.get(gps_tag_id, gps_tag_id)
                                        gps_data[gps_tag_name] = gps_value
                                    exif_data["GPSInfo"] = gps_data
                                else:
                                    exif_data[tag_name] = value
                            else:
                                if isinstance(value, bytes):
                                    exif_data[tag_name] = value.decode('utf-8', errors='ignore')
                                elif isinstance(value, (np.integer, np.floating)):
                                    exif_data[tag_name] = value.item()
                                elif isinstance(value, np.ndarray):
                                    exif_data[tag_name] = value.tolist()
                                else:
                                    exif_data[tag_name] = value
                        source = "pillow"
            except Exception:
                pass

        self.metadata["exif"] = exif_data
        self.metadata["exif_source"] = source
        return exif_data

    def apply_exif_orientation(self) -> 'ImageHandler':
        """Auto-rotate based on EXIF orientation."""
        self._ensure_loaded()
        ImageLoader.apply_exif_orientation(self._ctx)
        return self

    def strip_exif(self) -> 'ImageHandler':
        """Remove EXIF metadata."""
        self._ensure_loaded()
        ImageLoader.strip_exif(self._ctx)
        return self

    def get_exif(self) -> dict:
        """Get EXIF data."""
        self._ensure_loaded()
        return ImageLoader.get_exif(self._ctx)

    def get_metadata(self) -> dict:
        """Get comprehensive metadata."""
        self._ensure_loaded()
        return ImageLoader.get_metadata(self._ctx)

    def save_with_metadata(self, output_path: Union[str, Path], **kwargs) -> 'ImageHandler':
        """Save with metadata preservation."""
        self._ensure_loaded()
        ImageLoader.save_with_metadata(self._ctx, output_path, **kwargs)
        return self

    def format_convert(self, target_format: str) -> 'ImageHandler':
        """Convert image format."""
        self._ensure_loaded()
        ImageLoader.format_convert(self._ctx, target_format)
        return self

    def copy(self) -> 'ImageHandler':
        """Create deep copy."""
        self._ensure_loaded()
        new_handler = ImageHandler(self.path)
        new_handler._ctx = ImageLoader.copy(self._ctx)
        return new_handler

    def reset(self, force_reload: bool = True) -> 'ImageHandler':
        """Reset to original state."""
        ImageLoader.reset(self._ctx, force_reload)
        return self

    # ========================================================================
    # SPECIAL METHODS (DUNDER METHODS)
    # ========================================================================

    def __repr__(self) -> str:
        """
        Developer-friendly representation.

        Returns:
            String representation suitable for debugging

        Example:
            >>> handler = ImageHandler.open("photo.jpg")
            >>> repr(handler)
            "ImageHandler('photo.jpg', loaded=True, valid=True)"
        """
        filename = self._ctx.path.name
        is_loaded = self._ctx.is_loaded()
        is_valid = self.is_valid()

        return f"ImageHandler('{filename}', loaded={is_loaded}, valid={is_valid})"

    def __str__(self) -> str:
        """
        User-friendly string representation with detailed information.

        Format: "filename | format | mode | dimensions | filesize | load_status | validity"

        Returns:
            Formatted string with image details

        Example:
            >>> handler = ImageHandler.open("photo.jpg")
            >>> str(handler)
            "photo.jpg | JPEG | RGB | 1920x1080 | 2.45MB | Loaded | Valid"

            >>> handler = ImageHandler("photo.jpg")  # Not loaded yet
            >>> str(handler)
            "photo.jpg | Unknown | Unknown | Unknown | 2.45MB | Not Loaded | Valid"
        """
        components = []

        # 1. Filename (always available)
        components.append(self._ctx.path.name)

        # 2. Image format (only when loaded)
        if self._ctx.is_loaded() and self._ctx.img:
            # Try to get format from PIL Image, fallback to file extension
            img_format = self._ctx.img.format
            if img_format:
                components.append(img_format)
            else:
                # Fallback to file extension
                ext = self._ctx.path.suffix.upper()[1:]  # Remove the dot
                if ext in ('JPG', 'JPEG', 'PNG', 'GIF', 'BMP', 'TIFF', 'WEBP'):
                    components.append('JPEG' if ext == 'JPG' else ext)
                else:
                    components.append("Unknown")
        else:
            components.append("Unknown")

        # 3. Image mode (only when loaded)
        if self._ctx.mode:
            components.append(self._ctx.mode)
        else:
            components.append("Unknown")

        # 4. Dimensions (only when loaded)
        if self._ctx.size:
            width, height = self._ctx.size
            components.append(f"{width}x{height}")
        else:
            components.append("Unknown")

        # 5. File size (always available if file exists)
        file_size_str = self._format_file_size()
        components.append(file_size_str)

        # 6. Load status
        load_status = "Loaded" if self._ctx.is_loaded() else "Not Loaded"
        components.append(load_status)

        # 7. Validity status
        validity_status = "Valid" if self.is_valid() else "Invalid"
        components.append(validity_status)

        return " | ".join(components)

    def __enter__(self) -> 'ImageHandler':
        """
        Enter context manager.

        Returns:
            self for use in with statement

        Example:
            >>> with ImageHandler.open("photo.jpg") as handler:
            ...     handler.resize_aspect(width=800).save("output.jpg")
            # Automatically unloads after with block
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager and cleanup resources.

        Automatically unloads the image to free memory, regardless of
        whether an exception occurred.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised

        Example:
            >>> with ImageHandler.open("photo.jpg") as handler:
            ...     handler.show()
            # Image automatically unloaded here
        """
        # Always unload to free memory
        self.unload()
        # Return None to propagate exceptions (default behavior)

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _format_file_size(self) -> str:
        """
        Format file size in human-readable format (MB with 2 decimal places).

        Returns:
            Formatted file size string (e.g., "2.45MB")

        Example:
            >>> handler._format_file_size()
            "2.45MB"
        """
        try:
            # Get file size in bytes without loading image
            size_bytes = self._ctx.path.stat().st_size
            # Convert to MB with 2 decimal places
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f}MB"
        except (OSError, FileNotFoundError):
            # File doesn't exist or can't access
            return "Unknown"