# ImageHandler API Reference

## Table of Contents

- [Overview](#overview)
- [Quick Reference](#quick-reference)
- [Class Signature](#class-signature)
- [Methods](#methods)
  - [Constructors](#constructors)
  - [I/O Operations](#io-operations)
  - [Transformations](#transformations)
  - [Annotations](#annotations)
  - [Analysis](#analysis)
  - [EXIF & Metadata](#exif--metadata)
  - [Special Methods](#special-methods)
- [Properties](#properties)
- [Related Documentation](#related-documentation)

---

## Overview

Unified interface for image operations using ImageContext. Provides a fluent API for loading, transforming, annotating, analyzing, and saving images with method chaining support.

Primary use case: Single image processing with chainable operations for computer vision and ML workflows.

**Module:** `image_toolkit.handler`
**Import:** `from image_toolkit.handler import ImageHandler`

---

## Quick Reference

| Method | Purpose | Returns |
|--------|---------|---------|
| `open(path)` | Load image from file | `ImageHandler` |
| `load(force)` | Load image data | `self` |
| `save(output_path)` | Save image to disk | `self` |
| `resize_aspect(width, height)` | Resize maintaining aspect ratio | `self` |
| `square_pad(size)` | Resize and pad to square | `self` |
| `crop(box)` | Crop to region | `self` |
| `adjust(brightness, contrast)` | Adjust image properties | `self` |
| `to_grayscale()` | Convert to grayscale | `self` |
| `draw_bbox(box, label)` | Draw bounding box | `self` |
| `draw_bboxes(boxes)` | Draw multiple bounding boxes | `self` |
| `get_stats()` | Get image statistics | `dict` |
| `to_array()` | Convert to NumPy array | `np.ndarray` |
| `to_tensor()` | Convert to PyTorch tensor | `torch.Tensor` |
| `read_exif()` | Read EXIF metadata | `dict` |

**Legend:**
- Methods returning `self` support chaining
- See [individual method documentation](#methods) for detailed parameters

---

## Class Signature

```python
class ImageHandler:
    """
    Unified interface for image operations using ImageContext.

    All operations work on a shared ImageContext.
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize ImageHandler with a file path."""
        ...
```

**Parameters:**
- `path`: Path to the image file (str or Path object)

**Attributes:**
- `img`: Current PIL Image object (None if not loaded)
- `path`: Image file path (Path object)
- `metadata`: Image metadata dictionary

---

## Methods

### Constructors

#### `__init__(path: Union[str, Path])`

Initialize ImageHandler with a file path.

**Parameters:**
- `path`: Path to the image file

**Example:**
```python
handler = ImageHandler("photo.jpg")
```

---

#### `open(path: Union[str, Path]) -> ImageHandler`

Create instance and load image (class method).

**Parameters:**
- `path`: Path to the image file

**Returns:** `ImageHandler` - Loaded ImageHandler instance

**Example:**
```python
handler = ImageHandler.open("photo.jpg")
```

---

### I/O Operations

#### `is_valid() -> bool`

Check if image file is valid.

**Returns:** `bool` - True if image can be opened

**Example:**
```python
if handler.is_valid():
    handler.load()
```

---

#### `load(force: bool = False) -> ImageHandler`

Load image into memory.

**Parameters:**
- `force`: Force reload if already loaded (default: `False`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.load()
```

---

#### `unload() -> ImageHandler`

Free memory by unloading image data.

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.unload()
```

---

#### `save(output_path: Union[str, Path], quality: int = 95) -> ImageHandler`

Save image to disk.

**Parameters:**
- `output_path`: Where to save the file
- `quality`: JPEG quality 1-100 (default: `95`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.save("output.jpg", quality=90)
```

---

### Transformations

#### `resize_aspect(width: Optional[int] = None, height: Optional[int] = None, padding_color: Tuple[int, int, int] = (0, 0, 0)) -> ImageHandler`

Resize image while maintaining aspect ratio with optional padding.

**Parameters:**
- `width`: Target width (optional)
- `height`: Target height (optional)
- `padding_color`: RGB tuple for padding (default: `(0, 0, 0)`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.resize_aspect(width=800, height=600)
```

---

#### `square_pad(size: int, fill_color: Tuple[int, int, int] = (0, 0, 0)) -> ImageHandler`

Resize and pad to square dimensions.

**Parameters:**
- `size`: Target size (width and height)
- `fill_color`: RGB tuple for padding (default: `(0, 0, 0)`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.square_pad(224)
```

---

#### `add_margin(top: int = 0, right: int = 0, bottom: int = 0, left: int = 0, color: Tuple[int, int, int] = (0, 0, 0)) -> ImageHandler`

Add colored border around image.

**Parameters:**
- `top`: Top margin in pixels (default: `0`)
- `right`: Right margin in pixels (default: `0`)
- `bottom`: Bottom margin in pixels (default: `0`)
- `left`: Left margin in pixels (default: `0`)
- `color`: RGB tuple for margin color (default: `(0, 0, 0)`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.add_margin(top=10, bottom=10, color=(255, 255, 255))
```

---

#### `pad_to_size(target_w: int, target_h: int, color: Tuple[int, int, int] = (0, 0, 0)) -> ImageHandler`

Pad to exact dimensions without resizing.

**Parameters:**
- `target_w`: Target width
- `target_h`: Target height
- `color`: RGB tuple for padding (default: `(0, 0, 0)`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.pad_to_size(1920, 1080)
```

---

#### `adjust(brightness: float = 1.0, contrast: float = 1.0) -> ImageHandler`

Adjust brightness and contrast.

**Parameters:**
- `brightness`: Brightness factor (1.0 = unchanged, >1.0 = brighter)
- `contrast`: Contrast factor (1.0 = unchanged, >1.0 = more contrast)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.adjust(brightness=1.2, contrast=1.1)
```

---

#### `filter_blur(radius: int = 2) -> ImageHandler`

Apply Gaussian blur filter.

**Parameters:**
- `radius`: Blur radius in pixels (default: `2`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.filter_blur(radius=5)
```

---

#### `to_grayscale(keep_2d: bool = False) -> ImageHandler`

Convert image to grayscale.

**Parameters:**
- `keep_2d`: Keep single channel mode if True (default: `False`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.to_grayscale()
```

---

#### `crop(box: Tuple[int, int, int, int], normalized: bool = False) -> ImageHandler`

Crop image to region.

**Parameters:**
- `box`: Bounding box (x1, y1, x2, y2)
- `normalized`: Use normalized coordinates 0.0-1.0 (default: `False`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.crop((100, 100, 500, 500))
```

---

#### `extract_crops(boxes: List[Union[Tuple, Dict]], normalized: bool = False) -> List[Image.Image]`

Extract multiple crop regions as separate images.

**Parameters:**
- `boxes`: List of bounding boxes or dicts with 'box' key
- `normalized`: Use normalized coordinates (default: `False`)

**Returns:** `List[Image.Image]` - List of cropped PIL images

**Example:**
```python
crops = handler.extract_crops([(0, 0, 100, 100), (200, 200, 300, 300)])
```

---

#### `flip_horizontal() -> ImageHandler`

Flip image horizontally (mirror).

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.flip_horizontal()
```

---

#### `flip_vertical() -> ImageHandler`

Flip image vertically.

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.flip_vertical()
```

---

#### `rotate(angle: float, **kwargs) -> ImageHandler`

Rotate image by angle in degrees.

**Parameters:**
- `angle`: Rotation angle in degrees (counterclockwise)
- `**kwargs`: Additional arguments passed to PIL rotate

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.rotate(90)
```

---

#### `to_rgba() -> ImageHandler`

Convert image to RGBA mode.

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.to_rgba()
```

---

### Annotations

#### `draw_bbox(box: Tuple[float, float, float, float], label: Optional[str] = None, **kwargs) -> ImageHandler`

Draw single bounding box on image.

**Parameters:**
- `box`: Bounding box coordinates (x1, y1, x2, y2)
- `label`: Optional text label (default: `None`)
- `**kwargs`: Additional drawing parameters (color, width, etc.)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.draw_bbox((100, 100, 300, 300), label="Object")
```

---

#### `draw_bboxes(boxes: List[Union[Tuple, Dict]], **kwargs) -> ImageHandler`

Draw multiple bounding boxes on image.

**Parameters:**
- `boxes`: List of boxes or dicts with 'box' and optional 'label' keys
- `**kwargs`: Additional drawing parameters

**Returns:** `self` (supports chaining)

**Example:**
```python
boxes = [{'box': (100, 100, 200, 200), 'label': 'A'}]
handler.draw_bboxes(boxes)
```

---

#### `draw_polygon(points: List[Tuple[int, int]], **kwargs) -> ImageHandler`

Draw polygon on image.

**Parameters:**
- `points`: List of (x, y) coordinate tuples
- `**kwargs`: Drawing parameters (outline, fill, width)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.draw_polygon([(0, 0), (100, 0), (50, 100)])
```

---

#### `draw_mask(mask: np.ndarray, **kwargs) -> ImageHandler`

Draw segmentation mask overlay.

**Parameters:**
- `mask`: Binary or multi-class mask array
- `**kwargs`: Overlay parameters (color, alpha)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.draw_mask(mask_array, color=(255, 0, 0), alpha=0.5)
```

---

#### `draw_keypoints(keypoints: List[Tuple[int, int]], **kwargs) -> ImageHandler`

Draw keypoints on image.

**Parameters:**
- `keypoints`: List of (x, y) keypoint coordinates
- `**kwargs`: Drawing parameters (radius, color)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.draw_keypoints([(100, 150), (200, 250)])
```

---

#### `draw_text(text: str, position: Tuple[int, int], **kwargs) -> ImageHandler`

Draw text on image.

**Parameters:**
- `text`: Text string to draw
- `position`: (x, y) position
- `**kwargs`: Text parameters (font, color, size)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.draw_text("Hello", (50, 50), color=(255, 255, 255))
```

---

### Analysis

#### `get_stats() -> dict`

Get comprehensive image statistics.

**Returns:** `dict` - Dictionary with width, height, mode, format, size statistics

**Example:**
```python
stats = handler.get_stats()
```

---

#### `to_array(normalize: bool = True) -> np.ndarray`

Convert image to NumPy array.

**Parameters:**
- `normalize`: Scale values to [0.0, 1.0] if True (default: `True`)

**Returns:** `np.ndarray` - Image as numpy array

**Example:**
```python
arr = handler.to_array(normalize=True)
```

---

#### `to_tensor(normalize: bool = True, device: str = "cpu")`

Convert image to PyTorch tensor.

**Parameters:**
- `normalize`: Scale to [0.0, 1.0] (default: `True`)
- `device`: Target device (default: `"cpu"`)

**Returns:** `torch.Tensor` - Image as PyTorch tensor (C, H, W)

**Example:**
```python
tensor = handler.to_tensor(device="cuda")
```

---

#### `show(title: Optional[str] = None) -> ImageHandler`

Display image using system viewer.

**Parameters:**
- `title`: Window title (default: `None`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.show(title="My Image")
```

---

#### `inspect(title: Optional[str] = None, block: bool = True)`

Display image using Matplotlib with controls.

**Parameters:**
- `title`: Plot title (default: `None`)
- `block`: Block execution until window closed (default: `True`)

**Returns:** Plot figure if `block=False`, else `self`

**Example:**
```python
handler.inspect(title="Analysis")
```

---

#### `normalize(mean: Optional[List[float]] = None, std: Optional[List[float]] = None, **kwargs) -> ImageHandler`

Apply normalization (for ML preprocessing).

**Parameters:**
- `mean`: Mean values per channel (default: ImageNet means)
- `std`: Std deviation per channel (default: ImageNet stds)
- `**kwargs`: Additional normalization parameters

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

#### `denormalize(mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> ImageHandler`

Reverse normalization.

**Parameters:**
- `mean`: Mean values used in normalization (default: ImageNet means)
- `std`: Std values used in normalization (default: ImageNet stds)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.denormalize()
```

---

#### `get_channel_stats() -> dict`

Get per-channel statistics (mean, std, min, max).

**Returns:** `dict` - Statistics for each color channel

**Example:**
```python
channel_stats = handler.get_channel_stats()
```

---

#### `is_grayscale_mode() -> bool`

Check if image mode is grayscale.

**Returns:** `bool` - True if mode is 'L' or '1'

**Example:**
```python
if handler.is_grayscale_mode():
    print("Grayscale image")
```

---

#### `is_grayscale_content(tolerance: float = 0.01) -> bool`

Analyze if image content is grayscale.

**Parameters:**
- `tolerance`: Color variation tolerance (default: `0.01`)

**Returns:** `bool` - True if effectively grayscale

**Example:**
```python
is_gray = handler.is_grayscale_content(tolerance=0.02)
```

---

#### `compute_histogram(bins: int = 256) -> dict`

Compute color histogram.

**Parameters:**
- `bins`: Number of histogram bins (default: `256`)

**Returns:** `dict` - Histogram data per channel

**Example:**
```python
hist = handler.compute_histogram(bins=128)
```

---

#### `detect_dominant_colors(n_colors: int = 5) -> List[Tuple[int, int, int]]`

Extract dominant colors using clustering.

**Parameters:**
- `n_colors`: Number of colors to extract (default: `5`)

**Returns:** `List[Tuple[int, int, int]]` - List of RGB color tuples

**Example:**
```python
colors = handler.detect_dominant_colors(n_colors=3)
```

---

### EXIF & Metadata

#### `read_exif(prefer_exiftool: bool = True) -> Dict[str, Any]`

Read EXIF metadata from image.

**Parameters:**
- `prefer_exiftool`: Use exiftool if available (default: `True`)

**Returns:** `dict` - EXIF metadata dictionary

**Example:**
```python
exif = handler.read_exif()
```

---

#### `apply_exif_orientation() -> ImageHandler`

Auto-rotate image based on EXIF orientation tag.

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.apply_exif_orientation()
```

---

#### `strip_exif() -> ImageHandler`

Remove all EXIF metadata from image.

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.strip_exif()
```

---

#### `get_exif() -> dict`

Get EXIF data from loaded image.

**Returns:** `dict` - EXIF data dictionary

**Example:**
```python
exif_data = handler.get_exif()
```

---

#### `get_metadata() -> dict`

Get comprehensive metadata including EXIF, format, size.

**Returns:** `dict` - Complete metadata dictionary

**Example:**
```python
metadata = handler.get_metadata()
```

---

#### `save_with_metadata(output_path: Union[str, Path], **kwargs) -> ImageHandler`

Save image while preserving metadata.

**Parameters:**
- `output_path`: Output file path
- `**kwargs`: Additional save parameters

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.save_with_metadata("output.jpg")
```

---

#### `format_convert(target_format: str) -> ImageHandler`

Convert image to different format.

**Parameters:**
- `target_format`: Target format (e.g., 'PNG', 'JPEG')

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.format_convert("PNG")
```

---

#### `copy() -> ImageHandler`

Create deep copy of handler with duplicated image data.

**Returns:** `ImageHandler` - Independent copy

**Example:**
```python
copy = handler.copy()
```

---

#### `reset(force_reload: bool = True) -> ImageHandler`

Reset to original state by reloading from disk.

**Parameters:**
- `force_reload`: Force reload from disk (default: `True`)

**Returns:** `self` (supports chaining)

**Example:**
```python
handler.reset()
```

---

### Special Methods

#### `__repr__() -> str`

Developer-friendly string representation.

**Returns:** `str` - String like "ImageHandler('photo.jpg', loaded=True, valid=True)"

**Example:**
```python
repr(handler)
```

---

#### `__str__() -> str`

User-friendly string with detailed information.

**Returns:** `str` - Formatted string with image details

**Example:**
```python
str(handler)
```

---

#### `__enter__() -> ImageHandler`

Enter context manager.

**Returns:** `self` for use in with statement

**Example:**
```python
with ImageHandler.open("photo.jpg") as handler:
    handler.resize_aspect(width=800)
```

---

#### `__exit__(exc_type, exc_val, exc_tb) -> None`

Exit context manager and cleanup resources.

**Parameters:**
- `exc_type`: Exception type if raised
- `exc_val`: Exception value if raised
- `exc_tb`: Exception traceback if raised

**Example:**
```python
with ImageHandler.open("photo.jpg") as handler:
    handler.show()
# Image automatically unloaded here
```

---

## Properties

#### `img`

**Type:** `Optional[Image.Image]`
**Access:** Read-only

Get current PIL Image object.

**Example:**
```python
pil_image = handler.img
```

---

#### `path`

**Type:** `Path`
**Access:** Read-only

Get image file path.

**Example:**
```python
file_path = handler.path
```

---

#### `metadata`

**Type:** `Dict`
**Access:** Read-only

Get image metadata dictionary.

**Example:**
```python
meta = handler.metadata
```

---

## Related Documentation

- **[ImageHandler Usage Guide](../guide/ImageHandler.md)** - Learn patterns, workflows, and best practices
- **[BatchImageHandler API](./BatchImageHandler.md)** - Batch processing for multiple images
- **[Getting Started Tutorial](../tutorials/getting_started.md)** - Step-by-step introduction

## See Also

- `BatchImageHandler` - Process multiple images in parallel
- `ImageContext` - Low-level image context management
