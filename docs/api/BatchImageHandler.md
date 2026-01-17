# BatchImageHandler API Reference

## Table of Contents

- [Overview](#overview)
- [Quick Reference](#quick-reference)
- [Class Signature](#class-signature)
- [Methods](#methods)
  - [Constructors](#constructors)
  - [Filtering Methods](#filtering-methods)
  - [Core Processing Methods](#core-processing-methods)
  - [Convenience Batch Operations](#convenience-batch-operations)
  - [Save Operations](#save-operations)
  - [Analysis & Statistics](#analysis--statistics)
  - [Dataset Operations](#dataset-operations)
  - [Grid Visualization](#grid-visualization)
  - [Memory Management & Utilities](#memory-management--utilities)
  - [Special Methods](#special-methods)
- [Related Documentation](#related-documentation)

---

## Overview

Batch processing for multiple images with parallel execution support. Manages a collection of ImageContext instances and provides efficient batch operations including filtering, transformations, statistics, and ML dataset conversion.

**Module:** `image_toolkit.batch_handler`
**Import:** `from image_toolkit.batch_handler import BatchImageHandler`

---

## Quick Reference

| Method | Purpose | Returns |
|--------|---------|---------|
| `__init__(paths)` | Initialize with image paths | `None` |
| `from_directory(dir_path, pattern)` | Create batch from directory | `BatchImageHandler` |
| `from_glob(pattern)` | Create batch from glob pattern | `BatchImageHandler` |
| `filter_valid(parallel)` | Remove invalid images | `self` |
| `filter_by_size(min_width, max_width, ...)` | Filter by dimensions | `self` |
| `filter_by_aspect_ratio(min_ratio, max_ratio)` | Filter by aspect ratio | `self` |
| `filter_by_file_size(min_size, max_size)` | Filter by file size | `self` |
| `analyze_duplicates(hash_method, threshold)` | Analyze duplicate groups | `dict` |
| `filter_duplicates(hash_method, threshold)` | Remove duplicates | `self` |
| `sample(n, random_sample)` | Sample n images | `self` |
| `map(func, parallel)` | Apply custom function | `self` |
| `apply_transform(transform_name, **kwargs)` | Apply ImageHandler method | `self` |
| `chain_transforms(transforms)` | Apply multiple transforms | `self` |
| `resize(width, height)` | Resize all images | `self` |
| `adjust(brightness, contrast)` | Adjust brightness/contrast | `self` |
| `to_grayscale(keep_2d)` | Convert to grayscale | `self` |
| `save(output_dir, prefix, suffix)` | Save all images | `self` |
| `get_batch_stats()` | Get aggregated statistics | `dict` |
| `detect_outliers(metric, threshold)` | Detect outlier images | `List[Path]` |
| `verify_uniformity(check_size, ...)` | Check batch uniformity | `dict` |
| `to_dataset(format, normalized)` | Convert to ML dataset | `array/tensor` |
| `split_dataset(train_ratio, val_ratio, ...)` | Split into train/val/test | `dict` |
| `create_grid(rows, cols, cell_size)` | Create grid visualization | `Image` |
| `unload()` | Free memory | `self` |
| `process_in_chunks(chunk_size, func)` | Process in chunks | `self` |

**Legend:**
- Methods returning `self` support chaining
- See [individual method documentation](#methods) for detailed parameters

---

## Class Signature

```python
class BatchImageHandler:
    """
    Batch processing for multiple images with parallel execution support.

    Manages a collection of ImageContext instances and provides efficient batch operations
    including filtering, transformations, statistics, and ML dataset conversion.
    """

    def __init__(self, paths: List[Union[str, Path]]):
        """Initialize BatchImageHandler with list of image paths."""
        ...
```

**Parameters:**
- `paths`: List of paths to image files

**Attributes:**
- `_contexts`: List of ImageContext instances
- `_errors`: List of (path, exception) tuples encountered during processing
- `_progress_callback`: Optional progress callback function
- `_executor`: ParallelExecutor for parallel processing

---

## Methods

### Constructors

#### `__init__(paths: List[Union[str, Path]]) -> None`

Initialize BatchImageHandler with list of image paths.

**Parameters:**
- `paths`: List of paths to image files

**Example:**
```python
paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
batch = BatchImageHandler(paths)
```

---

#### `from_directory(dir_path: Union[str, Path], pattern: str = "*") -> BatchImageHandler`

Create batch from all images in a directory.

**Parameters:**
- `dir_path`: Directory path
- `pattern`: File pattern (e.g., "*.jpg", "*.png", "jpg") (default: `"*"`)

**Returns:** `BatchImageHandler` instance

**Example:**
```python
batch = BatchImageHandler.from_directory("photos/", "*.jpg")
```

---

#### `from_glob(pattern: str) -> BatchImageHandler`

Create batch from glob pattern.

**Parameters:**
- `pattern`: Glob pattern (e.g., "photos/**/*.jpg")

**Returns:** `BatchImageHandler` instance

**Example:**
```python
batch = BatchImageHandler.from_glob("photos/**/*.jpg")
```

---

### Filtering Methods

#### `filter_valid(parallel: bool = True) -> BatchImageHandler`

Remove corrupted/invalid images from the batch.

**Parameters:**
- `parallel`: Use parallel processing (default: `True`)

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.filter_valid()
```

---

#### `filter_by_size(min_width: Optional[int] = None, max_width: Optional[int] = None, min_height: Optional[int] = None, max_height: Optional[int] = None) -> BatchImageHandler`

Filter images by dimensions.

**Parameters:**
- `min_width`: Minimum width in pixels (optional)
- `max_width`: Maximum width in pixels (optional)
- `min_height`: Minimum height in pixels (optional)
- `max_height`: Maximum height in pixels (optional)

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.filter_by_size(min_width=800, max_width=2000, min_height=600)
```

---

#### `filter_by_aspect_ratio(min_ratio: Optional[float] = None, max_ratio: Optional[float] = None) -> BatchImageHandler`

Filter by aspect ratio (width/height).

**Parameters:**
- `min_ratio`: Minimum aspect ratio (optional)
- `max_ratio`: Maximum aspect ratio (optional)

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.filter_by_aspect_ratio(min_ratio=1.3)
```

---

#### `filter_by_file_size(min_size: Optional[Union[int, str]] = None, max_size: Optional[Union[int, str]] = None) -> BatchImageHandler`

Filter images by file size on disk.

**Parameters:**
- `min_size`: Minimum file size (int: bytes, str: "50KB", "1.5MB", "2GB") (optional)
- `max_size`: Maximum file size (same format as min_size) (optional)

**Returns:** `self` (supports chaining)

**Raises:**
- `ValueError`: If size format is invalid

**Example:**
```python
batch.filter_by_file_size(min_size='50KB', max_size='5MB')
```

---

#### `analyze_duplicates(hash_method: str = 'dhash', hash_size: int = 8, threshold: Optional[int] = None, parallel: bool = True) -> Dict[str, Any]`

Analyze images to identify duplicate groups and singleton images.

**Parameters:**
- `hash_method`: Hashing algorithm ('dhash', 'phash', 'ahash', 'whash') (default: `'dhash'`)
- `hash_size`: Hash size (8 = 64-bit, 16 = 256-bit) (default: `8`)
- `threshold`: Hamming distance threshold (None = use recommended default) (optional)
- `parallel`: Use parallel processing (default: `True`)

**Returns:** `dict` containing duplicate_groups, singleton_groups, hash_map, and stats

**Example:**
```python
analysis = batch.analyze_duplicates(hash_method='dhash', threshold=5)
```

---

#### `filter_duplicates(hash_method: str = 'dhash', hash_size: int = 8, threshold: Optional[int] = None, keep: str = 'first', parallel: bool = True) -> BatchImageHandler`

Filter duplicate using perceptual hashing and keep one from each group.

**Parameters:**
- `hash_method`: Hashing algorithm ('dhash', 'phash', 'ahash', 'whash') (default: `'dhash'`)
- `hash_size`: Hash size (8 = 64-bit, 16 = 256-bit) (default: `8`)
- `threshold`: Maximum Hamming distance to consider as duplicate (optional)
- `keep`: Which duplicate to keep ('first', 'last', 'largest', 'smallest') (default: `'first'`)
- `parallel`: Use parallel processing (default: `True`)

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.filter_duplicates(hash_method='dhash', threshold=8)
```

---

#### `sample(n: int, random_sample: bool = True) -> BatchImageHandler`

Sample n images from the batch.

**Parameters:**
- `n`: Number of images to sample
- `random_sample`: If True, random sampling; if False, takes first n (default: `True`)

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.sample(50, random_sample=True)
```

---

### Core Processing Methods

#### `map(func: Callable[[ImageHandler], ImageHandler], parallel: bool = True, workers: Optional[int] = None) -> BatchImageHandler`

Apply custom function to each image.

**Parameters:**
- `func`: Function that takes ImageHandler and returns ImageHandler
- `parallel`: Use parallel processing (default: `True`)
- `workers`: Number of worker threads (auto if None) (optional)

**Returns:** `self` (supports chaining)

**Example:**
```python
def my_pipeline(img):
    img.resize_aspect(width=512)
    return img
batch.map(my_pipeline, parallel=True)
```

---

#### `apply_transform(transform_name: str, parallel: bool = True, workers: Optional[int] = None, **kwargs) -> BatchImageHandler`

Apply any ImageHandler method to all images.

**Parameters:**
- `transform_name`: Name of the ImageHandler method
- `parallel`: Use parallel processing (default: `True`)
- `workers`: Number of worker threads (optional)
- `**kwargs`: Arguments to pass to the method

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.apply_transform('resize_aspect', width=800)
```

---

#### `chain_transforms(transforms: List[Tuple[str, dict]], parallel: bool = True) -> BatchImageHandler`

Apply multiple transformations in sequence.

**Parameters:**
- `transforms`: List of (method_name, kwargs) tuples
- `parallel`: Use parallel processing (default: `True`)

**Returns:** `self` (supports chaining)

**Example:**
```python
transforms = [
    ('resize_aspect', {'width': 800}),
    ('adjust', {'brightness': 1.2})
]
batch.chain_transforms(transforms)
```

---

### Convenience Batch Operations

#### `resize(width: Optional[int] = None, height: Optional[int] = None, parallel: bool = True, **kwargs) -> BatchImageHandler`

Resize all images.

**Parameters:**
- `width`: Target width (optional)
- `height`: Target height (optional)
- `parallel`: Use parallel processing (default: `True`)
- `**kwargs`: Additional arguments for resize_aspect

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.resize(width=800)
```

---

#### `adjust(brightness: float = 1.0, contrast: float = 1.0, parallel: bool = True) -> BatchImageHandler`

Adjust brightness/contrast for all images.

**Parameters:**
- `brightness`: Brightness factor (default: `1.0`)
- `contrast`: Contrast factor (default: `1.0`)
- `parallel`: Use parallel processing (default: `True`)

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.adjust(brightness=1.2, contrast=1.1)
```

---

#### `to_grayscale(keep_2d: bool = False, parallel: bool = True) -> BatchImageHandler`

Convert all images to grayscale.

**Parameters:**
- `keep_2d`: Keep single channel mode (default: `False`)
- `parallel`: Use parallel processing (default: `True`)

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.to_grayscale(keep_2d=False)
```

---

### Save Operations

#### `save(output_dir: Union[str, Path], prefix: str = "", suffix: str = "", quality: int = 95, overwrite: bool = False) -> BatchImageHandler`

Save all processed images.

**Parameters:**
- `output_dir`: Output directory path
- `prefix`: Prefix to add to filenames (default: `""`)
- `suffix`: Suffix to add to filenames (before extension) (default: `""`)
- `quality`: JPEG quality (1-100) (default: `95`)
- `overwrite`: If True, overwrites existing files (default: `False`)

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.save("output/", prefix="processed_", suffix="_800w")
```

---

### Analysis & Statistics

#### `get_batch_stats() -> dict`

Aggregate statistics across all images.

**Returns:** `dict` with aggregated statistics

**Example:**
```python
stats = batch.get_batch_stats()
```

---

#### `detect_outliers(metric: str = 'size', threshold: float = 2.0) -> List[Path]`

Detect outlier images based on statistics.

**Parameters:**
- `metric`: Metric to use ('size', 'width', 'height', 'aspect_ratio') (default: `'size'`)
- `threshold`: Number of standard deviations (default: `2.0`)

**Returns:** `List[Path]` - List of paths to outlier images

**Example:**
```python
outliers = batch.detect_outliers(metric='size', threshold=2.5)
```

---

#### `verify_uniformity(check_size: bool = True, check_format: bool = True, check_mode: bool = True, check_aspect_ratio: bool = False, aspect_tolerance: float = 0.01) -> Dict[str, Any]`

Verify uniformity of images in the batch.

**Parameters:**
- `check_size`: Check if all images have the same dimensions (default: `True`)
- `check_format`: Check if all images have the same format (default: `True`)
- `check_mode`: Check if all images have the same color mode (default: `True`)
- `check_aspect_ratio`: Check if all images have similar aspect ratios (default: `False`)
- `aspect_tolerance`: Tolerance for aspect ratio comparison (default: `0.01`)

**Returns:** `dict` containing uniformity report

**Example:**
```python
report = batch.verify_uniformity(check_size=True, check_format=True)
```

---

#### `visualize_distribution(save_path: Optional[str] = None) -> None`

Plot distribution of image properties.

**Parameters:**
- `save_path`: If provided, saves plot to this path instead of showing (optional)

**Example:**
```python
batch.visualize_distribution(save_path="distribution.png")
```

---

### Dataset Operations

#### `to_dataset(format: str = 'torch', normalized: bool = True, channels_first: bool = True)`

Convert batch to ML dataset format.

**Parameters:**
- `format`: Output format ('torch', 'numpy', or 'list') (default: `'torch'`)
- `normalized`: Scale to [0.0, 1.0] (default: `True`)
- `channels_first`: Return (C, H, W) instead of (H, W, C) (default: `True`)

**Returns:** Stacked tensor/array or list depending on format

**Example:**
```python
dataset = batch.to_dataset(format='torch', normalized=True)
```

---

#### `split_dataset(train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, shuffle: bool = True, random_seed: Optional[int] = None) -> Dict[str, BatchImageHandler]`

Split images into train/validation/test sets.

**Parameters:**
- `train_ratio`: Proportion of data for training (0.0-1.0) (default: `0.8`)
- `val_ratio`: Proportion of data for validation (0.0-1.0) (default: `0.1`)
- `test_ratio`: Proportion of data for testing (0.0-1.0) (default: `0.1`)
- `shuffle`: Whether to shuffle before splitting (default: `True`)
- `random_seed`: Random seed for reproducibility (optional)

**Returns:** `dict` with keys 'train', 'val', 'test' containing BatchImageHandler instances

**Raises:**
- `ValueError`: If ratios don't sum to 1.0 or are invalid

**Example:**
```python
splits = batch.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

---

### Grid Visualization

#### `create_grid(rows: int, cols: int, cell_size: Tuple[int, int] = (200, 200), padding: int = 5, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image`

Create a grid visualization of images.

**Parameters:**
- `rows`: Number of rows
- `cols`: Number of columns
- `cell_size`: (width, height) of each cell (default: `(200, 200)`)
- `padding`: Padding between cells in pixels (default: `5`)
- `background_color`: RGB color for background (default: `(255, 255, 255)`)

**Returns:** `Image.Image` - Single PIL Image containing the grid

**Example:**
```python
grid = batch.create_grid(4, 4, cell_size=(200, 200))
```

---

### Memory Management & Utilities

#### `unload() -> BatchImageHandler`

Free memory for all loaded images.

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.unload()
```

---

#### `process_in_chunks(chunk_size: int, func: Callable[[ImageHandler], ImageHandler], parallel: bool = True) -> BatchImageHandler`

Process large batches in smaller chunks to manage memory.

**Parameters:**
- `chunk_size`: Number of images per chunk
- `func`: Processing function (accepts and returns ImageHandler)
- `parallel`: Use parallel processing within chunks (default: `True`)

**Returns:** `self` (supports chaining)

**Example:**
```python
def transform(img):
    img.resize_aspect(width=800)
    return img
batch.process_in_chunks(100, transform)
```

---

#### `set_progress_callback(callback: Callable[[int, int], None]) -> BatchImageHandler`

Set custom progress callback.

**Parameters:**
- `callback`: Function(current, total) called during processing

**Returns:** `self` (supports chaining)

**Example:**
```python
def my_progress(current, total):
    print(f"Progress: {current}/{total}")
batch.set_progress_callback(my_progress)
```

---

#### `get_errors() -> List[Tuple[Path, Exception]]`

Get list of errors encountered during processing.

**Returns:** `List[Tuple[Path, Exception]]` - List of (path, exception) tuples

**Example:**
```python
errors = batch.get_errors()
```

---

#### `clear_errors() -> BatchImageHandler`

Clear error log.

**Returns:** `self` (supports chaining)

**Example:**
```python
batch.clear_errors()
```

---

#### `copy(deep: bool = True) -> BatchImageHandler`

Create an independent copy of this BatchImageHandler.

**Parameters:**
- `deep`: If True, deep copy all loaded image pixel data (default: `True`)

**Returns:** `BatchImageHandler` - New BatchImageHandler instance with copied state

**Example:**
```python
copy1 = original.copy(deep=True)
```

---

### Special Methods

#### `__len__() -> int`

Returns the number of images in the batch.

**Returns:** `int` - Number of images

**Example:**
```python
print(len(batch))
```

---

#### `__getitem__(key: Union[int, slice]) -> Union[ImageHandler, BatchImageHandler]`

Access images by index or slice.

**Parameters:**
- `key`: Integer index or slice object

**Returns:** For integer: `ImageHandler`; For slice: `BatchImageHandler`

**Raises:**
- `IndexError`: If index is out of range
- `TypeError`: If key is not int or slice

**Example:**
```python
img = batch[0]
first_10 = batch[0:10]
```

---

#### `__iter__()`

Iterate over images in the batch.

**Yields:** `ImageHandler` instances for each image

**Example:**
```python
for img in batch:
    print(img.path)
```

---

#### `__repr__() -> str`

String representation.

**Returns:** `str` - String representation

**Example:**
```python
print(batch)
```

---

#### `__enter__() -> BatchImageHandler`

Context manager support.

**Returns:** `self`

**Example:**
```python
with BatchImageHandler.from_directory("photos/") as batch:
    batch.resize(width=800)
```

---

#### `__exit__(exc_type, exc_val, exc_tb) -> None`

Context manager cleanup.

**Example:**
```python
with batch:
    batch.resize(width=800)
```

---

## Related Documentation

- **[BatchImageHandler Usage Guide](../guide/BatchImageHandler.md)** - Learn patterns, workflows, and best practices
- **[ImageHandler API Reference](ImageHandler.md)** - Single image operations reference
- **[Module Documentation](../README.md)** - Complete module reference

## See Also

- `ImageHandler` - Single image processing interface
- `ParallelExecutor` - Parallel execution engine
- `ImageContext` - Core image context object
