"""
BatchImageHandler - Efficient batch processing for multiple images.

Provides parallel execution, filtering, transformations, and analysis for collections of images.
"""
import glob
import random
from pathlib import Path
from typing import Union, List, Optional, Callable, Tuple, Dict, Any
from PIL import Image
import numpy as np

from .core import ImageContext, ImageLoader, ImageTransformer, ImageAnalyzer, ImageAnnotator
from .handler import ImageHandler
from .batch import ParallelExecutor, BatchOperations
from .batch.duplicate_analysis import analyze_duplicates_on_image_paths


class BatchImageHandler:
    """
    Batch processing for multiple images with parallel execution support.

    Manages a collection of ImageContext instances and provides efficient batch operations
    including filtering, transformations, statistics, and ML dataset conversion.

    Example:
        >>> batch = BatchImageHandler.from_directory("photos/", "*.jpg")
        >>> batch.filter_valid().resize(width=800)
        >>> batch.save("output/", prefix="processed_")
        >>> stats = batch.get_batch_stats()
    """

    def __init__(self, paths: List[Union[str, Path]]):
        """
        Initialize BatchImageHandler with list of image paths.

        Args:
            paths: List of paths to image files

        Example:
            >>> paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
            >>> batch = BatchImageHandler(paths)
        """
        self._contexts: List[ImageContext] = [ImageContext.from_path(p) for p in paths]
        self._errors: List[Tuple[Path, Exception]] = []
        self._progress_callback: Optional[Callable[[int, int], None]] = None
        self._executor: ParallelExecutor = ParallelExecutor()

    # ========================================================================
    # CONSTRUCTORS
    # ========================================================================

    @classmethod
    def from_directory(cls, dir_path: Union[str, Path], pattern: str = "*") -> 'BatchImageHandler':
        """
        Create batch from all images in a directory.

        Args:
            dir_path: Directory path
            pattern: File pattern (e.g., "*.jpg", "*.png", "jpg")

        Returns:
            BatchImageHandler instance

        Example:
            >>> batch = BatchImageHandler.from_directory("photos/", "*.jpg")
            >>> batch = BatchImageHandler.from_directory("photos/", "png")  # Auto-adds *
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Auto-expand pattern (e.g., "jpg" -> "*.jpg")
        if not pattern.startswith('*'):
            pattern = f"*.{pattern.lstrip('.')}"

        # Find all matching files
        paths = list(dir_path.glob(pattern))

        # Filter for common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
        paths = [p for p in paths if p.suffix.lower() in image_extensions and p.is_file()]

        if not paths:
            print(f"Warning: No images found in {dir_path} matching pattern {pattern}")

        return cls(paths)

    @classmethod
    def from_glob(cls, pattern: str) -> 'BatchImageHandler':
        """
        Create batch from glob pattern.

        Args:
            pattern: Glob pattern (e.g., "photos/**/*.jpg")

        Returns:
            BatchImageHandler instance

        Example:
            >>> batch = BatchImageHandler.from_glob("photos/**/*.jpg")
        """
        paths = [Path(p) for p in glob.glob(pattern, recursive=True)]

        # Filter for image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
        paths = [p for p in paths if p.suffix.lower() in image_extensions and p.is_file()]

        if not paths:
            print(f"Warning: No images found matching pattern {pattern}")

        return cls(paths)

    # ========================================================================
    # FILTERING METHODS
    # ========================================================================

    def filter_valid(self, parallel: bool = True) -> 'BatchImageHandler':
        """
        Remove corrupted/invalid images from the batch.

        Args:
            parallel: Use parallel processing

        Returns:
            self for method chaining

        Example:
            >>> batch.filter_valid()
        """
        def check_valid(ctx: ImageContext) -> Optional[ImageContext]:
            if ImageLoader.is_valid(ctx):
                return ctx
            else:
                raise ValueError(f"Invalid image: {ctx.path}")

        if parallel:
            results, errors = self._executor.map(check_valid, self._contexts, "Validating")
        else:
            results, errors = self._sequential_process(check_valid, self._contexts)

        self._contexts = results
        self._errors.extend(errors)

        return self

    def filter_by_size(self,
                      min_width: Optional[int] = None,
                      max_width: Optional[int] = None,
                      min_height: Optional[int] = None,
                      max_height: Optional[int] = None) -> 'BatchImageHandler':
        """
        Filter images by dimensions.

        Args:
            min_width: Minimum width in pixels
            max_width: Maximum width in pixels
            min_height: Minimum height in pixels
            max_height: Maximum height in pixels

        Returns:
            self for method chaining

        Example:
            >>> batch.filter_by_size(min_width=800, max_width=2000, min_height=600)
        """
        filtered = []

        for ctx in self._contexts:
            try:
                # Fast dimension check without loading full image
                with Image.open(ctx.path) as img:
                    w, h = img.size

                # Check constraints
                if min_width and w < min_width:
                    continue
                if max_width and w > max_width:
                    continue
                if min_height and h < min_height:
                    continue
                if max_height and h > max_height:
                    continue

                filtered.append(ctx)

            except Exception as e:
                self._errors.append((ctx.path, e))

        self._contexts = filtered
        return self

    def filter_by_aspect_ratio(self,
                               min_ratio: Optional[float] = None,
                               max_ratio: Optional[float] = None) -> 'BatchImageHandler':
        """
        Filter by aspect ratio (width/height).

        Args:
            min_ratio: Minimum aspect ratio
            max_ratio: Maximum aspect ratio

        Returns:
            self for method chaining

        Example:
            >>> # Filter for landscape images
            >>> batch.filter_by_aspect_ratio(min_ratio=1.3)
        """
        filtered = []

        for ctx in self._contexts:
            try:
                # Fast dimension check
                with Image.open(ctx.path) as img:
                    w, h = img.size
                    ratio = w / h if h > 0 else 0

                # Check constraints
                if min_ratio and ratio < min_ratio:
                    continue
                if max_ratio and ratio > max_ratio:
                    continue

                filtered.append(ctx)

            except Exception as e:
                self._errors.append((ctx.path, e))

        self._contexts = filtered
        return self

    def filter_by_file_size(self,
                           min_size: Optional[Union[int, str]] = None,
                           max_size: Optional[Union[int, str]] = None) -> 'BatchImageHandler':
        """
        Filter images by file size on disk.

        Useful for removing very small (possibly corrupted/thumbnails) or very large
        (uncompressed/RAW) images from the batch. Supports human-readable size formats.

        Args:
            min_size: Minimum file size. Can be:
                     - int: size in bytes
                     - str: human-readable format (e.g., '50KB', '1.5MB', '2GB')
                     Supported units: B, KB, MB, GB (case-insensitive)
            max_size: Maximum file size (same format as min_size)

        Returns:
            self for method chaining

        Raises:
            ValueError: If size format is invalid

        Example:
            >>> # Remove files smaller than 50KB (likely thumbnails/corrupted)
            >>> batch.filter_by_file_size(min_size='50KB')
            >>>
            >>> # Remove files larger than 5MB (likely uncompressed)
            >>> batch.filter_by_file_size(max_size='5MB')
            >>>
            >>> # Keep files between 100KB and 2MB
            >>> batch.filter_by_file_size(min_size='100KB', max_size='2MB')
            >>>
            >>> # Using bytes directly
            >>> batch.filter_by_file_size(min_size=51200, max_size=2097152)
            >>>
            >>> # Using GB for large files
            >>> batch.filter_by_file_size(max_size='1GB')
            >>>
            >>> # Decimal values supported
            >>> batch.filter_by_file_size(min_size='0.5MB', max_size='1.5GB')

        Supported Units:
            - B or bytes: 1 byte
            - KB or kilobytes: 1,024 bytes
            - MB or megabytes: 1,048,576 bytes (1024²)
            - GB or gigabytes: 1,073,741,824 bytes (1024³)

        Note:
            - Uses file size on disk (actual bytes stored)
            - Does not load images into memory (very fast)
            - Case-insensitive unit parsing ('kb', 'KB', 'Kb' all work)
        """
        # Parse size inputs
        try:
            min_bytes = self._parse_size_string(min_size) if min_size is not None else None
            max_bytes = self._parse_size_string(max_size) if max_size is not None else None
        except ValueError as e:
            raise ValueError(f"Invalid size parameter: {e}")

        # Check if any constraint specified
        if min_bytes is None and max_bytes is None:
            print("Warning: No file size constraints specified")
            return self

        # Validate min <= max
        if min_bytes is not None and max_bytes is not None and min_bytes > max_bytes:
            raise ValueError(
                f"min_size ({self._format_size(min_bytes)}) cannot be greater than "
                f"max_size ({self._format_size(max_bytes)})"
            )

        filtered = []
        removed_count = 0

        print(f"Filtering {len(self._contexts)} images by file size...")
        if min_bytes is not None:
            print(f"  Minimum: {self._format_size(min_bytes)}")
        if max_bytes is not None:
            print(f"  Maximum: {self._format_size(max_bytes)}")

        for ctx in self._contexts:
            try:
                # Get file size without loading image
                file_size = ctx.path.stat().st_size  # bytes

                # Check constraints
                if min_bytes is not None and file_size < min_bytes:
                    removed_count += 1
                    continue

                if max_bytes is not None and file_size > max_bytes:
                    removed_count += 1
                    continue

                filtered.append(ctx)

            except Exception as e:
                self._errors.append((ctx.path, e))
                removed_count += 1

        self._contexts = filtered

        print(f"Filtered: kept {len(filtered)}, removed {removed_count}")

        return self

    def analyze_duplicates(self,
                          hash_method: str = 'dhash',
                          hash_size: int = 8,
                          threshold: Optional[int] = None,
                          parallel: bool = True) -> Dict[str, Any]:
        """
        Analyze images to identify duplicate groups and singleton images.
        
        BatchImageHandler-specific wrapper around analyze_duplicates_on_image_paths().
        Returns ImageContext objects instead of Path objects.
        
        Args:
            hash_method: Hashing algorithm ('dhash', 'phash', 'ahash', 'whash')
            hash_size: Hash size (8 = 64-bit, 16 = 256-bit)
            threshold: Hamming distance threshold (None = use recommended default)
            parallel: Use parallel processing for hash computation
        
        Returns:
            Dictionary containing:
                - 'duplicate_groups': List[List[ImageContext]]
                - 'singleton_groups': List[List[ImageContext]]
                - 'hash_map': Dict[Path, Tuple[ImageContext, ImageHash]]
                - 'stats': Dict with summary statistics
        
        Example:
            >>> batch = BatchImageHandler.from_directory("photos/")
            >>> analysis = batch.analyze_duplicates(hash_method='dhash', threshold=5)
            >>> 
            >>> print(f"Duplicate groups: {analysis['stats']['num_duplicate_groups']}")
            >>> 
            >>> # Inspect first duplicate group
            >>> if analysis['duplicate_groups']:
            ...     first_group = analysis['duplicate_groups'][0]
            ...     for ctx in first_group:
            ...         print(f"  - {ctx.path.name}")
        """
        if not self._contexts:
            return {
                'duplicate_groups': [],
                'singleton_groups': [],
                'hash_map': {},
                'stats': {
                    'total_images': 0,
                    'num_duplicate_groups': 0,
                    'num_unique_images': 0,
                    'total_duplicates': 0,
                    'hash_method': hash_method,
                    'threshold': threshold
                }
            }
        
        # Extract image paths from contexts
        image_paths = [ctx.path for ctx in self._contexts]
        
        # Create mapping from Path to ImageContext for fast lookup
        path_to_context = {ctx.path: ctx for ctx in self._contexts}
        
        # Call generic utility function
        analysis = analyze_duplicates_on_image_paths(
            image_paths=image_paths,
            hash_method=hash_method,
            hash_size=hash_size,
            threshold=threshold,
            parallel=parallel,
            max_workers=None,
            progress_callback=self._progress_callback
        )
        
        # Add errors to BatchImageHandler's error list
        self._errors.extend(analysis['errors'])
        
        # Convert Path-based groups to ImageContext-based groups
        duplicate_groups_ctx = [
            [path_to_context[path] for path in group]
            for group in analysis['duplicate_groups']
        ]

        # Convert Path-based singleton groups to ImageContext-based groups
        singleton_groups_ctx = [
            [path_to_context[path] for path in group]
            for group in analysis['singleton_groups']
        ]

        # Convert hash_map from {Path: hash} to {Path: (ImageContext, hash)}
        hash_map_ctx = {
            path: (path_to_context[path], hash_val)
            for path, hash_val in analysis['hash_map'].items()
        }

        return {
            'duplicate_groups': duplicate_groups_ctx,
            'singleton_groups': singleton_groups_ctx,
            'hash_map': hash_map_ctx,
            'stats': analysis['stats']
        }

    def filter_duplicates(self,
                         hash_method: str = 'dhash',
                         hash_size: int = 8,
                         threshold: Optional[int] = None,
                         keep: str = 'first',
                         parallel: bool = True) -> 'BatchImageHandler':
        """
        Filter duplicate using perceptual hashing and keep one from each group based on the keep strategy.

        Supports multiple hashing algorithms to detect visually similar images.
        Images with hash distances below the threshold are considered duplicates.

        Args:
            hash_method: Hashing algorithm to use. Options:
                         - 'dhash' (default): Fast, good balance, robust to crops/resizes
                         - 'phash': Most accurate, slower, best for edited images
                         - 'ahash': Fastest, less robust, good for exact duplicates
                         - 'whash': Most robust, slowest, best for scale variations
            hash_size: Hash size (8 = 64-bit hash, 16 = 256-bit hash)
            threshold: Maximum Hamming distance to consider as duplicate (0-64 for hash_size=8)
                      If None, uses recommended default based on hash_method:
                      - ahash: 3, dhash: 5, phash: 8, whash: 10
            keep: Which duplicate to keep ('first', 'last', 'largest', 'smallest')
            parallel: Use parallel processing for hash computation

        Returns:
            self for method chaining

        Example:
            >>> # Remove duplicates (default behavior)
            >>> batch.filter_duplicates(hash_method='dhash', threshold=8)
            >>>
            >>> # Fast screening for exact duplicates
            >>> batch.filter_duplicates(hash_method='ahash', threshold=2)
            >>>
            >>> # Find similar images (resized, cropped)
            >>> batch.filter_duplicates(hash_method='dhash', threshold=8)
            >>>
            >>> # Find heavily edited versions
            >>> batch.filter_duplicates(hash_method='phash', threshold=12)
            >>>
            >>> # Match thumbnails to originals
            >>> batch.filter_duplicates(hash_method='whash', threshold=10)
            >>>
            >>> # Auto-set threshold based on method
            >>> batch.filter_duplicates(hash_method='phash')  # Uses threshold=8

        Algorithm Selection Guide:
            +----------+-------+----------+----------------+------------------------+
            | Method   | Speed | Accuracy | Best For       | Recommended Threshold  |
            +----------+-------+----------+----------------+------------------------+
            | ahash    | ★★★★★ | ★★☆☆☆    | Exact dupes    | 2-5 (default: 3)       |
            | dhash    | ★★★★☆ | ★★★★☆    | Similar images | 5-10 (default: 5)      |
            | phash    | ★★☆☆☆ | ★★★★★    | Edited images  | 8-15 (default: 8)      |
            | whash    | ★☆☆☆☆ | ★★★★★    | Multi-scale    | 10-20 (default: 10)    |
            +----------+-------+----------+----------------+------------------------+

        Note:
            - Lower threshold = stricter matching (fewer duplicates found)
            - Higher threshold = more lenient (more images flagged as duplicates)
        """
        if not self._contexts:
            return self
        
        # Use analyze_duplicates to get groups (which internally uses the generic function)
        analysis = self.analyze_duplicates(
            hash_method=hash_method,
            hash_size=hash_size,
            threshold=threshold,
            parallel=parallel
        )
        
        duplicate_groups = analysis['duplicate_groups']
        singleton_groups = analysis['singleton_groups']

        kept_contexts = []
        removed_count = 0

        # Keep non-duplicates (singletons) - each singleton_group has exactly 1 image
        for singleton_group in singleton_groups:
            kept_contexts.append(singleton_group[0])

        # Process duplicate groups - keep one from each
        for group in duplicate_groups:
            if keep == 'first':
                kept = group[0]
            elif keep == 'last':
                kept = group[-1]
            elif keep == 'largest':
                kept = max(group, key=lambda ctx: ctx.path.stat().st_size)
            elif keep == 'smallest':
                kept = min(group, key=lambda ctx: ctx.path.stat().st_size)
            else:
                raise ValueError(f"Unknown keep strategy: {keep}")
            
            kept_contexts.append(kept)
            removed_count += len(group) - 1
        
        print(f"Removed {removed_count} duplicate images, kept {len(kept_contexts)}")
        
        self._contexts = kept_contexts
        return self

    def sample(self, n: int, random_sample: bool = True) -> 'BatchImageHandler':
        """
        Sample n images from the batch.

        Args:
            n: Number of images to sample
            random_sample: If True, random sampling. If False, takes first n

        Returns:
            self for method chaining

        Example:
            >>> batch.sample(50, random_sample=True)
        """
        if n >= len(self._contexts):
            return self  # Already smaller than sample size

        if random_sample:
            self._contexts = random.sample(self._contexts, n)
        else:
            self._contexts = self._contexts[:n]

        return self

    # ========================================================================
    # CORE PROCESSING METHODS
    # ========================================================================

    def map(self,
            func: Callable[[ImageHandler], ImageHandler],
            parallel: bool = True,
            workers: Optional[int] = None) -> 'BatchImageHandler':
        """
        Apply custom function to each image.

        Args:
            func: Function that takes ImageHandler and returns ImageHandler
            parallel: Use parallel processing
            workers: Number of worker threads (auto if None)

        Returns:
            self for method chaining

        Example:
            >>> def my_pipeline(img):
            ...     img.resize_aspect(width=512)
            ...     img.adjust(brightness=1.15)
            ...     return img
            >>> batch.map(my_pipeline, parallel=True)
        """
        def apply_to_handler(ctx: ImageContext) -> ImageContext:
            # Create temporary handler with this context
            handler = ImageHandler(ctx.path)
            handler._ctx = ctx

            # Apply user function
            handler = func(handler)

            # Extract modified context
            return handler._ctx

        # Update executor workers if specified
        if workers:
            original_workers = self._executor.max_workers
            self._executor.max_workers = workers

        if parallel:
            results, errors = self._executor.map(apply_to_handler, self._contexts, "Mapping function")
        else:
            results, errors = self._sequential_process(apply_to_handler, self._contexts)

        # Restore workers
        if workers:
            self._executor.max_workers = original_workers

        self._contexts = results
        self._errors.extend(errors)

        return self

    def apply_transform(self,
                       transform_name: str,
                       parallel: bool = True,
                       workers: Optional[int] = None,
                       **kwargs) -> 'BatchImageHandler':
        """
        Apply any ImageHandler method to all images.

        Args:
            transform_name: Name of the ImageHandler method
            parallel: Use parallel processing
            workers: Number of worker threads
            **kwargs: Arguments to pass to the method

        Returns:
            self for method chaining

        Example:
            >>> batch.apply_transform('resize_aspect', width=800)
            >>> batch.apply_transform('adjust', brightness=1.2, contrast=0.9)
        """
        # Map transform name to component method
        transform_map = {
            # ImageTransformer methods
            'resize_aspect': (ImageTransformer, 'resize_aspect'),
            'square_pad': (ImageTransformer, 'square_pad'),
            'add_margin': (ImageTransformer, 'add_margin'),
            'pad_to_size': (ImageTransformer, 'pad_to_size'),
            'adjust': (ImageTransformer, 'adjust'),
            'filter_blur': (ImageTransformer, 'filter_blur'),
            'to_grayscale': (ImageTransformer, 'to_grayscale'),
            'crop': (ImageTransformer, 'crop'),
            'flip_horizontal': (ImageTransformer, 'flip_horizontal'),
            'flip_vertical': (ImageTransformer, 'flip_vertical'),
            'rotate': (ImageTransformer, 'rotate'),
            'to_rgba': (ImageTransformer, 'to_rgba'),
        }

        if transform_name not in transform_map:
            raise ValueError(f"Unknown transform: {transform_name}. "
                           f"Available: {', '.join(transform_map.keys())}")

        component_class, method_name = transform_map[transform_name]
        method = getattr(component_class, method_name)

        def transform_func(ctx: ImageContext) -> ImageContext:
            # Ensure loaded
            if not ctx.is_loaded():
                ImageLoader.load(ctx)
            # Apply transform
            return method(ctx, **kwargs)

        # Update executor workers if specified
        if workers:
            original_workers = self._executor.max_workers
            self._executor.max_workers = workers

        if parallel:
            results, errors = self._executor.map(transform_func, self._contexts,
                                                f"Applying {transform_name}")
        else:
            results, errors = self._sequential_process(transform_func, self._contexts)

        # Restore workers
        if workers:
            self._executor.max_workers = original_workers

        self._contexts = results
        self._errors.extend(errors)

        return self

    def chain_transforms(self,
                        transforms: List[Tuple[str, dict]],
                        parallel: bool = True) -> 'BatchImageHandler':
        """
        Apply multiple transformations in sequence.

        Args:
            transforms: List of (method_name, kwargs) tuples
            parallel: Use parallel processing

        Returns:
            self for method chaining

        Example:
            >>> transforms = [
            ...     ('resize_aspect', {'width': 800}),
            ...     ('adjust', {'brightness': 1.2}),
            ...     ('to_grayscale', {})
            ... ]
            >>> batch.chain_transforms(transforms)
        """
        def chain_func(ctx: ImageContext) -> ImageContext:
            # Ensure loaded
            if not ctx.is_loaded():
                ImageLoader.load(ctx)

            # Apply each transform in sequence
            for transform_name, kwargs in transforms:
                transform_map = {
                    'resize_aspect': (ImageTransformer, 'resize_aspect'),
                    'square_pad': (ImageTransformer, 'square_pad'),
                    'add_margin': (ImageTransformer, 'add_margin'),
                    'pad_to_size': (ImageTransformer, 'pad_to_size'),
                    'adjust': (ImageTransformer, 'adjust'),
                    'filter_blur': (ImageTransformer, 'filter_blur'),
                    'to_grayscale': (ImageTransformer, 'to_grayscale'),
                    'crop': (ImageTransformer, 'crop'),
                    'flip_horizontal': (ImageTransformer, 'flip_horizontal'),
                    'flip_vertical': (ImageTransformer, 'flip_vertical'),
                    'rotate': (ImageTransformer, 'rotate'),
                    'to_rgba': (ImageTransformer, 'to_rgba'),
                }

                if transform_name not in transform_map:
                    raise ValueError(f"Unknown transform: {transform_name}")

                component_class, method_name = transform_map[transform_name]
                method = getattr(component_class, method_name)
                ctx = method(ctx, **kwargs)

            return ctx

        if parallel:
            results, errors = self._executor.map(chain_func, self._contexts, "Chaining transforms")
        else:
            results, errors = self._sequential_process(chain_func, self._contexts)

        self._contexts = results
        self._errors.extend(errors)

        return self

    # ========================================================================
    # CONVENIENCE BATCH OPERATIONS
    # ========================================================================

    def resize(self,
               width: Optional[int] = None,
               height: Optional[int] = None,
               parallel: bool = True,
               **kwargs) -> 'BatchImageHandler':
        """
        Resize all images.

        Args:
            width: Target width
            height: Target height
            parallel: Use parallel processing
            **kwargs: Additional arguments for resize_aspect

        Returns:
            self for method chaining

        Example:
            >>> batch.resize(width=800)
        """
        return self.apply_transform('resize_aspect', parallel=parallel, width=width, height=height, **kwargs)

    def adjust(self,
               brightness: float = 1.0,
               contrast: float = 1.0,
               parallel: bool = True) -> 'BatchImageHandler':
        """
        Adjust brightness/contrast for all images.

        Args:
            brightness: Brightness factor
            contrast: Contrast factor
            parallel: Use parallel processing

        Returns:
            self for method chaining

        Example:
            >>> batch.adjust(brightness=1.2, contrast=1.1)
        """
        return self.apply_transform('adjust', parallel=parallel, brightness=brightness, contrast=contrast)

    def to_grayscale(self,
                     keep_2d: bool = False,
                     parallel: bool = True) -> 'BatchImageHandler':
        """
        Convert all images to grayscale.

        Args:
            keep_2d: Keep single channel mode
            parallel: Use parallel processing

        Returns:
            self for method chaining

        Example:
            >>> batch.to_grayscale(keep_2d=False)
        """
        return self.apply_transform('to_grayscale', parallel=parallel, keep_2d=keep_2d)

    # ========================================================================
    # SAVE OPERATIONS
    # ========================================================================

    def save(self,
             output_dir: Union[str, Path],
             prefix: str = "",
             suffix: str = "",
             quality: int = 95,
             overwrite: bool = False) -> 'BatchImageHandler':
        """
        Save all processed images.

        Args:
            output_dir: Output directory path
            prefix: Prefix to add to filenames
            suffix: Suffix to add to filenames (before extension)
            quality: JPEG quality (1-100)
            overwrite: If True, overwrites existing files

        Returns:
            self for method chaining

        Example:
            >>> batch.save("output/", prefix="processed_", suffix="_800w")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate save tasks
        save_tasks = []
        for ctx in self._contexts:
            # Generate filename: prefix + stem + suffix + ext
            stem = ctx.path.stem
            ext = ctx.path.suffix
            filename = f"{prefix}{stem}{suffix}{ext}"
            output_path = output_dir / filename

            # Check overwrite
            if output_path.exists() and not overwrite:
                self._errors.append((ctx.path,
                                   FileExistsError(f"File exists: {output_path}")))
                continue

            save_tasks.append((ctx, output_path))

        # Save in parallel
        def save_func(task):
            ctx, path = task
            if not ctx.is_loaded():
                ImageLoader.load(ctx)
            ImageLoader.save(ctx, path, quality=quality)
            return ctx

        results, errors = self._executor.map(save_func, save_tasks, "Saving")
        self._errors.extend(errors)

        return self

    # ========================================================================
    # ANALYSIS & STATISTICS
    # ========================================================================

    def get_batch_stats(self) -> dict:
        """
        Aggregate statistics across all images.

        Returns:
            Dictionary with aggregated statistics

        Example:
            >>> stats = batch.get_batch_stats()
            >>> print(stats['width']['mean'], stats['height']['mean'])
        """
        def get_stats(ctx: ImageContext) -> dict:
            if not ctx.is_loaded():
                ImageLoader.load(ctx)
            return ImageAnalyzer.get_stats(ctx)

        results, errors = self._executor.map(get_stats, self._contexts, "Analyzing")
        self._errors.extend(errors)

        # Aggregate with BatchOperations
        batch_stats = BatchOperations.compute_batch_stats(results)

        return batch_stats

    def detect_outliers(self, metric: str = 'size', threshold: float = 2.0) -> List[Path]:
        """
        Detect outlier images based on statistics.

        Args:
            metric: Metric to use ('size', 'width', 'height', 'aspect_ratio')
            threshold: Number of standard deviations

        Returns:
            List of paths to outlier images

        Example:
            >>> outliers = batch.detect_outliers(metric='size', threshold=2.5)
            >>> print(f"Found {len(outliers)} outliers")
        """
        # Get batch statistics
        batch_stats = self.get_batch_stats()

        # Detect outlier indices
        outlier_indices = BatchOperations.detect_outliers(batch_stats, metric, threshold)

        # Return paths
        outlier_paths = [self._contexts[i].path for i in outlier_indices]

        return outlier_paths

    def verify_uniformity(self,
                         check_size: bool = True,
                         check_format: bool = True,
                         check_mode: bool = True,
                         check_aspect_ratio: bool = False,
                         aspect_tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Verify uniformity of images in the batch.

        Checks if all images share common properties (size, format, color mode, aspect ratio).
        Useful for ensuring dataset consistency before training ML models.

        Args:
            check_size: Check if all images have the same dimensions
            check_format: Check if all images have the same format (JPEG, PNG, etc.)
            check_mode: Check if all images have the same color mode (RGB, L, RGBA, etc.)
            check_aspect_ratio: Check if all images have similar aspect ratios
            aspect_tolerance: Tolerance for aspect ratio comparison (default 0.01 = 1%)

        Returns:
            Dictionary containing:
            - 'uniform': bool - Whether batch is uniform
            - 'total_images': int - Total number of images
            - 'checks': dict - Results for each check performed
            - 'sizes': dict - Size distribution {(width, height): count}
            - 'formats': dict - Format distribution {format: count}
            - 'modes': dict - Color mode distribution {mode: count}
            - 'aspect_ratios': dict - Aspect ratio distribution (if checked)
            - 'violations': list - Images that don't match the majority

        Example:
            >>> batch = BatchImageHandler.from_directory("dataset/")
            >>> report = batch.verify_uniformity(check_size=True, check_format=True)
            >>>
            >>> if report['uniform']:
            ...     print("✓ All images are uniform")
            ... else:
            ...     print(f"✗ Found {len(report['violations'])} non-uniform images")
            ...     print(f"Size distribution: {report['sizes']}")
            ...     print(f"Format distribution: {report['formats']}")
            >>>
            >>> # Check aspect ratio uniformity
            >>> report = batch.verify_uniformity(check_aspect_ratio=True, aspect_tolerance=0.05)
            >>> print(f"Aspect ratios: {report['aspect_ratios']}")
            >>>
            >>> # Fix non-uniform batch
            >>> if not report['uniform']:
            ...     # Get the most common size
            ...     most_common_size = max(report['sizes'].items(), key=lambda x: x[1])[0]
            ...     print(f"Most common size: {most_common_size}")

        Note:
            - Does not load full images, only reads metadata (fast)
            - Violations list contains paths and reasons for non-uniformity
            - For visual distribution plots, export data to external tools
        """
        if not self._contexts:
            return {
                'uniform': True,
                'total_images': 0,
                'checks': {},
                'sizes': {},
                'formats': {},
                'modes': {},
                'aspect_ratios': {},
                'violations': []
            }

        # Initialize tracking
        sizes = {}
        formats = {}
        modes = {}
        aspect_ratios = {}
        violations = []

        print(f"Analyzing uniformity of {len(self._contexts)} images...")

        for ctx in self._contexts:
            try:
                # Get image properties without fully loading
                with Image.open(ctx.path) as img:
                    # Size
                    size = img.size
                    sizes[size] = sizes.get(size, 0) + 1

                    # Format
                    fmt = img.format or 'UNKNOWN'
                    formats[fmt] = formats.get(fmt, 0) + 1

                    # Mode
                    mode = img.mode
                    modes[mode] = modes.get(mode, 0) + 1

                    # Aspect ratio
                    if check_aspect_ratio:
                        w, h = size
                        if h > 0:
                            ratio = round(w / h, 3)
                            aspect_ratios[ratio] = aspect_ratios.get(ratio, 0) + 1

            except Exception as e:
                self._errors.append((ctx.path, e))
                violations.append({
                    'path': str(ctx.path),
                    'reason': f'Error reading image: {e}'
                })

        # Determine majority values
        majority_size = max(sizes.items(), key=lambda x: x[1])[0] if sizes else None
        majority_format = max(formats.items(), key=lambda x: x[1])[0] if formats else None
        majority_mode = max(modes.items(), key=lambda x: x[1])[0] if modes else None
        majority_ratio = max(aspect_ratios.items(), key=lambda x: x[1])[0] if aspect_ratios else None

        # Perform checks
        checks = {}
        uniform = True

        if check_size:
            size_uniform = len(sizes) == 1
            checks['size'] = {
                'uniform': size_uniform,
                'expected': majority_size,
                'unique_values': len(sizes)
            }
            if not size_uniform:
                uniform = False
                # Find violations
                for ctx in self._contexts:
                    try:
                        with Image.open(ctx.path) as img:
                            if img.size != majority_size:
                                violations.append({
                                    'path': str(ctx.path),
                                    'reason': f'Size mismatch: {img.size} != {majority_size}'
                                })
                    except:
                        pass

        if check_format:
            format_uniform = len(formats) == 1
            checks['format'] = {
                'uniform': format_uniform,
                'expected': majority_format,
                'unique_values': len(formats)
            }
            if not format_uniform:
                uniform = False
                # Find violations
                for ctx in self._contexts:
                    try:
                        with Image.open(ctx.path) as img:
                            fmt = img.format or 'UNKNOWN'
                            if fmt != majority_format:
                                violations.append({
                                    'path': str(ctx.path),
                                    'reason': f'Format mismatch: {fmt} != {majority_format}'
                                })
                    except:
                        pass

        if check_mode:
            mode_uniform = len(modes) == 1
            checks['mode'] = {
                'uniform': mode_uniform,
                'expected': majority_mode,
                'unique_values': len(modes)
            }
            if not mode_uniform:
                uniform = False
                # Find violations
                for ctx in self._contexts:
                    try:
                        with Image.open(ctx.path) as img:
                            if img.mode != majority_mode:
                                violations.append({
                                    'path': str(ctx.path),
                                    'reason': f'Mode mismatch: {img.mode} != {majority_mode}'
                                })
                    except:
                        pass

        if check_aspect_ratio and majority_ratio is not None:
            # Check if aspect ratios are within tolerance
            ratio_uniform = all(
                abs(ratio - majority_ratio) <= aspect_tolerance
                for ratio in aspect_ratios.keys()
            )
            checks['aspect_ratio'] = {
                'uniform': ratio_uniform,
                'expected': majority_ratio,
                'tolerance': aspect_tolerance,
                'unique_values': len(aspect_ratios)
            }
            if not ratio_uniform:
                uniform = False
                # Find violations
                for ctx in self._contexts:
                    try:
                        with Image.open(ctx.path) as img:
                            w, h = img.size
                            if h > 0:
                                ratio = round(w / h, 3)
                                if abs(ratio - majority_ratio) > aspect_tolerance:
                                    violations.append({
                                        'path': str(ctx.path),
                                        'reason': f'Aspect ratio mismatch: {ratio:.3f} != {majority_ratio:.3f} (tolerance: {aspect_tolerance})'
                                    })
                    except:
                        pass

        # Build report
        report = {
            'uniform': uniform,
            'total_images': len(self._contexts),
            'checks': checks,
            'sizes': dict(sorted(sizes.items(), key=lambda x: x[1], reverse=True)),
            'formats': dict(sorted(formats.items(), key=lambda x: x[1], reverse=True)),
            'modes': dict(sorted(modes.items(), key=lambda x: x[1], reverse=True)),
            'aspect_ratios': dict(sorted(aspect_ratios.items(), key=lambda x: x[1], reverse=True)),
            'violations': violations
        }

        # Print summary
        if uniform:
            print("✓ Batch is uniform")
        else:
            print(f"✗ Batch has uniformity issues:")
            for check_name, check_result in checks.items():
                if not check_result['uniform']:
                    print(f"  - {check_name}: {check_result['unique_values']} unique values")
            print(f"  - Total violations: {len(violations)}")

        return report

    def visualize_distribution(self, save_path: Optional[str] = None):
        """
        Plot distribution of image properties.

        Args:
            save_path: If provided, saves plot to this path instead of showing

        Example:
            >>> batch.visualize_distribution(save_path="distribution.png")
        """
        batch_stats = self.get_batch_stats()
        BatchOperations.visualize_distribution(batch_stats, save_path)

    # ========================================================================
    # DATASET CONVERSION
    # ========================================================================

    def to_dataset(self,
                   format: str = 'torch',
                   normalized: bool = True,
                   channels_first: bool = True):
        """
        Convert batch to ML dataset format.

        Args:
            format: Output format ('torch', 'numpy', or 'list')
            normalized: Scale to [0.0, 1.0]
            channels_first: Return (C, H, W) instead of (H, W, C)

        Returns:
            Stacked tensor/array or list depending on format

        Example:
            >>> batch = BatchImageHandler.from_directory("photos/", "*.jpg")
            >>> batch.resize(width=224, height=224)
            >>> dataset = batch.to_dataset(format='torch', normalized=True)
            >>> print(dataset.shape)  # torch.Size([N, 3, 224, 224])
        """
        # Verify all images loaded and same size
        sizes = set()
        for ctx in self._contexts:
            if not ctx.is_loaded():
                ImageLoader.load(ctx)
            sizes.add(ctx.size)

        if len(sizes) > 1:
            raise ValueError(f"All images must be same size for dataset conversion. "
                           f"Found sizes: {sizes}. Use resize() first.")

        # Convert based on format
        if format == 'numpy':
            arrays = [ImageAnalyzer.to_numpy(ctx, normalized=normalized)
                     for ctx in self._contexts]
            dataset = np.stack(arrays)
            if channels_first and dataset.ndim == 4:
                dataset = np.transpose(dataset, (0, 3, 1, 2))
            return dataset

        elif format == 'torch':
            try:
                import torch
            except ImportError:
                raise ImportError("PyTorch required. Install with: pip install torch")

            tensors = [ImageAnalyzer.to_tensor(ctx, normalized=normalized,
                                              channels_first=channels_first)
                      for ctx in self._contexts]
            return torch.stack(tensors)

        elif format == 'list':
            return [ImageAnalyzer.to_numpy(ctx, normalized=normalized)
                   for ctx in self._contexts]

        else:
            raise ValueError(f"Unknown format: {format}. Use 'numpy', 'torch', or 'list'.")

    # ========================================================================
    # DATASET OPERATIONS
    # ========================================================================

    def split_dataset(self,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      shuffle: bool = True,
                      random_seed: Optional[int] = None) -> Dict[str, 'BatchImageHandler']:
        """
        Split images into train/validation/test sets.

        Creates independent BatchImageHandler instances for each split, useful for
        machine learning dataset preparation.

        Args:
            train_ratio: Proportion of data for training (0.0-1.0)
            val_ratio: Proportion of data for validation (0.0-1.0)
            test_ratio: Proportion of data for testing (0.0-1.0)
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with keys 'train', 'val', 'test' containing BatchImageHandler instances

        Raises:
            ValueError: If ratios don't sum to 1.0 or are invalid

        Example:
            >>> batch = BatchImageHandler.from_directory("dataset/")
            >>> splits = batch.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
            >>>
            >>> train_batch = splits['train']
            >>> val_batch = splits['val']
            >>> test_batch = splits['test']
            >>>
            >>> print(f"Train: {len(train_batch)}, Val: {len(val_batch)}, Test: {len(test_batch)}")
            >>>
            >>> # Save splits to different directories
            >>> train_batch.save("output/train/")
            >>> val_batch.save("output/val/")
            >>> test_batch.save("output/test/")
            >>>
            >>> # Reproducible splitting
            >>> splits = batch.split_dataset(random_seed=42)

        Note:
            - Ratios must sum to 1.0 (within small tolerance)
            - Each split is an independent BatchImageHandler instance
            - Original batch remains unchanged
            - For class-balanced splitting, manually group images first
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        if any(r < 0 or r > 1 for r in [train_ratio, val_ratio, test_ratio]):
            raise ValueError("All ratios must be between 0.0 and 1.0")

        if not self._contexts:
            raise ValueError("Cannot split empty batch")

        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Copy contexts to avoid modifying original
        contexts = self._contexts.copy()

        # Shuffle if requested
        if shuffle:
            random.shuffle(contexts)

        total = len(contexts)

        # Calculate split sizes
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        # test_size gets the remainder to ensure all images are included

        print(f"Splitting {total} images: "
              f"train={train_size} ({train_ratio:.1%}), "
              f"val={val_size} ({val_ratio:.1%}), "
              f"test={total - train_size - val_size} ({test_ratio:.1%})")

        # Split contexts
        train_contexts = contexts[:train_size]
        val_contexts = contexts[train_size:train_size + val_size]
        test_contexts = contexts[train_size + val_size:]

        # Create new BatchImageHandler instances
        train_batch = BatchImageHandler([])
        train_batch._contexts = train_contexts
        train_batch._executor = ParallelExecutor()

        val_batch = BatchImageHandler([])
        val_batch._contexts = val_contexts
        val_batch._executor = ParallelExecutor()

        test_batch = BatchImageHandler([])
        test_batch._contexts = test_contexts
        test_batch._executor = ParallelExecutor()

        return {
            'train': train_batch,
            'val': val_batch,
            'test': test_batch
        }

    # ========================================================================
    # GRID VISUALIZATION
    # ========================================================================

    def create_grid(self,
                   rows: int,
                   cols: int,
                   cell_size: Tuple[int, int] = (200, 200),
                   padding: int = 5,
                   background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Create a grid visualization of images.

        Args:
            rows: Number of rows
            cols: Number of columns
            cell_size: (width, height) of each cell
            padding: Padding between cells in pixels
            background_color: RGB color for background

        Returns:
            Single PIL Image containing the grid

        Example:
            >>> grid = batch.create_grid(4, 4, cell_size=(200, 200))
            >>> grid.save("grid.jpg")
        """
        # Load images if needed
        images = []
        for ctx in self._contexts[:rows * cols]:
            if not ctx.is_loaded():
                ImageLoader.load(ctx)
            images.append(ctx.img)

        return BatchOperations.create_grid(images, rows, cols, cell_size, padding, background_color)

    # ========================================================================
    # MEMORY MANAGEMENT & UTILITIES
    # ========================================================================

    def unload(self) -> 'BatchImageHandler':
        """
        Free memory for all loaded images.

        Returns:
            self for method chaining

        Example:
            >>> batch.unload()
        """
        for ctx in self._contexts:
            ImageLoader.unload(ctx)
        return self

    def process_in_chunks(self,
                         chunk_size: int,
                         func: Callable[[ImageHandler], ImageHandler],
                         parallel: bool = True) -> 'BatchImageHandler':
        """
        Process large batches in smaller chunks to manage memory.

        Args:
            chunk_size: Number of images per chunk
            func: Processing function (accepts and returns ImageHandler)
            parallel: Use parallel processing within chunks

        Returns:
            self for method chaining

        Example:
            >>> def transform(img):
            ...     img.resize_aspect(width=800)
            ...     return img
            >>> batch.process_in_chunks(100, transform)
        """
        # Split contexts into chunks
        chunks = [self._contexts[i:i+chunk_size]
                 for i in range(0, len(self._contexts), chunk_size)]

        all_results = []

        for chunk_idx, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {chunk_idx}/{len(chunks)} ({len(chunk)} images)...")

            # Create temporary batch for chunk
            temp_batch = BatchImageHandler([])
            temp_batch._contexts = chunk
            temp_batch._executor = self._executor

            # Process chunk
            temp_batch.map(func, parallel=parallel)

            # Collect results and errors
            all_results.extend(temp_batch._contexts)
            self._errors.extend(temp_batch._errors)

            # Unload chunk to free memory
            temp_batch.unload()

        self._contexts = all_results
        return self

    def set_progress_callback(self, callback: Callable[[int, int], None]) -> 'BatchImageHandler':
        """
        Set custom progress callback.

        Args:
            callback: Function(current, total) called during processing

        Returns:
            self for method chaining

        Example:
            >>> def my_progress(current, total):
            ...     print(f"Progress: {current}/{total}")
            >>> batch.set_progress_callback(my_progress)
        """
        self._progress_callback = callback
        self._executor.progress_callback = callback
        return self

    def get_errors(self) -> List[Tuple[Path, Exception]]:
        """
        Get list of errors encountered during processing.

        Returns:
            List of (path, exception) tuples

        Example:
            >>> errors = batch.get_errors()
            >>> for path, error in errors:
            ...     print(f"Error processing {path}: {error}")
        """
        return self._errors.copy()

    def clear_errors(self) -> 'BatchImageHandler':
        """
        Clear error log.

        Returns:
            self for method chaining

        Example:
            >>> batch.clear_errors()
        """
        self._errors = []
        return self

    # ========================================================================
    # SPECIAL METHODS
    # ========================================================================

    def __len__(self) -> int:
        """
        Returns the number of images in the batch.

        Example:
            >>> print(len(batch))
        """
        return len(self._contexts)

    def __getitem__(self, key: Union[int, slice]) -> Union['ImageHandler', 'BatchImageHandler']:
        """
        Access images by index or slice.

        Args:
            key: Integer index or slice object

        Returns:
            - For integer index: ImageHandler wrapping the image
            - For slice: New BatchImageHandler with sliced images

        Raises:
            IndexError: If index is out of range
            TypeError: If key is not int or slice

        Example:
            >>> batch = BatchImageHandler.from_directory("photos/")
            >>>
            >>> # Single item access
            >>> img = batch[0]  # Returns ImageHandler
            >>> img.resize(width=800).to_grayscale()
            >>>
            >>> # Negative indexing
            >>> last = batch[-1]
            >>>
            >>> # Slicing
            >>> first_10 = batch[0:10]  # Returns BatchImageHandler
            >>> first_10.resize(width=800)
            >>>
            >>> # Step slicing
            >>> every_other = batch[::2]  # Returns BatchImageHandler
        """
        # Handle integer indexing
        if isinstance(key, int):
            # Support negative indexing
            if key < 0:
                key = len(self._contexts) + key

            # Bounds checking
            if key < 0 or key >= len(self._contexts):
                raise IndexError(f"Index {key} out of range for batch with {len(self._contexts)} images")

            # Get the context
            ctx = self._contexts[key]

            # Create ImageHandler wrapping this context (two-step pattern)
            handler = ImageHandler(ctx.path)
            handler._ctx = ctx

            return handler

        # Handle slice indexing
        elif isinstance(key, slice):
            # Get sliced contexts
            sliced_contexts = self._contexts[key]

            # Create new BatchImageHandler
            new_batch = BatchImageHandler([])
            new_batch._contexts = sliced_contexts
            new_batch._executor = ParallelExecutor()
            # Don't copy errors or progress callback for slices

            return new_batch

        else:
            raise TypeError(f"Indices must be integers or slices, not {type(key).__name__}")

    def __iter__(self):
        """
        Iterate over images in the batch.

        Yields:
            ImageHandler instances for each image

        Example:
            >>> batch = BatchImageHandler.from_directory("photos/")
            >>> for img in batch:
            ...     print(img.path)
            ...     img.resize(width=800)
        """
        for ctx in self._contexts:
            handler = ImageHandler(ctx.path)
            handler._ctx = ctx
            yield handler

    def __repr__(self) -> str:
        """
        String representation.

        Example:
            >>> print(batch)  # BatchImageHandler(50 images)
        """
        return f"BatchImageHandler({len(self._contexts)} images)"

    def __enter__(self) -> 'BatchImageHandler':
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.unload()

    def copy(self, deep: bool = True) -> 'BatchImageHandler':
        """
        Create an independent copy of this BatchImageHandler.

        Args:
            deep: If True, deep copy all loaded image pixel data in memory.
                  If False, create new contexts but don't copy pixel data
                  (unloaded images remain unloaded, loaded images stay loaded
                  but share memory until modified).

        Returns:
            New BatchImageHandler instance with copied state

        Example:
            >>> original = BatchImageHandler.from_directory("photos/", "*.jpg")
            >>> original.resize(width=800)
            >>>
            >>> # Create independent copy for different processing
            >>> copy1 = original.copy(deep=True)
            >>> copy1.to_grayscale()  # Won't affect original
            >>>
            >>> # Create memory-efficient copy
            >>> copy2 = original.copy(deep=False)
            >>> copy2.filter_by_size(min_width=1000)  # Filters without copying pixels
        """
        # Create new batch with empty list
        new_batch = BatchImageHandler([])

        # Copy contexts based on deep parameter
        if deep:
            # Deep copy: duplicate all pixel data for loaded images
            new_batch._contexts = [ImageLoader.copy(ctx) for ctx in self._contexts]
        else:
            # Lazy copy: new contexts without copying pixel data
            new_batch._contexts = [
                ImageContext(
                    path=ctx.path,
                    img=ctx.img,  # Share reference (or None if unloaded)
                    metadata=ctx.metadata.copy()
                )
                for ctx in self._contexts
            ]

        # Copy all state
        new_batch._errors = self._errors.copy()
        new_batch._progress_callback = self._progress_callback

        # Create NEW executor (don't share thread pools)
        new_batch._executor = ParallelExecutor()

        return new_batch


    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _compute_hash(self, ctx: ImageContext, hash_method: str, hash_size: int):
        """Compute perceptual hash for an image."""
        import imagehash

        try:
            # Load image if not already loaded
            if not ctx.is_loaded():
                ImageLoader.load(ctx)

            # Dispatch to appropriate hash function
            hash_functions = {
                'dhash': lambda img: imagehash.dhash(img, hash_size=hash_size),
                'phash': lambda img: imagehash.phash(img, hash_size=hash_size),
                'ahash': lambda img: imagehash.average_hash(img, hash_size=hash_size),
                'whash': lambda img: imagehash.whash(img, hash_size=hash_size)
            }

            if hash_method not in hash_functions:
                valid_methods = ', '.join(hash_functions.keys())
                raise ValueError(f"Unknown hash method '{hash_method}'. Valid options: {valid_methods}")

            return hash_functions[hash_method](ctx.img)

        except Exception as e:
            raise RuntimeError(f"Failed to compute {hash_method} for {ctx.path}: {e}")

    def _parse_size_string(self, size_str: Union[int, str]) -> int:
        """
        Parse size string to bytes.

        Args:
            size_str: Size as int (bytes) or string (e.g., '50KB', '1.5MB')

        Returns:
            Size in bytes

        Raises:
            ValueError: If format is invalid
        """
        import re

        # If already an integer, return as-is
        if isinstance(size_str, int):
            if size_str < 0:
                raise ValueError(f"Size cannot be negative: {size_str}")
            return size_str

        # Parse string format
        # Remove whitespace and convert to uppercase for easier parsing
        size_str = str(size_str).strip().upper()

        # Match pattern: number + optional unit
        pattern = r'^([\d.]+)\s*([A-Z]*)$'
        match = re.match(pattern, size_str)

        if not match:
            raise ValueError(
                f"Invalid size format: '{size_str}'. "
                f"Expected format: number + unit (e.g., '50KB', '1.5MB')"
            )

        number_str, unit = match.groups()

        try:
            number = float(number_str)
        except ValueError:
            raise ValueError(f"Invalid number in size: '{number_str}'")

        if number < 0:
            raise ValueError(f"Size cannot be negative: {size_str}")

        # Unit conversion to bytes
        unit_multipliers = {
            '': 1,                      # No unit = bytes
            'B': 1,                     # Bytes
            'BYTE': 1,
            'BYTES': 1,
            'KB': 1024,                 # Kilobytes
            'KILOBYTE': 1024,
            'KILOBYTES': 1024,
            'MB': 1024 ** 2,            # Megabytes
            'MEGABYTE': 1024 ** 2,
            'MEGABYTES': 1024 ** 2,
            'GB': 1024 ** 3,            # Gigabytes
            'GIGABYTE': 1024 ** 3,
            'GIGABYTES': 1024 ** 3,
        }

        if unit not in unit_multipliers:
            valid_units = [u for u in unit_multipliers.keys() if u]
            raise ValueError(
                f"Unknown unit: '{unit}'. "
                f"Valid units: {', '.join(valid_units)}"
            )

        # Calculate bytes
        bytes_value = number * unit_multipliers[unit]

        return int(bytes_value)

    def _format_size(self, bytes_value: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0 or unit == 'GB':
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} GB"

    def _sequential_process(self,
                           func: Callable,
                           contexts: List[ImageContext]) -> Tuple[List[ImageContext], List[Tuple[Path, Exception]]]:
        """Sequential processing fallback."""
        results = []
        errors = []

        for ctx in contexts:
            try:
                result = func(ctx)
                results.append(result)
            except Exception as e:
                errors.append((ctx.path, e))

        return results, errors
