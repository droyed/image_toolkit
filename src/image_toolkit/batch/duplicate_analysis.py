from typing import List, Union, Optional, Callable, Dict, Any, Tuple
from pathlib import Path


def _compute_hash_for_path(args: Tuple[Path, str, int]) -> Tuple[Path, Any]:
    """
    Compute hash for a single image path (module-level for pickling).

    Args:
        args: Tuple of (path, hash_method, hash_size)

    Returns:
        Tuple of (path, hash_value)

    Raises:
        RuntimeError: If hash computation fails
    """
    import imagehash
    from PIL import Image

    path, hash_method, hash_size = args

    try:
        with Image.open(path) as img:
            # Compute hash based on method
            if hash_method == 'dhash':
                hash_val = imagehash.dhash(img, hash_size=hash_size)
            elif hash_method == 'phash':
                hash_val = imagehash.phash(img, hash_size=hash_size)
            elif hash_method == 'ahash':
                hash_val = imagehash.average_hash(img, hash_size=hash_size)
            elif hash_method == 'whash':
                hash_val = imagehash.whash(img, hash_size=hash_size)
            else:
                raise ValueError(f"Unknown hash method: {hash_method}")

            return (path, hash_val)
    except Exception as e:
        raise RuntimeError(f"Failed to compute {hash_method} for {path}: {e}")


def analyze_duplicates_on_image_paths(
    image_paths: List[Union[str, Path]],
    hash_method: str = 'dhash',
    hash_size: int = 8,
    threshold: Optional[int] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Analyze duplicate images from a list of image paths (generic utility function).
    
    This is a standalone function that works on raw image paths without requiring
    BatchImageHandler. Useful for custom workflows or integration with other tools.
    
    Args:
        image_paths: List of paths to image files
        hash_method: Hashing algorithm ('dhash', 'phash', 'ahash', 'whash')
        hash_size: Hash size (8 = 64-bit, 16 = 256-bit)
        threshold: Hamming distance threshold (None = use recommended default)
        parallel: Use parallel processing for hash computation
        max_workers: Maximum number of parallel workers (None = CPU count)
        progress_callback: Optional callback function(current, total) for progress tracking
    
    Returns:
        Dictionary containing:
            - 'duplicate_groups': List[List[Path]] - groups with 2+ similar images
            - 'singleton_groups': List[List[Path]] - groups with exactly 1 image (no duplicates)
            - 'hash_map': Dict[Path, ImageHash] - computed hashes
            - 'errors': List[Tuple[Path, Exception]] - processing errors
            - 'stats': Dict with summary statistics
    
    Example:
        >>> from pathlib import Path
        >>> from image_toolkit.batch_handler import analyze_duplicates_on_image_paths
        >>> 
        >>> # Get list of image paths
        >>> image_paths = list(Path("photos/").glob("*.jpg"))
        >>> 
        >>> # Analyze duplicates
        >>> analysis = analyze_duplicates_on_image_paths(
        ...     image_paths,
        ...     hash_method='dhash',
        ...     threshold=5
        ... )
        >>> 
        >>> print(f"Found {len(analysis['duplicate_groups'])} duplicate groups")
        >>> print(f"Found {len(analysis['singleton_groups'])} singleton groups")
        >>>
        >>> for group in analysis['duplicate_groups']:
        ...     print(f"Group: {[p.name for p in group]}")
        >>>
        >>> # Use in custom workflow
        >>> duplicates_to_remove = []
        >>> for group in analysis['duplicate_groups']:
        ...     # Keep first, mark rest for removal
        ...     duplicates_to_remove.extend(group[1:])

    Note:
        - Returns Path objects, not ImageContext objects
        - Completely independent of BatchImageHandler
        - Can be used in any Python script or workflow
        - singleton_groups is a list of single-item lists (for consistency with duplicate_groups)
    """
    import imagehash
    from PIL import Image
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path
    
    # Convert to Path objects
    image_paths = [Path(p) for p in image_paths]
    
    if not image_paths:
        return {
            'duplicate_groups': [],
            'singleton_groups': [],
            'hash_map': {},
            'errors': [],
            'stats': {
                'total_images': 0,
                'num_duplicate_groups': 0,
                'num_unique_images': 0,
                'total_duplicates': 0,
                'hash_method': hash_method,
                'threshold': threshold
            }
        }
    
    # Validate hash method
    valid_methods = ['dhash', 'phash', 'ahash', 'whash']
    if hash_method not in valid_methods:
        raise ValueError(
            f"Invalid hash_method '{hash_method}'. "
            f"Must be one of: {', '.join(valid_methods)}"
        )
    
    # Set default threshold based on hash method if not provided
    if threshold is None:
        default_thresholds = {
            'ahash': 3,
            'dhash': 5,
            'phash': 8,
            'whash': 10
        }
        threshold = default_thresholds[hash_method]
        print(f"Using default threshold={threshold} for {hash_method}")
    
    # Warn if threshold seems unusually high
    max_recommended = {
        'ahash': 10,
        'dhash': 15,
        'phash': 20,
        'whash': 25
    }
    if threshold > max_recommended[hash_method]:
        print(f"⚠️  Warning: threshold={threshold} is high for {hash_method}, "
              f"may produce false positives (recommended max: {max_recommended[hash_method]})")

    # Step 1: Compute hashes for all images
    print(f"Computing {hash_method} hashes for {len(image_paths)} images...")
    
    hash_map = {}
    errors = []
    
    if parallel:
        # Parallel hash computation
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_hash_for_path, (path, hash_method, hash_size)): path
                      for path in image_paths}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(image_paths))
                
                try:
                    path, hash_val = future.result()
                    hash_map[path] = hash_val
                except Exception as e:
                    original_path = futures[future]
                    errors.append((original_path, e))
    else:
        # Sequential hash computation
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, len(image_paths))

            try:
                path, hash_val = _compute_hash_for_path((path, hash_method, hash_size))
                hash_map[path] = hash_val
            except Exception as e:
                errors.append((path, e))
    
    # Step 2: Find duplicate groups
    print(f"Finding duplicates with threshold={threshold}...")
    duplicate_groups = []
    singleton_groups = []
    processed = set()

    paths = list(hash_map.keys())
    for i, path1 in enumerate(paths):
        if path1 in processed:
            continue

        hash1 = hash_map[path1]
        group = [path1]

        # Compare with remaining images
        for path2 in paths[i+1:]:
            if path2 in processed:
                continue

            hash2 = hash_map[path2]
            distance = hash1 - hash2  # Hamming distance

            if distance <= threshold:
                group.append(path2)
                processed.add(path2)

        if len(group) > 1:
            duplicate_groups.append(group)
        else:
            singleton_groups.append([path1])  # Wrap in list for consistency
        processed.add(path1)
    
    # Build statistics
    total_duplicates = sum(len(group) for group in duplicate_groups)

    stats = {
        'total_images': len(image_paths),
        'num_duplicate_groups': len(duplicate_groups),
        'num_unique_images': len(singleton_groups),
        'total_duplicates': total_duplicates,
        'num_errors': len(errors),
        'hash_method': hash_method,
        'threshold': threshold
    }

    print(f"Found {len(duplicate_groups)} duplicate groups, "
          f"{len(singleton_groups)} singleton groups, "
          f"{len(errors)} errors")

    return {
        'duplicate_groups': duplicate_groups,
        'singleton_groups': singleton_groups,
        'hash_map': hash_map,
        'errors': errors,
        'stats': stats
    }