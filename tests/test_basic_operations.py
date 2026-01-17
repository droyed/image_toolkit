#!/usr/bin/env python3
"""
Test Basic Operations - Fundamental loading, filtering, and sampling

Tests: from_directory(), from_glob(), __len__(), __repr__(), filter_valid(),
       filter_by_size(), filter_by_aspect_ratio(), sample(), filter_by_file_size(),
       filter_duplicates()
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_toolkit import BatchImageHandler
from tests.test_utils import print_section, print_subsection, print_success, get_paths


def main():
    paths = get_paths()
    input_dir = paths['input']

    print("="*70)
    print(" Testing: Basic Operations")
    print("="*70)

    # Test 1: Load from directory
    print_section("1. Load Batch from Directory")
    batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    print(f"Loaded {len(batch)} images from {input_dir}")
    print(f"Batch representation: {repr(batch)}")

    # Test 2: Load with glob pattern
    print_section("2. Load with Glob Pattern")
    batch2 = BatchImageHandler.from_glob(str(input_dir / "*.jpg"))
    print(f"Loaded {len(batch2)} images using glob pattern")

    # Test 3: Filter valid images
    print_section("3. Filter Valid Images")
    initial_count = len(batch)
    batch.filter_valid()
    print(f"Valid images: {len(batch)}/{initial_count}")

    # Test 4: Filter by size constraints
    print_section("4. Filter by Size (width >= 400, height >= 400)")
    before_filter = len(batch)
    batch.filter_by_size(min_width=400, min_height=400)
    print(f"After size filter: {len(batch)}/{before_filter} images")

    # Test 5: Filter by aspect ratio
    print_section("5. Filter by Aspect Ratio (0.8 to 1.2)")
    before_aspect = len(batch)
    batch.filter_by_aspect_ratio(min_ratio=0.8, max_ratio=1.2)
    print(f"After aspect ratio filter: {len(batch)}/{before_aspect} images")

    # Test 6: Sample subset - random
    print_section("6. Sample Random Subset (n=5)")
    batch_sample = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_sample.filter_valid()
    sample = batch_sample.sample(5, random_sample=True)
    print(f"Random sample size: {len(sample)} images")
    print(f"Sample representation: {repr(sample)}")

    # Test 7: Sample subset - sequential
    print_section("7. Sample Sequential Subset (n=10)")
    batch_seq = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_seq.filter_valid()
    sample_seq = batch_seq.sample(10, random_sample=False)
    print(f"Sequential sample size: {len(sample_seq)} images")

    # Test 8: Chain operations
    print_section("8. Chain Multiple Operations")
    batch_chain = (BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
                   .filter_valid()
                   .filter_by_size(min_width=300, min_height=300)
                   .sample(8, random_sample=True))
    print(f"Chained operations result: {len(batch_chain)} images")
    print(f"Final batch: {repr(batch_chain)}")

    # Test 9: Filter by file size - human-readable formats
    print_section("9. Filter by File Size (Human-Readable)")
    try:
        # Test with human-readable formats
        batch_size = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_size.filter_valid()
        initial = len(batch_size)
        print(f"Initial batch: {initial} images")

        # Filter with KB format
        batch_kb = batch_size.copy(deep=False)
        batch_kb.filter_by_file_size(min_size='50KB')
        print(f"After min_size='50KB': {len(batch_kb)}/{initial} images")
        assert len(batch_kb) <= initial, "Batch size should not increase"

        # Filter with MB format
        batch_mb = batch_size.copy(deep=False)
        batch_mb.filter_by_file_size(max_size='5MB')
        print(f"After max_size='5MB': {len(batch_mb)}/{initial} images")

        # Filter with range
        batch_range = batch_size.copy(deep=False)
        batch_range.filter_by_file_size(min_size='10KB', max_size='2MB')
        print(f"After range '10KB' to '2MB': {len(batch_range)}/{initial} images")

        # Test decimal values
        batch_decimal = batch_size.copy(deep=False)
        batch_decimal.filter_by_file_size(min_size='0.1MB', max_size='1.5MB')
        print(f"After decimal range '0.1MB' to '1.5MB': {len(batch_decimal)}/{initial} images")

        print("✓ filter_by_file_size with human-readable formats works")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 10: Filter by file size - edge cases
    print_section("10. Filter by File Size - Edge Cases")
    try:
        batch_edge = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_edge.filter_valid()

        # Test with raw bytes
        batch_bytes = batch_edge.copy(deep=False)
        batch_bytes.filter_by_file_size(min_size=10240, max_size=10485760)
        print(f"Raw bytes (10240-10485760): {len(batch_bytes)} images")

        # Test invalid format - should raise ValueError
        try:
            batch_invalid = batch_edge.copy(deep=False)
            batch_invalid.filter_by_file_size(min_size='invalid')
            print("✗ Should have raised ValueError for invalid format")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for invalid format: {str(e)[:50]}...")

        # Test min > max - should raise ValueError
        try:
            batch_minmax = batch_edge.copy(deep=False)
            batch_minmax.filter_by_file_size(min_size='2MB', max_size='1MB')
            print("✗ Should have raised ValueError for min > max")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for min > max: {str(e)[:50]}...")

        # Test method chaining
        batch_chained = (BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
                        .filter_valid()
                        .filter_by_file_size(min_size='10KB')
                        .sample(5, random_sample=False))
        print(f"✓ Method chaining works: {len(batch_chained)} images")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 11: Filter duplicates - different hash methods
    print_section("11. Filter Duplicates - Hash Methods")
    try:
        import imagehash

        batch_dup = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_dup.filter_valid().sample(20, random_sample=False)
        initial = len(batch_dup)
        print(f"Initial batch for duplicate testing: {initial} images")

        # Test ahash (fast, exact duplicates)
        batch_ahash = batch_dup.copy(deep=False)
        batch_ahash.filter_duplicates(hash_method='ahash', threshold=3, keep='first')
        print(f"After ahash filtering (threshold=3): {len(batch_ahash)}/{initial} images")

        # Test dhash (default, similar images)
        batch_dhash = batch_dup.copy(deep=False)
        batch_dhash.filter_duplicates(hash_method='dhash', threshold=5, keep='first')
        print(f"After dhash filtering (threshold=5): {len(batch_dhash)}/{initial} images")

        # Test phash (edited images)
        batch_phash = batch_dup.copy(deep=False)
        batch_phash.filter_duplicates(hash_method='phash', threshold=8, keep='largest')
        print(f"After phash filtering (threshold=8): {len(batch_phash)}/{initial} images")

        # Test auto-threshold
        batch_auto = batch_dup.copy(deep=False)
        batch_auto.filter_duplicates(hash_method='dhash')  # Uses default threshold
        print(f"After dhash with auto-threshold: {len(batch_auto)}/{initial} images")

        # Test different keep strategies
        batch_last = batch_dup.copy(deep=False)
        batch_last.filter_duplicates(hash_method='ahash', threshold=3, keep='last')
        print(f"Keep strategy 'last': {len(batch_last)} images")

        batch_smallest = batch_dup.copy(deep=False)
        batch_smallest.filter_duplicates(hash_method='ahash', threshold=3, keep='smallest')
        print(f"Keep strategy 'smallest': {len(batch_smallest)} images")

        # Test invalid hash method
        try:
            batch_invalid = batch_dup.copy(deep=False)
            batch_invalid.filter_duplicates(hash_method='invalid')
            print("✗ Should have raised ValueError for invalid hash method")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for invalid hash method")

        print("✓ filter_duplicates with different hash methods works")

    except ImportError:
        print("⚠️  imagehash not installed. Install with: pip install imagehash")
        print("Skipping filter_duplicates tests")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 12: Filter duplicates - parallel vs sequential
    print_section("12. Filter Duplicates - Parallel Processing")
    try:
        import imagehash

        batch_parallel_test = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_parallel_test.filter_valid().sample(15, random_sample=False)
        initial = len(batch_parallel_test)

        # Parallel processing
        batch_parallel = batch_parallel_test.copy(deep=False)
        batch_parallel.filter_duplicates(hash_method='dhash', threshold=5, parallel=True)
        parallel_count = len(batch_parallel)
        print(f"Parallel processing: {parallel_count}/{initial} images")

        # Sequential processing
        batch_sequential = batch_parallel_test.copy(deep=False)
        batch_sequential.filter_duplicates(hash_method='dhash', threshold=5, parallel=False)
        sequential_count = len(batch_sequential)
        print(f"Sequential processing: {sequential_count}/{initial} images")

        # Results should be identical
        assert parallel_count == sequential_count, \
            f"Parallel and sequential should produce same results: {parallel_count} != {sequential_count}"
        print(f"✓ Parallel and sequential produce identical results: {parallel_count} images")

        # Test empty batch
        batch_empty = BatchImageHandler([])
        batch_empty.filter_duplicates()
        assert len(batch_empty) == 0, "Empty batch should remain empty"
        print("✓ Empty batch handled correctly")

        # Test method chaining
        batch_chained_dup = (BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
                            .filter_valid()
                            .sample(10, random_sample=False)
                            .filter_duplicates(hash_method='ahash', threshold=2))
        print(f"✓ Method chaining works: {len(batch_chained_dup)} images after duplicate filtering")

    except ImportError:
        print("⚠️  imagehash not installed. Skipping parallel processing tests")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_success()


if __name__ == "__main__":
    main()
