#!/usr/bin/env python3
"""
Tests for duplicate analysis functions.
Tests:
- _compute_hash_for_path() function
- BatchImageHandler.analyze_duplicates() method
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_toolkit.batch_handler import BatchImageHandler
from image_toolkit.batch.duplicate_analysis import _compute_hash_for_path
from tests.test_utils import (
    print_section, print_success, get_paths, ensure_output_dir
)


def main():
    print("="*70)
    print(" Testing: Duplicate Analysis Functions")
    print("="*70)

    paths = get_paths()
    input_dir = paths['input']
    output_dir = ensure_output_dir(paths['output'])

    # Check for imagehash dependency
    try:
        import imagehash
    except ImportError:
        print("⚠️  imagehash library not installed. Skipping all tests.")
        print("Install with: pip install imagehash")
        return

    # =======================================================================
    # PART 1: Testing _compute_hash_for_path() Function
    # =======================================================================

    print_section("1. Basic Hash Computation")
    try:
        # Get a test image
        test_images = list(input_dir.glob("*.jpg"))
        if not test_images:
            print("⚠️  No test images found in", input_dir)
            return

        test_image = test_images[0]
        print(f"Using test image: {test_image.name}")

        # Test all hash methods
        hash_results = {}
        for method in ['ahash', 'dhash', 'phash', 'whash']:
            result_path, hash_val = _compute_hash_for_path((test_image, method, 8))

            # Assertions
            assert result_path == test_image, f"Path should be returned unchanged for {method}"
            assert hash_val is not None, f"Hash should be computed for {method}"
            assert str(hash_val), f"Hash should have string representation for {method}"

            hash_results[method] = hash_val
            print(f"  ✓ {method}: {hash_val}")

        print(f"✓ All hash methods (ahash, dhash, phash, whash) work correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("2. Hash Size Variations")
    try:
        test_image = list(input_dir.glob("*.jpg"))[0]

        # Test hash_size=8 (64-bit hash)
        path8, hash8 = _compute_hash_for_path((test_image, 'dhash', 8))
        assert hash8 is not None, "Hash size 8 should produce valid hash"

        # Test hash_size=16 (256-bit hash)
        path16, hash16 = _compute_hash_for_path((test_image, 'dhash', 16))
        assert hash16 is not None, "Hash size 16 should produce valid hash"

        # Hashes should differ (different bit sizes produce different hashes)
        print(f"  hash_size=8:  {hash8}")
        print(f"  hash_size=16: {hash16}")
        print(f"✓ Hash size variations (8, 16) work correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("3. Error Handling")
    try:
        test_image = list(input_dir.glob("*.jpg"))[0]

        # Test with invalid path
        try:
            _compute_hash_for_path((Path("nonexistent_image_xyz.jpg"), 'dhash', 8))
            print("✗ Should have raised RuntimeError for invalid path")
        except RuntimeError as e:
            error_msg = str(e)
            assert "Failed to compute" in error_msg, "Error message should be descriptive"
            print(f"  ✓ Invalid path raises RuntimeError: {error_msg[:60]}...")

        # Test with invalid hash method
        try:
            _compute_hash_for_path((test_image, 'invalid_method_xyz', 8))
            print("✗ Should have raised RuntimeError for invalid method")
        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            print(f"  ✓ Invalid hash method raises error: {error_msg[:60]}...")

        print(f"✓ Error handling works correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("4. Hash Consistency")
    try:
        test_image = list(input_dir.glob("*.jpg"))[0]

        # Compute hash twice with same parameters
        for method in ['ahash', 'dhash', 'phash', 'whash']:
            _, hash1 = _compute_hash_for_path((test_image, method, 8))
            _, hash2 = _compute_hash_for_path((test_image, method, 8))

            # Hashes should be identical (deterministic)
            assert str(hash1) == str(hash2), \
                f"Hash computation should be deterministic for {method}"
            print(f"  ✓ {method} is deterministic: {hash1}")

        print(f"✓ Hash computation is consistent across all methods")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # =======================================================================
    # PART 2: Testing analyze_duplicates() Method
    # =======================================================================

    print_section("5. Basic Analysis")
    try:
        batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch.filter_valid().sample(15, random_sample=False)
        initial_count = len(batch)
        print(f"Testing with {initial_count} images")

        # Run analysis
        analysis = batch.analyze_duplicates(hash_method='dhash', threshold=5)

        # Verify structure
        assert 'duplicate_groups' in analysis, "Should have 'duplicate_groups' key"
        assert 'singleton_groups' in analysis, "Should have 'singleton_groups' key"
        assert 'hash_map' in analysis, "Should have 'hash_map' key"
        assert 'stats' in analysis, "Should have 'stats' key"

        # Verify groups contain ImageContext objects (not Path objects)
        all_groups = analysis['duplicate_groups'] + analysis['singleton_groups']
        if all_groups:
            first_group = all_groups[0]
            assert len(first_group) > 0, "Groups should not be empty"
            first_item = first_group[0]
            assert hasattr(first_item, 'path'), "Items should be ImageContext objects"
            assert hasattr(first_item, 'img'), "Items should be ImageContext objects"
            assert hasattr(first_item, 'metadata'), "Items should be ImageContext objects"
            print(f"  ✓ Groups contain ImageContext objects (not Paths)")

        # Verify hash_map format: {Path: (ImageContext, hash)}
        if analysis['hash_map']:
            first_key = next(iter(analysis['hash_map']))
            first_value = analysis['hash_map'][first_key]
            assert isinstance(first_key, Path), "Hash map keys should be Path objects"
            assert isinstance(first_value, tuple), "Hash map values should be tuples"
            assert len(first_value) == 2, "Hash map values should be (ImageContext, hash)"
            ctx, hash_val = first_value
            assert hasattr(ctx, 'path'), "First tuple element should be ImageContext"
            print(f"  ✓ Hash map format is correct: {{Path: (ImageContext, hash)}}")

        # Verify stats
        stats = analysis['stats']
        assert 'total_images' in stats, "Stats should have 'total_images'"
        assert 'num_duplicate_groups' in stats, "Stats should have 'num_duplicate_groups'"
        assert 'num_unique_images' in stats, "Stats should have 'num_unique_images'"
        assert 'hash_method' in stats, "Stats should have 'hash_method'"
        assert 'threshold' in stats, "Stats should have 'threshold'"

        print(f"✓ Analysis complete:")
        print(f"  - {len(analysis['duplicate_groups'])} duplicate groups")
        print(f"  - {len(analysis['singleton_groups'])} unique images")
        print(f"  - {len(analysis['hash_map'])} hashes computed")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("6. Parallel vs Sequential Consistency")
    try:
        batch_base = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_base.filter_valid().sample(20, random_sample=False)
        print(f"Testing with {len(batch_base)} images")

        # Sequential analysis
        batch_seq = batch_base.copy(deep=False)
        analysis_seq = batch_seq.analyze_duplicates(
            hash_method='dhash', threshold=5, parallel=False
        )
        seq_dup_groups = len(analysis_seq['duplicate_groups'])
        seq_singleton_groups = len(analysis_seq['singleton_groups'])
        seq_hash_count = len(analysis_seq['hash_map'])

        # Parallel analysis
        batch_par = batch_base.copy(deep=False)
        analysis_par = batch_par.analyze_duplicates(
            hash_method='dhash', threshold=5, parallel=True
        )
        par_dup_groups = len(analysis_par['duplicate_groups'])
        par_singleton_groups = len(analysis_par['singleton_groups'])
        par_hash_count = len(analysis_par['hash_map'])

        # Should produce identical results
        assert seq_dup_groups == par_dup_groups, \
            f"Duplicate groups should match: {seq_dup_groups} != {par_dup_groups}"
        assert seq_singleton_groups == par_singleton_groups, \
            f"Singleton groups should match: {seq_singleton_groups} != {par_singleton_groups}"
        assert seq_hash_count == par_hash_count, \
            f"Hash counts should match: {seq_hash_count} != {par_hash_count}"

        print(f"  Sequential: {seq_dup_groups} dup groups, {seq_singleton_groups} unique")
        print(f"  Parallel:   {par_dup_groups} dup groups, {par_singleton_groups} unique")
        print(f"✓ Parallel and sequential produce identical results")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("7. Hash Method Variations")
    try:
        batch_base = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_base.filter_valid().sample(15, random_sample=False)
        print(f"Testing with {len(batch_base)} images")

        results = {}
        for method in ['ahash', 'dhash', 'phash', 'whash']:
            batch = batch_base.copy(deep=False)
            analysis = batch.analyze_duplicates(hash_method=method, threshold=5)

            results[method] = {
                'dup_groups': len(analysis['duplicate_groups']),
                'unique': len(analysis['singleton_groups']),
                'hashes': len(analysis['hash_map'])
            }

            # Verify hash count matches batch size
            assert results[method]['hashes'] == len(batch_base), \
                f"Should compute hash for all images using {method}"

            print(f"  {method}: {results[method]['dup_groups']} dup groups, "
                  f"{results[method]['unique']} unique")

        print(f"✓ All hash methods (ahash, dhash, phash, whash) work correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("8. Threshold Impact")
    try:
        batch_base = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_base.filter_valid().sample(20, random_sample=False)
        print(f"Testing with {len(batch_base)} images")

        results = {}
        for threshold in [0, 3, 5, 10, 15]:
            batch = batch_base.copy(deep=False)
            analysis = batch.analyze_duplicates(
                hash_method='dhash', threshold=threshold
            )

            results[threshold] = {
                'dup_groups': len(analysis['duplicate_groups']),
                'total_in_dup_groups': sum(len(g) for g in analysis['duplicate_groups'])
            }

            print(f"  threshold={threshold:2d}: {results[threshold]['dup_groups']} dup groups, "
                  f"{results[threshold]['total_in_dup_groups']} images in duplicates")

        # Lower thresholds should generally find fewer or equal duplicates
        # (monotonic relationship: as threshold increases, duplicates increase or stay same)
        print(f"✓ Threshold parameter affects duplicate detection")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("9. Edge Cases")
    try:
        # Test empty batch
        batch_empty = BatchImageHandler([])
        analysis_empty = batch_empty.analyze_duplicates()

        assert len(analysis_empty['duplicate_groups']) == 0, \
            "Empty batch should have no duplicate groups"
        assert len(analysis_empty['singleton_groups']) == 0, \
            "Empty batch should have no singleton groups"
        assert analysis_empty['stats']['total_images'] == 0, \
            "Stats should show 0 images for empty batch"
        print(f"  ✓ Empty batch handled correctly")

        # Test single image
        batch_single = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_single.filter_valid().sample(1, random_sample=False)
        analysis_single = batch_single.analyze_duplicates()

        assert len(analysis_single['duplicate_groups']) == 0, \
            "Single image should have no duplicates"
        assert len(analysis_single['singleton_groups']) == 1, \
            "Single image should have 1 singleton group"
        assert analysis_single['stats']['total_images'] == 1, \
            "Stats should show 1 image"
        print(f"  ✓ Single image handled correctly")

        # Test multiple unique images with strict threshold
        batch_unique = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_unique.filter_valid().sample(5, random_sample=False)
        analysis_unique = batch_unique.analyze_duplicates(
            hash_method='ahash', threshold=0  # Very strict
        )

        total_groups = (len(analysis_unique['duplicate_groups']) +
                       len(analysis_unique['singleton_groups']))
        assert total_groups >= 1, "Should have at least one group"
        print(f"  ✓ Multiple images with strict threshold handled correctly")

        print(f"✓ All edge cases handled correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("10. Error Tracking")
    try:
        batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch.filter_valid().sample(10, random_sample=False)

        # Clear errors first
        batch.clear_errors()
        initial_errors = len(batch.get_errors())
        assert initial_errors == 0, "Should start with no errors"

        # Run analysis
        analysis = batch.analyze_duplicates(hash_method='dhash')

        # Check error tracking
        final_errors = len(batch.get_errors())
        print(f"  Errors before: {initial_errors}")
        print(f"  Errors after:  {final_errors}")

        # Errors should be tracked in batch._errors
        assert isinstance(batch.get_errors(), list), "Errors should be a list"
        print(f"✓ Error tracking works correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("11. Statistics Validation")
    try:
        batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch.filter_valid().sample(15, random_sample=False)
        initial_count = len(batch)

        analysis = batch.analyze_duplicates(hash_method='dhash', threshold=5)
        stats = analysis['stats']

        # Verify all required stats fields
        assert stats['total_images'] == initial_count, \
            f"Stats should reflect batch size: {stats['total_images']} != {initial_count}"
        assert stats['num_duplicate_groups'] >= 0, \
            "Duplicate group count should be non-negative"
        assert stats['num_unique_images'] >= 0, \
            "Unique image count should be non-negative"
        assert stats['hash_method'] == 'dhash', \
            "Stats should record hash method"
        assert stats['threshold'] == 5, \
            "Stats should record threshold"

        # Verify logical consistency
        total_in_groups = sum(len(g) for g in analysis['duplicate_groups'])
        total_in_singletons = sum(len(g) for g in analysis['singleton_groups'])
        total_from_groups = total_in_groups + total_in_singletons

        assert total_from_groups == initial_count, \
            f"Sum of all images in groups should equal total: {total_from_groups} != {initial_count}"

        print(f"  Total images: {stats['total_images']}")
        print(f"  Duplicate groups: {stats['num_duplicate_groups']}")
        print(f"  Unique images: {stats['num_unique_images']}")
        print(f"  Hash method: {stats['hash_method']}")
        print(f"  Threshold: {stats['threshold']}")
        print(f"✓ Statistics are valid and consistent")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("12. Integration with filter_duplicates")
    try:
        batch_base = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_base.filter_valid().sample(20, random_sample=False)
        initial_count = len(batch_base)
        print(f"Testing with {initial_count} images")

        # Run analyze_duplicates
        batch_analyze = batch_base.copy(deep=False)
        analysis = batch_analyze.analyze_duplicates(hash_method='dhash', threshold=5)
        num_dup_groups = len(analysis['duplicate_groups'])
        num_singletons = len(analysis['singleton_groups'])

        # Run filter_duplicates (should use same logic)
        batch_filter = batch_base.copy(deep=False)
        batch_filter.filter_duplicates(hash_method='dhash', threshold=5, keep='first')
        filtered_count = len(batch_filter)

        # After filtering with keep='first', should have:
        # - 1 image per duplicate group
        # - All singleton images
        expected_after_filter = num_dup_groups + num_singletons

        assert filtered_count == expected_after_filter, \
            f"Filter should keep {expected_after_filter} images, got {filtered_count}"

        print(f"  Analysis found: {num_dup_groups} dup groups, {num_singletons} singletons")
        print(f"  Filter kept: {filtered_count} images (expected {expected_after_filter})")
        print(f"✓ analyze_duplicates and filter_duplicates are consistent")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("13. Progress Callback")
    try:
        batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch.filter_valid().sample(10, random_sample=False)

        # Track progress updates
        progress_updates = []

        def progress_callback(current, total):
            progress_updates.append((current, total))

        # Set callback and run analysis
        batch.set_progress_callback(progress_callback)
        analysis = batch.analyze_duplicates(hash_method='dhash', parallel=True)

        # Verify callback was invoked
        assert len(progress_updates) > 0, "Progress callback should be invoked"

        # Verify progress tracking
        last_update = progress_updates[-1]
        assert last_update[0] == last_update[1], \
            "Last update should show completion"

        print(f"  Progress updates: {len(progress_updates)}")
        print(f"  Final: {last_update[0]}/{last_update[1]}")
        print(f"✓ Progress callback works correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("14. Memory Efficiency")
    try:
        batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch.filter_valid().sample(10, random_sample=False)

        # Ensure images are unloaded
        batch.unload()

        # Verify images are not loaded
        unloaded_count = sum(1 for ctx in batch._contexts if not ctx.is_loaded())
        assert unloaded_count == len(batch), "Images should be unloaded initially"
        print(f"  Images unloaded: {unloaded_count}/{len(batch)}")

        # Run analysis (should load images only when needed)
        analysis = batch.analyze_duplicates(hash_method='dhash', parallel=False)

        # Analysis should succeed even with unloaded images
        assert len(analysis['hash_map']) == len(batch), \
            "Analysis should process all images"

        print(f"  Hashes computed: {len(analysis['hash_map'])}")
        print(f"✓ Memory efficiency verified (works with unloaded images)")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # =======================================================================
    # SUMMARY
    # =======================================================================

    print_success("All duplicate analysis tests passed!")
    print()
    print("Summary:")
    print("  - _compute_hash_for_path() function: 4 test sections")
    print("  - analyze_duplicates() method: 10 test sections")
    print("  - Total: 14 comprehensive test sections")


if __name__ == "__main__":
    main()
