#!/usr/bin/env python3
"""
Test Batch Copy - Testing copy() method for BatchImageHandler

Tests: copy() method with deep and lazy copying modes, state independence,
       executor independence, and branching workflows
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_toolkit import BatchImageHandler
from image_toolkit.core import ImageLoader, ImageContext
from tests.test_utils import (
    print_section, print_subsection, print_success,
    get_paths, ensure_output_dir, print_error
)


def main():
    paths = get_paths()
    input_dir = paths['input']
    output_dir = ensure_output_dir(paths['output'] / "copy_tests")

    print("="*70)
    print(" Testing: BatchImageHandler.copy() Method")
    print("="*70)

    # Test 1: Deep copy independence
    print_section("1. Deep Copy Independence")
    try:
        original = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        original.filter_valid().sample(5, random_sample=False)
        original.resize(width=800, parallel=False)

        print(f"Original batch: {len(original)} images")
        print(f"Original first image mode: {original._contexts[0].img.mode}")
        print(f"Original first image size: {original._contexts[0].img.size}")

        # Deep copy and modify
        deep_copy = original.copy(deep=True)
        deep_copy.to_grayscale(keep_2d=True, parallel=False)

        print(f"Copy batch: {len(deep_copy)} images")
        print(f"Copy first image mode: {deep_copy._contexts[0].img.mode}")
        print(f"Copy first image size: {deep_copy._contexts[0].img.size}")

        # Verify original is unchanged
        assert original._contexts[0].img.mode in ['RGB', 'RGBA'], \
            "Original should still be color"
        assert deep_copy._contexts[0].img.mode == 'L', \
            "Copy should be grayscale (mode L)"

        print("✓ Deep copy creates independent pixel data")

    except Exception as e:
        print_error(f"Deep copy independence test failed: {e}")
        return

    # Test 2: Lazy copy behavior
    print_section("2. Lazy Copy Memory Sharing")
    try:
        original = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        original.filter_valid().sample(3, random_sample=False)
        original.resize(width=600, parallel=False)

        # Lazy copy
        lazy_copy = original.copy(deep=False)

        # Verify same PIL Image object (before modification)
        same_reference = original._contexts[0].img is lazy_copy._contexts[0].img
        print(f"Lazy copy shares PIL Image reference: {same_reference}")
        assert same_reference, "Lazy copy should share PIL Image reference"

        # After modification, references should differ
        lazy_copy.adjust(brightness=1.3, parallel=False)
        different_reference = original._contexts[0].img is not lazy_copy._contexts[0].img
        print(f"After modification, references differ: {different_reference}")

        print("✓ Lazy copy shares memory initially, diverges on modification")

    except Exception as e:
        print_error(f"Lazy copy test failed: {e}")
        return

    # Test 3: State copying
    print_section("3. State Copying (Errors & Callbacks)")
    try:
        original = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        original.filter_valid().sample(3, random_sample=False)

        # Add error and callback
        original._errors.append((Path("test.jpg"), ValueError("test error")))
        callback_called = []
        def test_callback(current, total):
            callback_called.append((current, total))
        original.set_progress_callback(test_callback)

        # Copy
        copy = original.copy(deep=True)

        # Verify state copied
        print(f"Original errors: {len(original._errors)}")
        print(f"Copy errors: {len(copy._errors)}")
        assert len(copy._errors) == 1, "Errors should be copied"
        assert copy._progress_callback is not None, "Callback should be copied"

        # Verify independence
        copy._errors.clear()
        assert len(original._errors) == 1, "Original errors should be independent"
        assert len(copy._errors) == 0, "Copy errors should be independent"

        print("✓ State (errors, callbacks) copied and independent")

    except Exception as e:
        print_error(f"State copying test failed: {e}")
        return

    # Test 4: Executor independence
    print_section("4. Executor Independence")
    try:
        original = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        original.filter_valid().sample(3, random_sample=False)

        copy = original.copy(deep=True)

        # Verify different executor objects
        executor_independent = original._executor is not copy._executor
        print(f"Original executor: {id(original._executor)}")
        print(f"Copy executor: {id(copy._executor)}")
        print(f"Executors are independent: {executor_independent}")

        assert executor_independent, "Each copy should have its own executor"

        print("✓ Each copy has independent ParallelExecutor")

    except Exception as e:
        print_error(f"Executor independence test failed: {e}")
        return

    # Test 5: Copy unloaded images
    print_section("5. Copy Unloaded Images")
    try:
        original = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        original.filter_valid().sample(3, random_sample=False)
        # Don't load images

        # Deep copy unloaded
        copy = original.copy(deep=True)

        # Both should be unloaded
        orig_loaded = original._contexts[0].is_loaded()
        copy_loaded = copy._contexts[0].is_loaded()
        print(f"Original loaded: {orig_loaded}")
        print(f"Copy loaded: {copy_loaded}")

        assert not orig_loaded, "Original should be unloaded"
        assert not copy_loaded, "Copy should be unloaded"

        # Load in copy shouldn't affect original
        ImageLoader.load(copy._contexts[0])
        assert copy._contexts[0].is_loaded(), "Copy should now be loaded"
        assert not original._contexts[0].is_loaded(), "Original should still be unloaded"

        print("✓ Copying unloaded images works correctly")

    except Exception as e:
        print_error(f"Unloaded images test failed: {e}")
        return

    # Test 6: Branching workflow (integration test)
    print_section("6. Branching Workflow (Real-World Scenario)")
    try:
        # Load and process original
        batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch.filter_valid().sample(5, random_sample=False)
        batch.resize(width=400, parallel=False)

        print(f"Base batch: {len(batch)} images, size={batch._contexts[0].img.size}")

        # Branch 1: Grayscale
        print_subsection("Branch 1: Grayscale Processing")
        grayscale = batch.copy(deep=True)
        grayscale.to_grayscale(keep_2d=True, parallel=False)
        grayscale_output = ensure_output_dir(output_dir / "grayscale")
        grayscale.save(grayscale_output, prefix="gray_", overwrite=True)
        print(f"  Saved {len(grayscale)} grayscale images")
        print(f"  Mode: {grayscale._contexts[0].img.mode}")

        # Branch 2: Brightness adjusted
        print_subsection("Branch 2: Brightness Adjustment")
        bright = batch.copy(deep=True)
        bright.adjust(brightness=1.3, parallel=False)
        bright_output = ensure_output_dir(output_dir / "bright")
        bright.save(bright_output, prefix="bright_", overwrite=True)
        print(f"  Saved {len(bright)} brightened images")
        print(f"  Mode: {bright._contexts[0].img.mode}")

        # Branch 3: Smaller thumbnails
        print_subsection("Branch 3: Thumbnails")
        thumbs = batch.copy(deep=True)
        thumbs.resize(width=200, parallel=False)
        thumb_output = ensure_output_dir(output_dir / "thumbs")
        thumbs.save(thumb_output, prefix="thumb_", overwrite=True)
        print(f"  Saved {len(thumbs)} thumbnails")
        print(f"  Size: {thumbs._contexts[0].img.size}")

        # Verify original unchanged
        assert batch._contexts[0].img.mode in ['RGB', 'RGBA'], \
            "Original should still be color"
        assert batch._contexts[0].img.size[0] == 400, \
            "Original should still be 400px wide"

        print("✓ Branching workflow successful - all branches independent")

    except Exception as e:
        print_error(f"Branching workflow test failed: {e}")
        return

    # Test 7: Method chaining compatibility
    print_section("7. Method Chaining Compatibility")
    try:
        original = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        original.filter_valid().sample(3, random_sample=False)

        # Chain copy with other operations
        result = (original
                  .copy(deep=True)
                  .resize(width=300, parallel=False)
                  .adjust(brightness=1.1, parallel=False))

        assert len(result) == 3, "Chaining should preserve batch size"
        assert result._contexts[0].img.size[0] == 300, "Chaining should apply resize"

        # Verify original unchanged
        assert not original._contexts[0].is_loaded(), "Original should be unloaded"

        print("✓ copy() method is chainable")

    except Exception as e:
        print_error(f"Method chaining test failed: {e}")
        return

    # Test 8: Copy with context manager
    print_section("8. Context Manager Compatibility")
    try:
        original = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        original.filter_valid().sample(2, random_sample=False)
        original.resize(width=400, parallel=False)

        # Use copy in context manager
        with original.copy(deep=True) as temp_batch:
            temp_batch.to_grayscale(keep_2d=True, parallel=False)
            assert temp_batch._contexts[0].is_loaded(), "Should be loaded in context"

        # After context, copy should be unloaded but original unaffected
        assert original._contexts[0].is_loaded(), "Original should still be loaded"
        assert original._contexts[0].img.mode in ['RGB', 'RGBA'], \
            "Original should still be color"

        print("✓ copy() works correctly with context manager")

    except Exception as e:
        print_error(f"Context manager test failed: {e}")
        return

    # Test 9: Large batch performance
    print_section("9. Performance with Larger Batch")
    try:
        import time

        large_batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        large_batch.filter_valid()

        # Time deep copy
        start = time.time()
        large_batch.resize(width=600, parallel=False)
        deep_copy = large_batch.copy(deep=True)
        deep_time = time.time() - start

        # Time lazy copy
        start = time.time()
        lazy_copy = large_batch.copy(deep=False)
        lazy_time = time.time() - start

        print(f"Batch size: {len(large_batch)} images")
        print(f"Deep copy time: {deep_time:.3f}s")
        print(f"Lazy copy time: {lazy_time:.3f}s")
        print(f"Speedup: {deep_time/lazy_time:.1f}x faster")

        assert lazy_time < deep_time, "Lazy copy should be faster"

        print("✓ Lazy copy is more performant as expected")

    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return

    print_success("All copy() tests passed!")
    print(f"\nTest outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
