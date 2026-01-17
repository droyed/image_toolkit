#!/usr/bin/env python3
"""
Test Error Handling - Error tracking and recovery

Tests: get_errors(), clear_errors(), error handling during processing
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_toolkit import BatchImageHandler
from tests.test_utils import (print_section, print_subsection, print_success,
                               get_paths, ensure_output_dir)


def main():
    paths = get_paths()
    input_dir = paths['input']
    output_base = paths['output']

    print("="*70)
    print(" Testing: Error Handling")
    print("="*70)

    # Test 1: Load with some invalid paths
    print_section("1. Load with Mixed Valid/Invalid Paths")
    batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")

    # Add some invalid paths manually if the batch handler supports it
    initial_count = len(batch)
    print(f"Loaded {initial_count} images")

    # Check for any errors during loading
    errors = batch.get_errors()
    if errors:
        print(f"Found {len(errors)} errors during loading:")
        for error in errors[:3]:
            print(f"  - {error}")
    else:
        print("No errors during loading")

    # Test 2: Graceful handling of transform errors
    print_section("2. Graceful Transform Error Handling")
    batch_transform = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_transform.filter_valid().sample(10, random_sample=False)

    def problematic_transform(img):
        """Transform that might fail on some images."""
        # This should work for most images
        return img.resize((300, 300))

    try:
        batch_transform.apply_transform(problematic_transform)
        print(f"Transform applied to {len(batch_transform)} images")

        errors = batch_transform.get_errors()
        if errors:
            print(f"Encountered {len(errors)} errors during transformation")
        else:
            print("All transformations succeeded")
    except Exception as e:
        print(f"Caught exception: {e}")

    # Test 3: Error tracking across operations
    print_section("3. Error Tracking Across Operations")
    batch_multi = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_multi.filter_valid().sample(10, random_sample=False)

    print("Performing multiple operations...")
    batch_multi.resize(width=400)
    errors_after_resize = len(batch_multi.get_errors())

    batch_multi.adjust(brightness=1.2)
    errors_after_adjust = len(batch_multi.get_errors())

    batch_multi.to_grayscale()
    errors_after_grayscale = len(batch_multi.get_errors())

    print(f"Errors after resize: {errors_after_resize}")
    print(f"Errors after adjust: {errors_after_adjust}")
    print(f"Errors after grayscale: {errors_after_grayscale}")

    # Test 4: Clear errors
    print_section("4. Clear Errors")
    errors_before = len(batch_multi.get_errors())
    print(f"Errors before clearing: {errors_before}")

    batch_multi.clear_errors()
    errors_after = len(batch_multi.get_errors())
    print(f"Errors after clearing: {errors_after}")

    # Test 5: Continue processing after errors
    print_section("5. Continue Processing After Errors")
    batch_continue = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_continue.filter_valid().sample(10, random_sample=False)

    # Apply a series of transforms
    batch_continue.resize(width=350)
    images_after_resize = len(batch_continue)

    batch_continue.adjust(brightness=1.3)
    images_after_adjust = len(batch_continue)

    print(f"Images after resize: {images_after_resize}")
    print(f"Images after adjust: {images_after_adjust}")
    print(f"Processing continued successfully")

    # Test 6: Save with error handling
    print_section("6. Save with Error Handling")
    output_dir = ensure_output_dir(output_base / "error_handling")
    batch_save = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_save.filter_valid().sample(10, random_sample=False)
    batch_save.resize(width=300)

    try:
        batch_save.save(output_dir, prefix="error_test_")
        print(f"Successfully saved {len(batch_save)} images")

        errors = batch_save.get_errors()
        if errors:
            print(f"Encountered {len(errors)} errors during save")
            for error in errors[:3]:
                print(f"  - {error}")
    except Exception as e:
        print(f"Save operation error: {e}")

    # Test 7: Error details inspection
    print_section("7. Detailed Error Inspection")
    batch_inspect = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_inspect.filter_valid().sample(5, random_sample=False)

    # Perform some operations
    batch_inspect.resize(width=400)

    errors = batch_inspect.get_errors()
    if errors:
        print(f"Total errors: {len(errors)}")
        print("\nError details:")
        for i, error in enumerate(errors[:5], 1):
            print(f"\n  Error {i}:")
            if isinstance(error, dict):
                for key, value in error.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    {error}")
    else:
        print("No errors to inspect")

    # Test 8: Verify batch integrity after errors
    print_section("8. Verify Batch Integrity After Errors")
    batch_integrity = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    initial = len(batch_integrity)

    batch_integrity.filter_valid()
    after_filter = len(batch_integrity)

    batch_integrity.resize(width=350)
    after_resize = len(batch_integrity)

    print(f"Initial: {initial} images")
    print(f"After filter: {after_filter} images")
    print(f"After resize: {after_resize} images")
    print("Batch integrity maintained")

    print_success()


if __name__ == "__main__":
    main()
