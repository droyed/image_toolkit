#!/usr/bin/env python3
"""
Test Transformations - All transformation methods

Tests: resize(), adjust(), to_grayscale(), apply_transform(),
       chain_transforms(), save()
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
    print(" Testing: Transformations")
    print("="*70)

    # Load and prepare batch
    print_section("Loading Test Batch")
    batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch.filter_valid().sample(10, random_sample=False)
    print(f"Working with {len(batch)} images")

    # Test 1: Resize all
    print_section("1. Resize All (width=512)")
    output_dir = ensure_output_dir(output_base / "resized")
    batch_resize = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_resize.filter_valid().sample(10, random_sample=False)
    batch_resize.resize(width=512)
    batch_resize.save(output_dir, prefix="resized_")
    print(f"Resized and saved {len(batch_resize)} images to {output_dir}")

    # Test 2: Adjust brightness and contrast
    print_section("2. Adjust All (brightness=1.2, contrast=1.1)")
    output_dir = ensure_output_dir(output_base / "adjusted")
    batch_adjust = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_adjust.filter_valid().sample(10, random_sample=False)
    batch_adjust.adjust(brightness=1.2, contrast=1.1)
    batch_adjust.save(output_dir, prefix="adjusted_")
    print(f"Adjusted and saved {len(batch_adjust)} images to {output_dir}")

    # Test 3: Convert to grayscale
    print_section("3. Convert to Grayscale")
    output_dir = ensure_output_dir(output_base / "grayscale")
    batch_gray = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_gray.filter_valid().sample(10, random_sample=False)
    batch_gray.to_grayscale()
    batch_gray.save(output_dir, prefix="gray_")
    print(f"Converted and saved {len(batch_gray)} images to {output_dir}")

    # Test 4: Apply transform (rotate 90)
    print_section("4. Apply Transform (rotate 90)")
    output_dir = ensure_output_dir(output_base / "rotated")
    batch_rotate = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_rotate.filter_valid().sample(10, random_sample=False)

    batch_rotate.apply_transform('rotate', angle=90, expand=True)
    batch_rotate.save(output_dir, prefix="rotated_")
    print(f"Rotated and saved {len(batch_rotate)} images to {output_dir}")

    # Test 5: Chain multiple transforms
    print_section("5. Chain Transforms (resize + adjust + grayscale)")
    output_dir = ensure_output_dir(output_base / "chained")
    batch_chain = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_chain.filter_valid().sample(10, random_sample=False)

    transforms = [
        lambda img: img.resize((400, 400)),
        lambda img: img
    ]

    batch_chain.resize(width=400)
    batch_chain.adjust(brightness=1.15)
    batch_chain.to_grayscale()
    batch_chain.save(output_dir, prefix="chained_")
    print(f"Applied chain and saved {len(batch_chain)} images to {output_dir}")

    # Test 6: Save with different formats
    print_section("6. Save with Different Formats")
    output_dir = ensure_output_dir(output_base / "formats")
    batch_format = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_format.filter_valid().sample(3, random_sample=False)

    # Save as PNG
    batch_format.save(output_dir, prefix="format_", suffix=".png")
    print(f"Saved {len(batch_format)} images as PNG to {output_dir}")

    print_success()


if __name__ == "__main__":
    main()
