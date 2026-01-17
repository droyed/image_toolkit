#!/usr/bin/env python3
"""
Test Visualization - Grid creation and visualization

Tests: create_grid()
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
    print(" Testing: Visualization")
    print("="*70)

    # Ensure output directory
    output_dir = ensure_output_dir(output_base / "grids")

    # Test 1: Create standard grid (5x6 = 30 images)
    print_section("1. Create 5x6 Grid (30 images)")
    batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch.filter_valid()

    if len(batch) >= 30:
        batch_grid = batch.sample(30, random_sample=False)
    else:
        batch_grid = batch

    grid = batch_grid.create_grid(rows=5, cols=6, cell_size=(150, 150))
    grid_path = output_dir / "grid_5x6.jpg"
    grid.save(grid_path, quality=90)
    print(f"Created 5x6 grid with {len(batch_grid)} images")
    print(f"Saved to {grid_path}")
    print(f"Grid size: {grid.size}")

    # Test 2: Create smaller grid (4x4 = 16 images)
    print_section("2. Create 4x4 Grid (16 images)")
    batch_small = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_small.filter_valid().sample(16, random_sample=True)

    grid_small = batch_small.create_grid(rows=4, cols=4, cell_size=(200, 200))
    grid_path = output_dir / "grid_4x4.jpg"
    grid_small.save(grid_path, quality=90)
    print(f"Created 4x4 grid with {len(batch_small)} images")
    print(f"Saved to {grid_path}")
    print(f"Grid size: {grid_small.size}")

    # Test 3: Create grid with padding
    print_section("3. Create Grid with Padding")
    batch_padded = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_padded.filter_valid().sample(12, random_sample=False)

    grid_padded = batch_padded.create_grid(
        rows=3, cols=4, cell_size=(180, 180), padding=10
    )
    grid_path = output_dir / "grid_padded.jpg"
    grid_padded.save(grid_path, quality=90)
    print(f"Created 3x4 grid with padding")
    print(f"Saved to {grid_path}")
    print(f"Grid size: {grid_padded.size}")

    # Test 4: Create grid with background color
    print_section("4. Create Grid with Custom Background")
    batch_bg = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_bg.filter_valid().sample(9, random_sample=False)

    grid_bg = batch_bg.create_grid(
        rows=3, cols=3, cell_size=(200, 200),
        padding=15, background_color=(50, 50, 50)
    )
    grid_path = output_dir / "grid_background.jpg"
    grid_bg.save(grid_path, quality=90)
    print(f"Created 3x3 grid with custom background color")
    print(f"Saved to {grid_path}")

    # Test 5: Create wide grid (2x8)
    print_section("5. Create Wide Grid (2x8)")
    batch_wide = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_wide.filter_valid().sample(16, random_sample=False)

    grid_wide = batch_wide.create_grid(rows=2, cols=8, cell_size=(150, 150))
    grid_path = output_dir / "grid_wide.jpg"
    grid_wide.save(grid_path, quality=90)
    print(f"Created 2x8 wide grid")
    print(f"Saved to {grid_path}")
    print(f"Grid size: {grid_wide.size}")

    # Test 6: Create tall grid (8x2)
    print_section("6. Create Tall Grid (8x2)")
    batch_tall = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_tall.filter_valid().sample(16, random_sample=True)

    grid_tall = batch_tall.create_grid(rows=8, cols=2, cell_size=(150, 150))
    grid_path = output_dir / "grid_tall.jpg"
    grid_tall.save(grid_path, quality=90)
    print(f"Created 8x2 tall grid")
    print(f"Saved to {grid_path}")
    print(f"Grid size: {grid_tall.size}")

    # Test 7: Create grid with transformed images
    print_section("7. Create Grid with Transformed Images")
    batch_transform = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_transform.filter_valid().sample(12, random_sample=False)
    batch_transform.to_grayscale()
    batch_transform.adjust(brightness=1.2)

    grid_transform = batch_transform.create_grid(
        rows=3, cols=4, cell_size=(180, 180), padding=5
    )
    grid_path = output_dir / "grid_transformed.jpg"
    grid_transform.save(grid_path, quality=90)
    print(f"Created grid with grayscale + adjusted images")
    print(f"Saved to {grid_path}")

    # Test 8: Create contact sheet (all images)
    print_section("8. Create Contact Sheet (All Images)")
    batch_all = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_all.filter_valid()

    # Calculate grid dimensions
    total = len(batch_all)
    cols = 6
    rows = (total + cols - 1) // cols  # Ceiling division

    if total > 0:
        grid_all = batch_all.create_grid(
            rows=rows, cols=cols, cell_size=(120, 120), padding=5
        )
        grid_path = output_dir / "contact_sheet.jpg"
        grid_all.save(grid_path, quality=85)
        print(f"Created contact sheet with all {total} images ({rows}x{cols})")
        print(f"Saved to {grid_path}")
        print(f"Grid size: {grid_all.size}")

    print_success()


if __name__ == "__main__":
    main()
