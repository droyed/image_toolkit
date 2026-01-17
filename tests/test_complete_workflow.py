#!/usr/bin/env python3
"""
Test Complete Workflow - End-to-end realistic workflow

Tests: Complete pipeline combining multiple operations
"""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_toolkit import BatchImageHandler
from tests.test_utils import (print_section, print_subsection, print_success,
                               print_stats, get_paths, ensure_output_dir)


def generate_report(batch, stats, output_dir, duration):
    """Generate a summary report."""
    report_path = output_dir / "workflow_report.txt"

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" Batch Image Processing Report\n")
        f.write("="*70 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing Duration: {duration:.2f} seconds\n\n")

        f.write("Batch Statistics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Images: {stats['count']}\n\n")

        f.write("Dimensions:\n")
        f.write(f"  Width:  {stats['width']['min']:.0f} - {stats['width']['max']:.0f} "
               f"(mean: {stats['width']['mean']:.0f})\n")
        f.write(f"  Height: {stats['height']['min']:.0f} - {stats['height']['max']:.0f} "
               f"(mean: {stats['height']['mean']:.0f})\n\n")

        f.write("Aspect Ratios:\n")
        f.write(f"  Range: {stats['aspect_ratio']['min']:.2f} - {stats['aspect_ratio']['max']:.2f}\n")
        f.write(f"  Mean:  {stats['aspect_ratio']['mean']:.2f}\n\n")

        f.write("Output Location:\n")
        f.write(f"  {output_dir}\n\n")

        f.write("="*70 + "\n")

    return report_path


def main():
    import time

    paths = get_paths()
    input_dir = paths['input']
    output_base = paths['output']

    print("="*70)
    print(" Testing: Complete Workflow")
    print("="*70)
    print("\nExecuting end-to-end image processing pipeline...\n")

    start_time = time.time()

    # Step 1: Load and validate
    print_section("Step 1: Load and Validate Images")
    batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    print(f"Found {len(batch)} images")

    batch.filter_valid()
    print(f"Valid images: {len(batch)}")

    # Step 2: Filter by constraints
    print_section("Step 2: Apply Size Constraints")
    batch.filter_by_size(min_width=300, min_height=300)
    print(f"After size filter: {len(batch)} images")

    batch.filter_by_aspect_ratio(min_ratio=0.5, max_ratio=2.0)
    print(f"After aspect filter: {len(batch)} images")

    # Step 3: Get statistics before processing
    print_section("Step 3: Analyze Original Images")
    stats_original = batch.get_batch_stats()
    print_stats(stats_original)

    # Detect outliers
    outliers = batch.detect_outliers(metric='size', threshold=2.0)
    if outliers:
        print(f"\nFound {len(outliers)} size outliers")

    # Step 4: Sample subset for processing
    print_section("Step 4: Sample Images for Processing")
    sample_size = min(20, len(batch))
    batch_sample = batch.sample(sample_size, random_sample=True)
    print(f"Selected {len(batch_sample)} images for processing")

    # Step 5: Apply transformations
    print_section("Step 5: Apply Transformations")

    # Resize to standard size
    print("Resizing to 512x512...")
    batch_sample.resize(width=512, height=512)

    # Adjust brightness and contrast
    print("Adjusting brightness and contrast...")
    batch_sample.adjust(brightness=1.15, contrast=1.1)

    # Step 6: Create processed output
    print_section("Step 6: Save Processed Images")
    output_dir = ensure_output_dir(output_base / "workflow" / "processed")
    batch_sample.save(output_dir, prefix="processed_", quality=90)
    print(f"Saved {len(batch_sample)} processed images to {output_dir}")

    # Step 7: Create grayscale versions
    print_section("Step 7: Create Grayscale Versions")
    batch_gray = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_gray.filter_valid().sample(sample_size, random_sample=True)
    batch_gray.resize(width=512, height=512)
    batch_gray.to_grayscale()

    output_gray = ensure_output_dir(output_base / "workflow" / "grayscale")
    batch_gray.save(output_gray, prefix="gray_", quality=90)
    print(f"Saved {len(batch_gray)} grayscale images to {output_gray}")

    # Step 8: Create visualization grid
    print_section("Step 8: Create Visualization Grid")
    grid_size = min(16, len(batch_sample))
    if grid_size >= 16:
        batch_grid = batch_sample.sample(16, random_sample=False)
        rows, cols = 4, 4
    elif grid_size >= 9:
        batch_grid = batch_sample.sample(9, random_sample=False)
        rows, cols = 3, 3
    else:
        batch_grid = batch_sample
        rows = cols = int(grid_size ** 0.5)

    if len(batch_grid) > 0:
        grid = batch_grid.create_grid(rows=rows, cols=cols, cell_size=(180, 180), padding=5)
        grid_path = ensure_output_dir(output_base / "workflow" / "grids") / "workflow_grid.jpg"
        grid.save(grid_path, quality=90)
        print(f"Created {rows}x{cols} grid: {grid_path}")
        print(f"Grid dimensions: {grid.size}")

    # Step 9: Generate statistics
    print_section("Step 9: Generate Processing Statistics")
    stats_processed = batch_sample.get_batch_stats()
    print("\nOriginal Statistics:")
    print_stats(stats_original)
    print("\nProcessed Statistics:")
    print_stats(stats_processed)

    # Step 10: Create distribution visualizations
    print_section("Step 10: Create Distribution Plots")
    dist_dir = ensure_output_dir(output_base / "workflow" / "distributions")

    dist_plot = dist_dir / "all_distributions.png"
    batch_sample.visualize_distribution(save_path=str(dist_plot))
    print(f"Distribution plots (width, height, aspect, size): {dist_plot}")

    # Step 11: Generate report
    print_section("Step 11: Generate Summary Report")
    elapsed = time.time() - start_time
    report_dir = ensure_output_dir(output_base / "workflow")
    report_path = generate_report(batch_sample, stats_processed, report_dir, elapsed)
    print(f"Report saved to: {report_path}")

    # Step 12: Summary
    print_section("Workflow Summary")
    print(f"Total Processing Time: {elapsed:.2f} seconds")
    print(f"Images Processed: {len(batch_sample)}")
    print(f"Outputs Generated:")
    print(f"  - Processed images: {output_dir}")
    print(f"  - Grayscale images: {output_gray}")
    print(f"  - Visualization grid: {grid_path if len(batch_grid) > 0 else 'N/A'}")
    print(f"  - Distribution plots: {dist_dir}")
    print(f"  - Summary report: {report_path}")

    errors = batch_sample.get_errors()
    if errors:
        print(f"\nWarning: {len(errors)} errors encountered during processing")
    else:
        print("\nNo errors encountered")

    print_success("Complete Workflow Executed Successfully!")


if __name__ == "__main__":
    main()
