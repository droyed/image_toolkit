#!/usr/bin/env python3
"""
Test Advanced Processing - Custom pipelines and chunked processing

Tests: map(), process_in_chunks(), set_progress_callback(), context manager
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
    print(" Testing: Advanced Processing")
    print("="*70)

    # Test 1: Map with custom function
    print_section("1. Map Custom Function")
    batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch.filter_valid().sample(10, random_sample=False)

    def custom_pipeline(handler):
        """Custom processing pipeline."""
        # Resize to square and then to standard size
        handler.resize_aspect(width=300, height=300)
        return handler

    batch.map(custom_pipeline)
    output_dir = ensure_output_dir(output_base / "mapped")
    batch.save(output_dir, prefix="mapped_")
    print(f"Applied custom function to {len(batch)} images")

    # Test 2: Process in chunks
    print_section("2. Process in Chunks (chunk_size=5)")
    batch_chunks = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_chunks.filter_valid()

    def chunk_processor(handler):
        """Process each image in chunk."""
        handler.resize_aspect(width=400)
        return handler

    print(f"Processing {len(batch_chunks)} images in chunks of 5")
    batch_chunks.process_in_chunks(chunk_size=5, func=chunk_processor)
    output_dir = ensure_output_dir(output_base / "chunked")
    batch_chunks.save(output_dir, prefix="chunked_")
    print(f"Processed and saved all chunks")

    # Test 3: Custom progress callback
    print_section("3. Custom Progress Callback")
    batch_progress = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_progress.filter_valid().sample(15, random_sample=False)

    progress_steps = []

    def custom_progress(current, total):
        """Custom progress tracking."""
        percentage = (current / total) * 100
        progress_steps.append(percentage)
        if current % 5 == 0 or current == total:
            print(f"  Processing: {current}/{total} ({percentage:.1f}%)")

    batch_progress.set_progress_callback(custom_progress)
    batch_progress.resize(width=350)
    print(f"Tracked {len(progress_steps)} progress updates")

    # Test 4: Context manager
    print_section("4. Context Manager Usage")
    output_dir = ensure_output_dir(output_base / "context")

    with BatchImageHandler.from_directory(input_dir, pattern="*.jpg") as batch_ctx:
        batch_ctx.filter_valid().sample(8, random_sample=False)
        batch_ctx.resize(width=450)
        batch_ctx.save(output_dir, prefix="context_")
        print(f"Processed {len(batch_ctx)} images in context manager")

    print("Context manager properly closed")

    # Test 5: Complex pipeline with map
    print_section("5. Complex Pipeline with Map")
    batch_complex = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_complex.filter_valid().sample(10, random_sample=False)

    def complex_pipeline(handler):
        """Multi-step processing pipeline."""
        # Resize
        handler.resize_aspect(width=400, height=400)
        # Adjust brightness and contrast
        handler.adjust(brightness=1.2, contrast=1.1)
        # Convert to grayscale and back to RGB
        handler.to_grayscale()
        handler.to_rgba()
        return handler

    batch_complex.map(complex_pipeline)
    output_dir = ensure_output_dir(output_base / "complex")
    batch_complex.save(output_dir, prefix="complex_")
    print(f"Applied complex pipeline to {len(batch_complex)} images")

    # Test 6: Parallel processing with map
    print_section("6. Parallel Map Processing")
    batch_parallel = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_parallel.filter_valid().sample(20, random_sample=False)

    def simple_transform(handler):
        """Simple transformation for parallel testing."""
        handler.resize_aspect(width=350, height=350)
        return handler

    print(f"Processing {len(batch_parallel)} images in parallel")
    batch_parallel.map(simple_transform, parallel=True, workers=4)
    output_dir = ensure_output_dir(output_base / "parallel_map")
    batch_parallel.save(output_dir, prefix="parallel_")
    print(f"Parallel processing completed")

    print_success()


if __name__ == "__main__":
    main()
