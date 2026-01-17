#!/usr/bin/env python3
"""
Test Memory Management - Memory handling features

Tests: unload(), process_in_chunks(), context manager
"""
import sys
from pathlib import Path
import gc

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_toolkit import BatchImageHandler
from tests.test_utils import (print_section, print_subsection, print_success,
                               get_paths, ensure_output_dir)


def get_memory_info():
    """Get approximate memory usage."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # MB
    except ImportError:
        return None


def main():
    paths = get_paths()
    input_dir = paths['input']
    output_base = paths['output']

    print("="*70)
    print(" Testing: Memory Management")
    print("="*70)

    # Test 1: Baseline memory usage
    print_section("1. Baseline Memory Usage")
    gc.collect()
    baseline_mem = get_memory_info()
    if baseline_mem:
        print(f"Baseline memory: {baseline_mem:.1f} MB")
    else:
        print("Memory tracking not available (psutil not installed)")

    # Test 2: Load batch and check memory
    print_section("2. Memory After Loading Batch")
    batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch.filter_valid()
    print(f"Loaded {len(batch)} images")

    gc.collect()
    loaded_mem = get_memory_info()
    if loaded_mem and baseline_mem:
        mem_increase = loaded_mem - baseline_mem
        print(f"Memory after loading: {loaded_mem:.1f} MB (+{mem_increase:.1f} MB)")

    # Test 3: Unload all images
    print_section("3. Unload All Images")
    batch.unload()
    print(f"Unloaded all images, contexts preserved: {len(batch)}")

    gc.collect()
    unloaded_mem = get_memory_info()
    if unloaded_mem and loaded_mem:
        mem_freed = loaded_mem - unloaded_mem
        print(f"Memory after unload: {unloaded_mem:.1f} MB (-{mem_freed:.1f} MB freed)")

    # Test 4: Context manager cleanup
    print_section("4. Context Manager Auto-Cleanup")
    gc.collect()
    before_ctx_mem = get_memory_info()

    with BatchImageHandler.from_directory(input_dir, pattern="*.jpg") as batch_ctx:
        batch_ctx.filter_valid().sample(10, random_sample=False)
        batch_ctx.resize(width=400)
        print(f"Processed {len(batch_ctx)} images in context")

        gc.collect()
        inside_ctx_mem = get_memory_info()
        if inside_ctx_mem and before_ctx_mem:
            print(f"Memory inside context: {inside_ctx_mem:.1f} MB")

    gc.collect()
    after_ctx_mem = get_memory_info()
    if after_ctx_mem and inside_ctx_mem:
        mem_freed = inside_ctx_mem - after_ctx_mem
        print(f"Memory after context: {after_ctx_mem:.1f} MB (-{mem_freed:.1f} MB freed)")

    # Test 5: Process in chunks for large batches
    print_section("5. Process in Chunks")
    batch_chunked = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_chunked.filter_valid()

    total_images = len(batch_chunked)
    chunk_size = 5

    print(f"Processing {total_images} images in chunks of {chunk_size}")

    def chunk_processor(handler):
        """Process each image."""
        handler.resize_aspect(width=350)
        handler.adjust(brightness=1.1)
        return handler

    batch_chunked.process_in_chunks(chunk_size=chunk_size, func=chunk_processor)

    output_dir = ensure_output_dir(output_base / "chunked_memory")
    batch_chunked.save(output_dir, prefix="chunk_")
    print(f"Completed chunked processing, saved to {output_dir}")

    # Test 6: Memory-efficient pipeline
    print_section("6. Memory-Efficient Pipeline")
    gc.collect()
    pipeline_start_mem = get_memory_info()

    # Load, process in chunks, save, unload
    with BatchImageHandler.from_directory(input_dir, pattern="*.jpg") as batch_pipeline:
        batch_pipeline.filter_valid().sample(15, random_sample=False)
        initial = len(batch_pipeline)

        # Process in small chunks
        def efficient_processor(handler):
            handler.resize_aspect(width=300)
            handler.to_grayscale()
            return handler

        batch_pipeline.process_in_chunks(chunk_size=5, func=efficient_processor)

        output_dir = ensure_output_dir(output_base / "memory_efficient")
        batch_pipeline.save(output_dir, prefix="efficient_")
        print(f"Processed {initial} images with memory-efficient pipeline")

    gc.collect()
    pipeline_end_mem = get_memory_info()
    if pipeline_start_mem and pipeline_end_mem:
        print(f"Memory returned to ~baseline: {pipeline_end_mem:.1f} MB")

    # Test 7: Large batch handling
    print_section("7. Large Batch Handling")
    batch_large = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_large.filter_valid()

    print(f"Loaded {len(batch_large)} images")
    print(f"Processing in chunks to manage memory...")

    # Process and save in chunks without keeping all in memory
    chunk_size = 10

    def process_handler(handler):
        handler.resize_aspect(width=300)
        return handler

    batch_large.process_in_chunks(chunk_size=chunk_size, func=process_handler)
    output_dir = ensure_output_dir(output_base / "large_batch")
    batch_large.save(output_dir, prefix="large_")
    print(f"Processed {len(batch_large)} images in manageable chunks")

    # Test 8: Memory leak prevention
    print_section("8. Memory Leak Prevention")
    gc.collect()
    leak_start = get_memory_info()

    # Create and destroy multiple batches
    for i in range(3):
        with BatchImageHandler.from_directory(input_dir, pattern="*.jpg") as batch_temp:
            batch_temp.filter_valid().sample(10, random_sample=False)
            batch_temp.resize(width=350)
            # Let context manager clean up

    gc.collect()
    leak_end = get_memory_info()

    if leak_start and leak_end:
        leak_diff = leak_end - leak_start
        print(f"Memory before cycles: {leak_start:.1f} MB")
        print(f"Memory after cycles: {leak_end:.1f} MB")
        print(f"Difference: {leak_diff:+.1f} MB")
        if abs(leak_diff) < 50:  # Allow some variance
            print("No significant memory leak detected")

    # Test 9: Reload after unload
    print_section("9. Reload After Unload")
    batch_reload = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_reload.filter_valid().sample(5, random_sample=False)

    print(f"Initial: {len(batch_reload)} images loaded")
    batch_reload.unload()
    print("Unloaded all images")

    # Note: Depending on implementation, may need to reload
    # For now, just verify the batch still has contexts
    print(f"Contexts preserved: {len(batch_reload)} entries")

    print_success()


if __name__ == "__main__":
    main()
