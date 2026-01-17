#!/usr/bin/env python3
"""
Benchmark Parallel Processing - Compare parallel vs sequential performance

Tests: Performance comparison across different worker counts
"""
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_toolkit import BatchImageHandler
from tests.test_utils import (print_section, print_subsection, print_success,
                               get_paths, time_operation)


def benchmark_resize(batch, width, workers=None):
    """Benchmark resize operation."""
    if workers:
        batch.resize(width=width, parallel=True, workers=workers)
    else:
        batch.resize(width=width, parallel=False)


def benchmark_adjust(batch, workers=None):
    """Benchmark adjust operation."""
    if workers:
        batch.adjust(brightness=1.2, contrast=1.1, parallel=True, workers=workers)
    else:
        batch.adjust(brightness=1.2, contrast=1.1, parallel=False)


def benchmark_grayscale(batch, workers=None):
    """Benchmark grayscale conversion."""
    if workers:
        batch.to_grayscale(parallel=True, workers=workers)
    else:
        batch.to_grayscale(parallel=False)


def benchmark_map(batch, transform_func, workers=None):
    """Benchmark map operation."""
    if workers:
        batch.map(transform_func, parallel=True, workers=workers)
    else:
        batch.map(transform_func, parallel=False)


def main():
    paths = get_paths()
    input_dir = paths['input']

    print("="*70)
    print(" Benchmark: Parallel Processing")
    print("="*70)
    print("\nComparing sequential vs parallel performance...\n")

    # Load test batch
    print_section("Loading Test Batch")
    base_batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    base_batch.filter_valid()
    total_images = len(base_batch)
    print(f"Loaded {total_images} images for benchmarking")

    # Benchmark 1: Resize operations
    print_section("Benchmark 1: Resize Operations (width=512)")

    # Sequential
    batch_seq = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_seq.filter_valid()
    start = time.time()
    batch_seq.resize(width=512, parallel=False)
    time_seq = time.time() - start
    print(f"Sequential: {time_seq:.3f}s")

    # Parallel - default workers
    batch_par = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_par.filter_valid()
    start = time.time()
    batch_par.resize(width=512, parallel=True)
    time_par = time.time() - start
    print(f"Parallel (auto workers): {time_par:.3f}s")

    speedup = time_seq / time_par if time_par > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

    # Benchmark 2: Adjust operations
    print_section("Benchmark 2: Adjust Operations")

    # Sequential
    batch_seq2 = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_seq2.filter_valid()
    start = time.time()
    batch_seq2.adjust(brightness=1.2, contrast=1.1, parallel=False)
    time_seq2 = time.time() - start
    print(f"Sequential: {time_seq2:.3f}s")

    # Parallel
    batch_par2 = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_par2.filter_valid()
    start = time.time()
    batch_par2.adjust(brightness=1.2, contrast=1.1, parallel=True)
    time_par2 = time.time() - start
    print(f"Parallel: {time_par2:.3f}s")

    speedup2 = time_seq2 / time_par2 if time_par2 > 0 else 0
    print(f"Speedup: {speedup2:.2f}x")

    # Benchmark 3: Grayscale conversion
    print_section("Benchmark 3: Grayscale Conversion")

    # Sequential
    batch_seq3 = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_seq3.filter_valid()
    start = time.time()
    batch_seq3.to_grayscale(parallel=False)
    time_seq3 = time.time() - start
    print(f"Sequential: {time_seq3:.3f}s")

    # Parallel
    batch_par3 = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_par3.filter_valid()
    start = time.time()
    batch_par3.to_grayscale(parallel=True)
    time_par3 = time.time() - start
    print(f"Parallel: {time_par3:.3f}s")

    speedup3 = time_seq3 / time_par3 if time_par3 > 0 else 0
    print(f"Speedup: {speedup3:.2f}x")

    # Benchmark 4: Custom map operation
    print_section("Benchmark 4: Custom Map Operation")

    def complex_transform(handler):
        """Complex multi-step transformation."""
        handler.resize_aspect(width=400, height=400)
        handler.rotate(0)  # No-op rotation
        return handler

    # Sequential
    batch_seq4 = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_seq4.filter_valid()
    start = time.time()
    batch_seq4.map(complex_transform, parallel=False)
    time_seq4 = time.time() - start
    print(f"Sequential: {time_seq4:.3f}s")

    # Parallel
    batch_par4 = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_par4.filter_valid()
    start = time.time()
    batch_par4.map(complex_transform, parallel=True)
    time_par4 = time.time() - start
    print(f"Parallel: {time_par4:.3f}s")

    speedup4 = time_seq4 / time_par4 if time_par4 > 0 else 0
    print(f"Speedup: {speedup4:.2f}x")

    # Benchmark 5: Variable worker counts
    print_section("Benchmark 5: Variable Worker Counts (Resize)")

    worker_counts = [1, 2, 4, 8]
    times = []

    for workers in worker_counts:
        batch_workers = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_workers.filter_valid()

        start = time.time()
        if workers == 1:
            batch_workers.resize(width=450, parallel=False)
        else:
            batch_workers.resize(width=450,
                                    parallel=True, workers=workers)
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"Workers={workers}: {elapsed:.3f}s")

    # Find optimal
    best_idx = times.index(min(times))
    best_workers = worker_counts[best_idx]
    best_time = times[best_idx]
    baseline_time = times[0]
    best_speedup = baseline_time / best_time if best_time > 0 else 0

    print(f"\nOptimal: {best_workers} workers ({best_time:.3f}s, {best_speedup:.2f}x speedup)")

    # Summary
    print_section("Performance Summary")
    print(f"Test Images: {total_images}")
    print(f"\nOperation Speedups:")
    print(f"  Resize:     {speedup:.2f}x")
    print(f"  Adjust:     {speedup2:.2f}x")
    print(f"  Grayscale:  {speedup3:.2f}x")
    print(f"  Custom Map: {speedup4:.2f}x")
    print(f"\nAverage Speedup: {(speedup + speedup2 + speedup3 + speedup4) / 4:.2f}x")

    print(f"\nOptimal Configuration:")
    print(f"  Workers: {best_workers}")
    print(f"  Speedup: {best_speedup:.2f}x")

    print(f"\nConclusion:")
    avg_speedup = (speedup + speedup2 + speedup3 + speedup4) / 4
    if avg_speedup > 1.5:
        print(f"  Parallel processing provides significant benefits ({avg_speedup:.2f}x)")
    elif avg_speedup > 1.1:
        print(f"  Parallel processing provides moderate benefits ({avg_speedup:.2f}x)")
    else:
        print(f"  Parallel overhead may outweigh benefits for this batch size")

    print_success("Benchmark Completed!")


if __name__ == "__main__":
    main()
