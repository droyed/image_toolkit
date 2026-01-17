#!/usr/bin/env python3
"""
Run all BatchImageHandler tests.

Executes all test scripts in sequence and reports results.
"""
import sys
import subprocess
from pathlib import Path
import time


def main():
    tests_dir = Path(__file__).parent / "tests"

    test_scripts = [
        ("test_basic_operations.py", "Basic Operations"),
        ("test_indexing.py", "Indexing & Slicing"),
        ("test_transformations.py", "Transformations"),
        ("test_advanced_processing.py", "Advanced Processing"),
        ("test_analysis.py", "Analysis & Statistics"),
        ("test_visualization.py", "Grid Visualization"),
        ("test_ml_dataset.py", "ML Dataset Conversion"),
        ("test_error_handling.py", "Error Handling"),
        ("test_memory_management.py", "Memory Management"),
        ("test_complete_workflow.py", "Complete Workflow"),
        ("test_batch_copy.py", "Copy Methods (Deep/Lazy)"),
        ("test_duplicate_analysis.py", "Duplicate Detection"),
        ("test_indexing_usage.py", "Indexing Usage Demos"),
        ("benchmark_parallel.py", "Performance Benchmark")
    ]

    print("="*70)
    print(" Running All BatchImageHandler Tests")
    print("="*70)
    print(f"\nTest suite: {len(test_scripts)} test scripts")
    print(f"Test directory: {tests_dir}\n")

    results = []
    start_time = time.time()

    for i, (test_script, description) in enumerate(test_scripts, 1):
        test_path = tests_dir / test_script

        print(f"\n{'='*70}")
        print(f" [{i}/{len(test_scripts)}] {description}")
        print(f" Script: {test_script}")
        print(f"{'='*70}\n")

        test_start = time.time()
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=False
        )
        test_elapsed = time.time() - test_start

        if result.returncode != 0:
            print(f"\n✗ {test_script} FAILED!")
            results.append((test_script, "FAILED", test_elapsed))

            # Ask if should continue
            response = input("\nContinue with remaining tests? (y/n): ")
            if response.lower() != 'y':
                print("\nTest suite aborted by user.")
                sys.exit(1)
        else:
            print(f"\n✓ {test_script} PASSED ({test_elapsed:.2f}s)")
            results.append((test_script, "PASSED", test_elapsed))

    total_elapsed = time.time() - start_time

    # Print summary
    print("\n" + "="*70)
    print(" TEST SUITE SUMMARY")
    print("="*70)

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")

    print(f"\nResults:")
    for script, status, elapsed in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"  {symbol} {script:40s} {status:8s} ({elapsed:.2f}s)")

    print(f"\n{'='*70}")
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Time: {total_elapsed:.2f}s")
    print(f"{'='*70}")

    if failed == 0:
        print("\n🎉 All test scripts completed successfully!")
        return 0
    else:
        print(f"\n⚠ {failed} test script(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
