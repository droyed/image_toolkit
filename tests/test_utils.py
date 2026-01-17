"""Common utilities for test scripts."""
from pathlib import Path
import time
from typing import Callable, Any


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


def print_subsection(title: str):
    """Print formatted subsection header."""
    print(f"\n{'-'*70}")
    print(f" {title}")
    print(f"{'-'*70}")


def print_stats(stats: dict):
    """Pretty print batch statistics."""
    print(f"Batch Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Width:  min={stats['width']['min']:.0f} "
          f"max={stats['width']['max']:.0f} "
          f"mean={stats['width']['mean']:.0f}")
    print(f"  Height: min={stats['height']['min']:.0f} "
          f"max={stats['height']['max']:.0f} "
          f"mean={stats['height']['mean']:.0f}")
    print(f"  Aspect: min={stats['aspect_ratio']['min']:.2f} "
          f"max={stats['aspect_ratio']['max']:.2f} "
          f"mean={stats['aspect_ratio']['mean']:.2f}")


def time_operation(name: str, func: Callable) -> tuple[Any, float]:
    """Time an operation and print result."""
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.3f}s")
    return result, elapsed


def get_paths() -> dict:
    """Get standard input/output paths."""
    base = Path(__file__).parent.parent
    return {
        'input': base / "assets" / "examples",
        'output': base / "test_outputs"
    }


def print_success(message: str = "All tests completed successfully!"):
    """Print success message."""
    print(f"\n{'='*70}")
    print(f" {message}")
    print(f"{'='*70}\n")


def print_error(message: str):
    """Print error message."""
    print(f"\n{'!'*70}")
    print(f" ERROR: {message}")
    print(f"{'!'*70}\n")


def ensure_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists and return it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
