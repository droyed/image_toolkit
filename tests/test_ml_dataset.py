#!/usr/bin/env python3
"""
Test ML Dataset Conversion - Convert to ML-ready formats

Tests: to_dataset(), split_dataset()
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_toolkit import BatchImageHandler
from tests.test_utils import print_section, print_subsection, print_success, get_paths


def main():
    paths = get_paths()
    input_dir = paths['input']

    print("="*70)
    print(" Testing: ML Dataset Conversion")
    print("="*70)

    # Load and prepare uniform-sized batch
    print_section("Preparing Uniform Batch")
    batch = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch.filter_valid().sample(10, random_sample=False)
    batch.resize(width=224, height=224)
    print(f"Prepared {len(batch)} images at 224x224")

    # Test 1: Convert to numpy array
    print_section("1. Convert to NumPy Array")
    try:
        import numpy as np
        dataset_np = batch.to_dataset(format='numpy', normalized=False)
        print(f"NumPy array shape: {dataset_np.shape}")
        print(f"Data type: {dataset_np.dtype}")
        print(f"Value range: [{dataset_np.min():.1f}, {dataset_np.max():.1f}]")
    except ImportError:
        print("NumPy not available - skipping")

    # Test 2: Convert to normalized numpy
    print_section("2. Convert to Normalized NumPy Array")
    try:
        import numpy as np
        dataset_norm = batch.to_dataset(format='numpy', normalized=True)
        print(f"Normalized array shape: {dataset_norm.shape}")
        print(f"Data type: {dataset_norm.dtype}")
        print(f"Value range: [{dataset_norm.min():.3f}, {dataset_norm.max():.3f}]")
    except ImportError:
        print("NumPy not available - skipping")

    # Test 3: Convert to PyTorch tensor
    print_section("3. Convert to PyTorch Tensor")
    try:
        import torch
        dataset_torch = batch.to_dataset(format='torch', normalized=True)
        print(f"Torch tensor shape: {dataset_torch.shape}")
        print(f"Data type: {dataset_torch.dtype}")
        print(f"Device: {dataset_torch.device}")
        print(f"Value range: [{dataset_torch.min():.3f}, {dataset_torch.max():.3f}]")
    except ImportError:
        print("PyTorch not available - skipping")

    # Test 4: Convert to list
    print_section("4. Convert to List")
    dataset_list = batch.to_dataset(format='list', normalized=False)
    print(f"List length: {len(dataset_list)}")
    print(f"First image type: {type(dataset_list[0])}")
    if hasattr(dataset_list[0], 'shape'):
        print(f"First image shape: {dataset_list[0].shape}")

    # Test 5: Channels-first vs channels-last
    print_section("5. Compare Channel Ordering")
    try:
        import numpy as np
        # Channels last (default): (N, H, W, C)
        dataset_last = batch.to_dataset(format='numpy', channels_first=False)
        print(f"Channels last (H, W, C): {dataset_last.shape}")

        # Channels first: (N, C, H, W) - PyTorch standard
        dataset_first = batch.to_dataset(format='numpy', channels_first=True)
        print(f"Channels first (C, H, W): {dataset_first.shape}")
    except ImportError:
        print("NumPy not available - skipping")

    # Test 6: Grayscale dataset
    print_section("6. Grayscale Dataset")
    batch_gray = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_gray.filter_valid().sample(10, random_sample=False)
    batch_gray.resize(width=224, height=224)
    batch_gray.to_grayscale()

    try:
        import numpy as np
        dataset_gray = batch_gray.to_dataset(format='numpy', normalized=True)
        print(f"Grayscale dataset shape: {dataset_gray.shape}")
        print(f"Value range: [{dataset_gray.min():.3f}, {dataset_gray.max():.3f}]")
    except ImportError:
        print("NumPy not available - skipping")

    # Test 7: Different sizes
    print_section("7. Different Image Sizes")
    size_configs = [
        (64, 64),
        (128, 128),
        (256, 256),
    ]

    for width, height in size_configs:
        batch_size = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_size.filter_valid().sample(5, random_sample=False)
        batch_size.resize(width=width, height=height)

        try:
            import numpy as np
            dataset = batch_size.to_dataset(format='numpy', normalized=True)
            print(f"  {width}x{height}: {dataset.shape}, "
                  f"memory: {dataset.nbytes / 1024 / 1024:.2f} MB")
        except ImportError:
            print(f"  {width}x{height}: NumPy not available")

    # Test 8: Memory efficiency
    print_section("8. Memory Efficiency Test")
    try:
        import numpy as np
        import sys

        batch_mem = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_mem.filter_valid().sample(20, random_sample=False)
        batch_mem.resize(width=224, height=224)

        dataset_uint8 = batch_mem.to_dataset(format='numpy', normalized=False)
        dataset_float32 = batch_mem.to_dataset(format='numpy', normalized=True)

        print(f"uint8 size: {dataset_uint8.nbytes / 1024 / 1024:.2f} MB")
        print(f"float32 size: {dataset_float32.nbytes / 1024 / 1024:.2f} MB")
        print(f"Size increase: {dataset_float32.nbytes / dataset_uint8.nbytes:.1f}x")
    except ImportError:
        print("NumPy not available - skipping")

    # Test 9: Integration with data augmentation
    print_section("9. Pre-augmented Dataset")
    batch_aug = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
    batch_aug.filter_valid().sample(10, random_sample=False)
    batch_aug.resize(width=224, height=224)
    batch_aug.adjust(brightness=1.2, contrast=1.1)

    try:
        import numpy as np
        dataset_aug = batch_aug.to_dataset(format='numpy', normalized=True)
        print(f"Augmented dataset shape: {dataset_aug.shape}")
        print(f"Ready for ML training")
    except ImportError:
        print("NumPy not available - skipping")

    # Test 10: Split dataset - basic split
    print_section("10. Split Dataset - 70/15/15")
    try:
        batch_split = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_split.filter_valid()
        initial_count = len(batch_split)
        print(f"Initial batch: {initial_count} images")

        # Split with default ratios
        splits = batch_split.split_dataset(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            shuffle=True
        )

        print(f"Train: {len(splits['train'])} images")
        print(f"Val: {len(splits['val'])} images")
        print(f"Test: {len(splits['test'])} images")

        # Verify splits are BatchImageHandler instances
        assert isinstance(splits['train'], BatchImageHandler), "Train should be BatchImageHandler"
        assert isinstance(splits['val'], BatchImageHandler), "Val should be BatchImageHandler"
        assert isinstance(splits['test'], BatchImageHandler), "Test should be BatchImageHandler"

        # Verify total matches
        total_after = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total_after == initial_count, \
            f"Total should match: {total_after} != {initial_count}"
        print(f"✓ Total matches: {total_after} == {initial_count}")

        # Verify ratios are approximately correct
        expected_train = int(initial_count * 0.7)
        expected_val = int(initial_count * 0.15)
        assert abs(len(splits['train']) - expected_train) <= 1, "Train ratio approximately correct"
        assert abs(len(splits['val']) - expected_val) <= 1, "Val ratio approximately correct"
        print("✓ Split ratios are approximately correct")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 11: Split dataset - reproducible split
    print_section("11. Split Dataset - Reproducible")
    try:
        batch_repro = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_repro.filter_valid()

        # Split with random_seed twice
        splits1 = batch_repro.split_dataset(random_seed=42)
        splits2 = batch_repro.split_dataset(random_seed=42)

        # Verify sizes match
        assert len(splits1['train']) == len(splits2['train']), "Train sizes should match"
        assert len(splits1['val']) == len(splits2['val']), "Val sizes should match"
        assert len(splits1['test']) == len(splits2['test']), "Test sizes should match"

        print(f"Split 1 - Train: {len(splits1['train'])}, Val: {len(splits1['val'])}, Test: {len(splits1['test'])}")
        print(f"Split 2 - Train: {len(splits2['train'])}, Val: {len(splits2['val'])}, Test: {len(splits2['test'])}")

        # Verify first image path matches (order is same)
        if len(splits1['train']) > 0 and len(splits2['train']) > 0:
            path1 = splits1['train']._contexts[0].path
            path2 = splits2['train']._contexts[0].path
            assert path1 == path2, "First image path should match"
            print(f"✓ First image matches: {path1.name}")

        print("✓ Splits are reproducible with same random_seed")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 12: Split dataset - no shuffle
    print_section("12. Split Dataset - Sequential (No Shuffle)")
    try:
        batch_seq = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_seq.filter_valid().sample(20, random_sample=False)
        initial_paths = [ctx.path for ctx in batch_seq._contexts]

        # Split without shuffling
        splits_seq = batch_seq.split_dataset(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            shuffle=False
        )

        train_count = len(splits_seq['train'])
        val_count = len(splits_seq['val'])
        test_count = len(splits_seq['test'])

        print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")

        # Verify order is preserved (first N go to train)
        if train_count > 0:
            train_first_path = splits_seq['train']._contexts[0].path
            assert train_first_path == initial_paths[0], "Order should be preserved"
            print(f"✓ Order preserved: first image is {train_first_path.name}")

        # Verify sequential assignment
        if val_count > 0:
            val_first_path = splits_seq['val']._contexts[0].path
            expected_val_path = initial_paths[train_count]
            assert val_first_path == expected_val_path, "Val should start after train"
            print(f"✓ Sequential assignment: val starts at correct position")

        print("✓ Sequential split (no shuffle) works correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 13: Split dataset - edge ratios
    print_section("13. Split Dataset - Edge Ratios")
    try:
        batch_edge = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_edge.filter_valid().sample(20, random_sample=False)
        initial = len(batch_edge)

        # Test 90/5/5 split
        splits_90 = batch_edge.split_dataset(
            train_ratio=0.9,
            val_ratio=0.05,
            test_ratio=0.05
        )
        print(f"90/5/5 split: Train={len(splits_90['train'])}, "
              f"Val={len(splits_90['val'])}, Test={len(splits_90['test'])}")
        total_90 = len(splits_90['train']) + len(splits_90['val']) + len(splits_90['test'])
        assert total_90 == initial, f"Total should match: {total_90} != {initial}"

        # Test 80/20/0 split (no test set)
        splits_80 = batch_edge.split_dataset(
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0
        )
        print(f"80/20/0 split: Train={len(splits_80['train'])}, "
              f"Val={len(splits_80['val'])}, Test={len(splits_80['test'])}")
        assert len(splits_80['test']) == 0, "Test set should be empty"
        total_80 = len(splits_80['train']) + len(splits_80['val']) + len(splits_80['test'])
        assert total_80 == initial, f"Total should match: {total_80} != {initial}"

        print("✓ Edge ratio splits work correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 14: Split dataset - error handling
    print_section("14. Split Dataset - Error Handling")
    try:
        batch_err = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_err.filter_valid()

        # Test invalid ratios (sum != 1.0)
        try:
            splits_invalid = batch_err.split_dataset(
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum = 1.1
            )
            print("✗ Should have raised ValueError for ratios summing to 1.1")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for invalid ratios: {str(e)[:50]}...")

        # Test negative ratios
        try:
            splits_negative = batch_err.split_dataset(
                train_ratio=0.8,
                val_ratio=-0.1,
                test_ratio=0.3
            )
            print("✗ Should have raised ValueError for negative ratio")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for negative ratio: {str(e)[:50]}...")

        # Test empty batch
        try:
            batch_empty = BatchImageHandler([])
            splits_empty = batch_empty.split_dataset()
            print("✗ Should have raised ValueError for empty batch")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for empty batch: {str(e)[:50]}...")

        # Test independence of splits
        batch_ind = BatchImageHandler.from_directory(input_dir, pattern="*.jpg")
        batch_ind.filter_valid().sample(10, random_sample=False)
        splits_ind = batch_ind.split_dataset()

        # Modify train split
        original_train_count = len(splits_ind['train'])
        splits_ind['train'].sample(2, random_sample=False)

        # Verify original batch unchanged
        assert len(batch_ind) == 10, "Original batch should be unchanged"

        # Verify val and test unchanged
        assert len(splits_ind['val']) > 0 or len(splits_ind['test']) > 0, "Other splits should exist"
        print("✓ Splits are independent from original batch")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print_success()


if __name__ == "__main__":
    main()
