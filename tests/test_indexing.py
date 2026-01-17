"""
Test script for BatchImageHandler indexing and slicing support.

Tests:
1. Basic integer indexing (positive and negative)
2. Slicing (basic, step, negative)
3. Iteration
4. Error handling (out of bounds, invalid types)
5. Modification persistence
"""

import sys
from pathlib import Path
from image_toolkit.batch_handler import BatchImageHandler
from image_toolkit.handler import ImageHandler

def test_basic_indexing():
    """Test basic integer indexing."""
    print("\n=== Test 1: Basic Integer Indexing ===")

    # Create a small batch for testing
    batch = BatchImageHandler([])

    # Mock some contexts for testing
    from image_toolkit.core import ImageContext
    test_paths = [Path(f"test_image_{i}.jpg") for i in range(10)]
    batch._contexts = [ImageContext(path=p, img=None, metadata={}) for p in test_paths]

    # Test positive indexing
    first = batch[0]
    print(f"✓ batch[0] returns ImageHandler: {isinstance(first, ImageHandler)}")
    print(f"✓ batch[0].path = {first._ctx.path}")

    fifth = batch[4]
    print(f"✓ batch[4].path = {fifth._ctx.path}")

    # Test negative indexing
    last = batch[-1]
    print(f"✓ batch[-1].path = {last._ctx.path}")

    second_last = batch[-2]
    print(f"✓ batch[-2].path = {second_last._ctx.path}")

    # Verify correct contexts
    assert first._ctx.path == Path("test_image_0.jpg"), "First item path mismatch"
    assert last._ctx.path == Path("test_image_9.jpg"), "Last item path mismatch"
    assert second_last._ctx.path == Path("test_image_8.jpg"), "Second last item path mismatch"

    print("✓ All basic indexing tests passed")

def test_slicing():
    """Test slice indexing."""
    print("\n=== Test 2: Slicing ===")

    batch = BatchImageHandler([])
    from image_toolkit.core import ImageContext
    test_paths = [Path(f"test_image_{i}.jpg") for i in range(20)]
    batch._contexts = [ImageContext(path=p, img=None, metadata={}) for p in test_paths]

    # Basic slice
    subset = batch[0:5]
    print(f"✓ batch[0:5] returns BatchImageHandler: {isinstance(subset, BatchImageHandler)}")
    print(f"✓ batch[0:5] length: {len(subset)}")
    assert len(subset) == 5, f"Expected 5 items, got {len(subset)}"
    assert subset._contexts[0].path == Path("test_image_0.jpg"), "First item in slice incorrect"
    assert subset._contexts[-1].path == Path("test_image_4.jpg"), "Last item in slice incorrect"

    # Step slice
    every_other = batch[::2]
    print(f"✓ batch[::2] length: {len(every_other)}")
    assert len(every_other) == 10, f"Expected 10 items, got {len(every_other)}"
    assert every_other._contexts[0].path == Path("test_image_0.jpg"), "Step slice start incorrect"
    assert every_other._contexts[1].path == Path("test_image_2.jpg"), "Step slice second item incorrect"

    # Negative slice
    last_5 = batch[-5:]
    print(f"✓ batch[-5:] length: {len(last_5)}")
    assert len(last_5) == 5, f"Expected 5 items, got {len(last_5)}"
    assert last_5._contexts[0].path == Path("test_image_15.jpg"), "Negative slice start incorrect"
    assert last_5._contexts[-1].path == Path("test_image_19.jpg"), "Negative slice end incorrect"

    # Empty slice
    empty = batch[10:10]
    print(f"✓ batch[10:10] length (empty slice): {len(empty)}")
    assert len(empty) == 0, f"Expected 0 items, got {len(empty)}"

    print("✓ All slicing tests passed")

def test_iteration():
    """Test iteration support."""
    print("\n=== Test 3: Iteration ===")

    batch = BatchImageHandler([])
    from image_toolkit.core import ImageContext
    test_paths = [Path(f"test_image_{i}.jpg") for i in range(5)]
    batch._contexts = [ImageContext(path=p, img=None, metadata={}) for p in test_paths]

    # Test iteration
    count = 0
    for i, img in enumerate(batch):
        assert isinstance(img, ImageHandler), f"Item {i} is not ImageHandler"
        assert img._ctx.path == Path(f"test_image_{i}.jpg"), f"Item {i} path mismatch"
        count += 1

    print(f"✓ Iterated over {count} items")
    assert count == 5, f"Expected 5 iterations, got {count}"

    # Test list comprehension
    paths = [img._ctx.path.name for img in batch]
    print(f"✓ List comprehension: {paths}")
    assert len(paths) == 5, "List comprehension length mismatch"

    print("✓ All iteration tests passed")

def test_error_handling():
    """Test error handling."""
    print("\n=== Test 4: Error Handling ===")

    batch = BatchImageHandler([])
    from image_toolkit.core import ImageContext
    test_paths = [Path(f"test_image_{i}.jpg") for i in range(5)]
    batch._contexts = [ImageContext(path=p, img=None, metadata={}) for p in test_paths]

    # Test out of bounds (positive)
    try:
        _ = batch[999]
        assert False, "Should have raised IndexError"
    except IndexError as e:
        print(f"✓ Out of bounds positive index raises IndexError: {e}")

    # Test out of bounds (negative)
    try:
        _ = batch[-999]
        assert False, "Should have raised IndexError"
    except IndexError as e:
        print(f"✓ Out of bounds negative index raises IndexError: {e}")

    # Test invalid type
    try:
        _ = batch["invalid"]
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"✓ Invalid key type raises TypeError: {e}")

    # Test empty batch
    empty_batch = BatchImageHandler([])
    try:
        _ = empty_batch[0]
        assert False, "Should have raised IndexError for empty batch"
    except IndexError as e:
        print(f"✓ Empty batch indexing raises IndexError: {e}")

    print("✓ All error handling tests passed")

def test_modification_persistence():
    """Test that modifications via indexed handlers persist."""
    print("\n=== Test 5: Modification Persistence ===")

    batch = BatchImageHandler([])
    from image_toolkit.core import ImageContext
    from PIL import Image

    # Create contexts with actual Image objects
    test_paths = [Path(f"test_image_{i}.jpg") for i in range(3)]
    batch._contexts = [
        ImageContext(
            path=p,
            img=Image.new('RGB', (100, 100), color=(255, 0, 0)),
            metadata={}
        )
        for p in test_paths
    ]

    # Get handler via indexing
    img = batch[0]

    # Verify it's the same context object (shared reference)
    print(f"✓ Handler context is same object: {img._ctx is batch._contexts[0]}")
    assert img._ctx is batch._contexts[0], "Context should be shared reference"

    # Modify the image
    original_size = batch._contexts[0].img.size
    print(f"✓ Original size: {original_size}")

    # Resize via handler
    img._ctx.img = img._ctx.img.resize((200, 200))

    # Verify modification persisted to batch
    new_size = batch._contexts[0].img.size
    print(f"✓ New size after modification: {new_size}")
    assert new_size == (200, 200), "Modification should persist to batch"

    print("✓ All modification persistence tests passed")

def test_slice_independence():
    """Test that sliced batches are independent."""
    print("\n=== Test 6: Slice Independence ===")

    batch = BatchImageHandler([])
    from image_toolkit.core import ImageContext
    test_paths = [Path(f"test_image_{i}.jpg") for i in range(10)]
    batch._contexts = [ImageContext(path=p, img=None, metadata={}) for p in test_paths]

    # Create slice
    subset = batch[0:5]

    # Verify it's a new BatchImageHandler
    print(f"✓ Slice is new instance: {subset is not batch}")
    assert subset is not batch, "Slice should be a new instance"

    # Verify contexts are shared (not copied)
    print(f"✓ Contexts are shared: {subset._contexts[0] is batch._contexts[0]}")
    assert subset._contexts[0] is batch._contexts[0], "Contexts should be shared"

    # Verify executor is new
    print(f"✓ Executor is new instance: {subset._executor is not batch._executor}")
    assert subset._executor is not batch._executor, "Executor should be new instance"

    print("✓ All slice independence tests passed")

def test_combined_operations():
    """Test combining indexing with other operations."""
    print("\n=== Test 7: Combined Operations ===")

    batch = BatchImageHandler([])
    from image_toolkit.core import ImageContext
    test_paths = [Path(f"test_image_{i}.jpg") for i in range(10)]
    batch._contexts = [ImageContext(path=p, img=None, metadata={}) for p in test_paths]

    # Get slice, then index into it
    subset = batch[0:5]
    first_of_subset = subset[0]
    print(f"✓ batch[0:5][0].path = {first_of_subset._ctx.path}")
    assert first_of_subset._ctx.path == Path("test_image_0.jpg"), "Chained indexing incorrect"

    # Iterate over slice
    count = 0
    for img in batch[2:7]:
        count += 1
    print(f"✓ Iterated over batch[2:7]: {count} items")
    assert count == 5, f"Expected 5 iterations, got {count}"

    # Negative indexing on slice
    subset = batch[0:8]
    last_of_subset = subset[-1]
    print(f"✓ batch[0:8][-1].path = {last_of_subset._ctx.path}")
    assert last_of_subset._ctx.path == Path("test_image_7.jpg"), "Negative index on slice incorrect"

    print("✓ All combined operations tests passed")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing BatchImageHandler Indexing and Slicing Support")
    print("=" * 60)

    try:
        test_basic_indexing()
        test_slicing()
        test_iteration()
        test_error_handling()
        test_modification_persistence()
        test_slice_independence()
        test_combined_operations()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
