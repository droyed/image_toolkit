"""
Comprehensive test and demonstration of BatchImageHandler indexing capabilities.

This script combines demonstrations from three separate files to provide a complete
showcase of indexing features, modification persistence, and real-world usage patterns.

Sections:
1. Basic Indexing Features - Core indexing, slicing, and iteration operations
2. Modification Persistence - How changes propagate through shared contexts
3. Real-World Workflows - Practical scenarios and before/after comparisons

Total: 12 demonstration functions covering all aspects of the indexing system.
"""

from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont
from image_toolkit.batch_handler import BatchImageHandler
from image_toolkit.core import ImageContext, ImageLoader


# =============================================================================
# Helper Functions - Image Creation Utilities
# =============================================================================

def create_demo_images(output_dir: Path, count: int = 10):
    """Create sample images for basic demonstrations."""
    output_dir.mkdir(exist_ok=True)

    for i in range(count):
        # Create images with different sizes and colors
        width = 200 + (i * 50)
        height = 200 + (i * 30)
        color = (255 - i * 20, i * 25, 100 + i * 15)

        img = Image.new('RGB', (width, height), color=color)

        # Add some text to identify the image
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Image {i}", fill=(255, 255, 255))

        img.save(output_dir / f"demo_{i:02d}.jpg")

    print(f"✓ Created {count} demo images in {output_dir}")


def create_test_images():
    """Create test images for persistence demonstrations."""
    output_dir = Path("test_persistence")
    output_dir.mkdir(exist_ok=True)

    for i in range(5):
        img = Image.new('RGB', (200, 200), color=(i * 50, 100, 200 - i * 40))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Test {i}", fill=(255, 255, 255))
        img.save(output_dir / f"test_{i}.jpg")

    return output_dir


def create_photo_gallery(count=30):
    """Create a simulated photo gallery for real-world scenarios."""
    gallery_dir = Path("photo_gallery")
    gallery_dir.mkdir(exist_ok=True)

    for i in range(count):
        # Vary image sizes
        if i % 3 == 0:
            size = (3000, 2000)  # Large landscape
        elif i % 3 == 1:
            size = (2000, 3000)  # Large portrait
        else:
            size = (1500, 1500)  # Medium square

        # Create image
        color = ((i * 30) % 255, (i * 50) % 255, (i * 70) % 255)
        img = Image.new('RGB', size, color=color)

        # Add label
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"Photo {i:03d}", fill=(255, 255, 255))

        img.save(gallery_dir / f"photo_{i:03d}.jpg")

    print(f"✓ Created {count} photos in {gallery_dir}")
    return gallery_dir


# =============================================================================
# Section 1: Basic Indexing Features
# =============================================================================

def demo_basic_indexing():
    """Demonstrate basic indexing operations."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Indexing")
    print("=" * 60)

    # Create demo images
    demo_dir = Path("demo_images")
    create_demo_images(demo_dir, count=10)

    # Load batch
    batch = BatchImageHandler.from_directory(demo_dir, "*.jpg")
    print(f"\nLoaded batch with {len(batch)} images")

    # Access individual images
    print("\n--- Single Image Access ---")
    first_img = batch[0]
    print(f"First image: {first_img.path.name}")

    last_img = batch[-1]
    print(f"Last image: {last_img.path.name}")

    middle_img = batch[len(batch) // 2]
    print(f"Middle image: {middle_img.path.name}")

    # Show that you can call methods on indexed items
    print("\n--- Operations on Indexed Items ---")
    img = batch[0]
    print(f"Image {img.path.name}:")
    print(f"  - Current size: {img._ctx.img.size if img._ctx.img else 'Not loaded'}")

    # Load and show size
    ImageLoader.load(img._ctx)
    print(f"  - Loaded size: {img._ctx.img.size}")


def demo_slicing():
    """Demonstrate slicing operations."""
    print("\n" + "=" * 60)
    print("Demo 2: Slicing")
    print("=" * 60)

    demo_dir = Path("demo_images")
    batch = BatchImageHandler.from_directory(demo_dir, "*.jpg")
    print(f"\nOriginal batch: {len(batch)} images")

    # Get first 5 images
    print("\n--- First 5 Images ---")
    first_five = batch[0:5]
    print(f"batch[0:5]: {len(first_five)} images")
    for img in first_five:
        print(f"  - {img.path.name}")

    # Get every other image
    print("\n--- Every Other Image ---")
    every_other = batch[::2]
    print(f"batch[::2]: {len(every_other)} images")
    for img in every_other:
        print(f"  - {img.path.name}")

    # Get last 3 images
    print("\n--- Last 3 Images ---")
    last_three = batch[-3:]
    print(f"batch[-3:]: {len(last_three)} images")
    for img in last_three:
        print(f"  - {img.path.name}")


def demo_iteration():
    """Demonstrate iteration support."""
    print("\n" + "=" * 60)
    print("Demo 3: Iteration")
    print("=" * 60)

    demo_dir = Path("demo_images")
    batch = BatchImageHandler.from_directory(demo_dir, "*.jpg")

    print("\n--- For Loop Iteration ---")
    for i, img in enumerate(batch):
        if i < 3:  # Show first 3
            print(f"  {i}: {img.path.name}")
    print(f"  ... ({len(batch)} total images)")

    print("\n--- List Comprehension ---")
    names = [img.path.name for img in batch]
    print(f"All image names: {names[:3]}...")

    print("\n--- Filter Pattern ---")
    # Get images where index is even
    even_images = [img for i, img in enumerate(batch) if i % 2 == 0]
    print(f"Even-indexed images: {[img.path.name for img in even_images]}")


def demo_practical_workflows():
    """Demonstrate practical workflow patterns."""
    print("\n" + "=" * 60)
    print("Demo 4: Practical Workflows")
    print("=" * 60)

    demo_dir = Path("demo_images")
    batch = BatchImageHandler.from_directory(demo_dir, "*.jpg")

    # Workflow 1: Process first N images differently
    print("\n--- Workflow 1: Process Subset ---")
    print("Process first 3 images with grayscale, rest stay color")

    # This is now much easier with slicing!
    first_three = batch[0:3]
    rest = batch[3:]

    print(f"First subset: {len(first_three)} images")
    print(f"Second subset: {len(rest)} images")

    # Workflow 2: Inspect specific image before batch processing
    print("\n--- Workflow 2: Inspect Before Processing ---")
    sample = batch[0]
    ImageLoader.load(sample._ctx)
    print(f"Sample image size: {sample._ctx.img.size}")
    print(f"Sample image mode: {sample._ctx.img.mode}")
    print("Decision: All images should be resized to 512x512")

    # Workflow 3: Process every Nth image
    print("\n--- Workflow 3: Process Every 3rd Image ---")
    every_third = batch[::3]
    print(f"Selected {len(every_third)} images for special processing:")
    for img in every_third:
        print(f"  - {img.path.name}")

    # Workflow 4: Get random sample more easily
    print("\n--- Workflow 4: Process Last Few Images ---")
    last_batch = batch[-3:]
    print(f"Processing final {len(last_batch)} images:")
    for img in last_batch:
        print(f"  - {img.path.name}")


def demo_chaining():
    """Demonstrate chaining indexing with batch operations."""
    print("\n" + "=" * 60)
    print("Demo 5: Chaining Operations")
    print("=" * 60)

    demo_dir = Path("demo_images")
    batch = BatchImageHandler.from_directory(demo_dir, "*.jpg")

    # Slice, then iterate
    print("\n--- Slice Then Iterate ---")
    print("Get middle 4 images and print their names:")
    for img in batch[3:7]:
        print(f"  - {img.path.name}")

    # Slice, then index
    print("\n--- Slice Then Index ---")
    subset = batch[2:8]
    first_of_subset = subset[0]
    last_of_subset = subset[-1]
    print(f"Subset range: {first_of_subset.path.name} to {last_of_subset.path.name}")

    # Multiple slices from same batch
    print("\n--- Multiple Slices ---")
    train_set = batch[0:6]
    val_set = batch[6:8]
    test_set = batch[8:]
    print(f"Train: {len(train_set)} images")
    print(f"Val: {len(val_set)} images")
    print(f"Test: {len(test_set)} images")


# =============================================================================
# Section 2: Modification Persistence
# =============================================================================

def demo_modification_persistence():
    """Demonstrate that modifications persist."""
    print("\n" + "=" * 70)
    print("Demo 6: Modification Persistence via Indexing")
    print("=" * 70)

    # Create test images
    test_dir = create_test_images()
    print(f"\n✓ Created test images in {test_dir}")

    # Load batch
    batch = BatchImageHandler.from_directory(test_dir, "*.jpg")
    print(f"✓ Loaded batch with {len(batch)} images")

    # Load all images
    for ctx in batch._contexts:
        ImageLoader.load(ctx)

    print("\n--- Initial State ---")
    for i, ctx in enumerate(batch._contexts):
        print(f"Image {i}: {ctx.path.name} - Size: {ctx.img.size}")

    # Modify via indexing
    print("\n--- Modifying via batch[0] ---")
    img_handler = batch[0]
    print(f"Got handler for: {img_handler.path.name}")
    print(f"  Original size: {img_handler._ctx.img.size}")

    # Resize the image
    img_handler._ctx.img = img_handler._ctx.img.resize((400, 400))
    print(f"  Resized to: {img_handler._ctx.img.size}")

    # Check if modification persisted to batch
    print("\n--- Checking Batch After Modification ---")
    print(f"batch._contexts[0] size: {batch._contexts[0].img.size}")
    print(f"✓ Modification persisted! (Same as handler: {img_handler._ctx.img.size})")

    # Verify they're the same object
    print(f"\n✓ Same context object: {img_handler._ctx is batch._contexts[0]}")

    # Modify multiple images via iteration
    print("\n--- Modifying Multiple Images via Iteration ---")
    for i, img in enumerate(batch):
        if i < 3:
            original_size = img._ctx.img.size
            new_width = original_size[0] + 100
            new_height = original_size[1] + 50
            img._ctx.img = img._ctx.img.resize((new_width, new_height))
            print(f"  {img.path.name}: {original_size} -> {img._ctx.img.size}")

    print("\n--- Verifying Batch State After Iteration Modifications ---")
    for i, ctx in enumerate(batch._contexts):
        print(f"Image {i}: {ctx.path.name} - Size: {ctx.img.size}")

    # Demonstrate slicing still shares contexts
    print("\n--- Modifications via Slices ---")
    subset = batch[0:2]
    print(f"Created slice: batch[0:2] with {len(subset)} images")

    # Modify via slice
    slice_img = subset[0]
    slice_img._ctx.img = slice_img._ctx.img.resize((600, 600))
    print(f"Modified subset[0] to: {slice_img._ctx.img.size}")

    # Check original batch
    print(f"Original batch[0] size: {batch._contexts[0].img.size}")
    print(f"✓ Modification via slice persisted to original batch!")


def demo_independent_slices():
    """Show that while contexts are shared, slices are independent instances."""
    print("\n" + "=" * 70)
    print("Demo 7: Slice Independence (while sharing contexts)")
    print("=" * 70)

    # Create test images
    test_dir = create_test_images()
    batch = BatchImageHandler.from_directory(test_dir, "*.jpg")

    print(f"\nOriginal batch: {len(batch)} images")

    # Create two slices
    slice1 = batch[0:3]
    slice2 = batch[2:5]

    print(f"Slice 1 (batch[0:3]): {len(slice1)} images")
    print(f"Slice 2 (batch[2:5]): {len(slice2)} images")

    # They're different BatchImageHandler instances
    print(f"\n✓ slice1 is not batch: {slice1 is not batch}")
    print(f"✓ slice2 is not batch: {slice2 is not batch}")
    print(f"✓ slice1 is not slice2: {slice1 is not slice2}")

    # But they share the underlying context objects where they overlap
    print(f"\n✓ Overlapping contexts are shared:")
    print(f"  slice1._contexts[2] is slice2._contexts[0]: {slice1._contexts[2] is slice2._contexts[0]}")
    print(f"  slice1._contexts[2] is batch._contexts[2]: {slice1._contexts[2] is batch._contexts[2]}")

    # Each slice has its own executor
    print(f"\n✓ Independent executors:")
    print(f"  slice1._executor is not batch._executor: {slice1._executor is not batch._executor}")

    # Each slice has its own error list (starts empty)
    print(f"\n✓ Independent error lists:")
    print(f"  len(batch._errors): {len(batch._errors)}")
    print(f"  len(slice1._errors): {len(slice1._errors)}")


def demo_practical_use_case_persistence():
    """Show a practical use case combining indexing and modifications."""
    print("\n" + "=" * 70)
    print("Demo 8: Practical Use Case - Selective Processing")
    print("=" * 70)

    # Create test images
    test_dir = create_test_images()
    batch = BatchImageHandler.from_directory(test_dir, "*.jpg")

    print(f"\nScenario: Process images selectively based on inspection")
    print(f"Total images: {len(batch)}")

    # Load all images
    for ctx in batch._contexts:
        ImageLoader.load(ctx)

    # Inspect first image to decide on processing
    print("\n--- Inspecting First Image ---")
    sample = batch[0]
    print(f"Sample: {sample.path.name}")
    print(f"  Size: {sample._ctx.img.size}")
    print(f"  Mode: {sample._ctx.img.mode}")

    # Decision: Resize images larger than 200x200
    print("\n--- Processing Strategy ---")
    print("Resize images to 150x150 if they're 200x200 or larger")

    count_resized = 0
    for i, img in enumerate(batch):
        width, height = img._ctx.img.size
        if width >= 200 or height >= 200:
            img._ctx.img = img._ctx.img.resize((150, 150))
            print(f"  ✓ Resized {img.path.name}")
            count_resized += 1

    print(f"\n✓ Resized {count_resized} images")

    # Verify changes in batch
    print("\n--- Final State ---")
    for i, ctx in enumerate(batch._contexts):
        print(f"Image {i}: {ctx.path.name} - Size: {ctx.img.size}")

    # Could now save the processed batch
    output_dir = Path("processed_output")
    output_dir.mkdir(exist_ok=True)
    print(f"\n--- Saving Processed Images ---")
    batch.save(output_dir, prefix="processed_")
    print(f"✓ Saved to {output_dir}")


# =============================================================================
# Section 3: Real-World Workflows
# =============================================================================

def workflow_before_indexing():
    """Show the old way of doing things (without indexing)."""
    print("\n" + "=" * 70)
    print("Demo 9: BEFORE - Without Indexing Support")
    print("=" * 70)

    gallery = create_photo_gallery(30)
    batch = BatchImageHandler.from_directory(gallery, "*.jpg")

    print(f"\nTotal photos: {len(batch)}")

    # OLD WAY: Access first image
    print("\n--- Accessing First Image ---")
    first_ctx = batch._contexts[0]  # Direct context access
    print(f"First photo: {first_ctx.path.name}")
    print("❌ Not intuitive, exposes internals")

    # OLD WAY: Get subset
    print("\n--- Getting Subset ---")
    subset_contexts = batch._contexts[0:10]
    subset_batch = BatchImageHandler([])
    subset_batch._contexts = subset_contexts
    print(f"Manual subset creation: {len(subset_batch)} images")
    print("❌ Verbose and error-prone")

    # OLD WAY: Process each image
    print("\n--- Processing Each Image ---")
    print("Using map() with lambda (complex for simple iteration)")
    count = [0]  # Mutable counter for lambda
    def count_and_print(handler):
        count[0] += 1
        if count[0] <= 3:
            print(f"  Processing: {handler.path.name}")
        return handler
    batch.map(count_and_print, parallel=False)
    print("❌ Requires function definition even for simple operations")


def workflow_after_indexing():
    """Show the new way with indexing support."""
    print("\n" + "=" * 70)
    print("Demo 10: AFTER - With Indexing Support")
    print("=" * 70)

    gallery = create_photo_gallery(30)
    batch = BatchImageHandler.from_directory(gallery, "*.jpg")

    print(f"\nTotal photos: {len(batch)}")

    # NEW WAY: Access first image
    print("\n--- Accessing First Image ---")
    first = batch[0]
    print(f"First photo: {first.path.name}")
    print("✅ Clean, Pythonic, intuitive")

    # NEW WAY: Get subset
    print("\n--- Getting Subset ---")
    subset = batch[0:10]
    print(f"Subset: {len(subset)} images")
    print("✅ One line, just like Python lists")

    # NEW WAY: Process each image
    print("\n--- Processing Each Image ---")
    for i, img in enumerate(batch):
        if i < 3:
            print(f"  Processing: {img.path.name}")
    print("✅ Natural for-loop, no extra functions needed")


def practical_scenario_gallery():
    """A realistic photo processing scenario."""
    print("\n" + "=" * 70)
    print("Demo 11: Photo Gallery Optimization")
    print("=" * 70)

    print("\nScenario: Web gallery needs:")
    print("  1. Thumbnails (150x150) for gallery view")
    print("  2. Medium (800px wide) for lightbox")
    print("  3. Keep originals for download")

    # Create gallery
    gallery = create_photo_gallery(30)
    batch = BatchImageHandler.from_directory(gallery, "*.jpg")
    print(f"\n✓ Loaded {len(batch)} photos")

    # Inspect first photo to understand dimensions
    print("\n--- Inspection Phase ---")
    sample = batch[0]
    ImageLoader.load(sample._ctx)
    print(f"Sample photo: {sample.path.name}")
    print(f"  Size: {sample._ctx.img.size}")
    print(f"  Mode: {sample._ctx.img.mode}")
    print("  Decision: All photos need resizing")

    # Create output directories
    thumb_dir = Path("gallery_output/thumbnails")
    medium_dir = Path("gallery_output/medium")
    thumb_dir.mkdir(parents=True, exist_ok=True)
    medium_dir.mkdir(parents=True, exist_ok=True)

    # Strategy: Process in batches for memory efficiency
    print("\n--- Processing Strategy ---")
    chunk_size = 10
    total = len(batch)

    for chunk_idx in range(0, total, chunk_size):
        # Use slicing to get chunk
        end_idx = min(chunk_idx + chunk_size, total)
        chunk = batch[chunk_idx:end_idx]

        print(f"Processing chunk {chunk_idx//chunk_size + 1} "
              f"(images {chunk_idx}-{end_idx-1})...")

        # Create thumbnails from this chunk
        thumb_chunk = chunk.copy(deep=True)
        thumb_chunk.resize(width=150, height=150)
        thumb_chunk.save(thumb_dir, prefix="thumb_")

        # Create medium versions from this chunk
        medium_chunk = chunk.copy(deep=True)
        medium_chunk.resize(width=800)
        medium_chunk.save(medium_dir, prefix="medium_")

    print(f"\n✓ Processed {total} photos")
    print(f"  Thumbnails: {len(list(thumb_dir.glob('*.jpg')))}")
    print(f"  Medium: {len(list(medium_dir.glob('*.jpg')))}")

    # Show examples using indexing
    print("\n--- Quality Check (using indexing) ---")
    thumbnails = BatchImageHandler.from_directory(thumb_dir, "*.jpg")
    mediums = BatchImageHandler.from_directory(medium_dir, "*.jpg")

    # Check first, middle, and last
    for idx in [0, len(thumbnails)//2, -1]:
        thumb = thumbnails[idx]
        medium = mediums[idx]
        ImageLoader.load(thumb._ctx)
        ImageLoader.load(medium._ctx)
        print(f"  Image {idx}: "
              f"thumb={thumb._ctx.img.size}, "
              f"medium={medium._ctx.img.size}")


def advanced_scenario_categorization():
    """Advanced: Selective processing with inspection."""
    print("\n" + "=" * 70)
    print("Demo 12: Advanced Categorization and Processing")
    print("=" * 70)

    print("\nScenario: Mixed photo collection needs:")
    print("  - Landscape photos: resize to 1920x1080")
    print("  - Portrait photos: resize to 1080x1920")
    print("  - Square photos: resize to 1080x1080")

    # Create gallery
    gallery = create_photo_gallery(30)
    batch = BatchImageHandler.from_directory(gallery, "*.jpg")

    # Load all images
    for ctx in batch._contexts:
        ImageLoader.load(ctx)

    print(f"\n✓ Loaded {len(batch)} photos")

    # Categorize using indexing and iteration
    landscape = []
    portrait = []
    square = []

    for i, img in enumerate(batch):
        width, height = img._ctx.img.size
        if width > height:
            landscape.append(i)
        elif height > width:
            portrait.append(i)
        else:
            square.append(i)

    print(f"\nCategories:")
    print(f"  Landscape: {len(landscape)} photos")
    print(f"  Portrait: {len(portrait)} photos")
    print(f"  Square: {len(square)} photos")

    # Process each category
    print("\n--- Processing by Category ---")

    output_dir = Path("categorized_output")
    output_dir.mkdir(exist_ok=True)

    # Process landscapes
    if landscape:
        print(f"Processing {len(landscape)} landscape photos...")
        for idx in landscape[:3]:  # Show first 3
            img = batch[idx]
            print(f"  {img.path.name}: {img._ctx.img.size} -> (1920, 1080)")
            img._ctx.img = img._ctx.img.resize((1920, 1080))

    # Process portraits
    if portrait:
        print(f"Processing {len(portrait)} portrait photos...")
        for idx in portrait[:3]:  # Show first 3
            img = batch[idx]
            print(f"  {img.path.name}: {img._ctx.img.size} -> (1080, 1920)")
            img._ctx.img = img._ctx.img.resize((1080, 1920))

    # Process squares
    if square:
        print(f"Processing {len(square)} square photos...")
        for idx in square[:3]:  # Show first 3
            img = batch[idx]
            print(f"  {img.path.name}: {img._ctx.img.size} -> (1080, 1080)")
            img._ctx.img = img._ctx.img.resize((1080, 1080))

    # Save all
    batch.save(output_dir, prefix="processed_")
    print(f"\n✓ Saved all processed photos to {output_dir}")

    # Verify using indexing
    print("\n--- Verification (spot check) ---")
    processed = BatchImageHandler.from_directory(output_dir, "*.jpg")
    for idx in [0, len(processed)//2, -1]:
        img = processed[idx]
        ImageLoader.load(img._ctx)
        print(f"  {img.path.name}: {img._ctx.img.size}")


# =============================================================================
# Cleanup Utilities
# =============================================================================

def cleanup():
    """Remove all temporary directories created during demonstrations."""
    directories = [
        "demo_images",
        "test_persistence",
        "processed_output",
        "photo_gallery",
        "gallery_output",
        "categorized_output"
    ]

    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)

    print("\n✓ Cleaned up all temporary directories")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all demonstrations in sequence."""
    print("=" * 70)
    print("COMPREHENSIVE INDEXING DEMONSTRATION")
    print("BatchImageHandler - Complete Feature Showcase")
    print("=" * 70)
    print("\nThis script demonstrates all indexing capabilities through")
    print("12 demonstrations organized into 3 sections:")
    print("  Section 1: Basic Indexing (5 demos)")
    print("  Section 2: Modification Persistence (3 demos)")
    print("  Section 3: Real-World Workflows (4 demos)")
    print("=" * 70)

    try:
        # Section 1: Basic Indexing Features
        print("\n\n" + "#" * 70)
        print("# SECTION 1: BASIC INDEXING FEATURES")
        print("#" * 70)
        demo_basic_indexing()
        demo_slicing()
        demo_iteration()
        demo_practical_workflows()
        demo_chaining()

        # Section 2: Modification Persistence
        print("\n\n" + "#" * 70)
        print("# SECTION 2: MODIFICATION PERSISTENCE")
        print("#" * 70)
        demo_modification_persistence()
        demo_independent_slices()
        demo_practical_use_case_persistence()

        # Section 3: Real-World Examples
        print("\n\n" + "#" * 70)
        print("# SECTION 3: REAL-WORLD WORKFLOWS")
        print("#" * 70)
        workflow_before_indexing()
        workflow_after_indexing()
        practical_scenario_gallery()
        advanced_scenario_categorization()

        # Final Summary
        print("\n\n" + "=" * 70)
        print("✓ ALL 12 DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nKEY TAKEAWAYS:")
        print("  1. Indexing: batch[0], batch[-1] - Intuitive element access")
        print("  2. Slicing: batch[0:10], batch[::2] - Clean subset operations")
        print("  3. Iteration: for img in batch - Natural Python loops")
        print("  4. Persistence: Changes via indexing affect original batch")
        print("  5. Flexibility: Slice, index, iterate - all work seamlessly")
        print("  6. Real-world: Practical patterns for common workflows")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cleanup()


if __name__ == "__main__":
    main()
