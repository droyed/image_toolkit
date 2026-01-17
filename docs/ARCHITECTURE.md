# Architecture Documentation

## Introduction & Philosophy

The `image_toolkit` library is built on a foundation of **separation of concerns** and **functional programming principles**. Each component has a single, well-defined responsibility and operates through static methods, ensuring stateless operation. The central design revolves around **ImageContext**, a lightweight data container that serves as the single source of truth for image state throughout the processing pipeline.

The architecture embraces a **delegation pattern** where high-level handler classes (ImageHandler, BatchImageHandler) delegate to specialized components (ImageLoader, ImageTransformer, ImageAnalyzer, ImageAnnotator). This creates a **fluent API** that allows method chaining for readable, expressive image processing pipelines while maintaining composability and testability.

## Core Architecture Components

### ImageContext (core/context.py)
- Lightweight data container holding: file path, PIL Image object, metadata
- No business logic—purely state storage
- Shared across all components as the single source of truth
- Enables lazy loading: images only loaded when operations require them

### Core Components (Static Method Pattern)

**ImageLoader** (io.py)
- Load/save images with format conversion
- EXIF orientation handling and metadata preservation
- Dual-path optimization: TurboJPEG for JPEG files, PIL for others

**ImageTransformer** (transforms.py)
- Geometric transformations: resize, crop, rotate, flip
- Visual adjustments: brightness, contrast, saturation, sharpness
- All operations modify ImageContext in place

**ImageAnalyzer** (analysis.py)
- Read-only analysis and statistics
- Conversion to NumPy arrays and PyTorch tensors
- Histogram generation and image metrics

**ImageAnnotator** (annotations.py)
- Visual markup: bounding boxes, masks, keypoints
- Text rendering and overlay operations
- Non-destructive annotation layer

### Handler Classes

**ImageHandler** (handler.py)
- Facade for single-image operations
- Delegates to specialized components
- Provides fluent API with method chaining

**BatchImageHandler** (batch_handler.py)
- Batch processing with parallel execution
- Filtering and transformation pipelines
- Error handling and progress tracking

## Code Flow: ImageHandler

```
┌─────────────────────────────────────────────────────────────┐
│ User Code                                                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ImageHandler.open(path)
                           │
                           ▼
              ┌────────────────────────┐
              │ ImageContext created   │
              │ - path: stored         │
              │ - image: None (lazy)   │
              │ - metadata: {}         │
              └────────────────────────┘
                           │
                           ▼
       User calls methods (resize, adjust, annotate)
                           │
                           ▼
              ┌────────────────────────┐
              │ _ensure_loaded()       │
              └────────────────────────┘
                           │
                           ▼
              ImageLoader.load(ctx)
                           │
              ┌────────────┴────────────┐
              │ Load from disk          │
              │ Handle EXIF orientation │
              │ Store PIL.Image in ctx  │
              └─────────────────────────┘
                           │
                           ▼
        Delegate to appropriate component
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
ImageTransformer    ImageAnalyzer    ImageAnnotator
  .resize(ctx)       .get_stats(ctx)   .draw_bbox(ctx)
  .adjust(ctx)       .to_array(ctx)    .draw_mask(ctx)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
              Return self (method chaining)
                           │
                           ▼
              handler.resize(...).adjust(...).save(...)
                           │
                           ▼
              ImageLoader.save(ctx, path)
                           │
                           ▼
              ┌────────────────────────┐
              │ Write to disk          │
              │ Preserve metadata      │
              │ Format conversion      │
              └────────────────────────┘
```

## Code Flow: BatchImageHandler

```
┌─────────────────────────────────────────────────────────────┐
│ User Code                                                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
       BatchImageHandler.from_directory(dir)
                           │
                           ▼
              ┌────────────────────────┐
              │ Create ImageContext    │
              │ for each file found    │
              └────────────────────────┘
                           │
                           ▼
              List[ImageContext] created
                           │
                           ▼
       Filter operations (optional)
       ┌──────────────────────────────┐
       │ .filter_valid()              │
       │ .filter_by_size(...)         │
       │ .filter_by_format(...)       │
       └──────────────────────────────┘
                           │
                           ▼
       Transform operations
       ┌──────────────────────────────┐
       │ .resize(...)                 │
       │ .adjust(...)                 │
       │ .annotate(...)               │
       └──────────────────────────────┘
                           │
                           ▼
       ParallelExecutor.map(transform_func, contexts)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌─────────┐        ┌─────────┐        ┌─────────┐
  │ Thread 1│        │ Thread 2│   ...  │ Thread N│
  └─────────┘        └─────────┘        └─────────┘
        │                  │                  │
        ▼                  ▼                  ▼
   Load ctx1          Load ctx2          Load ctxN
        │                  │                  │
        ▼                  ▼                  ▼
   Transform          Transform          Transform
        │                  │                  │
        ▼                  ▼                  ▼
   Save result        Save result        Save result
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Collect results        │
              │ Track errors           │
              │ Generate summary       │
              └────────────────────────┘
                           │
                           ▼
              .save_all(output_dir)
                           │
                           ▼
              ┌────────────────────────┐
              │ Write all processed    │
              │ images to disk         │
              │ Maintain directory     │
              │ structure if requested │
              └────────────────────────┘
```

## Design Patterns Summary

- **Static Method Pattern**: All core components use static methods for functional, stateless operation
- **Delegation Pattern**: Handler classes delegate to specialized components rather than implementing logic directly
- **Fluent Interface**: Method chaining enables readable, expressive processing pipelines
- **Context Manager Pattern**: Resource cleanup and proper error handling with `with` statements
- **Lazy Loading Pattern**: Images loaded on-demand only when operations require them
- **Dual-Path Optimization**: TurboJPEG fast path for JPEG files, PIL fallback for other formats
- **Facade Pattern**: ImageHandler provides simplified interface to complex subsystem

## Key Architectural Benefits

- **Composable**: Components work together transparently through shared ImageContext, enabling flexible combinations
- **Testable**: Stateless components with static methods are trivial to test in isolation without mocking
- **Scalable**: Parallel processing built into BatchImageHandler with thread pooling and error recovery
- **Flexible**: New components can be added without modifying existing ones—just delegate from handlers
- **Maintainable**: Clear separation of concerns makes code easy to understand, debug, and extend

## Component Dependencies

```
ImageHandler ──┐
               ├──> ImageContext <──┐
BatchImageHandler ┘                 │
                                    │
ImageLoader ────────────────────────┤
ImageTransformer ───────────────────┤
ImageAnalyzer ──────────────────────┤
ImageAnnotator ─────────────────────┘
```

All components depend on ImageContext; no cross-dependencies between components.
