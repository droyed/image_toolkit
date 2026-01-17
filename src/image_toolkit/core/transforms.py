"""
ImageTransformer - Stateless transformation operations using ImageContext
"""
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Tuple, Optional, List, Union, Dict

from .context import ImageContext


class ImageTransformer:
    """
    Stateless transformation operations for images.

    All methods operate on ImageContext to maintain single source of truth.
    Transformations modify ctx.img directly.
    """

    # ========================================================================
    # GEOMETRIC TRANSFORMATIONS
    # ========================================================================

    @staticmethod
    def resize_aspect(ctx: ImageContext,
                     width: Optional[int] = None,
                     height: Optional[int] = None,
                     padding_color: Tuple[int, int, int] = (0, 0, 0)) -> ImageContext:
        """
        Resize maintaining aspect ratio with optional padding.

        Args:
            ctx: ImageContext with loaded image
            width: Target width (optional)
            height: Target height (optional)
            padding_color: RGB color for padding

        Returns:
            Same context with resized image

        Example:
            >>> ImageTransformer.resize_aspect(ctx, width=800)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        # Case 1: Both dimensions provided - Force exact size with padding
        if width is not None and height is not None:
            ctx.img = ImageOps.pad(
                ctx.img,
                (width, height),
                method=Image.Resampling.LANCZOS,
                color=padding_color
            )
            return ctx

        # Case 2: Standard proportional resizing (single dimension provided)
        curr_w, curr_h = ctx.img.size
        if width:
            scale = width / curr_w
            new_size = (width, int(curr_h * scale))
        elif height:
            scale = height / curr_h
            new_size = (int(curr_w * scale), height)
        else:
            return ctx  # No dimensions provided

        ctx.img = ctx.img.resize(new_size, Image.Resampling.LANCZOS)
        return ctx

    @staticmethod
    def square_pad(ctx: ImageContext, size: int,
                   fill_color: Tuple[int, int, int] = (0, 0, 0)) -> ImageContext:
        """
        Resizes image to fit within size and pads to make it a perfect square.

        Args:
            ctx: ImageContext with loaded image
            size: Target square size
            fill_color: RGB color for padding

        Returns:
            Same context with padded image

        Example:
            >>> ImageTransformer.square_pad(ctx, 512)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        ctx.img.thumbnail((size, size), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (size, size), fill_color)
        # Center the image
        new_img.paste(ctx.img, ((size - ctx.img.size[0]) // 2, (size - ctx.img.size[1]) // 2))
        ctx.img = new_img
        return ctx

    @staticmethod
    def add_margin(ctx: ImageContext, top: int = 0, right: int = 0,
                   bottom: int = 0, left: int = 0,
                   color: Tuple[int, int, int] = (0, 0, 0)) -> ImageContext:
        """
        Adds a colored border around the image.

        Args:
            ctx: ImageContext with loaded image
            top, right, bottom, left: Pixels to add to each side
            color: RGB color for the margin

        Returns:
            Same context with margin added

        Example:
            >>> ImageTransformer.add_margin(ctx, top=10, left=10, color=(255,0,0))
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        border = (left, top, right, bottom)
        ctx.img = ImageOps.expand(ctx.img, border=border, fill=color)
        return ctx

    @staticmethod
    def pad_to_size(ctx: ImageContext, target_w: int, target_h: int,
                    color: Tuple[int, int, int] = (0, 0, 0)) -> ImageContext:
        """
        Ensures the image is exactly target_w x target_h.

        Args:
            ctx: ImageContext with loaded image
            target_w: Target width
            target_h: Target height
            color: RGB color for padding

        Returns:
            Same context with padded image

        Example:
            >>> ImageTransformer.pad_to_size(ctx, 800, 600)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        # Scale to fit inside the box
        ctx.img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

        # Calculate remaining space
        curr_w, curr_h = ctx.img.size
        pad_w = target_w - curr_w
        pad_h = target_h - curr_h

        # Split padding evenly (center the image)
        left, top = pad_w // 2, pad_h // 2
        right, bottom = pad_w - left, pad_h - top

        return ImageTransformer.add_margin(ctx, top, right, bottom, left, color)

    @staticmethod
    def crop(ctx: ImageContext, box: Tuple[int, int, int, int],
             normalized: bool = False) -> ImageContext:
        """
        Extract a region from the image.

        Args:
            ctx: ImageContext with loaded image
            box: (left, top, right, bottom) coordinates
            normalized: If True, coordinates are 0.0-1.0

        Returns:
            Same context with cropped image

        Example:
            >>> ImageTransformer.crop(ctx, (100, 100, 300, 300))
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        left, top, right, bottom = box

        if normalized:
            w, h = ctx.img.size
            left, right = int(left * w), int(right * w)
            top, bottom = int(top * h), int(bottom * h)

        ctx.img = ctx.img.crop((left, top, right, bottom))
        return ctx

    @staticmethod
    def extract_crops(ctx: ImageContext, boxes: List[Union[Tuple, Dict]],
                  normalized: bool = False) -> List[Image.Image]:
        """
        Extract multiple regions from the image.

        Args:
            ctx: ImageContext with loaded image
            boxes: List of boxes (tuples or dicts)
            normalized: If True, coordinates are 0.0-1.0

        Returns:
            List of PIL Image objects (not contexts)

        Note:
            This does NOT modify ctx.img, just returns crops

        Example:
            >>> crops = ImageTransformer.extract_crops(ctx, [(0,0,100,100)])
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        crops = []
        w, h = ctx.img.size

        for item in boxes:
            # Parse box format
            if isinstance(item, dict):
                box = item.get("box")
                label = item.get("label", None)
            else:
                box = item
                label = None

            left, top, right, bottom = box

            # Convert normalized to absolute
            if normalized:
                left, right = int(left * w), int(right * w)
                top, bottom = int(top * h), int(bottom * h)

            # Extract crop
            crop_img = ctx.img.crop((left, top, right, bottom))

            # Optionally attach label as metadata
            if label:
                crop_img.info['label'] = label

            crops.append(crop_img)

        return crops

    @staticmethod
    def flip_horizontal(ctx: ImageContext) -> ImageContext:
        """
        Mirror image horizontally.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Same context with flipped image

        Example:
            >>> ImageTransformer.flip_horizontal(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        ctx.img = ctx.img.transpose(Image.FLIP_LEFT_RIGHT)
        return ctx

    @staticmethod
    def flip_vertical(ctx: ImageContext) -> ImageContext:
        """
        Mirror image vertically.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Same context with flipped image

        Example:
            >>> ImageTransformer.flip_vertical(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        ctx.img = ctx.img.transpose(Image.FLIP_TOP_BOTTOM)
        return ctx

    @staticmethod
    def rotate(ctx: ImageContext, angle: float, expand: bool = False,
               fill_color: Tuple[int, int, int] = (0, 0, 0)) -> ImageContext:
        """
        Rotate image by angle.

        Args:
            ctx: ImageContext with loaded image
            angle: Rotation angle in degrees
            expand: If True, expand canvas to fit
            fill_color: Background color

        Returns:
            Same context with rotated image

        Example:
            >>> ImageTransformer.rotate(ctx, 45, expand=True)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        ctx.img = ctx.img.rotate(angle, expand=expand, fillcolor=fill_color)
        return ctx

    # ========================================================================
    # COLOR & FILTERING
    # ========================================================================

    @staticmethod
    def adjust(ctx: ImageContext, brightness: float = 1.0,
               contrast: float = 1.0) -> ImageContext:
        """
        Adjust brightness and contrast.

        Args:
            ctx: ImageContext with loaded image
            brightness: Brightness factor (1.0 = original)
            contrast: Contrast factor (1.0 = original)

        Returns:
            Same context with adjusted image

        Example:
            >>> ImageTransformer.adjust(ctx, brightness=1.2, contrast=0.9)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(ctx.img)
            ctx.img = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(ctx.img)
            ctx.img = enhancer.enhance(contrast)

        return ctx

    @staticmethod
    def to_grayscale(ctx: ImageContext, keep_2d: bool = False) -> ImageContext:
        """
        Convert image to grayscale.

        Args:
            ctx: ImageContext with loaded image
            keep_2d: If True, keeps single channel

        Returns:
            Same context with grayscale image

        Example:
            >>> ImageTransformer.to_grayscale(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        gray = ctx.img.convert('L')
        if keep_2d:
            ctx.img = gray
        else:
            # Convert back to RGB for consistency
            ctx.img = gray.convert('RGB')

        return ctx

    @staticmethod
    def filter_blur(ctx: ImageContext, radius: int = 2) -> ImageContext:
        """
        Apply Gaussian blur.

        Args:
            ctx: ImageContext with loaded image
            radius: Blur radius

        Returns:
            Same context with blurred image

        Example:
            >>> ImageTransformer.filter_blur(ctx, radius=5)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        ctx.img = ctx.img.filter(ImageFilter.GaussianBlur(radius))
        return ctx

    @staticmethod
    def to_rgba(ctx: ImageContext) -> ImageContext:
        """
        Convert image to RGBA.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Same context with RGBA image

        Example:
            >>> ImageTransformer.to_rgba(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        if ctx.img.mode != 'RGBA':
            ctx.img = ctx.img.convert('RGBA')

        return ctx

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @staticmethod
    def is_grayscale_mode(ctx: ImageContext) -> bool:
        """
        Check if image mode is grayscale.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            True if mode is 'L' or 'LA'

        Example:
            >>> if ImageTransformer.is_grayscale_mode(ctx):
            ...     print("Grayscale mode")
        """
        if ctx.img is None:
            return False
        return ctx.img.mode in ('L', 'LA')
