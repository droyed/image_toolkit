"""
ImageLoader - Stateless I/O operations for images using ImageContext
"""
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Union, Optional

from .context import ImageContext

# Initialize TurboJPEG at module level for performance
try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    jpeg = TurboJPEG()
except (ImportError, AttributeError):
    jpeg = None
    TJPF_RGB = None


class ImageLoader:
    """
    Stateless I/O operations for images.

    All methods operate on ImageContext to maintain single source of truth - ctx.img.
    """

    # ========================================================================
    # CONTEXT-BASED METHODS (Primary API)
    # ========================================================================

    @staticmethod
    def load(ctx: ImageContext, force: bool = False) -> ImageContext:
        """
        Load image from disk into context.

        Args:
            ctx: ImageContext to load into
            force: If True, reloads even if already loaded

        Returns:
            Same context (for chaining)

        Example:
            >>> ctx = ImageContext.from_path('photo.jpg')
            >>> ImageLoader.load(ctx)
            >>> print(ctx.img.size)
        """
        if ctx.img is not None and not force:
            return ctx

        if force and ctx.img is not None:
            ImageLoader.unload(ctx)

        # Check if it's a JPEG - use TurboJPEG for speed if available
        if jpeg and ctx.path.suffix.lower() in ['.jpg', '.jpeg']:
            try:
                with open(ctx.path, 'rb') as f:
                    raw_data = f.read()
                    rgb_array = jpeg.decode(raw_data, pixel_format=TJPF_RGB)
                    ctx.img = Image.fromarray(rgb_array)
                return ctx
            except Exception as e:
                print(f"TurboJPEG failed for {ctx.path.name}, falling back to PIL. Error: {e}")

        # Fallback for PNG/WebP or if TurboJPEG fails
        try:
            ctx.img = Image.open(ctx.path).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not load image {ctx.path}: {e}")

        return ctx

    @staticmethod
    def save(ctx: ImageContext, output_path: Union[str, Path],
             quality: int = 95) -> ImageContext:
        """
        Save image from context to disk.

        Args:
            ctx: ImageContext with loaded image
            output_path: Where to save
            quality: JPEG quality (1-100)

        Returns:
            Same context (for chaining)

        Example:
            >>> ImageLoader.save(ctx, 'output.jpg', quality=95)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if the target is a JPEG - use TurboJPEG for speed if available
        if jpeg and save_path.suffix.lower() in ['.jpg', '.jpeg']:
            try:
                # Convert PIL image to NumPy array
                if ctx.img.mode != "RGB":
                    img_to_save = ctx.img.convert("RGB")
                else:
                    img_to_save = ctx.img

                arr = np.asarray(img_to_save)

                # Encode using TurboJPEG
                dest_buf = jpeg.encode(arr, quality=quality, pixel_format=TJPF_RGB)

                # Write raw bytes to disk
                with open(save_path, 'wb') as f:
                    f.write(dest_buf)

                print(f"🚀 File saved (TurboJPEG): {save_path.name}")
                return ctx
            except Exception as e:
                print(f"TurboJPEG save failed for {save_path.name}, falling back to PIL. Error: {e}")

        # Fallback to standard Pillow save for PNG, WebP, etc.
        ctx.img.save(save_path, quality=quality, optimize=True)
        print(f"✅ File saved (Pillow): {save_path.name}")
        return ctx

    @staticmethod
    def is_valid(ctx: ImageContext) -> bool:
        """
        Check if image file at context path is valid.

        Args:
            ctx: ImageContext with path

        Returns:
            True if valid, False otherwise

        Example:
            >>> ctx = ImageContext.from_path('photo.jpg')
            >>> if ImageLoader.is_valid(ctx):
            ...     ImageLoader.load(ctx)
        """
        try:
            with Image.open(ctx.path) as temp_img:
                temp_img.verify()
            return True
        except (IOError, SyntaxError) as e:
            print(f"⚠️ Corrupt image detected: {ctx.path.name} -> {e}")
            return False

    @staticmethod
    def unload(ctx: ImageContext) -> ImageContext:
        """
        Free memory by closing image in context.

        Args:
            ctx: ImageContext to unload

        Returns:
            Same context (for chaining)

        Example:
            >>> ImageLoader.unload(ctx)
        """
        if ctx.img:
            ctx.img.close()
            ctx.img = None
        return ctx

    @staticmethod
    def get_exif(ctx: ImageContext) -> dict:
        """
        Get EXIF data from image in context.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Dictionary with EXIF data

        Example:
            >>> exif = ImageLoader.get_exif(ctx)
            >>> print(exif.get('DateTime'))
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        exif_data = {}
        try:
            exif = ctx.img._getexif()
            if exif:
                from PIL import ExifTags
                exif_data = {
                    ExifTags.TAGS.get(key, key): value
                    for key, value in exif.items()
                }
        except AttributeError:
            pass  # No EXIF data

        return exif_data

    @staticmethod
    def apply_exif_orientation(ctx: ImageContext) -> ImageContext:
        """
        Auto-rotate image based on EXIF orientation tag.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Same context (for chaining)

        Example:
            >>> ImageLoader.apply_exif_orientation(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        try:
            ctx.img = ImageOps.exif_transpose(ctx.img)
            ctx.metadata['exif_orientation_applied'] = True
        except Exception as e:
            print(f"⚠️ Could not apply EXIF orientation: {e}")

        return ctx

    @staticmethod
    def strip_exif(ctx: ImageContext) -> ImageContext:
        """
        Remove EXIF metadata from image in context.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Same context (for chaining)

        Example:
            >>> ImageLoader.strip_exif(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        # Create new image without EXIF
        data = list(ctx.img.getdata())
        image_without_exif = Image.new(ctx.img.mode, ctx.img.size)
        image_without_exif.putdata(data)

        ctx.img = image_without_exif
        ctx.metadata['exif_stripped'] = True

        return ctx

    @staticmethod
    def format_convert(ctx: ImageContext, target_format: str) -> ImageContext:
        """
        Convert image format in context.

        Args:
            ctx: ImageContext with loaded image
            target_format: Target format ('RGB', 'RGBA', 'L', etc.)

        Returns:
            Same context with converted image

        Example:
            >>> ImageLoader.format_convert(ctx, 'RGBA')
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        if ctx.img.mode != target_format:
            ctx.img = ctx.img.convert(target_format)

        return ctx

    @staticmethod
    def copy(ctx: ImageContext) -> ImageContext:
        """
        Create deep copy of context with copied image.

        Args:
            ctx: ImageContext to copy

        Returns:
            New ImageContext with copied image

        Example:
            >>> new_ctx = ImageLoader.copy(ctx)
        """
        new_ctx = ImageContext(
            path=ctx.path,
            img=ctx.img.copy() if ctx.img else None,
            metadata=ctx.metadata.copy()
        )
        return new_ctx

    @staticmethod
    def reset(ctx: ImageContext, force_reload: bool = True) -> ImageContext:
        """
        Reset context to original state.

        Args:
            ctx: ImageContext to reset
            force_reload: If True, reloads from disk

        Returns:
            Same context (for chaining)

        Example:
            >>> ImageLoader.reset(ctx)
        """
        ctx.img = None
        ctx.metadata = {"original_path": str(ctx.path)}

        if force_reload:
            ImageLoader.load(ctx)

        return ctx

    @staticmethod
    def get_metadata(ctx: ImageContext) -> dict:
        """
        Get comprehensive metadata including EXIF, ICC profile, format info.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Dictionary with comprehensive metadata

        Example:
            >>> metadata = ImageLoader.get_metadata(ctx)
            >>> print(metadata['format'], metadata['size'])
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        metadata = {
            'format': ctx.img.format,
            'mode': ctx.img.mode,
            'size': ctx.img.size,
            'info': ctx.img.info.copy() if hasattr(ctx.img, 'info') else {},
            'exif': ImageLoader.get_exif(ctx),
        }

        # Add ICC profile if present
        if 'icc_profile' in ctx.img.info:
            metadata['has_icc_profile'] = True

        return metadata

    @staticmethod
    def save_with_metadata(ctx: ImageContext, output_path: Union[str, Path],
                          preserve_exif: bool = True,
                          preserve_icc: bool = True,
                          quality: int = 95) -> ImageContext:
        """
        Save image with selective metadata preservation.

        Args:
            ctx: ImageContext with loaded image
            output_path: Where to save
            preserve_exif: Keep EXIF data
            preserve_icc: Keep ICC color profile
            quality: JPEG quality (1-100)

        Returns:
            Same context (for chaining)

        Example:
            >>> ImageLoader.save_with_metadata(ctx, 'output.jpg', preserve_exif=True)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare save kwargs
        save_kwargs = {'quality': quality, 'optimize': True}

        # Handle EXIF
        if preserve_exif:
            exif_data = ctx.img.info.get('exif')
            if exif_data:
                save_kwargs['exif'] = exif_data

        # Handle ICC profile
        if preserve_icc:
            icc_profile = ctx.img.info.get('icc_profile')
            if icc_profile:
                save_kwargs['icc_profile'] = icc_profile

        # Save based on format
        if save_path.suffix.lower() in ['.jpg', '.jpeg']:
            try:
                # For JPEG with metadata, use PIL
                if preserve_exif or preserve_icc:
                    ctx.img.save(save_path, **save_kwargs)
                    print(f"✅ Saved with metadata: {save_path.name}")
                else:
                    # Use fast TurboJPEG path if no metadata needed
                    if jpeg:
                        arr = np.asarray(ctx.img if ctx.img.mode == "RGB" else ctx.img.convert("RGB"))
                        dest_buf = jpeg.encode(arr, quality=quality, pixel_format=TJPF_RGB)
                        with open(save_path, 'wb') as f:
                            f.write(dest_buf)
                        print(f"🚀 Saved (TurboJPEG): {save_path.name}")
                    else:
                        ctx.img.save(save_path, **save_kwargs)
                        print(f"✅ Saved: {save_path.name}")
            except Exception as e:
                print(f"Save failed, using PIL fallback: {e}")
                ctx.img.save(save_path, **save_kwargs)
        else:
            # PNG, WebP, etc.
            ctx.img.save(save_path, **save_kwargs)
            print(f"✅ Saved with metadata: {save_path.name}")

        return ctx
