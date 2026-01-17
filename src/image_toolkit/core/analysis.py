"""
ImageAnalyzer - Stateless analysis operations using ImageContext
"""
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt

from .context import ImageContext


class ImageAnalyzer:
    """
    Stateless analysis operations for images.

    All methods operate on ImageContext to maintain single source of truth.
    """

    @staticmethod
    def get_stats(ctx: ImageContext) -> dict:
        """
        Get statistics from image in context.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = ImageAnalyzer.get_stats(ctx)
            >>> print(stats['width'], stats['height'])
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        w, h = ctx.img.size

        stats = {
            'width': w,
            'height': h,
            'aspect_ratio': w / h,
            'mode': ctx.img.mode,
        }

        # Mean color for RGB/RGBA images
        if ctx.img.mode in ('RGB', 'RGBA'):
            arr = np.array(ctx.img)
            stats['mean_color'] = tuple(arr.mean(axis=(0, 1)).astype(int).tolist())

        return stats

    @staticmethod
    def to_numpy(ctx: ImageContext, normalized: bool = False, dtype=np.float32) -> np.ndarray:
        """
        Convert image in context to NumPy array.

        Args:
            ctx: ImageContext with loaded image
            normalized: If True, scales to [0.0, 1.0]
            dtype: NumPy data type

        Returns:
            NumPy array

        Example:
            >>> arr = ImageAnalyzer.to_numpy(ctx, normalized=True)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        arr = np.asarray(ctx.img)

        if normalized:
            return arr.astype(dtype) / 255.0
        return arr.astype(dtype) if dtype != np.uint8 else arr

    @staticmethod
    def to_tensor(ctx: ImageContext, normalized: bool = False,
                  device: str = 'cpu',
                  dtype=None,
                  channels_first: bool = True):
        """
        Convert image in context to PyTorch tensor.

        Args:
            ctx: ImageContext with loaded image
            normalized: If True, scales to [0.0, 1.0]
            device: PyTorch device
            dtype: PyTorch data type
            channels_first: If True, returns (C, H, W)

        Returns:
            PyTorch tensor

        Example:
            >>> tensor = ImageAnalyzer.to_tensor(ctx, normalized=True)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for to_tensor(). Install with: pip install torch")

        if dtype is None:
            dtype = torch.float32

        # Get array representation
        if normalized:
            arr = ImageAnalyzer.to_numpy(ctx, normalized=True, dtype=np.float32)
        else:
            arr = ImageAnalyzer.to_numpy(ctx, normalized=False)

        # Convert to Tensor
        tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)

        # Handle Shape Permutation
        if channels_first:
            if tensor.ndim == 3:
                tensor = tensor.permute(2, 0, 1)
            elif tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)

        return tensor

    @staticmethod
    def show(ctx: ImageContext, title: Optional[str] = None) -> ImageContext:
        """
        Display image using system's default viewer.

        Args:
            ctx: ImageContext with loaded image
            title: Optional window title

        Returns:
            Same context (for chaining)

        Example:
            >>> ImageAnalyzer.show(ctx, title="My Image")
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        ctx.img.show(title=title)
        return ctx

    @staticmethod
    def inspect(ctx: ImageContext, title: Optional[str] = None, block: bool = True):
        """
        Display image using Matplotlib.

        Args:
            ctx: ImageContext with loaded image
            title: Optional plot title
            block: If True, waits for key press

        Returns:
            If block=True: pressed key string. If block=False: context

        Example:
            >>> key = ImageAnalyzer.inspect(ctx, block=True)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        pressed_key = None

        if block:
            plt.ioff()
        else:
            plt.ion()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(np.asarray(ctx.img))

        base_title = title if title else "Image Inspection"
        mode_hint = "ANY KEY to Capture/Close" if block else "[Non-blocking]"
        ax.set_title(f"{base_title}\n{mode_hint} | Size: {ctx.img.size}")
        ax.axis('off')

        if block:
            def on_key(event):
                nonlocal pressed_key
                pressed_key = event.key
                plt.close(fig)

            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.tight_layout()
            print(f"🛑 BLOCKING: Press any key on the image window to proceed...")
            plt.show(block=True)
            return pressed_key
        else:
            plt.tight_layout()
            plt.show()
            plt.pause(0.1)
            print(f"⚡ NON-BLOCKING: Processing continues...")
            return ctx

    @staticmethod
    def normalize(ctx: ImageContext, mean: List[float] = [0.485, 0.456, 0.406],
                  std: List[float] = [0.229, 0.224, 0.225],
                  inplace: bool = True):
        """
        Apply ImageNet-style normalization.

        Args:
            ctx: ImageContext with loaded image
            mean: Per-channel mean
            std: Per-channel std
            inplace: If True, modify image in context

        Returns:
            Context if inplace=True, else normalized array

        Example:
            >>> ImageAnalyzer.normalize(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        arr = np.array(ctx.img, dtype=np.float32) / 255.0

        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)

        if arr.ndim == 2:
            normalized = (arr - mean[0]) / std[0]
        else:
            normalized = (arr - mean) / std

        if inplace:
            denorm = (normalized * std + mean) * 255.0
            denorm = np.clip(denorm, 0, 255).astype(np.uint8)
            ctx.img = Image.fromarray(denorm)
            return ctx
        else:
            return normalized

    @staticmethod
    def denormalize(ctx: ImageContext, mean: List[float] = [0.485, 0.456, 0.406],
                    std: List[float] = [0.229, 0.224, 0.225]) -> ImageContext:
        """
        Reverse ImageNet normalization.

        Args:
            ctx: ImageContext with loaded image
            mean: Per-channel mean
            std: Per-channel std

        Returns:
            Same context with denormalized image

        Example:
            >>> ImageAnalyzer.denormalize(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        arr = np.array(ctx.img, dtype=np.float32)

        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)

        denormalized = (arr * std + mean) * 255.0
        denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)

        ctx.img = Image.fromarray(denormalized)
        return ctx

    @staticmethod
    def get_channel_stats(ctx: ImageContext) -> dict:
        """
        Get per-channel statistics.

        Args:
            ctx: ImageContext with loaded image

        Returns:
            Dictionary with channel statistics

        Example:
            >>> stats = ImageAnalyzer.get_channel_stats(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        arr = np.array(ctx.img, dtype=np.float32) / 255.0

        if arr.ndim == 2:
            return {
                'channels': 1,
                'mean': [float(arr.mean())],
                'std': [float(arr.std())],
                'min': [float(arr.min())],
                'max': [float(arr.max())]
            }
        else:
            return {
                'channels': arr.shape[2],
                'mean': arr.mean(axis=(0, 1)).tolist(),
                'std': arr.std(axis=(0, 1)).tolist(),
                'min': arr.min(axis=(0, 1)).tolist(),
                'max': arr.max(axis=(0, 1)).tolist()
            }

    @staticmethod
    def is_grayscale_content(ctx: ImageContext, tolerance: float = 0.01) -> bool:
        """
        Analyze if image content is actually grayscale.

        Args:
            ctx: ImageContext with loaded image
            tolerance: Maximum allowed difference between channels

        Returns:
            True if image is effectively grayscale

        Example:
            >>> if ImageAnalyzer.is_grayscale_content(ctx):
            ...     print("Image is grayscale")
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        if ctx.img.mode in ('L', 'LA'):
            return True

        if ctx.img.mode not in ('RGB', 'RGBA'):
            return False

        arr = np.array(ctx.img)

        if arr.shape[2] >= 3:
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

            rg_diff = np.abs(r.astype(float) - g.astype(float)).mean() / 255.0
            gb_diff = np.abs(g.astype(float) - b.astype(float)).mean() / 255.0
            rb_diff = np.abs(r.astype(float) - b.astype(float)).mean() / 255.0

            return (rg_diff < tolerance and
                    gb_diff < tolerance and
                    rb_diff < tolerance)

        return False

    @staticmethod
    def compute_histogram(ctx: ImageContext, bins: int = 256) -> dict:
        """
        Compute RGB histograms.

        Args:
            ctx: ImageContext with loaded image
            bins: Number of bins

        Returns:
            Dictionary with histogram arrays

        Example:
            >>> hist = ImageAnalyzer.compute_histogram(ctx)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        arr = np.array(ctx.img)

        if arr.ndim == 2:
            hist, _ = np.histogram(arr, bins=bins, range=(0, 256))
            return {'gray': hist.tolist()}
        else:
            histograms = {}
            channel_names = ['red', 'green', 'blue']
            for i, name in enumerate(channel_names[:arr.shape[2]]):
                hist, _ = np.histogram(arr[:,:,i], bins=bins, range=(0, 256))
                histograms[name] = hist.tolist()
            return histograms

    @staticmethod
    def detect_dominant_colors(ctx: ImageContext, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Extract dominant color palette using k-means.

        Args:
            ctx: ImageContext with loaded image
            n_colors: Number of dominant colors

        Returns:
            List of (R, G, B) tuples

        Example:
            >>> colors = ImageAnalyzer.detect_dominant_colors(ctx, 5)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        arr = np.array(ctx.img)
        if arr.ndim == 2:
            arr = arr.reshape(-1, 1)
        else:
            arr = arr.reshape(-1, arr.shape[2])

        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(arr)

        colors = kmeans.cluster_centers_.astype(int)

        if colors.shape[1] == 1:
            return [(int(c[0]), int(c[0]), int(c[0])) for c in colors]
        else:
            return [tuple(c) for c in colors]
