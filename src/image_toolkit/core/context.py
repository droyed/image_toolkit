"""
ImageContext - Shared state container for image operations.

This is a pure data object with no business logic.
All components operate on this shared context.
"""
from PIL import Image
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ImageContext:
    """
    Lightweight container for image state.

    Provides a single source of truth that all components (loader, transformer, analyzer, annotator)
    operate on directly - ctx.img.

    Attributes:
        path: Path to the image file
        img: Current PIL Image object (may be transformed)
        metadata: Dictionary of metadata (EXIF, custom, etc.)
    """
    path: Path
    img: Optional[Image.Image] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> 'ImageContext':
        """
        Create ImageContext from a file path.

        Args:
            path: Path to the image file

        Returns:
            New ImageContext instance

        Example:
            >>> ctx = ImageContext.from_path('photo.jpg')
            >>> print(ctx.path)
        """
        return cls(
            path=Path(path),
            metadata={"original_path": str(path)}
        )

    @property
    def size(self) -> Optional[tuple]:
        """
        Get current image size (width, height).

        Returns:
            Tuple of (width, height) or None if not loaded
        """
        return self.img.size if self.img else None

    @property
    def mode(self) -> Optional[str]:
        """
        Get current image mode (RGB, RGBA, L, etc.).

        Returns:
            PIL image mode string or None if not loaded
        """
        return self.img.mode if self.img else None

    def is_loaded(self) -> bool:
        """
        Check if image is loaded in memory.

        Returns:
            True if image is loaded, False otherwise
        """
        return self.img is not None
