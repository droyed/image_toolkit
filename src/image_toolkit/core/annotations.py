"""
ImageAnnotator - Stateless annotation operations using ImageContext
"""
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Union, Dict
import numpy as np

from .context import ImageContext


class ImageAnnotator:
    """
    Stateless annotation operations for images.

    All methods operate on ImageContext to maintain single source of truth.
    """

    @staticmethod
    def draw_bbox(ctx: ImageContext, box: Tuple[float, float, float, float],
                  label: Optional[str] = None, color: str = "red",
                  width: int = 3, normalized: bool = False) -> ImageContext:
        """
        Draw bounding box on image in context.

        Args:
            ctx: ImageContext with loaded image
            box: (xmin, ymin, xmax, ymax)
            label: Optional text label
            color: Box color
            width: Line width
            normalized: If True, coordinates are 0.0-1.0

        Returns:
            Same context with annotated image

        Example:
            >>> ImageAnnotator.draw_bbox(ctx, (100, 100, 300, 300), "Person")
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        # Ensure RGB mode for drawing
        if ctx.img.mode != 'RGB':
            ctx.img = ctx.img.convert('RGB')

        draw = ImageDraw.Draw(ctx.img)
        xmin, ymin, xmax, ymax = box

        # Convert normalized to pixel coordinates
        if normalized:
            w, h = ctx.img.size
            xmin, xmax = xmin * w, xmax * w
            ymin, ymax = ymin * h, ymax * h

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)

        # Draw label if provided
        if label:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            text_pos = (xmin, ymin - 15 if ymin > 15 else ymin)
            draw.text(text_pos, label, fill=color, font=font)

        return ctx

    @staticmethod
    def draw_bboxes(ctx: ImageContext,
                    boxes: List[Union[Tuple[float, float, float, float], Dict]],
                    color: str = "red",
                    width: int = 3,
                    normalized: bool = False) -> ImageContext:
        """
        Draw multiple bounding boxes.

        Args:
            ctx: ImageContext with loaded image
            boxes: List of boxes (tuples or dicts)
            color: Default color
            width: Line thickness
            normalized: If True, coordinates are 0.0-1.0

        Returns:
            Same context with annotated image

        Example:
            >>> boxes = [(100,100,200,200), (300,300,400,400)]
            >>> ImageAnnotator.draw_bboxes(ctx, boxes)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        # Ensure RGB mode
        if ctx.img.mode != 'RGB':
            ctx.img = ctx.img.convert('RGB')

        draw = ImageDraw.Draw(ctx.img)
        w, h = ctx.img.size

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for item in boxes:
            # Handle both raw tuples and dictionaries
            if isinstance(item, dict):
                box = item.get("box")
                label = item.get("label")
                box_color = item.get("color", color)
            else:
                box = item
                label = None
                box_color = color

            xmin, ymin, xmax, ymax = box

            if normalized:
                xmin, xmax = xmin * w, xmax * w
                ymin, ymax = ymin * h, ymax * h

            # Draw box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=width)

            # Draw label
            if label:
                text_pos = (xmin, ymin - 12 if ymin > 12 else ymin)
                draw.text(text_pos, str(label), fill=box_color, font=font)

        return ctx

    @staticmethod
    def draw_polygon(ctx: ImageContext, points: List[Tuple[int, int]],
                     label: Optional[str] = None,
                     color: str = "red",
                     width: int = 2,
                     fill: Optional[str] = None) -> ImageContext:
        """
        Draw polygon on image in context.

        Args:
            ctx: ImageContext with loaded image
            points: List of (x, y) coordinates
            label: Optional text label
            color: Outline color
            width: Line width
            fill: Optional fill color

        Returns:
            Same context with annotated image

        Example:
            >>> points = [(100,100), (200,100), (150,200)]
            >>> ImageAnnotator.draw_polygon(ctx, points, "Triangle")
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        draw = ImageDraw.Draw(ctx.img)

        # Draw polygon
        if fill:
            draw.polygon(points, outline=color, fill=fill, width=width)
        else:
            draw.polygon(points, outline=color, width=width)

        # Add label if provided
        if label:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            text_pos = points[0]
            draw.text(text_pos, label, fill=color, font=font)

        return ctx

    @staticmethod
    def draw_mask(ctx: ImageContext, mask: np.ndarray,
                  color: Tuple[int, int, int] = (255, 0, 0),
                  alpha: float = 0.5) -> ImageContext:
        """
        Overlay a binary or semantic segmentation mask.

        Args:
            ctx: ImageContext with loaded image
            mask: 2D numpy array (H, W)
            color: RGB color for overlay
            alpha: Transparency (0.0 = invisible, 1.0 = opaque)

        Returns:
            Same context with mask overlaid

        Example:
            >>> mask = np.zeros((height, width))
            >>> mask[100:200, 100:200] = 1
            >>> ImageAnnotator.draw_mask(ctx, mask, (255, 0, 0), 0.5)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        # Ensure mask matches image size
        if mask.shape[:2] != ctx.img.size[::-1]:
            raise ValueError(f"Mask shape {mask.shape} doesn't match image size {ctx.img.size}")

        # Create colored mask
        mask_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_img[mask > 0] = color

        # Convert to PIL
        mask_pil = Image.fromarray(mask_img)

        # Blend with original
        ctx.img = Image.blend(ctx.img.convert('RGB'), mask_pil, alpha)

        return ctx

    @staticmethod
    def draw_keypoints(ctx: ImageContext, keypoints: List[Tuple[int, int]],
                       labels: Optional[List[str]] = None,
                       color: str = "red",
                       radius: int = 5) -> ImageContext:
        """
        Draw keypoint markers.

        Args:
            ctx: ImageContext with loaded image
            keypoints: List of (x, y) coordinates
            labels: Optional labels for each keypoint
            color: Color of markers
            radius: Radius of keypoint circles

        Returns:
            Same context with keypoints drawn

        Example:
            >>> keypoints = [(100, 100), (150, 120)]
            >>> ImageAnnotator.draw_keypoints(ctx, keypoints)
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        draw = ImageDraw.Draw(ctx.img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for i, (x, y) in enumerate(keypoints):
            # Draw circle
            bbox = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(bbox, fill=color, outline=color)

            # Add label if provided
            if labels and i < len(labels):
                draw.text((x + radius + 2, y), labels[i], fill=color, font=font)

        return ctx

    @staticmethod
    def draw_text(ctx: ImageContext, text: str,
                  position: Tuple[int, int],
                  font_size: int = 16,
                  color: str = "white",
                  background: Optional[str] = None) -> ImageContext:
        """
        Add text annotation to image.

        Args:
            ctx: ImageContext with loaded image
            text: Text to draw
            position: (x, y) coordinates
            font_size: Font size
            color: Text color
            background: Optional background color

        Returns:
            Same context with text drawn

        Example:
            >>> ImageAnnotator.draw_text(ctx, "Sample", (50, 50), background="black")
        """
        if ctx.img is None:
            raise ValueError("No image loaded in context")

        draw = ImageDraw.Draw(ctx.img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Draw background box if requested
        if background:
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(bbox, fill=background)

        # Draw text
        draw.text(position, text, fill=color, font=font)

        return ctx
