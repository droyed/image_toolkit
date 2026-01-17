"""
Batch-specific operations for statistics, grids, and visualization.

Provides stateless utility functions for batch image analysis and visualization.
"""
import numpy as np
from PIL import Image, ImageOps
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


class BatchOperations:
    """
    Stateless batch-specific operations.

    Provides utility methods for:
    - Statistics aggregation across batch
    - Outlier detection
    - Grid visualization
    - Distribution visualization

    All methods are static and operate on provided data.
    """

    @staticmethod
    def compute_batch_stats(individual_stats: List[dict]) -> dict:
        """
        Aggregate statistics from individual images.

        Args:
            individual_stats: List of per-image statistics dictionaries

        Returns:
            Dictionary with aggregated statistics:
            {
                'count': int,
                'width': {'min': ..., 'max': ..., 'mean': ..., 'median': ..., 'std': ...},
                'height': {...},
                'aspect_ratio': {...},
                'individual_stats': [...]
            }

        Example:
            >>> stats = [{'width': 800, 'height': 600, 'aspect_ratio': 1.33}, ...]
            >>> batch_stats = BatchOperations.compute_batch_stats(stats)
            >>> print(batch_stats['width']['mean'])
        """
        if not individual_stats:
            return {
                'count': 0,
                'width': {},
                'height': {},
                'aspect_ratio': {},
                'individual_stats': []
            }

        # Extract arrays for each metric
        widths = np.array([s['width'] for s in individual_stats])
        heights = np.array([s['height'] for s in individual_stats])
        aspect_ratios = np.array([s['aspect_ratio'] for s in individual_stats])

        def compute_metric_stats(arr: np.ndarray) -> dict:
            """Compute min, max, mean, median, std for an array."""
            return {
                'min': float(arr.min()),
                'max': float(arr.max()),
                'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'std': float(arr.std())
            }

        return {
            'count': len(individual_stats),
            'width': compute_metric_stats(widths),
            'height': compute_metric_stats(heights),
            'aspect_ratio': compute_metric_stats(aspect_ratios),
            'individual_stats': individual_stats
        }

    @staticmethod
    def detect_outliers(batch_stats: dict, metric: str, threshold: float) -> List[int]:
        """
        Detect outlier images based on z-score.

        Args:
            batch_stats: Statistics from compute_batch_stats()
            metric: Metric to analyze ('size', 'width', 'height', 'aspect_ratio')
            threshold: Z-score threshold (typically 2.0 or 3.0)

        Returns:
            List of indices of outlier images

        Example:
            >>> outlier_indices = BatchOperations.detect_outliers(stats, 'size', 2.0)
            >>> outlier_paths = [paths[i] for i in outlier_indices]
        """
        individual_stats = batch_stats['individual_stats']

        if not individual_stats:
            return []

        # Extract metric values
        if metric == 'size':
            values = np.array([s['width'] * s['height']
                             for s in individual_stats])
        elif metric in ['width', 'height', 'aspect_ratio']:
            values = np.array([s[metric] for s in individual_stats])
        else:
            raise ValueError(f"Unknown metric: {metric}. "
                           f"Use 'size', 'width', 'height', or 'aspect_ratio'.")

        # Calculate z-scores
        mean = values.mean()
        std = values.std()

        if std == 0:
            return []  # No outliers if no variance

        z_scores = np.abs((values - mean) / std)

        # Return indices where |z-score| > threshold
        outlier_indices = np.where(z_scores > threshold)[0].tolist()

        return outlier_indices

    @staticmethod
    def create_grid(images: List[Image.Image],
                   rows: int,
                   cols: int,
                   cell_size: Tuple[int, int],
                   padding: int,
                   background_color: Tuple[int, int, int]) -> Image.Image:
        """
        Create grid visualization from images.

        Args:
            images: List of PIL Images
            rows: Number of rows
            cols: Number of columns
            cell_size: (width, height) of each cell
            padding: Padding between cells in pixels
            background_color: RGB color for background

        Returns:
            Single PIL Image containing the grid

        Example:
            >>> grid = BatchOperations.create_grid(images, 4, 4, (200, 200), 5, (255, 255, 255))
            >>> grid.save('grid.jpg')
        """
        cell_w, cell_h = cell_size

        # Calculate grid dimensions
        grid_w = cols * cell_w + (cols - 1) * padding
        grid_h = rows * cell_h + (rows - 1) * padding

        # Create canvas
        grid = Image.new('RGB', (grid_w, grid_h), background_color)

        # Paste images
        for idx, img in enumerate(images[:rows * cols]):
            row = idx // cols
            col = idx % cols

            # Resize image to cell size (preserving aspect ratio, fitting)
            img_resized = ImageOps.fit(img, cell_size, Image.Resampling.LANCZOS)

            # Calculate position
            x = col * (cell_w + padding)
            y = row * (cell_h + padding)

            # Paste into grid
            grid.paste(img_resized, (x, y))

        return grid

    @staticmethod
    def visualize_distribution(batch_stats: dict, save_path: Optional[str] = None):
        """
        Plot distribution of image properties.

        Creates 2x2 grid of histograms:
        - Width distribution
        - Height distribution
        - Aspect ratio distribution
        - Size (total pixels) distribution

        Args:
            batch_stats: Statistics from compute_batch_stats()
            save_path: If provided, saves plot to this path. Otherwise shows plot.

        Example:
            >>> BatchOperations.visualize_distribution(stats, 'distribution.png')
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization. "
                            "Install with: pip install matplotlib")

        individual_stats = batch_stats['individual_stats']

        if not individual_stats:
            print("No images to visualize.")
            return

        # Extract data
        widths = [s['width'] for s in individual_stats]
        heights = [s['height'] for s in individual_stats]
        ratios = [s['aspect_ratio'] for s in individual_stats]
        sizes = [s['width'] * s['height'] for s in individual_stats]

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Width distribution
        axes[0, 0].hist(widths, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].set_xlabel('Width (px)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)

        # Height distribution
        axes[0, 1].hist(heights, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].set_xlabel('Height (px)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)

        # Aspect ratio distribution
        axes[1, 0].hist(ratios, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Aspect Ratio Distribution')
        axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)

        # Size distribution
        axes[1, 1].hist(sizes, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Image Size Distribution')
        axes[1, 1].set_xlabel('Size (pixels)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Distribution saved to {save_path}")
        else:
            plt.show()
