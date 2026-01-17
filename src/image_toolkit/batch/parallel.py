"""
Parallel execution utilities with progress tracking.

Provides ThreadPoolExecutor wrapper with tqdm integration for batch image processing.
"""
import os
from typing import Callable, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class ParallelExecutor:
    """
    Wrapper for parallel execution with progress tracking.

    Uses ThreadPoolExecutor for I/O-bound operations (image loading, saving, transforms).
    Integrates with tqdm for progress visualization and supports custom callbacks.

    Example:
        >>> executor = ParallelExecutor(max_workers=8, show_progress=True)
        >>> def process(item): return item * 2
        >>> results, errors = executor.map(process, [1, 2, 3], "Processing")
    """

    def __init__(self,
                 max_workers: Optional[int] = None,
                 show_progress: bool = True,
                 progress_callback: Optional[Callable[[int, int], None]] = None):
        """
        Initialize ParallelExecutor.

        Args:
            max_workers: Number of worker threads. Default: min(32, cpu_count+4)
            show_progress: Whether to show tqdm progress bar
            progress_callback: Custom progress callback(current, total)
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.show_progress = show_progress
        self.progress_callback = progress_callback

    def map(self,
            func: Callable,
            items: List[Any],
            description: str = "Processing") -> Tuple[List[Any], List[Tuple[Any, Exception]]]:
        """
        Apply function to items in parallel with progress tracking.

        Args:
            func: Function to apply to each item
            items: List of items to process
            description: Description for progress bar

        Returns:
            Tuple of (successful_results, errors)
            errors is list of (item, exception) tuples

        Example:
            >>> executor = ParallelExecutor()
            >>> results, errors = executor.map(lambda x: x*2, [1,2,3], "Doubling")
        """
        if not items:
            return [], []

        # Small batch optimization: use sequential processing
        if len(items) < 10 or self.max_workers == 1:
            return self._sequential_map(func, items, description)

        # Parallel processing with ThreadPoolExecutor
        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(self._safe_apply, func, item): item
                             for item in items}

            # Setup progress tracking
            if self.show_progress and self.progress_callback is None:
                iterator = tqdm(as_completed(future_to_item),
                              total=len(items),
                              desc=description)
            else:
                iterator = as_completed(future_to_item)

            # Collect results as they complete
            for i, future in enumerate(iterator, 1):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append((item, e))

                # Custom progress callback
                if self.progress_callback:
                    self.progress_callback(i, len(items))

        return results, errors

    def _safe_apply(self, func: Callable, item: Any) -> Any:
        """
        Apply function with error handling.

        Errors are caught by future.result() in map().
        """
        return func(item)

    def _sequential_map(self,
                       func: Callable,
                       items: List[Any],
                       description: str) -> Tuple[List[Any], List[Tuple[Any, Exception]]]:
        """
        Sequential fallback for small batches or single-threaded execution.

        Args:
            func: Function to apply
            items: Items to process
            description: Progress bar description

        Returns:
            Tuple of (results, errors)
        """
        results = []
        errors = []

        # Setup progress tracking
        if self.show_progress and self.progress_callback is None:
            iterator = tqdm(items, desc=description)
        else:
            iterator = items

        for i, item in enumerate(iterator, 1):
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                errors.append((item, e))

            # Custom progress callback
            if self.progress_callback:
                self.progress_callback(i, len(items))

        return results, errors
