"""
Logging utilities for experiment tracking.

This module provides:
- setup_logger: Configure Python logging with file and console handlers
- SafeCSVLogger: CSV logging with auto-flush for Colab safety
- CSVLogger: Simple CSV logging for IEEE report results (legacy)
- MetricsTracker: Running statistics for metrics
"""

import os
import csv
import sys
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path


# =============================================================================
# Python Logging Setup
# =============================================================================

def setup_logger(
    name: str = "vizdoom",
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with console and/or file handlers.

    Designed for Colab safety: file handler uses immediate flush
    to prevent data loss on session disconnect.

    Args:
        name: Logger name (e.g., "vizdoom.train", "vizdoom.agent")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to log to console (stdout)
        format_string: Custom format string (default includes timestamp)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("vizdoom.train", level="INFO", log_file="run/training.log")
        >>> logger.info("Training started")
        >>> logger.debug("Detailed debug info")  # Only shown if level=DEBUG
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Default format with timestamp for Colab traceability
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)

    # File handler with immediate flush for Colab safety
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # File captures all levels

        # Wrap write to force immediate flush (Colab safety)
        original_emit = file_handler.emit

        def emit_with_flush(record):
            original_emit(record)
            file_handler.flush()

        file_handler.emit = emit_with_flush
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "vizdoom") -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance (creates basic one if doesn't exist)
    """
    return logging.getLogger(name)


# =============================================================================
# Safe CSV Logger (Colab-Optimized)
# =============================================================================

class SafeCSVLogger:
    """
    CSV logger with auto-flush for Colab safety.

    Keeps file handle open for efficiency but flushes periodically
    to prevent data loss on Colab session disconnect.

    Args:
        filepath: Path to CSV file
        fieldnames: List of column names
        flush_every: Flush to disk every N rows (default: 10)

    Example:
        >>> logger = SafeCSVLogger("results/training.csv", ["episode", "reward"])
        >>> logger.log({"episode": 1, "reward": 50.0})
        >>> logger.log({"episode": 2, "reward": 60.0})
        >>> logger.close()  # Or use as context manager
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        fieldnames: List[str],
        flush_every: int = 10
    ) -> None:
        self.filepath = Path(filepath)
        self.fieldnames = fieldnames
        self.flush_every = flush_every
        self._count = 0
        self._closed = False

        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Open file and keep handle (efficiency)
        self._file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()  # Flush header immediately

    def log(self, row: Dict[str, Any]) -> None:
        """
        Log a single row to CSV.

        Auto-flushes every `flush_every` rows for Colab safety.

        Args:
            row: Dictionary with fieldname -> value
        """
        if self._closed:
            raise RuntimeError("Cannot log to closed SafeCSVLogger")

        self._writer.writerow(row)
        self._count += 1

        # Auto-flush for Colab safety
        if self._count % self.flush_every == 0:
            self._file.flush()

    def log_batch(self, rows: List[Dict[str, Any]]) -> None:
        """
        Log multiple rows and flush.

        Args:
            rows: List of row dictionaries
        """
        if self._closed:
            raise RuntimeError("Cannot log to closed SafeCSVLogger")

        self._writer.writerows(rows)
        self._count += len(rows)
        self._file.flush()  # Always flush after batch

    def flush(self) -> None:
        """Force flush to disk (call after checkpoints)."""
        if not self._closed:
            self._file.flush()

    def close(self) -> None:
        """Close file handle."""
        if not self._closed:
            self._file.flush()
            self._file.close()
            self._closed = True

    def __enter__(self) -> "SafeCSVLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures file is closed."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures file is closed."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    @property
    def row_count(self) -> int:
        """Number of rows logged."""
        return self._count


class CSVLogger:
    """
    Simple CSV logging for experiment results.

    Appends rows to a CSV file. Useful for generating data
    for IEEE report tables and figures.

    Args:
        filepath: Path to CSV file
        fieldnames: List of column names

    Example:
        >>> logger = CSVLogger("results/run1.csv", ["episode", "reward"])
        >>> logger.log({"episode": 100, "reward": 50.5})
    """

    def __init__(
        self,
        filepath: str,
        fieldnames: List[str]
    ) -> None:
        self.filepath = Path(filepath)
        self.fieldnames = fieldnames

        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write header
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """
        Append row to CSV file.

        Args:
            row: Dictionary with fieldname -> value
        """
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def log_batch(self, rows: List[Dict[str, Any]]) -> None:
        """
        Append multiple rows at once.

        Args:
            rows: List of row dictionaries
        """
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(rows)

    @staticmethod
    def aggregate_results(
        csv_files: List[str],
        output_path: str,
        group_by: List[str] = None
    ) -> None:
        """
        Aggregate multiple CSV files into summary statistics.

        Useful for combining results from multiple seeds.

        Args:
            csv_files: List of CSV file paths
            output_path: Output path for aggregated results
            group_by: Columns to group by (default: ['algorithm', 'scenario'])
        """
        try:
            import pandas as pd

            if group_by is None:
                group_by = ['algorithm', 'scenario']

            # Load and concatenate
            dfs = [pd.read_csv(f) for f in csv_files if os.path.exists(f)]
            if not dfs:
                print("No CSV files found to aggregate.")
                return

            combined = pd.concat(dfs, ignore_index=True)

            # Find numeric columns for aggregation
            numeric_cols = combined.select_dtypes(include=['number']).columns
            numeric_cols = [c for c in numeric_cols if c not in group_by]

            # Aggregate
            agg_funcs = {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols}
            summary = combined.groupby(group_by).agg(agg_funcs)

            # Flatten column names
            summary.columns = ['_'.join(col).strip() for col in summary.columns]
            summary = summary.reset_index()

            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_path, index=False)
            print(f"Aggregated results saved to: {output_path}")

        except ImportError:
            print("Warning: pandas required for aggregation. Install with: pip install pandas")
        except Exception as e:
            print(f"Error during aggregation: {e}")


class MetricsTracker:
    """
    Track and compute statistics for metrics during training.

    Maintains running statistics (mean, std, min, max) for
    a set of metrics.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update("reward", 50.0)
        >>> tracker.update("reward", 60.0)
        >>> stats = tracker.get_stats("reward")
        >>> print(stats["mean"])  # 55.0
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = {}

    def update(self, name: str, value: float) -> None:
        """Add a value to a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_stats(
        self,
        name: str,
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            name: Metric name
            window: Only consider last N values (None = all)

        Returns:
            Dictionary with mean, std, min, max, count
        """
        if name not in self.metrics or not self.metrics[name]:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        import numpy as np

        values = self.metrics[name]
        if window is not None:
            values = values[-window:]

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values)
        }

    def reset(self, name: Optional[str] = None) -> None:
        """Reset metrics. If name is None, reset all."""
        if name is None:
            self.metrics = {}
        elif name in self.metrics:
            self.metrics[name] = []
