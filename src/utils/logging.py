"""
Logging utilities for experiment tracking.

This module provides:
- WandbLogger: Weights & Biases integration for cloud logging
- CSVLogger: Simple CSV logging for IEEE report results
"""

import os
import csv
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path


class WandbLogger:
    """
    Weights & Biases logging wrapper.

    Provides a simple interface for logging metrics, videos, and
    other artifacts to WandB. Can be disabled for offline runs.

    Args:
        project: WandB project name
        config: Configuration dictionary to log
        run_name: Optional custom run name
        enabled: Whether to actually log to WandB

    Example:
        >>> logger = WandbLogger(project="my-project", config={"lr": 0.001})
        >>> logger.log({"loss": 0.5, "reward": 100}, step=1000)
        >>> logger.finish()
    """

    def __init__(
        self,
        project: str = "vizdoom-ablation",
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        enabled: bool = True
    ) -> None:
        self.enabled = enabled
        self.run = None

        if enabled:
            try:
                import wandb
                self.wandb = wandb

                # Generate run name if not provided
                if run_name is None:
                    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                self.run = wandb.init(
                    project=project,
                    config=config,
                    name=run_name,
                    reinit=True
                )
                print(f"WandB initialized: {run_name}")

            except ImportError:
                print("Warning: wandb not installed. Logging disabled.")
                self.enabled = False
            except Exception as e:
                print(f"Warning: WandB initialization failed: {e}")
                self.enabled = False

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step/episode number
        """
        if self.enabled and self.run:
            self.wandb.log(metrics, step=step)

    def log_video(
        self,
        frames: List,
        fps: int = 30,
        caption: str = "episode"
    ) -> None:
        """
        Log video of episode to WandB.

        Args:
            frames: List of frame arrays (H, W, C)
            fps: Frames per second
            caption: Video caption
        """
        if self.enabled and self.run:
            try:
                import numpy as np
                frames_array = np.array(frames)
                video = self.wandb.Video(frames_array, fps=fps, caption=caption)
                self.wandb.log({"video": video})
            except Exception as e:
                print(f"Warning: Failed to log video: {e}")

    def log_image(
        self,
        image,
        name: str = "image"
    ) -> None:
        """
        Log image to WandB.

        Args:
            image: Image array or PIL Image
            name: Image name
        """
        if self.enabled and self.run:
            try:
                self.wandb.log({name: self.wandb.Image(image)})
            except Exception as e:
                print(f"Warning: Failed to log image: {e}")

    def log_table(
        self,
        data: List[List],
        columns: List[str],
        name: str = "table"
    ) -> None:
        """
        Log table to WandB.

        Args:
            data: List of rows
            columns: Column names
            name: Table name
        """
        if self.enabled and self.run:
            try:
                table = self.wandb.Table(columns=columns, data=data)
                self.wandb.log({name: table})
            except Exception as e:
                print(f"Warning: Failed to log table: {e}")

    def finish(self) -> None:
        """Close WandB run."""
        if self.enabled and self.run:
            self.wandb.finish()
            print("WandB run finished.")


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
