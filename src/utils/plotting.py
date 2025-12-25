"""
Plotting utilities for visualization and analysis.

This module provides:
- plot_learning_curve: Single algorithm learning curve
- plot_comparison: Multiple algorithms comparison
- plot_ablation_heatmap: 2D parameter ablation heatmap
- plot_ablation_bars: Bar chart for ablation results
"""

import os
from typing import List, Dict, Optional, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Use non-interactive backend for headless environments
matplotlib.use('Agg')


def plot_learning_curve(
    rewards: List[float],
    window: int = 100,
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot learning curve with moving average.

    Args:
        rewards: List of episode rewards
        window: Moving average window size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: If provided, save figure to this path
        show: Whether to display the plot

    Example:
        >>> rewards = [random.random() * i for i in range(1000)]
        >>> plot_learning_curve(rewards, save_path="curve.png")
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    episodes = np.arange(len(rewards))

    # Raw rewards (transparent)
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')

    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(
            episodes[window - 1:],
            moving_avg,
            color='blue',
            linewidth=2,
            label=f'{window}-episode average'
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning curve saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results: Dict[str, List[float]],
    window: int = 100,
    title: str = "Algorithm Comparison",
    xlabel: str = "Episode",
    ylabel: str = "Reward (Moving Average)",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot multiple algorithms on same axes for comparison.

    Args:
        results: Dict mapping algorithm name to reward list
        window: Moving average window
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional save path
        show: Whether to display the plot

    Example:
        >>> results = {
        ...     "DQN": [r * 0.5 for r in range(1000)],
        ...     "DDQN": [r * 0.6 for r in range(1000)],
        ... }
        >>> plot_comparison(results, save_path="comparison.png")
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, rewards), color in zip(results.items(), colors):
        episodes = np.arange(len(rewards))

        # Plot moving average
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(
                episodes[window - 1:],
                moving_avg,
                color=color,
                linewidth=2,
                label=name
            )
        else:
            ax.plot(episodes, rewards, color=color, linewidth=2, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_ablation_heatmap(
    results: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    x_param: str,
    y_param: str,
    metric_name: str = "Final Reward",
    title: str = "Parameter Ablation",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create heatmap for 2D parameter ablation.

    Args:
        results: 2D array of metric values (y_params x x_params)
        x_labels: Labels for x-axis parameters
        y_labels: Labels for y-axis parameters
        x_param: Name of x parameter
        y_param: Name of y parameter
        metric_name: Name of metric being plotted
        title: Plot title
        save_path: Optional save path
        show: Whether to display the plot

    Example:
        >>> results = np.random.rand(3, 4) * 100
        >>> plot_ablation_heatmap(
        ...     results,
        ...     x_labels=["0.0001", "0.001", "0.01", "0.1"],
        ...     y_labels=["0.9", "0.95", "0.99"],
        ...     x_param="Learning Rate",
        ...     y_param="Gamma"
        ... )
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(results, cmap='viridis', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Labels
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(title)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric_name, rotation=-90, va='bottom')

    # Add text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            value = results[i, j]
            text_color = 'white' if value < results.mean() else 'black'
            ax.text(
                j, i, f'{value:.1f}',
                ha='center', va='center',
                color=text_color, fontsize=10
            )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_ablation_bars(
    results: Dict[str, Dict[str, float]],
    metric_name: str = "Final Reward",
    title: str = "Ablation Study Results",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create bar chart for ablation study results with error bars.

    Args:
        results: Dict of {condition: {"mean": float, "std": float}}
        metric_name: Name of metric
        title: Plot title
        save_path: Optional save path
        show: Whether to display the plot

    Example:
        >>> results = {
        ...     "DQN": {"mean": 50.0, "std": 5.0},
        ...     "DDQN": {"mean": 60.0, "std": 4.0},
        ...     "Dueling": {"mean": 55.0, "std": 6.0},
        ... }
        >>> plot_ablation_bars(results)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(results.keys())
    means = [results[c]["mean"] for c in conditions]
    stds = [results[c].get("std", 0) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.6

    bars = ax.bar(x, means, width, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

    ax.set_xlabel('Configuration')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.1f}',
            xy=(bar.get_x() + bar.get_width() / 2, height + std),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9
        )

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Bar chart saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot multiple training metrics in subplots.

    Args:
        metrics: Dict of {metric_name: values_list}
        title: Overall figure title
        save_path: Optional save path
        show: Whether to display the plot

    Example:
        >>> metrics = {
        ...     "reward": [...],
        ...     "loss": [...],
        ...     "epsilon": [...]
        ... }
        >>> plot_training_metrics(metrics)
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)

    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        episodes = np.arange(len(values))
        ax.plot(episodes, values, linewidth=1)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Episode')
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
