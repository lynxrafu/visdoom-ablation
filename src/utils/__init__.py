"""
Utility modules for training and evaluation.

Includes:
- Replay buffers (uniform and prioritized)
- Logging (CSV)
- Plotting (learning curves, comparisons)
- Factory functions (agent and buffer builders)
- Analysis (results comparison and reporting)
"""

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, SumTree
from .logging import CSVLogger
from .plotting import plot_learning_curve, plot_comparison, plot_ablation_heatmap
from .factory import build_agent, build_buffer
from .analysis import (
    ResultsAnalyzer,
    ExperimentResult,
    AggregatedResult,
    ComparisonResult,
    ResultPlotter,
    analyze_results,
)

__all__ = [
    # Replay buffers
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "SumTree",
    # Logging
    "CSVLogger",
    # Plotting
    "plot_learning_curve",
    "plot_comparison",
    "plot_ablation_heatmap",
    # Factory
    "build_agent",
    "build_buffer",
    # Analysis
    "ResultsAnalyzer",
    "ExperimentResult",
    "AggregatedResult",
    "ComparisonResult",
    "ResultPlotter",
    "analyze_results",
]
