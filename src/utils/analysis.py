"""
Results Analysis Module - Automated comparison and visualization.

SOLID Design:
- Single Responsibility: Separate classes for loading, aggregating, comparing, exporting
- Open/Closed: Easy to add new metrics and plot types
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Classes depend on abstractions

Usage:
    from src.utils.analysis import ResultsAnalyzer

    analyzer = ResultsAnalyzer("results/")
    analyzer.load_all()
    comparison = analyzer.compare_algorithms()
    analyzer.generate_report("results/report/")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Protocol
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentResult:
    """Single experiment run result."""
    algorithm: str
    scenario: str
    seed: int
    config: Dict[str, Any]
    episodes: List[int]
    rewards: List[float]
    losses: List[float]
    epsilons: List[float]
    training_time_hours: float

    @property
    def final_reward(self) -> float:
        """Average reward of last 100 episodes."""
        return np.mean(self.rewards[-100:]) if len(self.rewards) >= 100 else np.mean(self.rewards)

    @property
    def max_reward(self) -> float:
        """Maximum episode reward."""
        return max(self.rewards) if self.rewards else 0.0

    @property
    def convergence_episode(self) -> int:
        """Episode where 90% of max performance was reached."""
        if not self.rewards:
            return 0
        threshold = 0.9 * self.final_reward
        for i, r in enumerate(pd.Series(self.rewards).rolling(50).mean()):
            if r and r >= threshold:
                return i
        return len(self.rewards)

    @property
    def stability(self) -> float:
        """Reward standard deviation in last 100 episodes (lower = more stable)."""
        return np.std(self.rewards[-100:]) if len(self.rewards) >= 100 else np.std(self.rewards)


@dataclass
class AggregatedResult:
    """Aggregated results across multiple seeds."""
    algorithm: str
    scenario: str
    num_seeds: int

    # Reward statistics
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float

    # Convergence
    convergence_mean: float
    convergence_std: float

    # Training time
    time_mean_hours: float
    time_std_hours: float

    # Stability
    stability_mean: float
    stability_std: float

    # Raw data for plotting
    all_rewards: List[List[float]] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Comparison between algorithms."""
    scenario: str
    algorithms: List[str]
    results: Dict[str, AggregatedResult]

    # Rankings
    rank_by_reward: List[str] = field(default_factory=list)
    rank_by_speed: List[str] = field(default_factory=list)
    rank_by_stability: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.results:
            self.rank_by_reward = sorted(
                self.algorithms,
                key=lambda a: self.results[a].reward_mean,
                reverse=True
            )
            self.rank_by_speed = sorted(
                self.algorithms,
                key=lambda a: self.results[a].convergence_mean
            )
            self.rank_by_stability = sorted(
                self.algorithms,
                key=lambda a: self.results[a].stability_mean
            )


# =============================================================================
# Protocols (Interfaces)
# =============================================================================

class ResultLoader(Protocol):
    """Protocol for loading experiment results."""
    def load(self, path: Path) -> Optional[ExperimentResult]: ...


class ResultAggregator(Protocol):
    """Protocol for aggregating results."""
    def aggregate(self, results: List[ExperimentResult]) -> AggregatedResult: ...


class ResultExporter(Protocol):
    """Protocol for exporting results."""
    def export(self, data: Any, path: Path) -> None: ...


# =============================================================================
# Loaders
# =============================================================================

class CSVResultLoader:
    """Load results from CSV files."""

    def load(self, path: Path) -> Optional[ExperimentResult]:
        """Load a single CSV result file."""
        try:
            df = pd.read_csv(path)

            # Try to load metadata from parent directory first (new format)
            metadata_path = path.parent / "metadata.json"
            config_path = path.parent / "config.yaml"

            algorithm = "unknown"
            scenario = "unknown"
            seed = 1
            config = {}

            if metadata_path.exists():
                # New format: read from metadata.json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                algorithm = metadata.get('agent_type', 'unknown')
                scenario = metadata.get('scenario', 'unknown')
                seed = metadata.get('seed', 1)
            else:
                # Old format: parse from directory name or filename
                # Directory format: HHMMSS_agent_scenario_lrX_seedY
                dir_name = path.parent.name
                parts = dir_name.split('_')

                if len(parts) >= 3:
                    # Skip timestamp if present (6 digits)
                    start_idx = 1 if parts[0].isdigit() and len(parts[0]) == 6 else 0
                    algorithm = parts[start_idx] if len(parts) > start_idx else "unknown"

                    # Find scenario (contains 'Vizdoom' or 'v0')
                    for i, part in enumerate(parts[start_idx:], start_idx):
                        if 'vizdoom' in part.lower() or 'v0' in part.lower():
                            # Reconstruct scenario name
                            scenario = part.replace('_', '-')
                            break

                    # Extract seed
                    for part in parts:
                        if part.startswith('seed'):
                            seed = int(part.replace('seed', ''))
                            break

            # Load config if available
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                except:
                    pass

            return ExperimentResult(
                algorithm=algorithm,
                scenario=scenario,
                seed=seed,
                config=config,
                episodes=df['episode'].tolist() if 'episode' in df else list(range(len(df))),
                rewards=df['reward'].tolist() if 'reward' in df else [],
                losses=df['loss'].tolist() if 'loss' in df else [],
                epsilons=df['epsilon'].tolist() if 'epsilon' in df else [],
                training_time_hours=df['time_hours'].iloc[-1] if 'time_hours' in df else 0.0
            )
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None


# =============================================================================
# Aggregators
# =============================================================================

class StandardAggregator:
    """Standard aggregation across seeds."""

    def aggregate(self, results: List[ExperimentResult]) -> AggregatedResult:
        """Aggregate multiple experiment runs."""
        if not results:
            raise ValueError("No results to aggregate")

        algorithm = results[0].algorithm
        scenario = results[0].scenario

        final_rewards = [r.final_reward for r in results]
        convergences = [r.convergence_episode for r in results]
        times = [r.training_time_hours for r in results]
        stabilities = [r.stability for r in results]
        all_rewards = [r.rewards for r in results]

        return AggregatedResult(
            algorithm=algorithm,
            scenario=scenario,
            num_seeds=len(results),
            reward_mean=np.mean(final_rewards),
            reward_std=np.std(final_rewards),
            reward_min=np.min(final_rewards),
            reward_max=np.max(final_rewards),
            convergence_mean=np.mean(convergences),
            convergence_std=np.std(convergences),
            time_mean_hours=np.mean(times),
            time_std_hours=np.std(times),
            stability_mean=np.mean(stabilities),
            stability_std=np.std(stabilities),
            all_rewards=all_rewards
        )


# =============================================================================
# Exporters
# =============================================================================

class CSVExporter:
    """Export results to CSV."""

    def export(self, data: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        print(f"Exported CSV: {path}")


class LaTeXExporter:
    """Export results to LaTeX table."""

    def export(self, data: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        latex = data.to_latex(index=False, float_format="%.2f", escape=False)

        # Add booktabs formatting
        latex = latex.replace("\\toprule", "\\hline\\hline")
        latex = latex.replace("\\midrule", "\\hline")
        latex = latex.replace("\\bottomrule", "\\hline\\hline")

        with open(path, 'w') as f:
            f.write(latex)
        print(f"Exported LaTeX: {path}")


class JSONExporter:
    """Export results to JSON."""

    def export(self, data: Dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Exported JSON: {path}")


# =============================================================================
# Plotters
# =============================================================================

class ResultPlotter:
    """Generate plots for results."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available

        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def plot_learning_curves(
        self,
        results: Dict[str, AggregatedResult],
        title: str = "Learning Curves",
        save_path: Optional[Path] = None,
        window: int = 50
    ) -> None:
        """Plot learning curves with confidence bands."""
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, (algo, result) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]

            # Compute mean and std across seeds
            min_len = min(len(r) for r in result.all_rewards)
            rewards_array = np.array([r[:min_len] for r in result.all_rewards])

            mean_rewards = pd.Series(rewards_array.mean(axis=0)).rolling(window).mean()
            std_rewards = pd.Series(rewards_array.std(axis=0)).rolling(window).mean()

            episodes = np.arange(len(mean_rewards))

            ax.plot(episodes, mean_rewards, color=color, label=algo, linewidth=2)
            ax.fill_between(
                episodes,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                color=color,
                alpha=0.2
            )

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {save_path}")

        plt.close()

    def plot_comparison_bars(
        self,
        results: Dict[str, AggregatedResult],
        metric: str = "reward",
        title: str = "Algorithm Comparison",
        save_path: Optional[Path] = None
    ) -> None:
        """Plot bar chart comparing algorithms."""
        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = list(results.keys())
        x = np.arange(len(algorithms))
        width = 0.6

        if metric == "reward":
            means = [results[a].reward_mean for a in algorithms]
            stds = [results[a].reward_std for a in algorithms]
            ylabel = "Final Reward"
        elif metric == "convergence":
            means = [results[a].convergence_mean for a in algorithms]
            stds = [results[a].convergence_std for a in algorithms]
            ylabel = "Convergence Episode"
        elif metric == "time":
            means = [results[a].time_mean_hours for a in algorithms]
            stds = [results[a].time_std_hours for a in algorithms]
            ylabel = "Training Time (hours)"
        elif metric == "stability":
            means = [results[a].stability_mean for a in algorithms]
            stds = [results[a].stability_std for a in algorithms]
            ylabel = "Stability (Reward Std)"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      color=self.colors[:len(algorithms)], alpha=0.8)

        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.annotate(
                f'{mean:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9
            )

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {save_path}")

        plt.close()

    def plot_ablation_heatmap(
        self,
        data: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        x_name: str,
        y_name: str,
        title: str = "Ablation Heatmap",
        save_path: Optional[Path] = None
    ) -> None:
        """Plot heatmap for parameter ablation."""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data, cmap='viridis', aspect='auto')

        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(x_name, fontsize=12)
        ax.set_ylabel(y_name, fontsize=12)
        ax.set_title(title, fontsize=14)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Reward', rotation=-90, va='bottom')

        # Add text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text_color = 'white' if data[i, j] < data.mean() else 'black'
                ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                       color=text_color, fontsize=10)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {save_path}")

        plt.close()

    def plot_radar(
        self,
        results: Dict[str, AggregatedResult],
        title: str = "Algorithm Comparison (Radar)",
        save_path: Optional[Path] = None
    ) -> None:
        """Plot radar/spider chart comparing multiple metrics."""
        categories = ['Reward', 'Speed', 'Stability', 'Efficiency']
        num_vars = len(categories)

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, (algo, result) in enumerate(results.items()):
            # Normalize metrics to 0-1 scale
            all_rewards = [r.reward_mean for r in results.values()]
            all_conv = [r.convergence_mean for r in results.values()]
            all_stab = [r.stability_mean for r in results.values()]
            all_time = [r.time_mean_hours for r in results.values()]

            values = [
                (result.reward_mean - min(all_rewards)) / (max(all_rewards) - min(all_rewards) + 1e-8),
                1 - (result.convergence_mean - min(all_conv)) / (max(all_conv) - min(all_conv) + 1e-8),
                1 - (result.stability_mean - min(all_stab)) / (max(all_stab) - min(all_stab) + 1e-8),
                1 - (result.time_mean_hours - min(all_time)) / (max(all_time) - min(all_time) + 1e-8),
            ]
            values += values[:1]

            color = self.colors[i % len(self.colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {save_path}")

        plt.close()


# =============================================================================
# Main Analyzer Class
# =============================================================================

class ResultsAnalyzer:
    """
    Main class for analyzing experiment results.

    Provides a unified interface for:
    - Loading results from CSV files
    - Aggregating across seeds
    - Comparing algorithms
    - Generating plots and reports

    Example:
        >>> analyzer = ResultsAnalyzer("results/")
        >>> analyzer.load_all()
        >>> print(analyzer.summary())
        >>> analyzer.compare_algorithms("VizdoomBasic-v0")
        >>> analyzer.generate_report("results/report/")
    """

    def __init__(
        self,
        results_dir: str = "results",
        loader: Optional[ResultLoader] = None,
        aggregator: Optional[ResultAggregator] = None
    ):
        self.results_dir = Path(results_dir)
        self.loader = loader or CSVResultLoader()
        self.aggregator = aggregator or StandardAggregator()
        self.plotter = ResultPlotter()

        # Exporters
        self.csv_exporter = CSVExporter()
        self.latex_exporter = LaTeXExporter()
        self.json_exporter = JSONExporter()

        # Storage
        self.raw_results: List[ExperimentResult] = []
        self.aggregated: Dict[Tuple[str, str], AggregatedResult] = {}
        self.comparisons: Dict[str, ComparisonResult] = {}

    def load_all(self) -> int:
        """Load all CSV results from the results directory (recursively)."""
        self.raw_results = []

        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return 0

        # Recursively find all training_log.csv and *.csv files
        csv_files = list(self.results_dir.glob("**/training_log.csv"))
        csv_files.extend(list(self.results_dir.glob("**/*.csv")))
        # Remove duplicates and filter out non-training files
        csv_files = list(set(f for f in csv_files if 'summary' not in f.name))

        for csv_file in csv_files:
            result = self.loader.load(csv_file)
            if result and result.rewards:  # Only add if has data
                self.raw_results.append(result)

        print(f"Loaded {len(self.raw_results)} experiment results")
        self._aggregate_all()
        return len(self.raw_results)

    def _aggregate_all(self) -> None:
        """Aggregate all results by algorithm and scenario."""
        self.aggregated = {}

        # Group by (algorithm, scenario)
        groups: Dict[Tuple[str, str], List[ExperimentResult]] = {}

        for result in self.raw_results:
            key = (result.algorithm, result.scenario)
            if key not in groups:
                groups[key] = []
            groups[key].append(result)

        # Aggregate each group
        for key, results in groups.items():
            self.aggregated[key] = self.aggregator.aggregate(results)

        print(f"Aggregated into {len(self.aggregated)} algorithm-scenario combinations")

    def get_algorithms(self) -> List[str]:
        """Get list of unique algorithms."""
        return list(set(r.algorithm for r in self.raw_results))

    def get_scenarios(self) -> List[str]:
        """Get list of unique scenarios."""
        return list(set(r.scenario for r in self.raw_results))

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        rows = []

        for (algo, scenario), result in self.aggregated.items():
            rows.append({
                'Algorithm': algo,
                'Scenario': scenario,
                'Seeds': result.num_seeds,
                'Reward (mean)': f"{result.reward_mean:.2f}",
                'Reward (std)': f"{result.reward_std:.2f}",
                'Convergence': f"{result.convergence_mean:.0f}",
                'Time (hrs)': f"{result.time_mean_hours:.2f}",
                'Stability': f"{result.stability_mean:.2f}"
            })

        return pd.DataFrame(rows).sort_values(['Scenario', 'Algorithm'])

    def compare_algorithms(
        self,
        scenario: Optional[str] = None
    ) -> Dict[str, ComparisonResult]:
        """Compare algorithms for given scenario(s)."""
        scenarios = [scenario] if scenario else self.get_scenarios()

        for scen in scenarios:
            algorithms = []
            results = {}

            for (algo, s), result in self.aggregated.items():
                if s == scen:
                    algorithms.append(algo)
                    results[algo] = result

            if results:
                self.comparisons[scen] = ComparisonResult(
                    scenario=scen,
                    algorithms=algorithms,
                    results=results
                )

        return self.comparisons

    def print_comparison(self, scenario: str) -> None:
        """Print comparison for a scenario."""
        if scenario not in self.comparisons:
            self.compare_algorithms(scenario)

        if scenario not in self.comparisons:
            print(f"No results for scenario: {scenario}")
            return

        comp = self.comparisons[scenario]

        print(f"\n{'=' * 60}")
        print(f"COMPARISON: {scenario}")
        print('=' * 60)

        print(f"\n{'Algorithm':<15} {'Reward':>12} {'Convergence':>12} {'Time (h)':>10} {'Stability':>10}")
        print('-' * 60)

        for algo in comp.rank_by_reward:
            r = comp.results[algo]
            print(f"{algo:<15} {r.reward_mean:>8.2f} +/-{r.reward_std:>4.1f} "
                  f"{r.convergence_mean:>8.0f} {r.time_mean_hours:>10.2f} {r.stability_mean:>10.2f}")

        print(f"\n Rankings:")
        print(f"  By Reward:    {' > '.join(comp.rank_by_reward)}")
        print(f"  By Speed:     {' > '.join(comp.rank_by_speed)}")
        print(f"  By Stability: {' > '.join(comp.rank_by_stability)}")

    def generate_report(
        self,
        output_dir: str = "results/report",
        include_plots: bool = True,
        include_tables: bool = True
    ) -> None:
        """Generate complete analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating report in: {output_path}")

        # Ensure comparisons are computed
        self.compare_algorithms()

        # 1. Export summary table
        if include_tables:
            summary_df = self.summary()
            self.csv_exporter.export(summary_df, output_path / "summary.csv")
            self.latex_exporter.export(summary_df, output_path / "summary.tex")

        # 2. Generate plots for each scenario
        if include_plots:
            for scenario, comp in self.comparisons.items():
                scenario_clean = scenario.replace('-', '_').replace('/', '_')

                # Learning curves
                self.plotter.plot_learning_curves(
                    comp.results,
                    title=f"Learning Curves - {scenario}",
                    save_path=output_path / f"curves_{scenario_clean}.png"
                )

                # Bar charts
                for metric in ["reward", "convergence", "stability"]:
                    self.plotter.plot_comparison_bars(
                        comp.results,
                        metric=metric,
                        title=f"{metric.title()} Comparison - {scenario}",
                        save_path=output_path / f"bars_{metric}_{scenario_clean}.png"
                    )

                # Radar chart
                if len(comp.algorithms) >= 3:
                    self.plotter.plot_radar(
                        comp.results,
                        title=f"Multi-Metric Comparison - {scenario}",
                        save_path=output_path / f"radar_{scenario_clean}.png"
                    )

        # 3. Export detailed JSON
        report_data = {
            "summary": self.summary().to_dict(orient='records'),
            "comparisons": {
                scenario: {
                    "algorithms": comp.algorithms,
                    "rank_by_reward": comp.rank_by_reward,
                    "rank_by_speed": comp.rank_by_speed,
                    "rank_by_stability": comp.rank_by_stability,
                    "results": {
                        algo: {
                            "reward_mean": r.reward_mean,
                            "reward_std": r.reward_std,
                            "convergence_mean": r.convergence_mean,
                            "time_mean_hours": r.time_mean_hours,
                            "stability_mean": r.stability_mean
                        }
                        for algo, r in comp.results.items()
                    }
                }
                for scenario, comp in self.comparisons.items()
            }
        }
        self.json_exporter.export(report_data, output_path / "report.json")

        print(f"\nReport complete! Files saved to: {output_path}")

    def get_best_algorithm(
        self,
        scenario: str,
        metric: str = "reward"
    ) -> Tuple[str, AggregatedResult]:
        """Get the best algorithm for a scenario by given metric."""
        if scenario not in self.comparisons:
            self.compare_algorithms(scenario)

        comp = self.comparisons.get(scenario)
        if not comp:
            raise ValueError(f"No results for scenario: {scenario}")

        if metric == "reward":
            best = comp.rank_by_reward[0]
        elif metric == "speed":
            best = comp.rank_by_speed[0]
        elif metric == "stability":
            best = comp.rank_by_stability[0]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best, comp.results[best]


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_results(
    results_dir: str = "results",
    output_dir: str = "results/report"
) -> ResultsAnalyzer:
    """
    Convenience function to analyze results and generate report.

    Usage:
        analyzer = analyze_results("results/", "results/report/")
        print(analyzer.summary())
    """
    analyzer = ResultsAnalyzer(results_dir)
    count = analyzer.load_all()

    if count > 0:
        analyzer.compare_algorithms()
        analyzer.generate_report(output_dir)

        # Print summary to console
        for scenario in analyzer.get_scenarios():
            analyzer.print_comparison(scenario)
    else:
        print("No results found to analyze.")

    return analyzer
