#!/usr/bin/env python3
"""
Ablation study runner for ViZDoom deep RL experiments.

This script executes grid search experiments over parameter combinations.
Results are aggregated and exported for IEEE report generation.

Usage:
    python experiments/ablate.py --phase algorithms
    python experiments/ablate.py --phase lr
    python experiments/ablate.py --phase extensions
    python experiments/ablate.py --phase all

Phases:
    algorithms: Compare DQN vs Deep SARSA
    lr: Learning rate ablation
    gamma: Discount factor ablation
    nstep: N-step returns ablation
    extensions: DDQN, Dueling, PER ablation
    all: Run all ablations
"""

import os
import sys
import argparse
import subprocess
from itertools import product
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_experiment(
    args: List[str],
    dry_run: bool = False
) -> int:
    """
    Run a single training experiment.

    Args:
        args: Command line arguments for train.py
        dry_run: If True, just print command without running

    Returns:
        Return code from subprocess
    """
    cmd = ['python', 'experiments/train.py'] + args
    cmd_str = ' '.join(cmd)

    if dry_run:
        print(f"[DRY RUN] {cmd_str}")
        return 0

    print(f"\n{'=' * 60}")
    print(f"Running: {cmd_str}")
    print('=' * 60)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_algorithm_comparison(
    scenarios: List[str],
    seeds: List[int],
    num_episodes: int = 1000,
    dry_run: bool = False
) -> None:
    """
    Phase 1: Compare DQN vs Deep SARSA.

    Args:
        scenarios: List of ViZDoom scenarios
        seeds: Random seeds for multiple runs
        num_episodes: Training episodes per run
        dry_run: If True, just print commands
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Algorithm Comparison (DQN vs Deep SARSA)")
    print("=" * 60)

    algorithms = ['dqn', 'deep_sarsa']

    for algo, scenario, seed in product(algorithms, scenarios, seeds):
        args = [
            f'agent.type={algo}',
            f'env.scenario={scenario}',
            f'seed={seed}',
            f'training.num_episodes={num_episodes}',
            'logging.wandb_enabled=true'
        ]
        run_experiment(args, dry_run)


def run_lr_ablation(
    scenarios: List[str],
    seeds: List[int],
    num_episodes: int = 1000,
    dry_run: bool = False
) -> None:
    """
    Phase 2a: Learning rate ablation.
    """
    print("\n" + "=" * 60)
    print("PHASE 2a: Learning Rate Ablation")
    print("=" * 60)

    learning_rates = [0.0001, 0.001, 0.01]

    for lr, scenario, seed in product(learning_rates, scenarios, seeds):
        args = [
            f'agent.learning_rate={lr}',
            f'env.scenario={scenario}',
            f'seed={seed}',
            f'training.num_episodes={num_episodes}',
            'logging.wandb_enabled=true'
        ]
        run_experiment(args, dry_run)


def run_gamma_ablation(
    scenarios: List[str],
    seeds: List[int],
    num_episodes: int = 1000,
    dry_run: bool = False
) -> None:
    """
    Phase 2b: Discount factor ablation.
    """
    print("\n" + "=" * 60)
    print("PHASE 2b: Gamma (Discount Factor) Ablation")
    print("=" * 60)

    gammas = [0.9, 0.99]

    for gamma, scenario, seed in product(gammas, scenarios, seeds):
        args = [
            f'agent.gamma={gamma}',
            f'env.scenario={scenario}',
            f'seed={seed}',
            f'training.num_episodes={num_episodes}',
            'logging.wandb_enabled=true'
        ]
        run_experiment(args, dry_run)


def run_nstep_ablation(
    scenarios: List[str],
    seeds: List[int],
    num_episodes: int = 1000,
    dry_run: bool = False
) -> None:
    """
    Phase 2c: N-step returns ablation (TD vs MC-like).
    """
    print("\n" + "=" * 60)
    print("PHASE 2c: N-step Returns Ablation")
    print("=" * 60)

    n_steps = [1, 3]  # 1 = TD, 3 = more MC-like

    for n, scenario, seed in product(n_steps, scenarios, seeds):
        args = [
            f'agent.n_step={n}',
            f'env.scenario={scenario}',
            f'seed={seed}',
            f'training.num_episodes={num_episodes}',
            'logging.wandb_enabled=true'
        ]
        run_experiment(args, dry_run)


def run_extension_ablation(
    scenarios: List[str],
    seeds: List[int],
    num_episodes: int = 1000,
    dry_run: bool = False
) -> None:
    """
    Phase 3: Extension ablation (DDQN, Dueling, PER).
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Extension Ablation")
    print("=" * 60)

    # Configurations: (agent_type, prioritized, description)
    configs = [
        ('dqn', False, 'Baseline DQN'),
        ('ddqn', False, 'DQN + Double'),
        ('dueling', False, 'DQN + Dueling'),
        ('dqn', True, 'DQN + PER'),
        ('ddqn', True, 'DQN + Double + PER'),
        ('dueling', True, 'DQN + Dueling + PER'),
    ]

    for (algo, per, desc), scenario, seed in product(configs, scenarios, seeds):
        print(f"\nConfiguration: {desc}")
        args = [
            f'agent.type={algo}',
            f'buffer.prioritized={str(per).lower()}',
            f'env.scenario={scenario}',
            f'seed={seed}',
            f'training.num_episodes={num_episodes}',
            'logging.wandb_enabled=true'
        ]

        # For dueling, also enable double_dqn
        if algo == 'dueling':
            args.append('agent.double_dqn=true')

        run_experiment(args, dry_run)


def aggregate_results(output_dir: str = "results") -> None:
    """
    Aggregate results from all CSV files into summary.
    """
    print("\n" + "=" * 60)
    print("Aggregating Results")
    print("=" * 60)

    try:
        import pandas as pd
        import glob

        csv_files = glob.glob(f"{output_dir}/*.csv")
        if not csv_files:
            print("No CSV files found to aggregate.")
            return

        print(f"Found {len(csv_files)} CSV files")

        # Load and process each file
        all_data = []
        for f in csv_files:
            df = pd.read_csv(f)
            # Extract config from filename
            filename = Path(f).stem
            parts = filename.split('_')
            if len(parts) >= 2:
                df['algorithm'] = parts[0]
                df['scenario'] = parts[1] if len(parts) > 1 else 'unknown'
            all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)

        # Summary statistics
        summary = combined.groupby(['algorithm', 'scenario']).agg({
            'reward': ['mean', 'std', 'max'],
            'episode': 'max'
        }).round(2)

        print("\nSummary Statistics:")
        print(summary)

        # Save summary
        summary_path = f"{output_dir}/ablation_summary.csv"
        summary.to_csv(summary_path)
        print(f"\nSummary saved to: {summary_path}")

    except ImportError:
        print("Warning: pandas required for aggregation. Install with: pip install pandas")
    except Exception as e:
        print(f"Error during aggregation: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation studies for ViZDoom deep RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
    algorithms   - Compare DQN vs Deep SARSA
    lr           - Learning rate ablation (0.0001, 0.001, 0.01)
    gamma        - Discount factor ablation (0.9, 0.99)
    nstep        - N-step returns ablation (1, 3)
    extensions   - DDQN, Dueling, PER ablation
    all          - Run all ablation phases

Example:
    python experiments/ablate.py --phase lr --scenarios basic --episodes 500
    python experiments/ablate.py --phase extensions --dry-run
        """
    )

    parser.add_argument(
        '--phase',
        type=str,
        required=True,
        choices=['algorithms', 'lr', 'gamma', 'nstep', 'extensions', 'all'],
        help='Which ablation phase to run'
    )

    parser.add_argument(
        '--scenarios',
        type=str,
        nargs='+',
        default=['VizdoomBasic-v0', 'VizdoomTakeCover-v0'],
        help='ViZDoom scenarios to test'
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='Random seeds for multiple runs'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes per run'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )

    parser.add_argument(
        '--aggregate-only',
        action='store_true',
        help='Only aggregate existing results, skip training'
    )

    args = parser.parse_args()

    # Print configuration
    print("\n" + "=" * 60)
    print("ViZDoom Deep RL Ablation Study")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes: {args.episodes}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    if args.aggregate_only:
        aggregate_results()
        return

    # Run selected phase
    if args.phase == 'algorithms' or args.phase == 'all':
        run_algorithm_comparison(
            args.scenarios, args.seeds, args.episodes, args.dry_run
        )

    if args.phase == 'lr' or args.phase == 'all':
        run_lr_ablation(
            args.scenarios, args.seeds, args.episodes, args.dry_run
        )

    if args.phase == 'gamma' or args.phase == 'all':
        run_gamma_ablation(
            args.scenarios, args.seeds, args.episodes, args.dry_run
        )

    if args.phase == 'nstep' or args.phase == 'all':
        run_nstep_ablation(
            args.scenarios, args.seeds, args.episodes, args.dry_run
        )

    if args.phase == 'extensions' or args.phase == 'all':
        run_extension_ablation(
            args.scenarios, args.seeds, args.episodes, args.dry_run
        )

    # Aggregate results at the end
    if not args.dry_run:
        aggregate_results()

    print("\n" + "=" * 60)
    print("Ablation study complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
