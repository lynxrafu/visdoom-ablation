#!/usr/bin/env python3
"""
Main training script for ViZDoom deep RL experiments.

Uses Hydra for configuration management. Run with:
    python experiments/train.py
    python experiments/train.py agent.type=ddqn env.scenario=VizdoomTakeCover-v0
    python experiments/train.py --multirun agent.learning_rate=0.0001,0.001,0.01

For Colab, ensure Xvfb is running for headless mode.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

from src.envs import make_vizdoom_env
from src.utils.factory import build_agent, build_buffer
from src.utils.logging import WandbLogger, SafeCSVLogger, setup_logger, get_logger
from src.utils.plotting import plot_learning_curve
from src.utils.config_schema import validate_config, ConfigValidationError, print_config_summary


def set_seed(seed: int, env=None) -> Dict[str, Any]:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Master seed value
        env: Optional gymnasium environment for action/observation space seeding

    Returns:
        Dictionary of all seed values set (for metadata storage)
    """
    import random

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True

    # Environment seeds (if provided)
    env_action_seeded = False
    env_obs_seeded = False
    if env is not None:
        try:
            env.action_space.seed(seed)
            env_action_seeded = True
        except Exception:
            pass  # Some envs don't support seeding

        try:
            if hasattr(env, 'observation_space'):
                env.observation_space.seed(seed)
                env_obs_seeded = True
        except Exception:
            pass

    # For deterministic behavior (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    return {
        'master_seed': seed,
        'python_random': seed,
        'numpy': seed,
        'torch': seed,
        'torch_cuda': seed if cuda_seeded else None,
        'env_action_space': seed if env_action_seeded else None,
        'env_observation_space': seed if env_obs_seeded else None,
    }


def train_episode(
    env,
    agent,
    buffer,
    config: DictConfig,
    training: bool = True
) -> tuple:
    """
    Run a single training episode.

    Args:
        env: Gymnasium environment
        agent: RL agent
        buffer: Replay buffer
        config: Hydra configuration
        training: Whether to train (update) or just evaluate

    Returns:
        Tuple of (episode_reward, episode_length, metrics_dict)
    """
    state, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    losses = []
    mean_q_values = []
    grad_norms = []
    td_error_means = []

    # Track action distribution for exploration analysis
    action_counts = np.zeros(agent.num_actions, dtype=np.int32)

    for step in range(config.training.max_steps_per_episode):
        # Select action
        action = agent.select_action(state, training=training)
        action_counts[action] += 1

        # Environment step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if training:
            # For SARSA, get next action before storing
            # (needed for on-policy updates)
            next_action = None
            if hasattr(agent, 'is_on_policy') and agent.is_on_policy:
                next_action = agent.select_action(next_state, training=True)

            # Store transition in buffer
            buffer.push(state, action, reward, next_state, done, next_action)

            # Update if buffer has enough samples
            if len(buffer) >= config.training.min_buffer_size:
                if step % config.training.update_freq == 0:
                    # Sample batch
                    batch = buffer.sample(config.training.batch_size)

                    # Convert numpy arrays to tensors
                    batch_tensors = {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in batch.items()
                    }

                    # Gradient update
                    metrics = agent.update(batch_tensors)
                    losses.append(metrics['loss'])
                    mean_q_values.append(metrics.get('mean_q', 0.0))
                    grad_norms.append(metrics.get('grad_norm', 0.0))
                    td_error_means.append(metrics.get('td_error_mean', 0.0))

                    # Update PER priorities if using prioritized replay
                    if hasattr(buffer, 'update_priorities') and 'tree_indices' in batch:
                        buffer.update_priorities(
                            batch['tree_indices'],
                            metrics['td_errors']
                        )

        episode_reward += reward
        episode_length += 1
        state = next_state

        if done:
            break

    # Compute action distribution statistics
    total_actions = action_counts.sum()
    action_probs = action_counts / total_actions if total_actions > 0 else action_counts
    action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

    episode_metrics = {
        'mean_loss': np.mean(losses) if losses else 0.0,
        'mean_q': np.mean(mean_q_values) if mean_q_values else 0.0,
        'max_q': np.max(mean_q_values) if mean_q_values else 0.0,
        'num_updates': len(losses),
        'grad_norm': np.mean(grad_norms) if grad_norms else 0.0,
        'td_error_mean': np.mean(td_error_means) if td_error_means else 0.0,
        'td_error_std': np.std(td_error_means) if td_error_means else 0.0,
        'td_error_max': np.max(td_error_means) if td_error_means else 0.0,
        # Action distribution metrics
        'action_counts': action_counts,
        'action_entropy': action_entropy,
        'action_most_common': int(np.argmax(action_counts)),
    }

    return episode_reward, episode_length, episode_metrics


def evaluate(
    env,
    agent,
    num_episodes: int
) -> Dict[str, float]:
    """
    Evaluate agent without exploration or training.

    Args:
        env: Gymnasium environment
        agent: RL agent
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary of evaluation metrics
    """
    rewards = []
    lengths = []

    # Set to eval mode
    agent.set_training_mode(False)

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)

    # Back to train mode
    agent.set_training_mode(True)

    return {
        # Namespaced metrics for WandB
        'eval/reward_mean': np.mean(rewards),
        'eval/reward_std': np.std(rewards),
        'eval/reward_min': np.min(rewards),
        'eval/reward_max': np.max(rewards),
        'eval/length_mean': np.mean(lengths)
    }


def create_run_directory(config: DictConfig) -> tuple:
    """
    Create a unique, timestamped run directory for Colab/Drive persistence.

    Structure: results/YYYY-MM-DD/HH-MM-SS_agent_scenario_params_seed/

    The format is designed for:
    - Easy sorting by date and time
    - Quick identification of agent, scenario, and key params
    - Unique timestamps prevent any overwriting

    Returns:
        Tuple of (run_dir, run_name, run_id)
        - run_dir: Path to the run directory
        - run_name: Descriptive name for display
        - run_id: Unique ID with full timestamp (for WandB)
    """
    from datetime import datetime
    import hashlib

    # Create timestamp with readable format
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")  # HH-MM-SS for readability

    # Shorten scenario name (remove Vizdoom prefix and -v0 suffix)
    scenario_short = config.env.scenario.replace("Vizdoom", "").replace("-v0", "").replace("-", "")

    # Build params string with key hyperparameters
    params_parts = [f"lr{config.agent.learning_rate}"]
    if config.agent.gamma != 0.99:  # Only include if non-default
        params_parts.append(f"g{config.agent.gamma}")
    if config.agent.n_step != 1:  # Only include if non-default
        params_parts.append(f"n{config.agent.n_step}")
    if config.buffer.prioritized:  # Only include if using PER
        params_parts.append("per")
    params_str = "_".join(params_parts)

    # Create folder name: HH-MM-SS_agent_scenario_params_seed
    folder_name = f"{time_str}_{config.agent.type}_{scenario_short}_{params_str}_seed{config.seed}"

    # Run name for display (without time)
    run_name = f"{config.agent.type}_{scenario_short}_{params_str}_seed{config.seed}"

    # Unique run ID with full timestamp (for WandB)
    run_id = f"{date_str}_{time_str}_{run_name}"

    # Generate config hash for reproducibility verification
    config_yaml = OmegaConf.to_yaml(config)
    config_hash = hashlib.md5(config_yaml.encode()).hexdigest()[:8]

    # Full run directory path
    run_dir = Path("results") / date_str / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    # Save full config for reproducibility
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_yaml)

    # Save comprehensive run metadata
    import json
    metadata = {
        # Identification
        'run_id': run_id,
        'run_name': run_name,
        'config_hash': config_hash,
        'start_time': now.isoformat(),
        'date': date_str,
        'time': time_str,

        # Environment
        'scenario': config.env.scenario,
        'scenario_short': scenario_short,
        'frame_skip': config.env.frame_skip,
        'frame_stack': config.env.frame_stack,

        # Agent
        'agent_type': config.agent.type,
        'learning_rate': config.agent.learning_rate,
        'gamma': config.agent.gamma,
        'epsilon_start': config.agent.epsilon_start,
        'epsilon_end': config.agent.epsilon_end,
        'epsilon_decay': config.agent.epsilon_decay,
        'target_update_freq': config.agent.target_update_freq,
        'n_step': config.agent.n_step,

        # Buffer
        'buffer_capacity': config.buffer.capacity,
        'buffer_prioritized': config.buffer.prioritized,

        # Training
        'seed': config.seed,
        'num_episodes': config.training.num_episodes,
        'batch_size': config.training.batch_size,
        'update_freq': config.training.update_freq,
    }
    with open(run_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return run_dir, run_name, run_id


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig) -> float:
    """
    Main training function.

    Args:
        config: Hydra configuration

    Returns:
        Final average reward (for hyperparameter optimization)
    """
    # Validate configuration early (catch errors before training)
    try:
        validate_config(config)
    except ConfigValidationError as e:
        print(f"ERROR: {e}")
        return float('-inf')

    # Create unique run directory (timestamped, never overwrites)
    run_dir, run_name, run_id = create_run_directory(config)

    # Setup Python logger (file + console)
    log_level = config.logging.get('level', 'INFO')
    logger = setup_logger(
        name="vizdoom.train",
        level=log_level,
        log_file=run_dir / "training.log",
        console=True
    )

    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 60)

    # Print configuration summary
    print_config_summary(config)
    logger.debug("Full configuration:\n" + OmegaConf.to_yaml(config))

    # Set initial seed (before env creation for NumPy/Torch)
    set_seed(config.seed)
    logger.info(f"Initial random seed set to: {config.seed}")

    # Determine device
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    logger.info(f"Using device: {device}")

    # Create environment
    env = make_vizdoom_env(
        scenario=config.env.scenario,
        frame_skip=config.env.frame_skip,
        frame_stack=config.env.frame_stack,
        image_size=config.env.image_size,
        clip_rewards=config.env.clip_rewards
    )

    # Set seed again with environment (for action/observation space seeding)
    seed_info = set_seed(config.seed, env=env)
    logger.debug(f"Seed info: {seed_info}")

    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    logger.info(f"Environment: {config.env.scenario}")
    logger.info(f"State shape: {state_shape}, Actions: {num_actions}")

    # Build agent and buffer
    agent = build_agent(config, state_shape, num_actions)
    buffer = build_buffer(config, state_shape)
    logger.info(f"Agent: {agent}")
    logger.info(f"Buffer capacity: {config.buffer.capacity}")

    # Setup logging - save to run directory
    # Use run_id for WandB to ensure unique names across runs
    wandb_logger = WandbLogger(
        project=config.logging.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
        run_name=run_id,  # Unique timestamped name
        enabled=config.logging.wandb_enabled
    )

    # Safe CSV logger with auto-flush for Colab safety
    flush_every = config.logging.get('flush_every', 10)
    csv_logger = None
    if config.logging.csv_log:
        csv_logger = SafeCSVLogger(
            filepath=run_dir / "training_log.csv",
            fieldnames=[
                'episode', 'reward', 'length', 'loss',
                'epsilon', 'buffer_size', 'time_hours'
            ],
            flush_every=flush_every
        )
        logger.info(f"CSV logging enabled (flush every {flush_every} rows)")

    # Training loop
    episode_rewards = []
    start_time = time.time()
    best_eval_reward = float('-inf')

    logger.info("Starting training...")
    logger.info("-" * 60)

    for episode in range(config.training.num_episodes):
        # Train one episode
        reward, length, metrics = train_episode(
            env, agent, buffer, config, training=True
        )
        episode_rewards.append(reward)

        # Decay exploration
        agent.decay_epsilon()

        # Logging
        if episode % config.logging.log_freq == 0:
            elapsed_time = time.time() - start_time
            avg_reward_100 = np.mean(episode_rewards[-100:])

            # Namespaced WandB metrics for organized dashboard
            wandb_log_data = {
                # Training metrics
                'train/reward': reward,
                'train/reward_avg_100': avg_reward_100,
                'train/length': length,
                'train/loss': metrics['mean_loss'],
                'train/mean_q': metrics['mean_q'],
                'train/max_q': metrics.get('max_q', 0.0),
                # Agent state
                'agent/epsilon': agent.epsilon,
                'agent/grad_norm': metrics.get('grad_norm', 0.0),
                # Buffer state
                'buffer/size': len(buffer),
                # TD error statistics
                'td/error_mean': metrics.get('td_error_mean', 0.0),
                'td/error_std': metrics.get('td_error_std', 0.0),
                'td/error_max': metrics.get('td_error_max', 0.0),
                # Action distribution (for exploration analysis)
                'actions/entropy': metrics.get('action_entropy', 0.0),
                'actions/most_common': metrics.get('action_most_common', 0),
                # Time tracking
                'time/hours': elapsed_time / 3600,
            }

            # Log individual action counts if available
            action_counts = metrics.get('action_counts')
            if action_counts is not None:
                for i, count in enumerate(action_counts):
                    wandb_log_data[f'actions/action_{i}_count'] = int(count)

            wandb_logger.log(wandb_log_data, step=episode)

            # CSV uses flat structure for simplicity
            if csv_logger:
                csv_logger.log({
                    'episode': episode,
                    'reward': reward,
                    'length': length,
                    'loss': metrics['mean_loss'],
                    'epsilon': agent.epsilon,
                    'buffer_size': len(buffer),
                    'time_hours': elapsed_time / 3600
                })

            logger.info(
                f"Episode {episode:5d} | "
                f"Reward: {reward:7.2f} | "
                f"Avg100: {avg_reward_100:7.2f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Buffer: {len(buffer):6d}"
            )

        # Evaluation
        if episode % config.training.eval_freq == 0 and episode > 0:
            eval_metrics = evaluate(env, agent, config.training.eval_episodes)
            wandb_logger.log(eval_metrics, step=episode)

            logger.info(
                f"  [EVAL] Mean: {eval_metrics['eval/reward_mean']:.2f} +/- "
                f"{eval_metrics['eval/reward_std']:.2f}"
            )

            # Save best model
            if eval_metrics['eval/reward_mean'] > best_eval_reward:
                best_eval_reward = eval_metrics['eval/reward_mean']
                agent.save(str(run_dir / "checkpoints" / "best.pt"))
                logger.info("  [BEST] New best model saved!")
                # Force flush CSV after checkpoint save (Colab safety)
                if csv_logger:
                    csv_logger.flush()

        # Periodic checkpoint
        if episode % config.training.save_freq == 0 and episode > 0:
            agent.save(str(run_dir / "checkpoints" / f"ep{episode}.pt"))
            logger.info(f"  Checkpoint saved: ep{episode}.pt")
            # Force flush CSV after checkpoint save (Colab safety)
            if csv_logger:
                csv_logger.flush()

    # Final save
    agent.save(str(run_dir / "checkpoints" / "final.pt"))

    # Training complete
    total_time = time.time() - start_time
    final_avg_reward = np.mean(episode_rewards[-100:])

    logger.info("-" * 60)
    logger.info("Training complete!")
    logger.info(f"Total time: {total_time / 3600:.2f} hours")
    logger.info(f"Final avg reward (last 100): {final_avg_reward:.2f}")
    logger.info(f"Best eval reward: {best_eval_reward:.2f}")
    logger.info(f"Results saved to: {run_dir}")

    # Plot learning curve
    plot_learning_curve(
        episode_rewards,
        window=100,
        title=f"{config.agent.type} on {config.env.scenario}",
        save_path=str(run_dir / "plots" / "learning_curve.png")
    )

    # Save final summary with seed info for reproducibility verification
    import json
    summary = {
        'run_name': run_name,
        'run_id': run_id,
        'run_dir': str(run_dir),
        'total_time_hours': total_time / 3600,
        'num_episodes': config.training.num_episodes,
        'final_avg_reward_100': final_avg_reward,
        'best_eval_reward': best_eval_reward,
        'final_epsilon': agent.epsilon,
        'final_buffer_size': len(buffer),
        'seed_info': seed_info,
    }
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Cleanup - close all loggers and environment
    if csv_logger:
        csv_logger.close()
        logger.info("CSV logger closed.")
    wandb_logger.finish()
    env.close()

    logger.info(f"All outputs saved to: {run_dir}")
    return final_avg_reward


if __name__ == "__main__":
    main()
