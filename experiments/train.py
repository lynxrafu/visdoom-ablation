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
from src.utils.logging import WandbLogger, CSVLogger
from src.utils.plotting import plot_learning_curve


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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

    for step in range(config.training.max_steps_per_episode):
        # Select action
        action = agent.select_action(state, training=training)

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

    episode_metrics = {
        'mean_loss': np.mean(losses) if losses else 0.0,
        'mean_q': np.mean(mean_q_values) if mean_q_values else 0.0,
        'num_updates': len(losses)
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
        'eval_reward_mean': np.mean(rewards),
        'eval_reward_std': np.std(rewards),
        'eval_reward_min': np.min(rewards),
        'eval_reward_max': np.max(rewards),
        'eval_length_mean': np.mean(lengths)
    }


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig) -> float:
    """
    Main training function.

    Args:
        config: Hydra configuration

    Returns:
        Final average reward (for hyperparameter optimization)
    """
    # Print configuration
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))
    print("=" * 60)

    # Set seed for reproducibility
    set_seed(config.seed)

    # Determine device
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    print(f"Using device: {device}")

    # Create environment
    env = make_vizdoom_env(
        scenario=config.env.scenario,
        frame_skip=config.env.frame_skip,
        frame_stack=config.env.frame_stack,
        image_size=config.env.image_size,
        clip_rewards=config.env.clip_rewards
    )

    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print(f"Environment: {config.env.scenario}")
    print(f"State shape: {state_shape}, Actions: {num_actions}")

    # Build agent and buffer
    agent = build_agent(config, state_shape, num_actions)
    buffer = build_buffer(config, state_shape)
    print(f"Agent: {agent}")
    print(f"Buffer capacity: {config.buffer.capacity}")

    # Setup logging
    run_name = (
        f"{config.agent.type}_{config.env.scenario}_"
        f"lr{config.agent.learning_rate}_seed{config.seed}"
    )

    wandb_logger = WandbLogger(
        project=config.logging.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
        run_name=run_name,
        enabled=config.logging.wandb_enabled
    )

    csv_logger = None
    if config.logging.csv_log:
        os.makedirs("results", exist_ok=True)
        csv_logger = CSVLogger(
            filepath=f"results/{run_name}.csv",
            fieldnames=[
                'episode', 'reward', 'length', 'loss',
                'epsilon', 'buffer_size', 'time_hours'
            ]
        )

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    episode_rewards = []
    start_time = time.time()
    best_eval_reward = float('-inf')

    print("\nStarting training...")
    print("-" * 60)

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

            log_data = {
                'episode': episode,
                'reward': reward,
                'reward_avg_100': avg_reward_100,
                'length': length,
                'loss': metrics['mean_loss'],
                'mean_q': metrics['mean_q'],
                'epsilon': agent.epsilon,
                'buffer_size': len(buffer),
                'time_hours': elapsed_time / 3600
            }

            wandb_logger.log(log_data, step=episode)

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

            print(
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

            print(
                f"  [EVAL] Mean: {eval_metrics['eval_reward_mean']:.2f} +/- "
                f"{eval_metrics['eval_reward_std']:.2f}"
            )

            # Save best model
            if eval_metrics['eval_reward_mean'] > best_eval_reward:
                best_eval_reward = eval_metrics['eval_reward_mean']
                agent.save(f"checkpoints/{run_name}_best.pt")
                print(f"  [BEST] New best model saved!")

        # Periodic checkpoint
        if episode % config.training.save_freq == 0 and episode > 0:
            agent.save(f"checkpoints/{run_name}_ep{episode}.pt")

    # Final save
    agent.save(f"checkpoints/{run_name}_final.pt")

    # Training complete
    total_time = time.time() - start_time
    final_avg_reward = np.mean(episode_rewards[-100:])

    print("-" * 60)
    print("Training complete!")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Final avg reward (last 100): {final_avg_reward:.2f}")
    print(f"Best eval reward: {best_eval_reward:.2f}")

    # Plot learning curve
    os.makedirs("results", exist_ok=True)
    plot_learning_curve(
        episode_rewards,
        window=100,
        title=f"{config.agent.type} on {config.env.scenario}",
        save_path=f"results/{run_name}_curve.png"
    )

    # Cleanup
    wandb_logger.finish()
    env.close()

    return final_avg_reward


if __name__ == "__main__":
    main()
