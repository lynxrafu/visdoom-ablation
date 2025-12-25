#!/usr/bin/env python3
"""
Quick test script to verify ViZDoom installation and project setup.

Runs 10 training episodes on the Basic scenario to verify:
- ViZDoom environment works
- Agent training loop works
- Logging works

Usage:
    python experiments/test_setup.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        import numpy as np
        print(f"  numpy: {np.__version__}")
    except ImportError as e:
        print(f"  numpy: FAILED - {e}")
        return False

    try:
        import torch
        print(f"  torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  torch: FAILED - {e}")
        return False

    try:
        import gymnasium
        print(f"  gymnasium: {gymnasium.__version__}")
    except ImportError as e:
        print(f"  gymnasium: FAILED - {e}")
        return False

    try:
        import vizdoom
        print(f"  vizdoom: {vizdoom.__version__}")
    except ImportError as e:
        print(f"  vizdoom: FAILED - {e}")
        return False

    try:
        import hydra
        print(f"  hydra: {hydra.__version__}")
    except ImportError as e:
        print(f"  hydra: FAILED - {e}")
        return False

    try:
        import wandb
        print(f"  wandb: {wandb.__version__}")
    except ImportError as e:
        print(f"  wandb: FAILED (optional) - {e}")

    print("All core imports successful!\n")
    return True


def test_environment():
    """Test ViZDoom environment creation."""
    print("Testing environment...")

    try:
        from src.envs import make_vizdoom_env

        env = make_vizdoom_env(
            scenario="VizdoomBasic-v0",
            frame_skip=4,
            frame_stack=4,
            image_size=84
        )

        print(f"  Scenario: VizdoomBasic-v0")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Test reset
        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape}")

        # Test step
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Step result: reward={reward}, terminated={term}")

        env.close()
        print("Environment test passed!\n")
        return True

    except Exception as e:
        print(f"  Environment test FAILED: {e}\n")
        return False


def test_agent():
    """Test agent creation and update."""
    print("Testing agent...")

    try:
        import torch
        import numpy as np
        from src.agents import DQNAgent
        from src.utils.replay_buffer import ReplayBuffer

        state_shape = (4, 84, 84)
        num_actions = 3

        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            learning_rate=0.001,
            device="cpu"  # Use CPU for quick test
        )

        print(f"  Agent: {agent}")
        print(f"  Device: {agent.device}")

        # Test action selection
        dummy_state = np.random.randn(*state_shape).astype(np.float32)
        action = agent.select_action(dummy_state, training=True)
        print(f"  Action selected: {action}")

        # Test replay buffer
        buffer = ReplayBuffer(capacity=1000, state_shape=state_shape)
        for i in range(100):
            s = np.random.randn(*state_shape).astype(np.float32)
            a = np.random.randint(num_actions)
            r = np.random.randn()
            s_next = np.random.randn(*state_shape).astype(np.float32)
            done = i % 20 == 19
            buffer.push(s, a, r, s_next, done)

        print(f"  Buffer size: {len(buffer)}")

        # Test update
        batch = buffer.sample(32)
        batch_tensors = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in batch.items()
        }
        metrics = agent.update(batch_tensors)
        print(f"  Update metrics: loss={metrics['loss']:.4f}")

        print("Agent test passed!\n")
        return True

    except Exception as e:
        print(f"  Agent test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop(num_episodes: int = 10):
    """Test a short training loop."""
    print(f"Testing training loop ({num_episodes} episodes)...")

    try:
        import torch
        import numpy as np
        from src.envs import make_vizdoom_env
        from src.agents import DQNAgent
        from src.utils.replay_buffer import ReplayBuffer

        # Create environment
        env = make_vizdoom_env(
            scenario="VizdoomBasic-v0",
            frame_skip=4,
            frame_stack=4
        )

        state_shape = env.observation_space.shape
        num_actions = env.action_space.n

        # Create agent
        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            learning_rate=0.0001,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create buffer
        buffer = ReplayBuffer(capacity=10000, state_shape=state_shape)

        print(f"  State shape: {state_shape}")
        print(f"  Num actions: {num_actions}")
        print(f"  Device: {agent.device}")

        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0

            for step in range(300):  # Max 300 steps
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buffer.push(state, action, reward, next_state, done)

                # Update if buffer is ready
                if len(buffer) >= 100 and step % 4 == 0:
                    batch = buffer.sample(32)
                    batch_tensors = {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in batch.items()
                    }
                    agent.update(batch_tensors)

                episode_reward += reward
                state = next_state

                if done:
                    break

            agent.decay_epsilon()
            episode_rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, eps={agent.epsilon:.3f}")

        env.close()

        avg_reward = np.mean(episode_rewards)
        print(f"\n  Average reward: {avg_reward:.2f}")
        print("Training loop test passed!\n")
        return True

    except Exception as e:
        print(f"  Training loop test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ViZDoom Deep RL Setup Test")
    print("=" * 60 + "\n")

    results = {
        "Imports": test_imports(),
        "Environment": test_environment(),
        "Agent": test_agent(),
        "Training Loop": test_training_loop(10)
    }

    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\nAll tests passed! Your setup is ready.")
        print("Run 'python experiments/train.py' to start training.")
        return 0
    else:
        print("\nSome tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
