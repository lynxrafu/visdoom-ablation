"""
Factory functions for building agents and buffers from configuration.

This module provides:
- build_agent: Create agent instances from Hydra config
- build_buffer: Create replay buffer instances from Hydra config

These factories enable the config-driven pipeline where experiments
are defined entirely through YAML configuration files.
"""

from typing import Any, Tuple, Union
from omegaconf import DictConfig

from ..agents.dqn import DQNAgent
from ..agents.deep_sarsa import DeepSARSAAgent
from ..agents.extensions import DDQNAgent, DuelingDQNAgent
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


# Registry of available agent types
AGENT_REGISTRY = {
    'dqn': DQNAgent,
    'deep_sarsa': DeepSARSAAgent,
    'ddqn': DDQNAgent,
    'dueling': DuelingDQNAgent,
    'dueling_ddqn': DuelingDQNAgent,  # Alias with double_dqn=True
}


def build_agent(
    config: DictConfig,
    state_shape: Tuple[int, ...],
    num_actions: int
) -> Any:
    """
    Build an agent from Hydra configuration.

    Supports the following agent types:
    - 'dqn': Vanilla Deep Q-Network
    - 'deep_sarsa': On-policy Deep SARSA
    - 'ddqn': Double DQN
    - 'dueling': Dueling DQN
    - 'dueling_ddqn': Dueling + Double DQN

    Args:
        config: Hydra DictConfig with agent settings
        state_shape: Environment observation shape (C, H, W)
        num_actions: Number of discrete actions

    Returns:
        Initialized agent instance

    Raises:
        ValueError: If agent type is not recognized

    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({
        ...     'agent': {'type': 'dqn', 'learning_rate': 0.0001, ...},
        ...     'device': 'auto'
        ... })
        >>> agent = build_agent(config, (4, 84, 84), 3)
    """
    agent_type = config.agent.type.lower()

    if agent_type not in AGENT_REGISTRY:
        available = list(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent type: '{agent_type}'. "
            f"Available: {available}"
        )

    # Common parameters for all agents
    common_params = {
        'state_shape': state_shape,
        'num_actions': num_actions,
        'learning_rate': config.agent.learning_rate,
        'gamma': config.agent.gamma,
        'epsilon_start': config.agent.epsilon_start,
        'epsilon_end': config.agent.epsilon_end,
        'epsilon_decay': config.agent.epsilon_decay,
        'target_update_freq': config.agent.target_update_freq,
        'n_step': config.agent.n_step,
        'device': config.device,
    }

    # Build agent based on type
    if agent_type == 'dqn':
        return DQNAgent(**common_params)

    elif agent_type == 'deep_sarsa':
        return DeepSARSAAgent(**common_params)

    elif agent_type == 'ddqn':
        return DDQNAgent(**common_params)

    elif agent_type == 'dueling':
        # Check if also using DDQN
        double_dqn = config.agent.get('double_dqn', False)
        return DuelingDQNAgent(**common_params, double_dqn=double_dqn)

    elif agent_type == 'dueling_ddqn':
        # Dueling with Double DQN
        return DuelingDQNAgent(**common_params, double_dqn=True)

    else:
        # Fallback (shouldn't reach here due to registry check)
        raise ValueError(f"Unhandled agent type: {agent_type}")


def build_buffer(
    config: DictConfig,
    state_shape: Tuple[int, ...]
) -> Union[ReplayBuffer, PrioritizedReplayBuffer]:
    """
    Build a replay buffer from Hydra configuration.

    Creates either:
    - ReplayBuffer: Standard uniform sampling
    - PrioritizedReplayBuffer: Priority-based sampling (PER)

    The buffer type is determined by config.buffer.prioritized.

    Args:
        config: Hydra DictConfig with buffer settings
        state_shape: State observation shape for pre-allocation

    Returns:
        Initialized replay buffer instance

    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({
        ...     'agent': {'n_step': 1, 'gamma': 0.99},
        ...     'buffer': {'capacity': 100000, 'prioritized': False}
        ... })
        >>> buffer = build_buffer(config, (4, 84, 84))
    """
    # Common parameters
    capacity = config.buffer.capacity
    n_step = config.agent.n_step
    gamma = config.agent.gamma

    if config.buffer.prioritized:
        # Prioritized Experience Replay
        return PrioritizedReplayBuffer(
            capacity=capacity,
            state_shape=state_shape,
            alpha=config.buffer.per_alpha,
            beta_start=config.buffer.per_beta_start,
            beta_frames=config.buffer.per_beta_frames,
            n_step=n_step,
            gamma=gamma
        )
    else:
        # Standard uniform replay
        return ReplayBuffer(
            capacity=capacity,
            state_shape=state_shape,
            n_step=n_step,
            gamma=gamma
        )


def get_agent_info(agent_type: str) -> dict:
    """
    Get information about an agent type.

    Args:
        agent_type: Agent type string

    Returns:
        Dictionary with agent metadata
    """
    info = {
        'dqn': {
            'name': 'Deep Q-Network',
            'reference': 'Mnih et al., 2015',
            'description': 'Off-policy TD with CNN and target network',
            'extensions': []
        },
        'deep_sarsa': {
            'name': 'Deep SARSA',
            'reference': 'Based on SARSA (Rummery & Niranjan, 1994)',
            'description': 'On-policy TD using actual next action',
            'extensions': []
        },
        'ddqn': {
            'name': 'Double DQN',
            'reference': 'Van Hasselt et al., 2016',
            'description': 'Decouples action selection from evaluation',
            'extensions': ['reduces overestimation bias']
        },
        'dueling': {
            'name': 'Dueling DQN',
            'reference': 'Wang et al., 2016',
            'description': 'Separate value and advantage streams',
            'extensions': ['better state value estimation']
        },
        'dueling_ddqn': {
            'name': 'Dueling Double DQN',
            'reference': 'Wang et al., 2016 + Van Hasselt et al., 2016',
            'description': 'Combines Dueling and Double DQN',
            'extensions': ['dueling architecture', 'double Q-learning']
        }
    }

    return info.get(agent_type.lower(), {'name': 'Unknown', 'description': 'N/A'})


def list_agents() -> list:
    """List all available agent types."""
    return list(AGENT_REGISTRY.keys())
