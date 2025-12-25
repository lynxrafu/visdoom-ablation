"""
DQN extension implementations.

This module provides:
- DDQNAgent: Double DQN (Van Hasselt et al., 2016)
- DuelingQNetwork: Dueling network architecture (Wang et al., 2016)
- DuelingDQNAgent: DQN with Dueling architecture

These extensions address known issues with vanilla DQN:
- DDQN: Reduces overestimation bias by decoupling action selection/evaluation
- Dueling: Better state value estimation via separate V and A streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any

from .dqn import DQNAgent
from .base import QNetwork


class DDQNAgent(DQNAgent):
    """
    Double DQN agent (Van Hasselt et al., 2016).

    Double DQN addresses the overestimation bias in vanilla DQN by
    decoupling action selection from action evaluation:

    - Online network selects best action: a* = argmax_a Q_online(s', a)
    - Target network evaluates: Q_target(s', a*)

    This simple change significantly reduces overestimation while
    maintaining the same architecture and computational cost.

    Args:
        Same as DQNAgent

    Reference:
        Van Hasselt, H., Guez, A., & Silver, D. (2016).
        Deep reinforcement learning with double Q-learning.
        AAAI Conference on Artificial Intelligence.

    Example:
        >>> agent = DDQNAgent(
        ...     state_shape=(4, 84, 84),
        ...     num_actions=3
        ... )
    """

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Double DQN TD target.

        DDQN target:
            a* = argmax_a Q_online(s', a)
            y = r + gamma^n * Q_target(s', a*)

        Args:
            rewards: Batch of (n-step) rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Target Q-values
        """
        with torch.no_grad():
            # Online network selects best action
            online_next_q = self.online_network(next_states)
            best_actions = online_next_q.argmax(dim=1)

            # Target network evaluates
            target_next_q = self.target_network(next_states)
            next_q = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # N-step discount
            discount = self.gamma ** self.n_step

            # Double DQN target
            targets = rewards + discount * next_q * (1 - dones.float())

        return targets

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        config = super().get_config()
        config['agent_type'] = 'ddqn'
        return config


class DuelingQNetwork(nn.Module):
    """
    Dueling Network Architecture (Wang et al., 2016).

    Separates Q(s,a) into two streams:
    - V(s): State value (scalar) - how good is this state?
    - A(s,a): Advantage (per action) - how much better is action a?

    The Q-value is computed as:
        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    The subtraction of mean(A) ensures identifiability and improves
    stability of the learning process.

    Architecture:
        Shared conv backbone -> split into:
            Value stream:     FC(512) -> FC(1)
            Advantage stream: FC(512) -> FC(num_actions)
        Combined: Q = V + (A - mean(A))

    Args:
        input_channels: Number of input channels (4 for frame stacking)
        num_actions: Number of discrete actions

    Reference:
        Wang, Z., et al. (2016). Dueling network architectures for
        deep reinforcement learning. ICML.
    """

    def __init__(self, input_channels: int, num_actions: int) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions

        # Shared convolutional backbone (same as DQN)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate conv output size
        conv_out_size = self._compute_conv_output_size()

        # Value stream
        self.value_fc = nn.Linear(conv_out_size, 512)
        self.value_out = nn.Linear(512, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(conv_out_size, 512)
        self.advantage_out = nn.Linear(512, num_actions)

        # Initialize weights
        self._initialize_weights()

    def _compute_conv_output_size(self) -> int:
        """Calculate flattened conv output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, 84, 84)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.

        Args:
            x: Input tensor of shape (batch, channels, 84, 84)

        Returns:
            Q-values of shape (batch, num_actions)
        """
        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Value stream
        v = F.relu(self.value_fc(x))
        v = self.value_out(v)  # Shape: (batch, 1)

        # Advantage stream
        a = F.relu(self.advantage_fc(x))
        a = self.advantage_out(a)  # Shape: (batch, num_actions)

        # Combine: Q = V + (A - mean(A))
        # Subtracting mean ensures identifiability
        q = v + (a - a.mean(dim=1, keepdim=True))

        return q

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get state value V(s) only.

        Useful for analysis and debugging.

        Args:
            x: Input tensor

        Returns:
            State values of shape (batch, 1)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        v = F.relu(self.value_fc(x))
        v = self.value_out(v)

        return v


class DuelingDQNAgent(DQNAgent):
    """
    DQN with Dueling Network Architecture.

    Combines the benefits of DQN with the Dueling architecture.
    Can optionally use Double DQN target computation as well.

    Args:
        state_shape: Shape of observations (C, H, W)
        num_actions: Number of discrete actions
        double_dqn: Whether to use DDQN target computation
        ... other args same as DQNAgent

    Example:
        >>> # Dueling DQN only
        >>> agent = DuelingDQNAgent(
        ...     state_shape=(4, 84, 84),
        ...     num_actions=3,
        ...     double_dqn=False
        ... )
        >>> # Dueling + Double DQN (recommended)
        >>> agent = DuelingDQNAgent(
        ...     state_shape=(4, 84, 84),
        ...     num_actions=3,
        ...     double_dqn=True
        ... )
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        double_dqn: bool = False,
        **kwargs
    ) -> None:
        # Initialize parent (will create standard QNetwork)
        super().__init__(state_shape, num_actions, **kwargs)

        self.double_dqn = double_dqn

        # Replace networks with Dueling architecture
        input_channels = state_shape[0]
        self.online_network = DuelingQNetwork(input_channels, num_actions).to(self.device)
        self.target_network = DuelingQNetwork(input_channels, num_actions).to(self.device)

        # Sync target network
        self.sync_target_network()

        # Freeze target
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Recreate optimizer for new network
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.learning_rate
        )

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TD target (optionally using DDQN).

        If double_dqn=True, uses DDQN target.
        Otherwise, uses standard DQN target.

        Args:
            rewards: Batch of (n-step) rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Target Q-values
        """
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: online selects, target evaluates
                online_next_q = self.online_network(next_states)
                best_actions = online_next_q.argmax(dim=1)

                target_next_q = self.target_network(next_states)
                next_q = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: target selects max
                target_next_q = self.target_network(next_states)
                next_q = target_next_q.max(dim=1)[0]

            discount = self.gamma ** self.n_step
            targets = rewards + discount * next_q * (1 - dones.float())

        return targets

    def get_state_value(self, state: np.ndarray) -> float:
        """
        Get state value V(s) for a state.

        Useful for analyzing what the agent has learned.

        Args:
            state: State observation (C, H, W)

        Returns:
            State value as float
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.online_network.get_value(state_tensor)
            return value.item()

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        config = super().get_config()
        config['agent_type'] = 'dueling' if not self.double_dqn else 'dueling_ddqn'
        config['double_dqn'] = self.double_dqn
        config['architecture'] = 'dueling'
        return config
