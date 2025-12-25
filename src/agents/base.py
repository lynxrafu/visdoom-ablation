"""
Base classes for deep reinforcement learning agents.

This module provides:
- QNetwork: CNN-based Q-value network (Mnih et al. 2015 architecture)
- BaseAgent: Abstract base class defining the agent interface
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """
    CNN-based Q-Network for visual reinforcement learning.

    Architecture follows Mnih et al. (2015) "Human-level control through
    deep reinforcement learning":
    - Conv1: 32 filters, 8x8 kernel, stride 4
    - Conv2: 64 filters, 4x4 kernel, stride 2
    - Conv3: 64 filters, 3x3 kernel, stride 1
    - FC: 512 hidden units
    - Output: num_actions Q-values

    Args:
        input_channels: Number of input channels (4 for frame stacking)
        num_actions: Number of discrete actions

    Example:
        >>> net = QNetwork(input_channels=4, num_actions=3)
        >>> state = torch.randn(1, 4, 84, 84)
        >>> q_values = net(state)  # Shape: (1, 3)
    """

    def __init__(self, input_channels: int, num_actions: int) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened size after conv layers
        conv_out_size = self._compute_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

        # Initialize weights
        self._initialize_weights()

    def _compute_conv_output_size(self) -> int:
        """Calculate the flattened size after convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, 84, 84)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)

    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, 84, 84)
               Values should be normalized to [0, 1]

        Returns:
            Q-values tensor of shape (batch, num_actions)
        """
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values

    def get_action(self, state: torch.Tensor) -> int:
        """
        Get greedy action for a single state.

        Args:
            state: State tensor of shape (channels, 84, 84)

        Returns:
            Action index with highest Q-value
        """
        with torch.no_grad():
            state = state.unsqueeze(0)  # Add batch dimension
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class BaseAgent(ABC):
    """
    Abstract base class for deep RL agents.

    Defines the interface that all agents must implement:
    - select_action: Choose action given state
    - update: Perform gradient update from batch
    - save/load: Model persistence

    Also provides common functionality:
    - Epsilon decay for exploration
    - Target network synchronization
    - Device management

    Args:
        state_shape: Shape of observations (C, H, W)
        num_actions: Number of discrete actions
        learning_rate: Optimizer learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Multiplicative decay per episode
        device: 'cuda', 'cpu', or 'auto'
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = "auto"
    ) -> None:
        # Store configuration
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Current exploration rate
        self.epsilon = epsilon_start

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks will be initialized by subclasses
        self.online_network: Optional[nn.Module] = None
        self.target_network: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Training statistics
        self.update_count = 0
        self.episode_count = 0

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action index
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one gradient update step.

        Args:
            batch: Dictionary containing:
                - 'states': (batch, C, H, W)
                - 'actions': (batch,)
                - 'rewards': (batch,)
                - 'next_states': (batch, C, H, W)
                - 'dones': (batch,)
                - 'weights': (batch,) for PER

        Returns:
            Dictionary of metrics (loss, mean_q, etc.)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint to path."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint from path."""
        pass

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        self.episode_count += 1

    def sync_target_network(self) -> None:
        """
        Hard update: Copy online network weights to target network.

        This is called periodically during training to update
        the target network used for computing TD targets.
        """
        if self.target_network is not None and self.online_network is not None:
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

    def soft_update_target(self, tau: float = 0.005) -> None:
        """
        Soft update: Blend online weights into target network.

        target = tau * online + (1 - tau) * target

        Args:
            tau: Interpolation parameter (0 < tau << 1)
        """
        if self.target_network is not None and self.online_network is not None:
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.online_network.parameters()
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration as dictionary."""
        return {
            "state_shape": self.state_shape,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "device": str(self.device),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_actions={self.num_actions}, "
            f"lr={self.learning_rate}, "
            f"gamma={self.gamma}, "
            f"epsilon={self.epsilon:.3f})"
        )
