"""
Deep Q-Network (DQN) agent implementation.

Implements the DQN algorithm from Mnih et al. (2015):
"Human-level control through deep reinforcement learning"

Key features:
- CNN-based Q-network for visual input
- Target network for stable training
- Experience replay (handled externally)
- Epsilon-greedy exploration
- Support for n-step returns
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

from .base import BaseAgent, QNetwork


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent.

    The DQN agent learns a Q-function approximated by a CNN.
    It uses:
    - Experience replay for sample efficiency and decorrelation
    - Target network for stable Q-value estimation
    - Epsilon-greedy exploration that decays over training

    This class serves as the base for extensions (DDQN, Dueling).

    Args:
        state_shape: Shape of observations (C, H, W)
        num_actions: Number of discrete actions
        learning_rate: Adam optimizer learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Multiplicative decay per episode
        target_update_freq: Steps between target network updates
        n_step: N-step returns (1 = standard TD, >1 = more MC-like)
        device: 'cuda', 'cpu', or 'auto'

    Example:
        >>> agent = DQNAgent(
        ...     state_shape=(4, 84, 84),
        ...     num_actions=3,
        ...     learning_rate=0.0001
        ... )
        >>> action = agent.select_action(state)
        >>> metrics = agent.update(batch)
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
        target_update_freq: int = 1000,
        n_step: int = 1,
        device: str = "auto",
        grad_clip: float = 10.0
    ) -> None:
        super().__init__(
            state_shape=state_shape,
            num_actions=num_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            device=device
        )

        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        # Initialize networks
        input_channels = state_shape[0]
        self.online_network = QNetwork(input_channels, num_actions).to(self.device)
        self.target_network = QNetwork(input_channels, num_actions).to(self.device)

        # Copy weights to target network
        self.sync_target_network()

        # Freeze target network (no gradients)
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate
        )

        # Training statistics
        self.total_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        With probability epsilon, select random action.
        Otherwise, select action with highest Q-value.

        Args:
            state: Current state observation (C, H, W)
            training: If False, use pure greedy (epsilon=0)

        Returns:
            Selected action index
        """
        # Random action with probability epsilon
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TD target values.

        Standard DQN target:
            y = r + gamma^n * max_a Q_target(s', a)

        This method can be overridden by subclasses (e.g., DDQN).

        Args:
            rewards: Batch of (n-step) rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Target Q-values
        """
        with torch.no_grad():
            # Max Q-value from target network
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]

            # N-step discount
            discount = self.gamma ** self.n_step

            # TD target: r + gamma^n * max_a Q(s', a) * (1 - done)
            targets = rewards + discount * max_next_q * (1 - dones.float())

        return targets

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one DQN gradient update step.

        Args:
            batch: Dictionary containing:
                - 'states': (batch, C, H, W)
                - 'actions': (batch,)
                - 'rewards': (batch,)
                - 'next_states': (batch, C, H, W)
                - 'dones': (batch,)
                - 'weights': (batch,) importance sampling weights

        Returns:
            Dictionary with metrics:
                - 'loss': MSE loss value
                - 'td_errors': Absolute TD errors (for PER)
                - 'mean_q': Mean Q-value
                - 'max_q': Max Q-value
        """
        # Move batch to device
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)

        # Importance sampling weights (for PER, defaults to 1.0)
        weights = batch.get('weights', torch.ones_like(rewards)).to(self.device)

        # Current Q-values for taken actions
        q_values = self.online_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute TD targets
        targets = self.compute_td_target(rewards, next_states, dones)

        # TD errors (for PER priority updates)
        td_errors = (current_q - targets).detach().abs()

        # Weighted MSE loss (weights from importance sampling)
        elementwise_loss = F.mse_loss(current_q, targets, reduction='none')
        loss = (weights * elementwise_loss).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.online_network.parameters(),
                self.grad_clip
            )

        self.optimizer.step()

        # Update target network periodically
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.sync_target_network()

        # Return metrics
        return {
            'loss': loss.item(),
            'td_errors': td_errors.cpu().numpy(),
            'mean_q': q_values.mean().item(),
            'max_q': q_values.max().item()
        }

    def save(self, path: str) -> None:
        """
        Save model checkpoint.

        Saves:
        - Online network weights
        - Target network weights
        - Optimizer state
        - Training state (epsilon, steps)

        Args:
            path: File path for checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'config': self.get_config()
        }

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: File path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions given state.

        Useful for debugging and visualization.

        Args:
            state: State observation (C, H, W)

        Returns:
            Q-values array of shape (num_actions,)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            return q_values.squeeze(0).cpu().numpy()

    def set_training_mode(self, training: bool = True) -> None:
        """
        Set network training mode.

        Args:
            training: If True, set to training mode; else eval mode
        """
        if training:
            self.online_network.train()
            self.target_network.train()
        else:
            self.online_network.eval()
            self.target_network.eval()

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        config = super().get_config()
        config.update({
            'n_step': self.n_step,
            'target_update_freq': self.target_update_freq,
            'grad_clip': self.grad_clip,
            'agent_type': 'dqn'
        })
        return config
