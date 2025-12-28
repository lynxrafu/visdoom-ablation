"""
Deep SARSA agent implementation.

SARSA (State-Action-Reward-State-Action) is an on-policy TD learning
algorithm that uses the actual next action taken (rather than max)
for computing TD targets.

Key difference from DQN:
- DQN (off-policy):  Q(s,a) <- r + gamma * max_a' Q(s', a')
- SARSA (on-policy): Q(s,a) <- r + gamma * Q(s', a')  where a' is actual next action
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any

from .dqn import DQNAgent


class DeepSARSAAgent(DQNAgent):
    """
    Deep SARSA agent (on-policy TD learning).

    SARSA uses the actual next action taken by the policy for
    computing TD targets, making it on-policy. This means:
    - Must use the same policy for both action selection and update
    - Generally more conservative than Q-learning
    - Can be more stable in certain environments

    Note:
        Transitions must include 'next_actions' for proper SARSA updates.
        The replay buffer should store the actual next action taken.

    Args:
        Same as DQNAgent

    Example:
        >>> agent = DeepSARSAAgent(
        ...     state_shape=(4, 84, 84),
        ...     num_actions=3
        ... )
        >>> # During training, store next_action in buffer:
        >>> buffer.push(state, action, reward, next_state, done, next_action)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Flag to indicate this is an on-policy algorithm
        self.is_on_policy = True

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_actions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute SARSA TD target using actual next action.

        SARSA target:
            y = r + gamma^n * Q_target(s', a')

        where a' is the actual next action taken, not argmax.

        Args:
            rewards: Batch of (n-step) rewards
            next_states: Batch of next states
            dones: Batch of done flags
            next_actions: Batch of actual next actions taken

        Returns:
            Target Q-values
        """
        if next_actions is None:
            raise ValueError(
                "DeepSARSA requires next_actions for TD target computation. "
                "Make sure the replay buffer stores next_actions."
            )

        with torch.no_grad():
            # Get Q-values for next states
            next_q_values = self.target_network(next_states)

            # Select Q-value for actual next action (not max)
            next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # N-step discount
            discount = self.gamma ** self.n_step

            # SARSA target
            targets = rewards + discount * next_q * (1 - dones.float())

        return targets

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one SARSA gradient update step.

        Requires 'next_actions' key in batch for on-policy update.

        Args:
            batch: Dictionary containing:
                - 'states': (batch, C, H, W)
                - 'actions': (batch,)
                - 'rewards': (batch,)
                - 'next_states': (batch, C, H, W)
                - 'next_actions': (batch,) - REQUIRED for SARSA
                - 'dones': (batch,)
                - 'weights': (batch,) importance sampling weights

        Returns:
            Dictionary with metrics
        """
        # Move batch to device
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        next_actions = batch['next_actions'].to(self.device)
        dones = batch['dones'].to(self.device)
        weights = batch.get('weights', torch.ones_like(rewards)).to(self.device)

        # Current Q-values
        q_values = self.online_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # SARSA target with actual next action
        targets = self.compute_td_target(rewards, next_states, dones, next_actions)

        # TD errors
        td_errors = (current_q - targets).detach().abs()

        # Weighted loss
        elementwise_loss = F.mse_loss(current_q, targets, reduction='none')
        loss = (weights * elementwise_loss).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability and capture gradient norm
        grad_norm = 0.0
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.online_network.parameters(),
                self.grad_clip
            ).item()
        else:
            # Calculate grad norm without clipping
            total_norm = 0.0
            for p in self.online_network.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        self.optimizer.step()

        # Update target network
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.sync_target_network()

        # TD error statistics for scientific analysis
        td_errors_np = td_errors.cpu().numpy()

        # Return metrics (enhanced for scientific rigor)
        return {
            'loss': loss.item(),
            'td_errors': td_errors_np,
            'mean_q': q_values.mean().item(),
            'max_q': q_values.max().item(),
            'grad_norm': grad_norm,
            'td_error_mean': float(td_errors_np.mean()),
            'td_error_std': float(td_errors_np.std()),
            'td_error_max': float(td_errors_np.max()),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        config = super().get_config()
        config['agent_type'] = 'deep_sarsa'
        config['is_on_policy'] = True
        return config
