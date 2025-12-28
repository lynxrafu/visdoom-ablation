"""
Experience replay buffer implementations.

This module provides:
- ReplayBuffer: Standard uniform sampling with n-step returns
- SumTree: Data structure for efficient priority sampling
- PrioritizedReplayBuffer: PER with importance sampling (Schaul et al. 2016)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """
    Standard experience replay buffer with uniform sampling.

    Stores transitions (s, a, r, s', done) and supports n-step returns.
    Uses pre-allocated NumPy arrays for memory efficiency.

    Args:
        capacity: Maximum number of transitions to store
        state_shape: Shape of state observations (C, H, W)
        n_step: Number of steps for n-step returns (1 = standard TD)
        gamma: Discount factor for n-step return computation

    Example:
        >>> buffer = ReplayBuffer(100000, (4, 84, 84), n_step=3, gamma=0.99)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(32)
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        n_step: int = 1,
        gamma: float = 0.99
    ) -> None:
        self.capacity = capacity
        self.state_shape = state_shape
        self.n_step = n_step
        self.gamma = gamma

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        # For Deep SARSA (stores actual next action taken)
        self.next_actions = np.zeros(capacity, dtype=np.int64)

        # Buffer state
        self.position = 0
        self.size = 0

        # N-step buffer for accumulating transitions
        self.n_step_buffer: deque = deque(maxlen=n_step)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: Optional[int] = None
    ) -> None:
        """
        Add a transition to the buffer.

        For n-step > 1, transitions are accumulated before storage
        to compute n-step returns.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
            next_action: Next action taken (for SARSA)
        """
        # Add to n-step buffer
        transition = (state, action, reward, next_state, done, next_action)
        self.n_step_buffer.append(transition)

        # Only store when n-step buffer is full or episode ends
        if len(self.n_step_buffer) == self.n_step or done:
            # Compute n-step return
            n_step_return = 0.0
            for i, (_, _, r, _, d, _) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    break

            # Get first state/action and last next_state
            first_state, first_action, _, _, _, first_next_action = self.n_step_buffer[0]
            _, _, _, last_next_state, last_done, last_next_action = self.n_step_buffer[-1]

            # Store in buffer
            idx = self.position
            self.states[idx] = first_state
            self.actions[idx] = first_action
            self.rewards[idx] = n_step_return
            self.next_states[idx] = last_next_state
            self.dones[idx] = last_done

            if first_next_action is not None:
                self.next_actions[idx] = first_next_action

            # Update position (circular buffer)
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            # Clear n-step buffer on episode end
            if done:
                self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing:
                - 'states': (batch, C, H, W)
                - 'actions': (batch,)
                - 'rewards': (batch,)
                - 'next_states': (batch, C, H, W)
                - 'dones': (batch,)
                - 'next_actions': (batch,) for SARSA
                - 'indices': (batch,) sampled indices
                - 'weights': (batch,) uniform weights (all 1.0)
        """
        indices = np.random.choice(self.size, batch_size, replace=False)

        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'next_actions': self.next_actions[indices],
            'indices': indices,
            'weights': np.ones(batch_size, dtype=np.float32)
        }

    def reset(self) -> None:
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        self.n_step_buffer.clear()

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def save_state(self, filepath: str) -> None:
        """
        Save buffer state to compressed file.

        Saves current buffer contents with gzip compression for
        efficient storage and exact reproducibility.

        Args:
            filepath: Path to save file (will add .gz if not present)
        """
        import gzip
        import pickle
        from pathlib import Path

        filepath = Path(filepath)
        if not filepath.suffix == '.gz':
            filepath = Path(str(filepath) + '.gz')

        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_states': self.next_states[:self.size],
            'dones': self.dones[:self.size],
            'next_actions': self.next_actions[:self.size],
            'position': self.position,
            'size': self.size,
            'capacity': self.capacity,
            'state_shape': self.state_shape,
            'n_step': self.n_step,
            'gamma': self.gamma,
        }

        with gzip.open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, filepath: str) -> None:
        """
        Load buffer state from compressed file.

        Args:
            filepath: Path to saved buffer state file
        """
        import gzip
        import pickle
        from pathlib import Path

        filepath = Path(filepath)
        if not filepath.exists() and not filepath.suffix == '.gz':
            filepath = Path(str(filepath) + '.gz')

        with gzip.open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Verify compatibility
        if state['state_shape'] != self.state_shape:
            raise ValueError(
                f"State shape mismatch: buffer has {self.state_shape}, "
                f"loaded file has {state['state_shape']}"
            )

        # Restore state
        loaded_size = state['size']
        self.states[:loaded_size] = state['states']
        self.actions[:loaded_size] = state['actions']
        self.rewards[:loaded_size] = state['rewards']
        self.next_states[:loaded_size] = state['next_states']
        self.dones[:loaded_size] = state['dones']
        self.next_actions[:loaded_size] = state['next_actions']
        self.position = state['position']
        self.size = loaded_size
        self.n_step_buffer.clear()


class SumTree:
    """
    Binary sum tree for efficient priority-based sampling.

    Each leaf stores a priority value. Internal nodes store the sum
    of their children. This allows O(log n) sampling and updates.

    The tree structure:
        - Leaves: indices [capacity-1, 2*capacity-2]
        - Internal nodes: indices [0, capacity-2]
        - Root: index 0 (total sum of priorities)

    Args:
        capacity: Number of leaf nodes (max transitions)
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Tree array: internal nodes + leaves
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def update(self, tree_idx: int, priority: float) -> None:
        """
        Update priority at tree_idx and propagate change up.

        Args:
            tree_idx: Index in tree array
            priority: New priority value
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change to root
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority: float) -> int:
        """
        Add new priority to the next available slot.

        Args:
            priority: Priority value for new transition

        Returns:
            Data index (0 to capacity-1)
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)

        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        return data_idx

    def get(self, value: float) -> Tuple[int, float, int]:
        """
        Sample leaf by cumulative sum value.

        Traverses tree to find leaf where cumulative sum
        reaches the given value.

        Args:
            value: Random value in [0, total_sum]

        Returns:
            Tuple of (tree_idx, priority, data_idx)
        """
        tree_idx = 0  # Start at root

        while True:
            left = 2 * tree_idx + 1
            right = left + 1

            # Reached leaf
            if left >= len(self.tree):
                break

            # Go left if value <= left subtree sum
            if value <= self.tree[left]:
                tree_idx = left
            else:
                value -= self.tree[left]
                tree_idx = right

        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx

    @property
    def total(self) -> float:
        """Total sum of all priorities (root value)."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Maximum priority among leaves."""
        leaf_start = self.capacity - 1
        return self.tree[leaf_start:].max()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (Schaul et al. 2016).

    Samples transitions proportional to their TD error.
    Uses importance sampling weights to correct the bias.

    Key parameters:
    - alpha: Priority exponent (0 = uniform, 1 = full prioritization)
    - beta: Importance sampling exponent (annealed from beta_start to 1)

    Args:
        capacity: Maximum buffer size
        state_shape: Shape of state observations
        alpha: Priority exponent
        beta_start: Initial importance sampling exponent
        beta_frames: Frames over which to anneal beta to 1.0
        n_step: N-step returns
        gamma: Discount factor

    Example:
        >>> buffer = PrioritizedReplayBuffer(100000, (4, 84, 84))
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(32)
        >>> # After update:
        >>> buffer.update_priorities(batch['tree_indices'], td_errors)
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        n_step: int = 1,
        gamma: float = 0.99
    ) -> None:
        self.capacity = capacity
        self.state_shape = state_shape
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.n_step = n_step
        self.gamma = gamma

        # Current frame for beta annealing
        self.frame = 1

        # Sum tree for priority sampling
        self.tree = SumTree(capacity)

        # Data storage
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_actions = np.zeros(capacity, dtype=np.int64)

        # Buffer state
        self.size = 0
        self.max_priority = 1.0

        # N-step buffer
        self.n_step_buffer: deque = deque(maxlen=n_step)

    @property
    def beta(self) -> float:
        """Current importance sampling exponent (annealed)."""
        return min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: Optional[int] = None
    ) -> None:
        """
        Add transition with maximum priority.

        New transitions are added with max priority to ensure
        they are sampled at least once.
        """
        # Add to n-step buffer
        transition = (state, action, reward, next_state, done, next_action)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) == self.n_step or done:
            # Compute n-step return
            n_step_return = 0.0
            for i, (_, _, r, _, d, _) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    break

            first_state, first_action, _, _, _, first_next_action = self.n_step_buffer[0]
            _, _, _, last_next_state, last_done, _ = self.n_step_buffer[-1]

            # Add to tree with max priority
            data_idx = self.tree.add(self.max_priority ** self.alpha)

            # Store data
            self.states[data_idx] = first_state
            self.actions[data_idx] = first_action
            self.rewards[data_idx] = n_step_return
            self.next_states[data_idx] = last_next_state
            self.dones[data_idx] = last_done

            if first_next_action is not None:
                self.next_actions[data_idx] = first_next_action

            self.size = min(self.size + 1, self.capacity)

            if done:
                self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample batch proportional to priorities.

        Uses stratified sampling: divide total priority into
        batch_size segments and sample one from each.

        Returns:
            Dictionary with transitions, indices, and IS weights
        """
        indices = []
        tree_indices = []
        priorities = []

        # Stratified sampling
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)

            tree_idx, priority, data_idx = self.tree.get(value)

            tree_indices.append(tree_idx)
            indices.append(data_idx)
            priorities.append(priority)

        indices = np.array(indices)
        tree_indices = np.array(tree_indices)
        priorities = np.array(priorities)

        # Importance sampling weights
        # w_i = (N * P(i))^(-beta) / max(w)
        probs = priorities / self.tree.total
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Increment frame for beta annealing
        self.frame += 1

        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'next_actions': self.next_actions[indices],
            'indices': indices,
            'tree_indices': tree_indices,
            'weights': weights.astype(np.float32)
        }

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        Update priorities based on TD errors.

        Args:
            tree_indices: Indices in sum tree
            td_errors: Absolute TD errors from update
        """
        # Add small constant for stability
        priorities = np.abs(td_errors) + 1e-6

        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(tree_idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)

    def reset(self) -> None:
        """Clear the buffer."""
        self.tree = SumTree(self.capacity)
        self.size = 0
        self.max_priority = 1.0
        self.frame = 1
        self.n_step_buffer.clear()

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def save_state(self, filepath: str) -> None:
        """
        Save PER buffer state to compressed file.

        Saves current buffer contents including priorities with gzip
        compression for efficient storage and exact reproducibility.

        Args:
            filepath: Path to save file (will add .gz if not present)
        """
        import gzip
        import pickle
        from pathlib import Path

        filepath = Path(filepath)
        if not filepath.suffix == '.gz':
            filepath = Path(str(filepath) + '.gz')

        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'states': self.states[:self.size] if self.size > 0 else self.states[:1],
            'actions': self.actions[:self.size] if self.size > 0 else self.actions[:1],
            'rewards': self.rewards[:self.size] if self.size > 0 else self.rewards[:1],
            'next_states': self.next_states[:self.size] if self.size > 0 else self.next_states[:1],
            'dones': self.dones[:self.size] if self.size > 0 else self.dones[:1],
            'next_actions': self.next_actions[:self.size] if self.size > 0 else self.next_actions[:1],
            'tree_data': self.tree.tree.copy(),
            'tree_pointer': self.tree.data_pointer,
            'size': self.size,
            'max_priority': self.max_priority,
            'frame': self.frame,
            'capacity': self.capacity,
            'state_shape': self.state_shape,
            'alpha': self.alpha,
            'beta_start': self.beta_start,
            'beta_frames': self.beta_frames,
            'n_step': self.n_step,
            'gamma': self.gamma,
        }

        with gzip.open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, filepath: str) -> None:
        """
        Load PER buffer state from compressed file.

        Args:
            filepath: Path to saved buffer state file
        """
        import gzip
        import pickle
        from pathlib import Path

        filepath = Path(filepath)
        if not filepath.exists() and not filepath.suffix == '.gz':
            filepath = Path(str(filepath) + '.gz')

        with gzip.open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Verify compatibility
        if state['state_shape'] != self.state_shape:
            raise ValueError(
                f"State shape mismatch: buffer has {self.state_shape}, "
                f"loaded file has {state['state_shape']}"
            )

        # Restore state
        loaded_size = state['size']
        if loaded_size > 0:
            self.states[:loaded_size] = state['states']
            self.actions[:loaded_size] = state['actions']
            self.rewards[:loaded_size] = state['rewards']
            self.next_states[:loaded_size] = state['next_states']
            self.dones[:loaded_size] = state['dones']
            self.next_actions[:loaded_size] = state['next_actions']

        # Restore tree
        self.tree.tree = state['tree_data']
        self.tree.data_pointer = state['tree_pointer']

        self.size = loaded_size
        self.max_priority = state['max_priority']
        self.frame = state['frame']
        self.n_step_buffer.clear()
