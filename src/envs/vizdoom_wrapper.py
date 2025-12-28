"""
ViZDoom environment wrappers for visual reinforcement learning.

This module provides Gymnasium wrappers for preprocessing:
- PreprocessWrapper: Grayscale conversion, resizing, normalization
- FrameStackWrapper: Stack consecutive frames for temporal info
- SkipFrameWrapper: Action repeat with reward accumulation

Factory function:
- make_vizdoom_env: Create fully preprocessed environment
"""

import gymnasium as gym
import numpy as np
import cv2
import warnings
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

# Suppress ViZDoom screen format warning (it auto-converts to RGB24)
warnings.filterwarnings("ignore", message="Detected screen format")

# Register ViZDoom gymnasium environments
# This import registers all ViZDoom envs with gymnasium
try:
    import vizdoom.gymnasium_wrapper  # noqa: F401
except ImportError:
    # Fallback for older vizdoom versions
    try:
        from vizdoom import gymnasium_wrapper  # noqa: F401
    except ImportError:
        print("Warning: Could not import vizdoom.gymnasium_wrapper. "
              "ViZDoom environments may not be registered.")


# Available ViZDoom scenarios with their characteristics
VIZDOOM_SCENARIOS = {
    'VizdoomBasic-v0': {
        'description': 'Basic shooting task - kill monster in single room',
        'actions': 3,  # Turn left, turn right, shoot
        'max_steps': 300,
        'difficulty': 'easy'
    },
    'VizdoomCorridor-v0': {
        'description': 'Navigate corridor and kill enemies',
        'actions': 7,
        'max_steps': 2100,
        'difficulty': 'medium'
    },
    'VizdoomDefendCenter-v0': {
        'description': 'Defend against enemies from center',
        'actions': 3,
        'max_steps': 2100,
        'difficulty': 'medium'
    },
    'VizdoomDefendLine-v0': {
        'description': 'Defend from a line position',
        'actions': 3,
        'max_steps': 2100,
        'difficulty': 'medium'
    },
    'VizdoomHealthGathering-v0': {
        'description': 'Collect health packs to survive',
        'actions': 3,
        'max_steps': 2100,
        'difficulty': 'easy'
    },
    'VizdoomMyWayHome-v0': {
        'description': 'Navigate maze to find goal',
        'actions': 3,
        'max_steps': 2100,
        'difficulty': 'hard'
    },
    'VizdoomPredictPosition-v0': {
        'description': 'Predict and shoot moving target',
        'actions': 3,
        'max_steps': 300,
        'difficulty': 'medium'
    },
    'VizdoomTakeCover-v0': {
        'description': 'Survival task - dodge fireballs',
        'actions': 2,  # Move left, move right
        'max_steps': 2100,
        'difficulty': 'medium'
    },
    'VizdoomDeathmatch-v0': {
        'description': 'Combat task - kill enemies, survive',
        'actions': 15,  # Discretized from Dict space (see DictActionWrapper)
        'max_steps': 4200,
        'difficulty': 'hard'
    },
    'VizdoomHealthGatheringSupreme-v0': {
        'description': 'Advanced health gathering with hazards',
        'actions': 3,
        'max_steps': 2100,
        'difficulty': 'hard'
    }
}


class DictActionWrapper(gym.ActionWrapper):
    """
    Convert Dict/MultiDiscrete action space to Discrete.

    Some ViZDoom scenarios (like Deathmatch) have complex action spaces
    with multiple action dimensions. This wrapper flattens them into
    a single Discrete space by enumerating all valid action combinations.

    For example, if the action space has:
    - binary[0]: ATTACK (0 or 1)
    - binary[1]: MOVE_FORWARD (0 or 1)
    - binary[2]: TURN_LEFT (0 or 1)
    - binary[3]: TURN_RIGHT (0 or 1)

    We create discrete actions for useful combinations like:
    - 0: No action
    - 1: Attack
    - 2: Move forward
    - 3: Move forward + Attack
    - 4: Turn left
    - 5: Turn right
    - etc.

    Args:
        env: Gymnasium environment with Dict action space
    """

    # Predefined action combinations for Deathmatch-like scenarios
    # Each tuple represents: (ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, ...)
    DEATHMATCH_ACTIONS = [
        # Basic movement
        [0, 0, 0, 0, 0, 0, 0],  # 0: No action
        [1, 0, 0, 0, 0, 0, 0],  # 1: Attack
        [0, 1, 0, 0, 0, 0, 0],  # 2: Move forward
        [0, 0, 1, 0, 0, 0, 0],  # 3: Move backward
        [0, 0, 0, 1, 0, 0, 0],  # 4: Turn left
        [0, 0, 0, 0, 1, 0, 0],  # 5: Turn right
        # Attack + movement
        [1, 1, 0, 0, 0, 0, 0],  # 6: Attack + Move forward
        [1, 0, 0, 1, 0, 0, 0],  # 7: Attack + Turn left
        [1, 0, 0, 0, 1, 0, 0],  # 8: Attack + Turn right
        # Strafe
        [0, 0, 0, 0, 0, 1, 0],  # 9: Move left (strafe)
        [0, 0, 0, 0, 0, 0, 1],  # 10: Move right (strafe)
        # Combined
        [0, 1, 0, 1, 0, 0, 0],  # 11: Move forward + Turn left
        [0, 1, 0, 0, 1, 0, 0],  # 12: Move forward + Turn right
        [1, 1, 0, 1, 0, 0, 0],  # 13: Attack + Move forward + Turn left
        [1, 1, 0, 0, 1, 0, 0],  # 14: Attack + Move forward + Turn right
    ]

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        self.original_action_space = env.action_space
        self._setup_action_mapping()

        # Create new Discrete action space
        self.action_space = gym.spaces.Discrete(len(self.action_list))

    def _setup_action_mapping(self):
        """Setup action mapping based on original action space type."""
        if isinstance(self.original_action_space, gym.spaces.Dict):
            # Dict action space - get the binary actions
            self.action_keys = list(self.original_action_space.spaces.keys())
            self.num_binary = len(self.action_keys)

            # Use predefined actions, trimmed to match action space size
            self.action_list = []
            for action in self.DEATHMATCH_ACTIONS:
                if len(action) >= self.num_binary:
                    self.action_list.append(action[:self.num_binary])
                else:
                    # Pad with zeros if needed
                    padded = action + [0] * (self.num_binary - len(action))
                    self.action_list.append(padded)

        elif isinstance(self.original_action_space, gym.spaces.MultiDiscrete):
            # MultiDiscrete action space
            self.num_binary = len(self.original_action_space.nvec)
            self.action_keys = None

            self.action_list = []
            for action in self.DEATHMATCH_ACTIONS:
                if len(action) >= self.num_binary:
                    self.action_list.append(action[:self.num_binary])
                else:
                    padded = action + [0] * (self.num_binary - len(action))
                    self.action_list.append(padded)
        else:
            raise ValueError(f"Unsupported action space: {type(self.original_action_space)}")

    def action(self, action: int):
        """Convert discrete action to original action space format."""
        action_values = self.action_list[action]

        # ViZDoom gymnasium wrapper expects numpy array of button states
        # regardless of whether action_space is Dict or MultiDiscrete
        return np.array(action_values, dtype=np.float32)


class PreprocessWrapper(gym.ObservationWrapper):
    """
    Preprocess ViZDoom observations for neural network input.

    Processing steps:
    1. Convert to grayscale (if RGB)
    2. Resize to target dimensions (default 84x84)
    3. Normalize pixel values to [0, 1]

    Args:
        env: Gymnasium environment to wrap
        width: Target image width
        height: Target image height

    Note:
        Output observation is a 2D array (H, W), not 3D.
        Use with FrameStackWrapper to add channel dimension.
    """

    def __init__(
        self,
        env: gym.Env,
        width: int = 84,
        height: int = 84
    ) -> None:
        super().__init__(env)

        self.width = width
        self.height = height

        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(height, width),
            dtype=np.float32
        )

    def observation(self, obs) -> np.ndarray:
        """
        Process a single observation frame.

        Handles various input formats from ViZDoom:
        - dict: {'screen': array, ...} - extract 'screen' key
        - (H, W, C): Standard HWC format
        - (C, H, W): Channel-first format
        - (H, W): Already grayscale

        Args:
            obs: Raw observation from environment (dict or array)

        Returns:
            Preprocessed observation (H, W) normalized to [0, 1]
        """
        # Handle dictionary observations (ViZDoom gymnasium wrapper format)
        if isinstance(obs, dict):
            if 'screen' in obs:
                obs = obs['screen']
            elif 'rgb' in obs:
                obs = obs['rgb']
            else:
                # Use first array value found
                for v in obs.values():
                    if isinstance(v, np.ndarray):
                        obs = v
                        break

        # Ensure numpy array
        obs = np.asarray(obs)

        # Handle channel-first format (C, H, W)
        if len(obs.shape) == 3:
            if obs.shape[0] in [1, 3, 4]:  # Likely CHW
                obs = np.transpose(obs, (1, 2, 0))

        # Convert to grayscale if RGB
        if len(obs.shape) == 3:
            if obs.shape[2] == 3:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            elif obs.shape[2] == 1:
                obs = obs.squeeze(-1)
            elif obs.shape[2] == 4:  # RGBA
                obs = cv2.cvtColor(obs, cv2.COLOR_RGBA2GRAY)

        # Resize to target dimensions
        obs = cv2.resize(
            obs,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA
        )

        # Normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0

        return obs


class FrameStackWrapper(gym.Wrapper):
    """
    Stack consecutive frames for temporal information.

    Maintains a deque of recent frames and returns them stacked
    as a single observation. This helps the agent perceive
    motion and velocity.

    Args:
        env: Gymnasium environment to wrap
        n_frames: Number of frames to stack (default 4)

    Output shape: (n_frames, H, W)
    """

    def __init__(self, env: gym.Env, n_frames: int = 4) -> None:
        super().__init__(env)

        self.n_frames = n_frames
        self.frames: deque = deque(maxlen=n_frames)

        # Update observation space
        old_space = env.observation_space
        low = np.repeat(old_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(old_space.high[np.newaxis, ...], n_frames, axis=0)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Fill frame stack with initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs)

        return self._get_observation(), info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take step and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Stack frames into single observation."""
        return np.array(self.frames, dtype=np.float32)


class SkipFrameWrapper(gym.Wrapper):
    """
    Skip frames (action repeat) with reward accumulation.

    Executes the same action for multiple frames, accumulating
    rewards. This reduces computational cost and provides
    more meaningful rewards per decision.

    Args:
        env: Gymnasium environment to wrap
        skip: Number of frames to skip (repeat action)
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self.skip = skip

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action for skip frames, accumulating reward."""
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class ClipRewardWrapper(gym.RewardWrapper):
    """
    Clip rewards to [-1, 1] range.

    This stabilizes training by preventing large reward magnitudes
    from causing gradient explosion.

    Args:
        env: Gymnasium environment to wrap
    """

    def reward(self, reward: float) -> float:
        """Clip reward to [-1, 1]."""
        return np.clip(reward, -1.0, 1.0)


class EpisodeInfoWrapper(gym.Wrapper):
    """
    Track episode statistics (length, return).

    Adds 'episode' key to info dict at episode end with:
    - 'r': Total episode return
    - 'l': Episode length
    - 't': Episode time (optional)

    Args:
        env: Gymnasium environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.episode_return = 0.0
        self.episode_length = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset episode statistics."""
        self.episode_return = 0.0
        self.episode_length = 0
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Track statistics and add to info on episode end."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_return += reward
        self.episode_length += 1

        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_return,
                'l': self.episode_length
            }

        return obs, reward, terminated, truncated, info


def make_vizdoom_env(
    scenario: str = "VizdoomBasic-v0",
    frame_skip: int = 4,
    frame_stack: int = 4,
    image_size: int = 84,
    clip_rewards: bool = False,
    episode_info: bool = True,
    render_mode: Optional[str] = None
) -> gym.Env:
    """
    Create a fully preprocessed ViZDoom Gymnasium environment.

    Applies the following wrappers (in order):
    1. SkipFrameWrapper: Action repeat with reward accumulation
    2. PreprocessWrapper: Grayscale, resize, normalize
    3. FrameStackWrapper: Stack consecutive frames
    4. ClipRewardWrapper: Optional reward clipping
    5. EpisodeInfoWrapper: Optional episode statistics

    Args:
        scenario: ViZDoom environment ID (e.g., 'VizdoomBasic-v0')
        frame_skip: Number of frames to skip (action repeat)
        frame_stack: Number of frames to stack
        image_size: Size of square observation image
        clip_rewards: Whether to clip rewards to [-1, 1]
        episode_info: Whether to track episode statistics
        render_mode: 'human' for visualization, None for headless

    Returns:
        Preprocessed Gymnasium environment

    Example:
        >>> env = make_vizdoom_env('VizdoomBasic-v0')
        >>> obs, info = env.reset()
        >>> print(obs.shape)  # (4, 84, 84)
        >>> action = env.action_space.sample()
        >>> obs, reward, term, trunc, info = env.step(action)

    Raises:
        ValueError: If scenario is not a valid ViZDoom environment

    Note:
        For Colab/headless use, ensure Xvfb is running:
        >>> import os
        >>> os.system('Xvfb :1 -screen 0 1024x768x24 &')
        >>> os.environ['DISPLAY'] = ':1'
    """
    # Validate scenario
    if scenario not in VIZDOOM_SCENARIOS:
        available = list(VIZDOOM_SCENARIOS.keys())
        raise ValueError(
            f"Unknown scenario: {scenario}. "
            f"Available: {available}"
        )

    # Create base environment with proper screen format
    # Set frame_skip=1 here (we handle it in SkipFrameWrapper)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected screen format")
        env = gym.make(scenario, render_mode=render_mode, frame_skip=1)

    # Handle complex action spaces (Dict/MultiDiscrete -> Discrete)
    # This is needed for scenarios like Deathmatch
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env = DictActionWrapper(env)

    # Apply wrappers in order
    if frame_skip > 1:
        env = SkipFrameWrapper(env, skip=frame_skip)

    env = PreprocessWrapper(env, width=image_size, height=image_size)
    env = FrameStackWrapper(env, n_frames=frame_stack)

    if clip_rewards:
        env = ClipRewardWrapper(env)

    if episode_info:
        env = EpisodeInfoWrapper(env)

    return env


def get_scenario_info(scenario: str) -> Dict[str, Any]:
    """
    Get information about a ViZDoom scenario.

    Args:
        scenario: ViZDoom environment ID

    Returns:
        Dictionary with scenario metadata
    """
    if scenario not in VIZDOOM_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    return VIZDOOM_SCENARIOS[scenario].copy()


def list_scenarios() -> List[str]:
    """List all available ViZDoom scenarios."""
    return list(VIZDOOM_SCENARIOS.keys())
