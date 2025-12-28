"""
Configuration schema and validation for ViZDoom experiments.

This module provides:
- Dataclass schemas for all config sections
- Validation logic to catch invalid hyperparameters early
- validate_config() function to check entire config

Usage:
    from src.utils.config_schema import validate_config
    validate_config(config)  # Raises ConfigValidationError if invalid
"""

from dataclasses import dataclass
from typing import List, Optional
from omegaconf import DictConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


# =============================================================================
# Config Dataclasses with Validation
# =============================================================================

@dataclass
class EnvConfig:
    """Environment configuration schema."""
    scenario: str
    frame_skip: int
    frame_stack: int
    image_size: int
    clip_rewards: bool

    def __post_init__(self):
        # Validate scenario
        valid_scenarios = [
            "VizdoomBasic-v0",
            "VizdoomCorridor-v0",
            "VizdoomDefendCenter-v0",
            "VizdoomDefendLine-v0",
            "VizdoomHealthGathering-v0",
            "VizdoomMyWayHome-v0",
            "VizdoomPredictPosition-v0",
            "VizdoomTakeCover-v0",
            "VizdoomDeathmatch-v0",
            "VizdoomHealthGatheringSupreme-v0",
        ]
        # Allow custom scenarios but warn
        if self.scenario not in valid_scenarios:
            import logging
            logging.getLogger("vizdoom").warning(
                f"Scenario '{self.scenario}' not in known list. "
                f"Known scenarios: {valid_scenarios}"
            )

        # Validate frame_skip
        if not 1 <= self.frame_skip <= 10:
            raise ConfigValidationError(
                f"frame_skip must be in [1, 10], got {self.frame_skip}"
            )

        # Validate frame_stack
        if not 1 <= self.frame_stack <= 8:
            raise ConfigValidationError(
                f"frame_stack must be in [1, 8], got {self.frame_stack}"
            )

        # Validate image_size
        if not 32 <= self.image_size <= 256:
            raise ConfigValidationError(
                f"image_size must be in [32, 256], got {self.image_size}"
            )


@dataclass
class AgentConfig:
    """Agent configuration schema."""
    type: str
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    target_update_freq: int
    n_step: int
    double_dqn: bool = False

    def __post_init__(self):
        # Validate agent type
        valid_types = ['dqn', 'deep_sarsa', 'ddqn', 'dueling', 'dueling_ddqn']
        if self.type.lower() not in valid_types:
            raise ConfigValidationError(
                f"agent.type must be one of {valid_types}, got '{self.type}'"
            )

        # Validate learning_rate
        if not 0 < self.learning_rate <= 1:
            raise ConfigValidationError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )

        # Validate gamma
        if not 0 < self.gamma <= 1:
            raise ConfigValidationError(
                f"gamma must be in (0, 1], got {self.gamma}"
            )

        # Validate epsilon values
        if not 0 <= self.epsilon_start <= 1:
            raise ConfigValidationError(
                f"epsilon_start must be in [0, 1], got {self.epsilon_start}"
            )
        if not 0 <= self.epsilon_end <= 1:
            raise ConfigValidationError(
                f"epsilon_end must be in [0, 1], got {self.epsilon_end}"
            )
        if self.epsilon_end > self.epsilon_start:
            raise ConfigValidationError(
                f"epsilon_end ({self.epsilon_end}) must be <= epsilon_start ({self.epsilon_start})"
            )
        if not 0 < self.epsilon_decay <= 1:
            raise ConfigValidationError(
                f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}"
            )

        # Validate target_update_freq
        if not self.target_update_freq >= 1:
            raise ConfigValidationError(
                f"target_update_freq must be >= 1, got {self.target_update_freq}"
            )

        # Validate n_step
        if not 1 <= self.n_step <= 10:
            raise ConfigValidationError(
                f"n_step must be in [1, 10], got {self.n_step}"
            )


@dataclass
class BufferConfig:
    """Replay buffer configuration schema."""
    capacity: int
    prioritized: bool
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000

    def __post_init__(self):
        # Validate capacity
        if not 1000 <= self.capacity <= 10_000_000:
            raise ConfigValidationError(
                f"buffer.capacity must be in [1000, 10000000], got {self.capacity}"
            )

        # Validate PER parameters if prioritized
        if self.prioritized:
            if not 0 <= self.per_alpha <= 1:
                raise ConfigValidationError(
                    f"per_alpha must be in [0, 1], got {self.per_alpha}"
                )
            if not 0 <= self.per_beta_start <= 1:
                raise ConfigValidationError(
                    f"per_beta_start must be in [0, 1], got {self.per_beta_start}"
                )
            if not self.per_beta_frames >= 1:
                raise ConfigValidationError(
                    f"per_beta_frames must be >= 1, got {self.per_beta_frames}"
                )


@dataclass
class TrainingConfig:
    """Training configuration schema."""
    num_episodes: int
    max_steps_per_episode: int
    batch_size: int
    min_buffer_size: int
    update_freq: int
    eval_freq: int
    eval_episodes: int
    save_freq: int

    def __post_init__(self):
        # Validate num_episodes
        if not self.num_episodes >= 1:
            raise ConfigValidationError(
                f"num_episodes must be >= 1, got {self.num_episodes}"
            )

        # Validate max_steps_per_episode
        if not 1 <= self.max_steps_per_episode <= 100_000:
            raise ConfigValidationError(
                f"max_steps_per_episode must be in [1, 100000], got {self.max_steps_per_episode}"
            )

        # Validate batch_size
        if not 1 <= self.batch_size <= 1024:
            raise ConfigValidationError(
                f"batch_size must be in [1, 1024], got {self.batch_size}"
            )

        # Validate min_buffer_size >= batch_size
        if self.min_buffer_size < self.batch_size:
            raise ConfigValidationError(
                f"min_buffer_size ({self.min_buffer_size}) must be >= batch_size ({self.batch_size})"
            )

        # Validate update_freq
        if not 1 <= self.update_freq <= 100:
            raise ConfigValidationError(
                f"update_freq must be in [1, 100], got {self.update_freq}"
            )

        # Validate eval_freq
        if not self.eval_freq >= 1:
            raise ConfigValidationError(
                f"eval_freq must be >= 1, got {self.eval_freq}"
            )

        # Validate eval_episodes
        if not 1 <= self.eval_episodes <= 100:
            raise ConfigValidationError(
                f"eval_episodes must be in [1, 100], got {self.eval_episodes}"
            )


@dataclass
class LoggingConfig:
    """Logging configuration schema."""
    csv_log: bool
    log_freq: int
    level: str = "INFO"
    flush_every: int = 10

    def __post_init__(self):
        # Validate log_freq
        if not self.log_freq >= 1:
            raise ConfigValidationError(
                f"log_freq must be >= 1, got {self.log_freq}"
            )

        # Validate logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ConfigValidationError(
                f"logging.level must be one of {valid_levels}, got '{self.level}'"
            )

        # Validate flush_every
        if not 1 <= self.flush_every <= 100:
            raise ConfigValidationError(
                f"flush_every must be in [1, 100], got {self.flush_every}"
            )


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_config(config: DictConfig) -> None:
    """
    Validate entire configuration.

    Checks all sections for valid values and raises ConfigValidationError
    with a descriptive message if any validation fails.

    Args:
        config: Hydra DictConfig to validate

    Raises:
        ConfigValidationError: If any config value is invalid

    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.load("configs/default.yaml")
        >>> validate_config(config)  # Raises if invalid
    """
    errors = []

    # Validate env section
    try:
        EnvConfig(
            scenario=config.env.scenario,
            frame_skip=config.env.frame_skip,
            frame_stack=config.env.frame_stack,
            image_size=config.env.image_size,
            clip_rewards=config.env.clip_rewards,
        )
    except ConfigValidationError as e:
        errors.append(f"[env] {e}")
    except Exception as e:
        errors.append(f"[env] Unexpected error: {e}")

    # Validate agent section
    try:
        AgentConfig(
            type=config.agent.type,
            learning_rate=config.agent.learning_rate,
            gamma=config.agent.gamma,
            epsilon_start=config.agent.epsilon_start,
            epsilon_end=config.agent.epsilon_end,
            epsilon_decay=config.agent.epsilon_decay,
            target_update_freq=config.agent.target_update_freq,
            n_step=config.agent.n_step,
            double_dqn=config.agent.get('double_dqn', False),
        )
    except ConfigValidationError as e:
        errors.append(f"[agent] {e}")
    except Exception as e:
        errors.append(f"[agent] Unexpected error: {e}")

    # Validate buffer section
    try:
        BufferConfig(
            capacity=config.buffer.capacity,
            prioritized=config.buffer.prioritized,
            per_alpha=config.buffer.per_alpha,
            per_beta_start=config.buffer.per_beta_start,
            per_beta_frames=config.buffer.per_beta_frames,
        )
    except ConfigValidationError as e:
        errors.append(f"[buffer] {e}")
    except Exception as e:
        errors.append(f"[buffer] Unexpected error: {e}")

    # Validate training section
    try:
        TrainingConfig(
            num_episodes=config.training.num_episodes,
            max_steps_per_episode=config.training.max_steps_per_episode,
            batch_size=config.training.batch_size,
            min_buffer_size=config.training.min_buffer_size,
            update_freq=config.training.update_freq,
            eval_freq=config.training.eval_freq,
            eval_episodes=config.training.eval_episodes,
            save_freq=config.training.save_freq,
        )
    except ConfigValidationError as e:
        errors.append(f"[training] {e}")
    except Exception as e:
        errors.append(f"[training] Unexpected error: {e}")

    # Validate logging section
    try:
        LoggingConfig(
            csv_log=config.logging.csv_log,
            log_freq=config.logging.log_freq,
            level=config.logging.get('level', 'INFO'),
            flush_every=config.logging.get('flush_every', 10),
        )
    except ConfigValidationError as e:
        errors.append(f"[logging] {e}")
    except Exception as e:
        errors.append(f"[logging] Unexpected error: {e}")

    # Validate seed
    if not isinstance(config.seed, int) or config.seed < 0:
        errors.append(f"[seed] seed must be a non-negative integer, got {config.seed}")

    # Validate device
    valid_devices = ['auto', 'cuda', 'cpu']
    if config.device not in valid_devices:
        errors.append(f"[device] device must be one of {valid_devices}, got '{config.device}'")

    # Raise all errors together
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigValidationError(error_msg)


def print_config_summary(config: DictConfig) -> None:
    """
    Print a formatted summary of the configuration.

    Args:
        config: Hydra DictConfig to summarize
    """
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Agent:      {config.agent.type} (lr={config.agent.learning_rate}, gamma={config.agent.gamma})")
    print(f"Scenario:   {config.env.scenario}")
    print(f"Episodes:   {config.training.num_episodes}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Buffer:     {config.buffer.capacity} ({'PER' if config.buffer.prioritized else 'Uniform'})")
    print(f"Seed:       {config.seed}")
    print(f"Device:     {config.device}")
    print("=" * 60)
