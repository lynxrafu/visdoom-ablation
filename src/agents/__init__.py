"""
Agent implementations for deep reinforcement learning.

Available agents:
- DQNAgent: Vanilla Deep Q-Network
- DeepSARSAAgent: On-policy Deep SARSA
- DDQNAgent: Double DQN
- DuelingDQNAgent: Dueling architecture DQN
"""

from .base import BaseAgent, QNetwork
from .dqn import DQNAgent
from .deep_sarsa import DeepSARSAAgent
from .extensions import DDQNAgent, DuelingDQNAgent, DuelingQNetwork

__all__ = [
    "BaseAgent",
    "QNetwork",
    "DQNAgent",
    "DeepSARSAAgent",
    "DDQNAgent",
    "DuelingDQNAgent",
    "DuelingQNetwork",
]
