from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional

import numpy as np

ExtraState = Any


class Agent(ABC):
    """Base class for an agent for use in the Talos ecosystem."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def get_action(self, obs, extra_state: ExtraState = None) -> Tuple[int, ExtraState]:
        """Request an action."""
        pass

    def post_step(self, obs, action, next_obs, extra_state: ExtraState = None) -> ExtraState:
        """Perform any post-processing on the extra state."""
        return extra_state

    @abstractmethod
    def save(self) -> bytes:
        """Extract all policy data."""
        pass

    @abstractmethod
    def load(self, agent_data: bytes):
        """Load policy data."""
        pass

    def get_q_values(self, obs) -> Optional[np.ndarray]:
        """Get the Q-values for a given observation. Not all agents use Q-values."""
        return None
