from dataclasses import dataclass
from typing import Dict, Callable, Tuple, List

import gymnasium as gym
from gymnasium.envs.registration import register

from .agent import Agent
from .file.profile import ProfileConfig
from .error import AgentNotFound, WrapperNotFound


@dataclass
class AgentSpec:
    """A specification for creating agents with Talos."""
    id: str
    training_wrapper: Callable
    graphing_wrapper: Callable[[Dict, ProfileConfig], None]
    agent_factory: Callable[[int, int], Agent]


@dataclass
class WrapperSpec:
    """A specification for using environment wrappers with Talos."""
    id: str
    wrapper_factory: Callable[[gym.Env], gym.Wrapper]


@dataclass
class EnvSpec:
    """A specification for using environments with Talos. Wraps the OpenAI Gym registration ability."""
    id: str
    graphing_wrapper: Callable[[Dict, List[str], Tuple], None]


# Global registries for storing configs.
agent_registry: Dict[str, AgentSpec] = {}
wrapper_registry: Dict[str, WrapperSpec] = {}
env_registry: Dict[str, EnvSpec] = {}


def _dummy_training_wrapper(
        env_factory,
        agent,
        agent_config,
        training_artifacts
):
    print("Dummy training wrapper in use - no training will be done.")
    pass


def _dummy_agent_graphing_wrapper(
        artifacts,
        config: ProfileConfig
):
    print("Dummy graphing wrapper in use - no graphs produced.")
    pass


def _dummy_env_graphing_wrapper(
        agent_names: List[str],
        rewards: List[float],
        extra_infos: List[Dict]
):
    print("Dummy graphing wrapper in use - no graphs produced.")
    pass


def register_agent(
        agent_id: str,
        agent_factory: Callable[[], Agent],
        training_wrapper: Callable = _dummy_training_wrapper,
        graphing_wrapper: Callable[[Dict, ProfileConfig], None] = _dummy_agent_graphing_wrapper,
):
    """Register an agent with Talos."""
    global agent_registry

    agent_spec = AgentSpec(
        id=agent_id,
        training_wrapper=training_wrapper,
        graphing_wrapper=graphing_wrapper,
        agent_factory=agent_factory
    )

    agent_registry[agent_id] = agent_spec


def _get_spec(agent_id: str) -> AgentSpec:
    global agent_registry

    if agent_id in agent_registry:
        spec = agent_registry[agent_id]
        return spec
    else:
        raise AgentNotFound


def get_agent(
        agent_id: str
) -> Tuple[Callable, Callable]:
    spec = _get_spec(agent_id)
    return spec.agent_factory, spec.training_wrapper


def get_agent_graphing(agent_id: str) -> Callable[[Dict, ProfileConfig], None]:
    spec = _get_spec(agent_id)
    return spec.graphing_wrapper


def register_wrapper(
        wrapper_id: str,
        wrapper_factory: Callable[[gym.Env], gym.Wrapper]
):
    """Register a wrapper with Talos."""
    global wrapper_registry

    wrapper_spec = WrapperSpec(
        id=wrapper_id,
        wrapper_factory=wrapper_factory
    )

    wrapper_registry[wrapper_id] = wrapper_spec


def get_wrapper(
        id: str
) -> Callable:
    global wrapper_registry

    if id in wrapper_registry:
        spec = wrapper_registry[id]
        return spec.wrapper_factory
    else:
        raise WrapperNotFound


def register_env(
        env_id: str,
        entry_point: str,
        graphing_wrapper: Callable[[Dict, List[str], Tuple], None] = _dummy_env_graphing_wrapper
):
    """Register an environment with Talos."""
    global env_registry

    env_spec = EnvSpec(
        id=env_id,
        graphing_wrapper=graphing_wrapper
    )

    env_registry[env_id] = env_spec

    # Register the environment with the OpenAI Gym.
    register(
        id=env_id,
        entry_point=entry_point
    )


def get_env_graphing_wrapper(env_id: str):
    global env_registry

    if env_id in env_registry:
        spec = env_registry[env_id]
        return spec.graphing_wrapper
    else:
        raise WrapperNotFound
