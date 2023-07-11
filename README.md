# Crete

[![PyPI version](https://badge.fury.io/py/crete.svg)](https://badge.fury.io/py/crete)

```
               _       
  ___ _ __ ___| |_ ___ 
 / __| '__/ _ \ __/ _ \
| (__| | |  __/ ||  __/
 \___|_|  \___|\__\___|
Reinforcement Learning CLI assistant.

```

## Installation

Crete is now officially on PyPI!

```cmd

pip install crete

```

## Introduction

Crete is a CLI for RL training.
Named after the island that Talos, the first AI in human mythology, walked upon.

It can be used as easily as:

```
python -m crete --help
```

## Registering Agents, Environments and Wrappers

To start using Crete, you must register (at least) one agent.
Crete already comes preloaded with all the OpenAI Gym(nasium) environments.
You can register your own environments too, as long as they use the OpenAI Gym(nasium) API.
More information on making enivironment can be found on
the [Gym(nasium) website](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

Agents must inherit the `crete.Agent` class. For example;

```
 cool_rl_project
 +- crete_profile.yaml  <- config file, more on this later.
 +- module_a 
 |  +- agents
 |  |  +- __init__.py
 |  |  +- agent_1.py
 |  +- etc...
```

```python
# module_a/agent_1.py

from crete import Agent, ExtraState, ProfileConfig


class DQNAgent(Agent):
    def __init__(self, obs, n_actions, device='cpu') -> None:
        super().__init__("DQN")
        ...

    def get_action(self, obs, extra_state: ExtraState = None) -> Tuple[int, ExtraState]:
        """Request an action."""
        ...

    def save(self) -> bytes:
        """Extract all policy data."""
        ...

    def load(self, agent_data: bytes):
        """Load policy data."""
        ...


def dqn_training_wrapper(
        env_factory: Callable[[int], gym.Env],  # Function that creates the environment.
        agent: DQNAgent,  # The agent itself.
        config: ProfileConfig,  # Object with config values.
        artifacts: Dict,  # Empty map for storing training information.
        save_callback  # Function for saving agents, for instance at peak rewards.
):
    """The DQN training procedure."""
    # Note that this function is outside the agent definition.
    ...
```

And then register that agent in the module's initialisation file.
There is also `register_env` and `register_wrapper` which work the same way.
Environments and wrappers should use the OpenAI Gym(nasium) API.

```python
# module_a/__init__.py

from crete import register_agent
from .dqn_agent import DQNAgent, dqn_training_wrapper

register_agent(
    agent_id="DQN",
    agent_factory=lambda obs, n_actions, device: DQNAgent(obs, n_actions, device=device),
    training_wrapper=dqn_training_wrapper
)
```

Then, register that module with crete (this will create a `.crete.yaml` file);

```
python -m crete module add module_a.agents
```

Verify the module has been loaded correctly by running the following, which should list DQN.

```
python -m crete list agents
```

## Training Agents

Previously we saw the training wrapper for the example DQN agent. It contained 5 parameters;

- `env_factory: Callable[[int], gym.Env]`. This is the function that creates the environment.
    - A factory is given instead of the environment itself to allow fresh environments to be created for evaluation.
- `agent: Agent`. The agent itself.
- `config: ProfileConfig`. The configuration profile containing agent and training parameters.
- `artifacts: Dict`. An empty dictionary for storing training artifacts.
    - Training artifacts are things like loss over time. These artifacts are saved in the concrete file.
- `save_callback: SaveCallback`. Utility function for saving snapshots of the agent.
    - Used like `save_callback(agent, artifacts, step_iteration, "autosave-name")`

Agents can be trained using the `train` or `batch` command, which trains one or several 'profiles' respectively.
A 'profile' is contained within a `.yaml` file, and may look like the following;

```yaml
---
# crete_profile.yaml

defaults: # Contains default `config` values that can be overridden. 
  batch_size: 32
  init_epsilon: 1
  final_epsilon: 0.1
  eval_freq: 1000
  gather_freq: 50

final_dqn_map_0: # Arbitrary profile ID, can be anything.
  agent_id: DQN # The ID of the registered agent to use.
  env_id: CartPole-v1 # The ID of the environment to use.
  env_args: # Arbitrary environment arguments that will be passed to the constructor
    render_mode: human
  config: # Config values that can be accessed through the ProfileConfig object via the training wrapper.
    total_steps: 40000
    decay_steps: 38000
    replay_buffer_size: 10000
    batch_size: 128
    refresh_target_network_freq: 1000
    hidden_layers:
      - 64
      - 64
```

Training is engaged by;

```
python -m crete train crete_profile.yaml final_dqn_map_0 ; to train a specific profile or,
python -m crete batch crete_profile.yaml                 ; to train all profiles within a file.
```

## Saving Agents and their Performances

Crete allows the entire training process and outcomes to be saved, for replaying, evaluating and comparing.
All that is required is for the agent to implement the `save` and `load` methods, which serialise the agents weight data
to bytes.
Everything else is handled by Crete. 
One would usually use `pickle` to achieve this.

For example, with a PyTorch-based agent:

```python
def save(self) -> bytes:
    data = {
        "data": self.net.state_dict(),
        "layers": self.net.hidden_layers
    }

    return pickle.dumps(data)


def load(self, agent_data: bytes):
    agent_dict = pickle.loads(agent_data)
    data, layers = itemgetter("data", "layers")(agent_dict)

    self.net.set_hidden_layers(layers)
    self.net.load_state_dict(data)
```

Crete automatically saves the agent on completion of the training wrapper, or on keyboard interrupt.