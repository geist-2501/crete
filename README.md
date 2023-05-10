# Crete

```
               _       
  ___ _ __ ___| |_ ___ 
 / __| '__/ _ \ __/ _ \
| (__| | |  __/ ||  __/
 \___|_|  \___|\__\___|
Reinforcement Learning CLI assistant.

```

## Installation

Crete is not released for public use yet, and is not available of PyPI. 

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

Simply create a module containing an agent that inherits `crete.Agent`, for example;

```
 cool_rl_project
 +- crete_profile.yaml
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

    def save(self) -> Dict:
        """Extract all policy data."""
        ...

    def load(self, agent_data: Dict):
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
    ...
```

And then register that agent in the modules initialisation file.
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
  config: # Config values that can be accessed through the Profile object via the training wrapper.
    total_steps :  40000
    decay_steps :  38000
    replay_buffer_size: 10000
    batch_size: 128
    refresh_target_network_freq :  1000
    hidden_layers:
      - 64
      - 64
```

Training is engaged by;

```
python -m crete train crete_profile.yaml final_dqn_map_0 ; to train a specific profile or,
python -m crete batch crete_profile.yaml                 ; to train all profiles within a file.
```
