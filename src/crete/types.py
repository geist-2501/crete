import configparser
from typing import Callable, Dict, Optional, Any

from gymnasium import Env

from .agent import Agent

AgentFactory = Callable[[Any, int, str], Agent]
EnvFactory = Callable[[Optional[int]], Env]

Artifacts = Dict
SaveCallback = Callable[[Any, Artifacts, int, Optional[str]], None]
