import gymnasium as gym
import typer
from rich import print

from crete.core import load_extra_modules
from crete.file.crete_config import CreteConfig
from crete.registration import agent_registry, wrapper_registry

list_app = typer.Typer()


@list_app.callback()
def doc():
    """View registered entities for Crete."""
    pass


@list_app.command("agents")
def list_agents():
    """List all registered agents."""
    load_extra_modules()

    for agent in agent_registry.keys():
        print(f" - {agent}")


@list_app.command("wrappers")
def list_wrappers():
    """List all registered wrappers."""
    load_extra_modules()

    for wrapper in wrapper_registry.keys():
        print(f" - {wrapper}")


@list_app.command("envs")
def list_envs():
    """List all registered environments."""
    load_extra_modules()

    for env in gym.envs.registry.keys():
        print(f" - {env}")


@list_app.command(name="modules")
def list_modules():
    # Load the .crete file.
    conf = CreteConfig.try_read(CreteConfig.conf_filename)
    for module in conf.extra_modules:
        print(f" - {module}")
