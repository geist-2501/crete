import gymnasium as gym
import typer
from rich import print

from crete.core import load_extra_modules
from crete.registration import agent_registry, wrapper_registry

app = typer.Typer()


@app.callback()
def doc():
    """View registered entities for Talos."""
    pass


@app.command("agents")
def list_agents():
    """List all registered agents."""
    load_extra_modules()

    print("[bold]Currently registered agents[/]:")
    for agent in agent_registry.keys():
        print(" " + agent)


@app.command("wrappers")
def list_wrappers():
    """List all registered wrappers."""
    load_extra_modules()

    print("\n[bold]Currently registered wrappers[/]:")
    for wrapper in wrapper_registry.keys():
        print(" " + wrapper)


@app.command("envs")
def list_envs():
    """List all registered environments."""
    load_extra_modules()

    print("\n[bold]Currently registered environments[/]:")
    for env in gym.envs.registry.keys():
        print(" " + env)
