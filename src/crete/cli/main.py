from typing import Optional, List

import typer
from gym.utils.play import play as gym_play
from rich import print

from .cli_utils import _convert_to_key_value_list
from .config import app as config_app
from .list import app as list_app
from .talfile import talfile_app
from .profile import profile_app
from ..core import load_config, create_env_factory, get_device, create_agent, create_save_callback
from ..global_state import get_cli_state
from ..error import *
from ..file import TalFile
from ..profile import ProfileConfig

app = typer.Typer()
app.add_typer(config_app, name="config")
app.add_typer(list_app, name="list")
app.add_typer(talfile_app, name="talfile")
app.add_typer(profile_app, name="profile")

__app_name__ = "talos"
__version__ = "0.1.0"


def talos_app():
    app(prog_name=__app_name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(r"""
 _____  _    _     ___  ____  
|_   _|/ \  | |   / _ \/ ___| 
  | | / _ \ | |  | | | \___ \ 
  | |/ ___ \| |__| |_| |___) |
  |_/_/   \_\_____\___/|____/
  RL agent training assistant""")
        print(f"[bold green]  v{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
        debug_mode: bool = typer.Option(
            False,
            "--dbg",
            help="Active debug mode. Triggers things like extra logging."
        ),
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            help="Show the application's version and exit.",
            callback=_version_callback,
            is_eager=True,
        )
) -> None:
    get_cli_state().debug_mode = debug_mode


@app.command()
def train(
        opt_agent: str = typer.Option(
            "DQN",
            "--agent",
            "-a",
            help="Set the agent to train with",
            prompt="Agent to train with?"
        ),
        opt_config: str = typer.Option(
            "talos_settings.ini",
            "--config",
            "-c",
            help="Set the path of the config to use",
            prompt="Configuration to use?"
        ),
        opt_wrapper: str = typer.Option(
            None,
            "--wrapper",
            "-w",
            help="Set the wrapper to use (currently only 1 supported)"
        ),
        opt_env: str = typer.Option(
            "CartPole-v1",
            "--env",
            "-e",
            help="Set the environment to use",
            prompt="Environment to train in?"
        ),
        opt_env_args: List[str] = typer.Option(
            [],
            "--env-arg",
            help="Set the arguments to be passed to the environment. Write as `key=val`."
        ),
        opt_autosave: Optional[str] = typer.Option(
            None,
            "--autosave-path",
            help="Set the path to save the agent to automatically"
        ),
        opt_seed: int = typer.Option(
            0,
            "--seed",
            "-s",
            help="Set the random seed to use."
        )
) -> None:
    """Train an agent on a given environment."""
    opt_env_args = _convert_to_key_value_list(opt_env_args)

    device = get_device()
    print(f"Using device [bold white]{device}.[/]")

    # Load config.
    try:
        print(f"Loading config `{opt_config}`... ", end="")
        config = load_config(opt_config)
        print("[bold green]success![/]")
    except ConfigNotFound:
        print("[bold green]failure![/]")
        raise typer.Abort()

    env_factory = create_env_factory(opt_env, opt_wrapper, env_args=opt_env_args, base_seed=opt_seed)
    agent, training_wrapper = create_agent(env_factory, opt_agent, device=device)

    config_section = config[opt_agent] if opt_agent in config.sections() else config['DEFAULT']
    agent_config = ProfileConfig(dict(config_section), {})
    print(f"\nProceeding to train a {opt_agent} on {opt_env} with config values:")
    print(agent_config.to_dict())

    if typer.confirm("Ready to proceed?", default=True) is False:
        return

    training_artifacts = {}
    try:
        save_callback = create_save_callback(opt_agent, agent_config.to_dict(), opt_wrapper, opt_env, opt_env_args)
        training_wrapper(env_factory, agent, agent_config, training_artifacts, save_callback)
    except KeyboardInterrupt:
        print("[bold red]Training interrupted[/bold red].")

    if opt_autosave or typer.confirm("Save agent to disk?"):
        try:
            if opt_autosave:
                path = opt_autosave
            else:
                path = typer.prompt("Enter a path to save to")
            print(f"Saving agent to disk ([italic]{path}[/]) ...")
            data = agent.save()
            talfile = TalFile(
                id=opt_agent,
                env_name=opt_env,
                agent_data=data,
                training_artifacts=training_artifacts,
                used_wrappers=opt_wrapper,
                config=agent_config.to_dict(),
                env_args=opt_env_args
            )
            talfile.write(path)
        except OSError as ex:
            print("[bold red]Saving failed![/] " + ex.strerror)


@app.command()
def play(
        arg_env: str = typer.Argument(
            "CartPole-v1",
            help="The environment to play in"
        ),
        opt_wrapper: str = typer.Option(
            None,
            "--wrapper",
            "-w"
        ),
        opt_seed: int = typer.Option(
            None,
            "--seed",
            "-s"
        ),
        opt_env_args: List[str] = typer.Option(
            [],
            "--env-arg",
        ),
        opt_fps: int = typer.Option(
            None,
            "--fps"
        )
):
    """Play the environment as a human. (Not for procrastination!)"""
    opt_env_args = _convert_to_key_value_list(opt_env_args)

    env_factory = create_env_factory(arg_env, opt_wrapper, render_mode='rgb_array', env_args=opt_env_args)
    env = env_factory(opt_seed)
    gym_play(env, fps=opt_fps)
