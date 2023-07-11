import os
from typing import Optional, List

import typer
from gymnasium.utils.play import play as gym_play
from rich import print

from crete.cli.cli_utils import _convert_to_key_value_list
from crete.cli.list import list_app
from crete.cli.module import module_app
from crete.cli.concfile import concfile_app
from crete.core import create_env_factory, create_agent, create_save_callback, load_extra_modules
from crete.error import ProfilePropertyNotFound
from crete.global_state import get_cli_state
from crete.file.profile import read_profile, Profile
from crete.file.concrete import ConcreteFile
from crete.util import print_err, print_ex

app = typer.Typer()
app.add_typer(list_app, name="list")
app.add_typer(concfile_app, name="concfile")
app.add_typer(module_app, name="module")

__app_name__ = "crete"
__version__ = "0.2.0"
__app_logo__ = r"""
               _       
  ___ _ __ ___| |_ ___ 
 / __| '__/ _ \ __/ _ \
| (__| | |  __/ ||  __/
 \___|_|  \___|\__\___|
 """


def crete_app():
    app(prog_name=__app_name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__app_logo__)
        print(f"[bold black]RL Training Assistant[/] [green]v{__version__}[/]")
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
        arg_profile_path: str,
        arg_target_profile: str = typer.Argument(
            ...,
            help="The name of the profile to use"
        ),
        opt_out_dir: str = typer.Option(
            ".",
            "--out",
            "-o",
            help="Directory to place trained agents."
        ),
        opt_as: str = typer.Option(
            None,
            "--as",
            help="Filename override to save as."
        ),
        opt_override: bool = typer.Option(
            False,
            "--override",
            "-x",
            help="If false, don't run retrain profiles that already have an file."
        )
):
    """
    Train an agent on an environment.
    """

    load_extra_modules()

    # Load config.
    try:
        print(f"Loading profiles `{arg_profile_path}`... ", end="")
        profiles = read_profile(arg_profile_path)
        print("[bold green]success![/]")
    except RuntimeError:
        print("[bold red]failure![/]")
        raise typer.Abort()

    if arg_target_profile not in profiles:
        print(f"Profile {arg_target_profile} doesn't exist in {arg_profile_path}! Choices are;")
        print(profiles.keys())
        raise typer.Abort()

    target_profile = profiles[arg_target_profile]

    _train_with_profile(target_profile, halt=True, out_dir=opt_out_dir, save_path=opt_as, override=opt_override)


@app.command()
def batch(
        profile_path: str,
        opt_out_dir: str = typer.Option(
            ".",
            "--out",
            "-o",
            help="Directory to place trained agents."
        ),
        opt_override: bool = typer.Option(
            False,
            "--override",
            "-x",
            help="If false, don't run retrain profiles that already have an file."
        )
):
    """
    Train all configurations within a profile as a batch.
    """

    load_extra_modules()

    # Load config.
    try:
        print(f"Loading profiles `{profile_path}`... ", end="")
        profiles = read_profile(profile_path)
        print("[bold green]success![/]")
    except RuntimeError:
        print("[bold green]failure![/]")
        raise typer.Abort()

    # Train all profiles.
    for _, target_profile in profiles.items():
        _train_with_profile(target_profile, halt=False, out_dir=opt_out_dir, override=opt_override)


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

    load_extra_modules()

    opt_env_args = _convert_to_key_value_list(opt_env_args)

    env_factory = create_env_factory(arg_env, opt_wrapper, render_mode='rgb_array', env_args=opt_env_args)
    env = env_factory(opt_seed)
    gym_play(env, fps=opt_fps)


def _train_with_profile(
        target_profile: Profile,
        halt: bool = False,
        out_dir: str = ".",
        save_path: str = None,
        override=False
):

    if save_path is None:
        save_path = f"{target_profile.name}.cnc"

    path = os.path.join(out_dir, save_path)
    if override is False and os.path.exists(path):
        print(f"Profile {target_profile.name} already exists! To overwrite, --override, or set an alias with --as")
        return

    env_factory = create_env_factory(
        target_profile.env_id,
        target_profile.env_wrapper,
        env_args=target_profile.env_args
    )
    agent, training_wrapper = create_agent(env_factory, target_profile.agent_id)

    print(f"\nProceeding to train a {target_profile.agent_id} on {target_profile.env_id} with config values:")
    print(target_profile.config.to_dict())

    if halt:
        if typer.confirm("Ready to proceed?", default=True) is False:
            return

    training_artifacts = {}
    try:
        save_callback = create_save_callback(
            target_profile.agent_id,
            target_profile.config.to_dict(),
            target_profile.env_wrapper,
            target_profile.env_id,
            target_profile.env_args
        )

        training_wrapper(env_factory, agent, target_profile.config, training_artifacts, save_callback)
    except ProfilePropertyNotFound as ex:
        print_ex(ex)
    except KeyboardInterrupt:
        print("[bold red]Training interrupted[/bold red].")

    if halt:
        if typer.confirm("Save agent to disk?") is False:
            return

    try:
        print(f"Saving agent to disk ([italic]{path}[/]) ...")
        data = agent.save()
        concfile = ConcreteFile(
            id=target_profile.agent_id,
            env_name=target_profile.env_id,
            agent_data=data,
            training_artifacts=training_artifacts,
            used_wrappers=target_profile.env_wrapper,
            config=target_profile.config.to_dict(),
            env_args=target_profile.env_args
        )
        concfile.write(path)
    except OSError as ex:
        print("[bold red]Saving failed![/] " + ex.strerror)