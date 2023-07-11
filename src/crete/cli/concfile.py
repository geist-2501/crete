from typing import List, Optional

import numpy as np
import typer
from gymnasium.wrappers import RecordVideo
from rich import print

from ..file.profile import ProfileConfig
from .cli_utils import _convert_to_key_value_list
from ..core import create_env_factory, create_agent, play_agent, evaluate_agents, graph_agent, graph_env_results, \
    dump_scores_to_csv, load_extra_modules
from ..error import ConcfileLoadError, AgentNotFound
from ..file.concrete import ConcreteFile

concfile_app = typer.Typer()


@concfile_app.callback()
def doc():
    """Perform operations on Concrete files (.cnc), like replaying and viewing."""
    pass


@concfile_app.command()
def view(path: str):
    """View the metadata of a concfile"""
    try:
        concfile = ConcreteFile.read(path)
    except ConcfileLoadError as ex:
        print(f"Couldn't load concfile {path}, " + str(ex))
        raise typer.Abort()

    print(f"[bold white]Agent name[/]:       {concfile.id}")
    print(f"[bold white]Environment used[/]: {concfile.env_name}")
    print(f"[bold white]Wrapper used[/]:     {concfile.used_wrappers}")
    print(f"[bold white]Env args[/]:         {concfile.env_args}")
    print("[bold white]Configuration used[/]:")
    print(concfile.config)


@concfile_app.command()
def replay(
        path: str,
        opt_env: str = typer.Option(
            None,
            "--env",
            "-e"
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
        opt_save_to: Optional[str] = typer.Option(
            None,
            "--save-to",
            "-s"
        ),
        opt_num_eps: int = typer.Option(
            1,
            "--num-eps",
            "-n"
        ),
        opt_render: str = typer.Option(
            "human",
            "--render-as"
        ),
        opt_target_fps: int = typer.Option(
            30,
            "--fps"
        ),
        opt_wait_for_keypress: bool = typer.Option(
            False,
            "--wait-for-keypress",
            "-k"
        )
):
    """Play the agent in the environment it was trained in."""

    load_extra_modules()

    opt_env_args = _convert_to_key_value_list(opt_env_args)

    try:
        concfile = ConcreteFile.read(path)
    except ConcfileLoadError as ex:
        print(f"Couldn't load concfile {path}, " + str(ex))
        raise typer.Abort()

    if opt_env is None:
        opt_env = concfile.env_name
    else:
        print(f"Overwriting environment specified in concfile ({concfile.env_name}) with {opt_env}")

    if opt_wrapper is None:
        opt_wrapper = concfile.used_wrappers
    else:
        print(f"Overwriting wrapper specified in concfile ({concfile.used_wrappers}) with {opt_wrapper}")

    opt_env_args = {**opt_env_args, **concfile.env_args}

    env_factory = create_env_factory(opt_env, opt_wrapper, render_mode=opt_render, env_args=opt_env_args)
    agent, _ = create_agent(env_factory, concfile.id)
    agent.load(concfile.agent_data)

    env = env_factory(opt_seed)

    if opt_save_to:
        file_prefix = f"{agent.name}-{opt_env}"
        print(f"Recording to {opt_save_to}/{file_prefix}")
        env = RecordVideo(
            env=env,
            video_folder=opt_save_to,
            episode_trigger=lambda ep_num: True,
            name_prefix=file_prefix
        )

    try:
        for _ in range(opt_num_eps):
            reward_history = play_agent(agent, env, wait_time=1/opt_target_fps, wait_for_keypress=opt_wait_for_keypress)
            print(f"Final score: {sum(reward_history):.2f}")
    except KeyboardInterrupt:
        env.close()
        raise typer.Abort()

    env.close()


@concfile_app.command()
def graph(path: str):
    """Produce graphs of the agent gathered during training."""

    load_extra_modules()

    try:
        concfile = ConcreteFile.read(path)
    except ConcfileLoadError as ex:
        print(f"Couldn't load concfile {path}, " + str(ex))
        raise typer.Abort()

    graph_agent(concfile.id, concfile.training_artifacts, ProfileConfig(concfile.config, {}))


@concfile_app.command()
def compare(
        paths: List[str] = typer.Argument(
            None,
            help="Concfiles of agents to compare against each other.cnc."
        ),
        opt_env_args: List[str] = typer.Option(
            [],
            "--env-arg",
            help="Arguments to pass to the environment in the form `param=value`."
        ),
        opt_n_episodes: int = typer.Option(
            3,
            "--num-episodes",
            "-n",
            help="The number of evaluation episodes to run for each agent."
        ),
        opt_save_as: Optional[str] = typer.Option(
            None,
            "--save-as",
            "-s",
            help="Prefix to save the graphs and data as."
        )
):
    """
    Compare several agents against each other.cnc in an environment.
    Agents must have a common environment, but can have different wrappers.
    """

    load_extra_modules()

    opt_env_args = _convert_to_key_value_list(opt_env_args)

    loaded_agents = []
    for agent_concfile in paths:
        print(f" > {agent_concfile}... ", end="")
        try:
            # Load concfile.
            concfile = ConcreteFile.read(agent_concfile)

            # Recreate the env factory and wrapper.
            env_args = {**concfile.env_args, **opt_env_args}
            agent_env_factory = create_env_factory(concfile.env_name, concfile.used_wrappers, env_args=env_args)
            agent, _ = create_agent(agent_env_factory, concfile.id)
            agent.load(concfile.agent_data)
            loaded_agents.append({
                "agent_name": f"{agent_concfile} ({concfile.id})",
                "agent_id": concfile.id,
                "agent": agent,
                "env_name": concfile.env_name,
                "env_factory": agent_env_factory
            })
            extra_info = f"Uses {concfile.env_name}"
            extra_info += f" with wrapper {concfile.used_wrappers}." if concfile.used_wrappers else "."
            print(f"[bold green]success![/] {extra_info}")
        except RuntimeError as ex:
            print("[bold red]failed![/] Couldn't load .cnc file. " + str(ex))
        except AgentNotFound:
            print("[bold red]failed![/] Couldn't find agent definition. Make sure it's been registered.")

    # Check all the environments are the same.
    common_env_id = loaded_agents[0]["env_name"]
    if not all(agent["env_name"] == common_env_id for agent in loaded_agents):
        print("[bold red]Failure![/] Agents don't all use the same environment. Cannot proceed.")
        raise typer.Abort()

    if len(loaded_agents) == len(paths):
        should_continue = typer.confirm("All agents loaded, ready to proceed?", default=True)
    else:
        should_continue = typer.confirm("Only some agents loaded, ready to proceed?", default=False)

    if should_continue:
        scores = evaluate_agents(loaded_agents, n_episodes=opt_n_episodes)
        graph_env_results(common_env_id, opt_env_args, loaded_agents, scores)
        if opt_save_as is not None:
            dump_scores_to_csv(f"{opt_save_as}.csv", [a["agent_id"] for a in loaded_agents], scores)


@concfile_app.command()
def prune(
        arg_concfile_path: str = typer.Argument(
            ...,
            help="Path to concfile to edit."
        ),
        opt_artifact_name: Optional[str] = typer.Option(
            None,
            "--name",
            "-n",
            help="Name of artifact to prune. If nested, use notation `name.index`"
        ),
        opt_prune_on: Optional[int] = typer.Option(
            None,
            "--on",
            "-o",
            help="Number of steps taken between removing a frame in the artifact."
        )
):
    """
    Removes data from a concfile's training artifacts to reduce it's filesize.
    """
    try:
        # Load concfile.
        print(f" > Loading {arg_concfile_path}... ", end="")
        concfile = ConcreteFile.read(arg_concfile_path)
        print(f"[bold green]success![/]")

    except RuntimeError as ex:
        print("[bold red]failed![/] Couldn't load .conc file. " + str(ex))
        raise typer.Abort()

    if opt_artifact_name is None:
        # List sizes.
        print("Concfile size:")
        for artifact_name, artifact_values in concfile.training_artifacts.items():
            if type(artifact_values) is tuple:
                print(f" {artifact_name} -> ", end="")
                for tuple_val in artifact_values:
                    print(f"({len(tuple_val)}, {type(tuple_val)}) ", end="")
                print()
            else:
                print(f" {artifact_name} -> {len(artifact_values)}, {type(artifact_values)}")
    else:
        assert opt_prune_on is not None, "Must have a prune on rate if pruning!"
        path = opt_artifact_name.split('.')
        artifact_part = concfile.get_artifact(path)

        artifact_part_len = len(artifact_part)
        print(f" {opt_artifact_name} -> {artifact_part_len} entries")
        est_len = artifact_part_len - artifact_part_len / opt_prune_on
        print(f" Pruning on {opt_prune_on} would leave ~{est_len :.2f} entries")

        typer.confirm("Continue?", default=True, abort=True)

        pruned_part = np.array([x for i, x in enumerate(artifact_part) if i % opt_prune_on == 0])

        concfile.set_artifact(path, pruned_part)
        concfile.write(arg_concfile_path)

        print(f"Prune complete, left with {len(pruned_part)} entries.")


@concfile_app.command()
def squeeze(
        arg_concfile_path: str = typer.Argument(
            ...,
            help="Path to concfile to edit."
        )
):
    """
    Removes excess array dimensions from a concfile's artifacts.
    """
    try:
        # Load concfile.
        print(f" > Loading {arg_concfile_path}... ", end="")
        concfile = ConcreteFile.read(arg_concfile_path)
        print(f"[bold green]success![/]")

    except RuntimeError as ex:
        print("[bold red]failed![/] Couldn't load .conc file. " + str(ex))
        raise typer.Abort()

    concfile.artifact_apply(lambda l: np.squeeze(l))
    concfile.write(arg_concfile_path)

    print(f"Squeeze complete.")
