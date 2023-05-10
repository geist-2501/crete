import typer

from crete.file.crete_config import CreteConfig
from crete.util import print_err

module_app = typer.Typer()


@module_app.command()
def add(
        arg_module_name: str = typer.Argument(
            ...,
            help="Path of module to add for Crete to load."
        )
):
    # Load the .crete file.
    conf = CreteConfig.try_read(CreteConfig.conf_filename)

    # Add to it.
    if arg_module_name not in conf.extra_modules:
        conf.extra_modules.append(arg_module_name)
    else:
        print_err(f"Path {arg_module_name} already exists in `{CreteConfig.conf_filename}`")

    # Overwrite it.
    conf.write(CreteConfig.conf_filename)


@module_app.command()
def remove(
        arg_module_name: str = typer.Argument(
            ...,
            help="Path of module to add for Crete to load."
        )
):
    # Load the .crete file.
    conf = CreteConfig.try_read(CreteConfig.conf_filename)

    # Remove from it.
    if arg_module_name in conf.extra_modules:
        conf.extra_modules.remove(arg_module_name)
    else:
        print_err(f"Path {arg_module_name} does not exist in `{CreteConfig.conf_filename}`")

    # Overwrite it.
    conf.write(CreteConfig.conf_filename)


@module_app.command(name="list")
def list_modules():
    # Load the .crete file.
    conf = CreteConfig.try_read(CreteConfig.conf_filename)
    print(conf.extra_modules)