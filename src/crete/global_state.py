from dataclasses import dataclass

_can_graph = True

try:
    import tkinter
except ModuleNotFoundError:
    import matplotlib as mpl
    print("Using headless backend!")
    mpl.use("agg")
    _can_graph = False


@dataclass
class GlobalState:
    debug_mode = False
    can_graph = _can_graph


cli_state = GlobalState()


def get_cli_state() -> GlobalState:
    global cli_state
    return cli_state
