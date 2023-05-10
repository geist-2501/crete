from dataclasses import dataclass


@dataclass
class GlobalState:
    debug_mode = False


cli_state = GlobalState()


def get_cli_state() -> GlobalState:
    global cli_state
    return cli_state
