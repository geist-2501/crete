from typing import List, Dict, Tuple, Optional


def _convert_to_key_value_list(args: List[str]) -> Dict[str, str]:

    key_values = {}
    for arg in args:
        parts = arg.split('=')
        assert len(parts) == 2
        key, value = parts
        key_values[key] = value

    return key_values


def _convert_to_key_list_value(arg: Optional[str]) -> Optional[Tuple[str, List[str]]]:
    if arg is None:
        return None

    parts = arg.split("=")
    assert len(parts) == 2, "Require a key-value pair, like `etc=1,2,3"
    key, val = parts
    vals = val.split(",")

    return key, vals
