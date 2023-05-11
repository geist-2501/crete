from collections import defaultdict
from typing import List
from rich import print

import numpy as np

from crete.error import CreteException


def print_err(text: str):
    print(f"[bold red]Error[/]\t{text}")


def print_ex(ex: CreteException):
    text = f"[bold red]Exception\t{ex.name}[/]"
    if ex.message is not None:
        print(f"{text} | {ex.message}")
    else:
        print(text)


def pad(array, pad_value=0) -> np.ndarray:
    dimensions = get_max_shape(array)
    result = np.full(dimensions, pad_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result


def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError:
        pass


def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError:
        yield (*index, slice(len(array))), array


def get_max_shape(array):
    dimensions = defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


def std_err(data) -> float:
    return np.std(data, ddof=1) / np.sqrt(np.size(data))


def parse_int_list(raw: str) -> List[int]:
    parts = raw.split(",")
    return list(map(lambda part: int(part), parts))


def to_camel_case(text: str, delim: str = "_") -> str:
    parts = text.split(delim)
    new_text = [parts[0], *[p.capitalize() for p in parts[1:]]]
    return ''.join(new_text)