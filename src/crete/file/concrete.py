import pickle
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Callable
from zipfile import ZipFile

import numpy as np
import torch

from crete.error import ConcfileLoadError


@dataclass()
class ConcreteFile:
    id: str
    agent_data: bytes

    training_artifacts: Dict
    config: Dict
    used_wrappers: str = None
    env_name: str = None
    env_args: Dict = field(default_factory=dict)

    @classmethod
    def read(cls, path: str) -> Any:
        try:
            with ZipFile(path, 'r') as file:
                meta_bytes = file.read("meta")
                data_bytes = file.read("data")

                meta = pickle.loads(meta_bytes)

                return ConcreteFile(**{**meta, "agent_data": data_bytes})

        except OSError as ex:
            raise ConcfileLoadError(str(ex))

    def write(self, path: str):
        with ZipFile(path, 'w') as file:
            file_meta = pickle.dumps({
                "id": self.id,
                "training_artifacts": self.training_artifacts,
                "config": self.config,
                "used_wrappers": self.used_wrappers,
                "env_name": self.env_name,
                "env_args": self.env_args
            })
            file.writestr("meta", file_meta)
            file.writestr("data", self.agent_data)

    def get_artifact(self, path: List[str]) -> Any:
        root = self.training_artifacts
        for part in path:
            if type(root) is dict:
                root = root[part]
            elif type(root) is tuple:
                root = root[int(part)]
            else:
                raise RuntimeError("No more appropriate entries to index!")

        return root

    def artifact_apply(self, func: Callable[[Any], Any]):
        for k, v in self.training_artifacts.items():
            if type(v) is tuple:
                new_v = tuple(func(x) for x in v)
            elif type(v) is list or type(v) is np.ndarray:
                new_v = func(v)
            else:
                raise RuntimeError

            self.training_artifacts[k] = new_v

    def set_artifact(self, path: List[str], data: Any):
        root = self.training_artifacts
        self._set_artifact_rec(path, 0, [root], data)

    def _set_artifact_rec(self, path: List[str], depth: int, roots: List, data: List):
        root = roots[-1]
        if depth == len(path) - 1:
            # Set.
            if type(root) is dict:
                key = path[depth]
                root[key] = data
            elif type(root) is tuple:
                # Have to go back a step and replace the whole tuple since they're immutable.
                key = int(path[depth])
                data_tuple = root[:key] + tuple([data]) + root[key+1:]
                self._set_artifact_rec(path[:-1], depth - 1, roots[:-1], data_tuple)
            else:
                raise RuntimeError("Cannot set on anything but a dict or tuple!")

        else:
            # Recurse.
            if type(root) is dict:
                key = path[depth]
                new_root = root[key]
                self._set_artifact_rec(path, depth + 1, [*roots, new_root], data)
            elif type(root) is tuple:
                key = int(path[depth])
                new_root = root[key]
                self._set_artifact_rec(path, depth + 1, [*roots, new_root], data)
            else:
                raise RuntimeError("Invalid path!")
