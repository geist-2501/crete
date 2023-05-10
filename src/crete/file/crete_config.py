from dataclasses import dataclass, field
from typing import List

import yaml
from yaml import Loader


@dataclass()
class CreteConfig:
    conf_filename = ".crete.yaml"

    extra_modules: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "config": {
                "extra_modules": self.extra_modules
            }
        }

    def write(self, path: str):
        with open(path, 'w') as file:
            yaml.dump(self.to_dict(), file)

    @classmethod
    def read(cls, path: str):
        with open(path, 'r') as file:
            doc = yaml.load(file, Loader=Loader)
            assert type(doc) is dict
            return cls(**doc["config"])

    @classmethod
    def try_read(cls, path: str):
        try:
            return cls.read(path)
        except OSError:
            return cls()
