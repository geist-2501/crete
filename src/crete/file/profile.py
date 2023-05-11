from dataclasses import dataclass
from typing import Dict, List, TypeVar, Optional

import yaml
from yaml import Loader

from ..error import ProfilePropertyNotFound


class ProfileConfig:

    def __init__(self, base_conf: Dict, profile_conf: Dict) -> None:
        self._conf = {**base_conf, **profile_conf}

    def getint(self, name: str, required=True) -> int:
        return self._get(name, int, required)

    def getstr(self, name: str, required=True) -> str:
        return self._get(name, str, required)

    def getfloat(self, name: str, required=True) -> float:
        return self._get(name, float, required)

    def getbool(self, name: str, required=True) -> bool:
        return self._get(name, self._to_bool, required)

    def getlist(self, name: str, required=True) -> List:
        return self._get(name, required=required)

    def _get(self, name: str, conv=None, required=True):
        if name not in self._conf:
            if required:
                raise ProfilePropertyNotFound(name)
            else:
                return None
        else:
            prop = self._conf[name]
            return conv(prop) if conv is not None else prop

    def to_dict(self) -> Dict:
        return dict(self._conf)  # Make a copy.

    @staticmethod
    def _to_bool(raw: str) -> bool:
        raw = str(raw)
        if raw in ['True', 'true', 'Yes', 'yes', '1']:
            return True
        elif raw in ['False', 'false', 'No', 'no', '0']:
            return False

        raise RuntimeError(f"'{raw}' is not a valid boolean property.")


@dataclass
class Profile:
    name: str
    agent_id: str
    env_id: str
    env_args: Dict
    env_wrapper: Optional[str]

    config: ProfileConfig


T = TypeVar('T')


def _get_or(d: Dict[str, T], key: str, other: T) -> T:
    return d[key] if key in d else other


def read_profile(path: str) -> Dict[str, Profile]:
    with open(path, 'r') as file:
        doc = yaml.load(file, Loader=Loader)

        assert type(doc) is dict

        base_conf = doc["defaults"]
        profiles = {}
        for profile_name, profile_doc in doc.items():
            if profile_name == "defaults":
                continue

            assert all([key in profile_doc for key in ("agent_id", "env_id", "config")])

            profile_conf = ProfileConfig(base_conf, profile_doc["config"])
            profile = Profile(
                name=profile_name,
                agent_id=profile_doc["agent_id"],
                env_id=profile_doc["env_id"],
                env_args=_get_or(profile_doc, "env_args", {}),
                env_wrapper=_get_or(profile_doc, "env_wrapper", None),
                config=profile_conf
            )

            profiles[profile_name] = profile

        return profiles
