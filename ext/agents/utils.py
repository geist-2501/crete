from typing import List, Callable

import gymnasium as gym
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

from crete import Agent


class StaticLinearDecay:
    def __init__(self, start_value, final_value, max_steps):
        self.start_value = start_value
        self.final_value = final_value
        self.max_steps = max_steps

    def get(self, step):
        step = min(step, self.max_steps)
        upper = self.start_value * (self.max_steps - step)
        lower = self.final_value * step
        return (upper + lower) / self.max_steps


def smoothen(data):
    return uniform_filter1d(data, size=30)


def evaluate(
        env: gym.Env,
        agent: Agent,
        n_episodes=1,
        max_episode_steps=10000
) -> float:
    total_ep_rewards = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total_ep_reward = 0
        extra_state = None
        for _ in range(max_episode_steps):
            a, extra_state = agent.get_action(s, extra_state)
            s, r, done, _, _ = env.step(a)
            total_ep_reward += r

            if done:
                break

        total_ep_rewards.append(total_ep_reward)
    return np.mean(total_ep_rewards).item()


def label_values(
        values: np.ndarray,
        name_func: Callable[[int], str] = None,
        name_list: List[str] = None
) -> List[str]:
    out = []
    if name_func is not None:
        names = [name_func(action) for action in range(len(values))]
    else:
        names = name_list

    for name, value in zip(names, values):
        out.append(f"{name}: {value:.2f}")

    return out
