# gymnasium is recommended for RLLIB
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from hiv_patient import HIVPatient


# Define the environment that extends HIVPatient and gym.Env
class HIVPatientGym(HIVPatient, gym.Env):
    def __init__(self, clipping=True, logscale=False):
        super().__init__(clipping, logscale)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=float("-inf"), high=float("inf"), shape=(6,), dtype=np.float32
        )
        self._counter_step = 0

    def step(self, a_index):
        state, reward, done, info = super().step(a_index)

        if info is None:
            info = {}

        self._counter_step += 1
        if self._counter_step >= 200:
            self._counter_step = 0
            done = True

        return state, reward, done, info

    def reset(self, *args, **kwargs):
        self._counter_step = 0
        return super().reset(*args, **kwargs)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
