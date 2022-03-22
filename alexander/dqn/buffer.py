from collections import deque, namedtuple
from typing import Tuple

import numpy as np
from tqdm import tqdm

from dqn.hiv_patient import HIVPatient

Experience = namedtuple(
    "Experience",
    field_names=[
        "state",
        "action",
        "reward",
        "done",
        "new_state",
    ],
)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, new_states = zip(
            *(self.buffer[idx] for idx in indices)
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(new_states),
        )

    def sample_experience(self) -> Experience:
        idx = np.random.randint(0, len(self.buffer))
        return self.buffer[idx]

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for experience in self.buffer:
            state, action, reward, _, _ = experience
            x.append(np.concatenate([state, action]))
            y.append(reward)
        return np.array(x), np.array(y)

    def populate(
        self,
        patient,
        num_episodes,
        steps_per_episode,
        patient_mode="unhealthy",
        one_hot=False,
    ):
        for _ in tqdm(range(num_episodes)):
            state = patient.reset(mode=patient_mode)
            for _ in range(steps_per_episode):
                act_index = np.random.randint(4)
                new_state, reward, done, _ = patient.step(act_index)
                if one_hot:
                    action = np.zeros(4)
                    action[act_index] = 1
                else:
                    action = act_index
                experience = Experience(state, action, reward, done, new_state)
                self.append(experience)
                state = new_state


if __name__ == "__main__":

    import torch

    replay_buffer = ReplayBuffer(100)
    replay_buffer.populate(HIVPatient(), 10, 10)

    states, actions, rewards, _, _ = replay_buffer.sample(batch_size=8)

    print(type(states))
    print(states.shape)

    print(actions)
    print(type(actions))
    print(actions.dtype)