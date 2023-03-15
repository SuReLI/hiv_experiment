import numpy as np
import random
from torch import Tensor
from typing import List


class ReplayBuffer:
    """
        A dynamic memory to store RL transitions to be used in the DQN
        algorithm.
    """

    def __init__(self, capacity: int):
        """
            Give a maximum capacity for the replay memory.
        """
        self.capacity = capacity  # capacity of the buffer.
        self.data = []            # the actual data transitions.
        self.index = 0            # index of the next cell to be filled.

    def append(self, x, a, r, y, d):
        """
            Appends a (s, a, r, s', d) transition to the replay buffer.
            Wraps around the FIFO structure when full of transitions.
        """
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (x, a, r, y, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int, flatten: bool = True, device: str = "cpu") -> List[Tensor]:
        """
            Returns a sample from the replay buffer to learn from it.
            The batch size can be changed and the device can be forced
            to something different from the 'cpu'.

            Returns an array of the form:
                [
                    [x1, ..., xB],  # B samples of the state.
                    [a1, ..., aB],  # B samples of the action from x.
                    [r1, ..., rB],  # B samples of the reward from (x, a).
                    [y1, ..., yB],  # B samples of the next state from (x, a).
                    [d1, ..., dB],  # B samples of the terminality of y.
                ]
                with B being the size of the batch.
        """
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: Tensor(x).to(device), list(zip(*batch)))) if flatten else np.array(batch, dtype=object)

    def __len__(self) -> int:
        """
            Gives the length of the replay buffer.
            Grows from 0 to the capacity of the buffer.
        """
        return len(self.data)


# if __name__ == "__main__":
#     # Testing insertion in the ReplayBuffer class
#     from tqdm import trange

#     replay_buffer_size = int(1e4)
#     nb_samples = int(2e4)
#     memory = ReplayBuffer(replay_buffer_size)

#     cartpole = gym.make('CartPole-v1')
#     state = cartpole.reset()

#     for _ in trange(nb_samples):
#         action = cartpole.action_space.sample()
#         next_state, reward, done, _ = cartpole.step(action)
#         memory.append(state, action, reward, next_state, done)
#         if done:
#             state = cartpole.reset()
#         else:
#             state = next_state

#     print(len(memory))

#     # Testing sampling in the ReplayBuffer class
#     nb_batches = int(1e4)
#     batch_size = 50

#     for _ in trange(nb_batches):
#         batch = memory.sample(batch_size)

#     for sample in memory.sample(2):
#         print(sample)
