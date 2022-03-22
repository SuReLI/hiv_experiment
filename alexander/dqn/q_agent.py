import numpy as np
import torch
from torch import nn

from dqn.buffer import Experience, ReplayBuffer
from dqn.hiv_patient import HIVPatient


class Agent:
    def __init__(self, patient: HIVPatient) -> None:
        self.patient = patient

        self.state = None
        self.reset()

    def reset(self) -> None:
        self.state = self.patient.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        if np.random.random() > epsilon:
            action = self.patient.sample_action_space()
        else:
            state = torch.tensor(np.array([self.state]), device=device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
        return action

    def play_step(
        self,
        net: nn.Module,
        epsilon: float,
        device: str,
        replay_buffer: ReplayBuffer,
    ) -> float:
        with torch.no_grad():
            action = self.get_action(net, epsilon, device)

            # do step in the environment
            new_state, reward, done, _ = self.patient.step(action)
            exp = Experience(self.state, action, reward, done, new_state)
            replay_buffer.append(exp)

            self.state = new_state
            return reward
