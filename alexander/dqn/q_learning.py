from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import optim
from tqdm import tqdm

from dqn.buffer import ReplayBuffer
from dqn.hiv_patient import HIVPatient
from dqn.q_agent import Agent


@dataclass
class Epsilon:
    start: float
    end: float


@dataclass
class QLearningCongfig:
    gamma: float
    batch_size: int
    learning_rate: float
    num_episodes: int
    steps_per_episode: int
    epsilon: Epsilon
    target_update_rate: int


class DQN(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x.float())


class Qlearner:
    def __init__(
        self,
        conf: QLearningCongfig,
        patient: HIVPatient,
        agent: Agent,
        device: str,
    ) -> None:
        self.conf = conf

        self.patient = patient
        self.agent = agent
        self.device = device

        self.epsilon = self.conf.epsilon.start

        obs_size = self.patient.get_state_size()
        action_size = self.patient.get_action_size()

        self.policy_net = DQN(obs_size, action_size)
        self.target_net = DQN(obs_size, action_size)

        self.total_reward = 0
        self.episode_reward = 0

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.conf.learning_rate,
        )

    def dqn_loss(self, batch):
        states, actions, rewards, _, next_states = batch

        state_batch = torch.tensor(states)
        action_batch = torch.tensor(actions).unsqueeze(-1)
        reward_batch = torch.tensor(rewards)
        next_state_batch = torch.tensor(next_states)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        state_action_values = state_action_values.squeeze(-1)

        next_state_values, _ = torch.max(self.target_net(next_state_batch), dim=1)
        next_state_values = next_state_values.detach()

        exp_state_action_values = next_state_values * self.conf.gamma + reward_batch

        return self.loss_fn(state_action_values, exp_state_action_values)

    def gradient_step(self, memory: ReplayBuffer) -> torch.Tensor:
        if len(memory) < self.conf.batch_size:
            return

        batch = memory.sample(self.conf.batch_size)
        loss = self.dqn_loss(batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss

    def train(self, memory: ReplayBuffer):
        self.patient.reset()

        episode_rewards = []
        losses = []

        for episode in range(self.conf.num_episodes):
            self.patient.reset()
            episode_reward = 0
            episode_losses = []

            for step in tqdm(range(self.conf.steps_per_episode), unit="batch"):
                reward = self.agent.play_step(
                    self.policy_net,
                    self.epsilon,
                    self.device,
                )
                episode_reward += reward

                loss = self.gradient_step(memory)
                episode_losses.append(loss.item())

                if step % self.conf.target_update_rate == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            episode_rewards.append(episode_reward)
            logger.info(
                f"episode = {episode},  "
                f"reward = {episode_reward}, "
                f"average loss = {np.mean(episode_losses)}"
            )
            losses.extend(episode_losses)

        return episode_rewards, losses
