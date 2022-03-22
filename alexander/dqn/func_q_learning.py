import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from alexander.dqn.buffer import ReplayBuffer
from alexander.dqn.hiv_patient import HIVPatient
from alexander.dqn.q_agent import Agent
from dataclasses import dataclass


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


@dataclass
class Epsilon:
    max: float
    min: float
    delay: int


@dataclass
class QLearningCongfig:
    gamma: float
    batch_size: int
    learning_rate: float
    num_episodes: int
    steps_per_episode: int
    target_update_rate: int


def anneal_cos(start, end, pct):
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def anneal_linear(start, end, pct):
    return (end - start) * pct + start


def loss_dqn(
    batch: Tuple[np.array, ...],
    gamma: float,
    loss_fn: Callable,
    policy_net: nn.Module,
    target_net: nn.Module,
) -> torch.Tensor:
    states, actions, rewards, _, next_states = batch

    state_batch = torch.tensor(states)
    action_batch = torch.tensor(actions).unsqueeze(-1)
    reward_batch = torch.tensor(rewards)
    next_state_batch = torch.tensor(next_states)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_action_values = state_action_values.squeeze(-1)

    next_state_values, _ = torch.max(target_net(next_state_batch), dim=1)
    next_state_values = next_state_values.detach()

    exp_state_action_values = next_state_values * gamma + reward_batch
    return loss_fn(state_action_values, exp_state_action_values)


def gradient_step_dqn(
    memory: ReplayBuffer,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    policy_net: nn.Module,
    target_net: nn.Module,
    loss_fn: Callable,
) -> torch.Tensor:
    if len(memory) < batch_size:
        return

    batch = memory.sample(batch_size)

    loss = loss_dqn(
        batch=batch,
        gamma=gamma,
        loss_fn=loss_fn,
        policy_net=policy_net,
        target_net=target_net,
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # for param in self.policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)

    optimizer.step()
    return loss


def train_dqn(
    patient: HIVPatient,
    memory: ReplayBuffer,
    agent: Agent,
    policy_net: nn.Module,
    target_net: nn.Module,
    q_conf: QLearningCongfig,
    eps_conf: Epsilon,
):

    optimizer = torch.optim.Adam(
        params=policy_net.parameters(),
        lr=q_conf.learning_rate,
    )
    loss_fn = nn.SmoothL1Loss()

    patient.reset()

    episode_rewards = []
    losses = []
    epsilons = []

    global_step = 0

    epsilon = eps_conf.max

    for episode in tqdm(range(q_conf.num_episodes), unit="episode"):
        patient.reset()
        episode_reward = 0

        if global_step > eps_conf.delay:
            epsilon = anneal_cos(
                eps_conf.max,
                eps_conf.min,
                episode / q_conf.num_episodes,
            )

        for _ in range(q_conf.steps_per_episode):

            reward, done = agent.play_step(policy_net, epsilon, memory)
            episode_reward += reward

            loss = gradient_step_dqn(
                memory=memory,
                batch_size=q_conf.batch_size,
                optimizer=optimizer,
                gamma=q_conf.gamma,
                policy_net=policy_net,
                target_net=target_net,
                loss_fn=loss_fn,
            )
            losses.append(loss.item())
            epsilons.append(epsilon)

            if global_step % q_conf.target_update_rate == 0:
                target_net.load_state_dict(policy_net.state_dict())

            global_step += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        # logger.info(
        #     f"episode = {episode},  "
        #     f"reward = {episode_reward}, "
        #     f"average loss = {np.mean(episode_losses)}"
        # )

    return episode_rewards, losses, epsilons
