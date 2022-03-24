#!/usr/bin/env python

import gym
from gym.wrappers import FrameStack
from hiv_patient import HIVPatient

import matplotlib.pyplot as plt

from models import GymFF
from models import AtariCNN
from utils import greedy_action
from wrappers import WarpFrame

import numpy as np
import torch
from copy import deepcopy
from tqdm import trange
import logging

from buffer import ReplayBuffer
from utils import plot_traj
from utils import plot_gradients


class DQN_agent:
    """
        TODO
    """

    def __init__(self, config, model, device="cpu"):
        """
            TODO
        """
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.nb_actions = config["nb_actions"]
        self.memory = ReplayBuffer(config["buffer_size"])
        self.learn_from = config["learn_from"]
        self.ep_length = config["ep_length"]
        self.patient_learning_mode = config["patient_learning_mode"]
        self.patient_learning_extras = config["patient_learning_extras"]
        self.epsilon_max = config["epsilon_max"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_period = config["epsilon_decay_period"]
        self.epsilon_delay = config["epsilon_delay_decay"]
        self.epsilon_step = (
            (self.epsilon_max-self.epsilon_min)/self.epsilon_period
        )
        self.total_steps = 0
        self.model = model
        self.target = deepcopy(self.model).to(device)
        self.criterion = config["criterion"]
        self.optimizer = config["optimizer"](
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.nb_gradient_steps = config["nb_gradient_steps"]
        self.target_update = config["target_update"]
        self.evaluate_period = config["evaluate_period"]
        self.evaluation_nb_eps = config["evaluation_nb_eps"]
        self.gradient_viz_period = config["gradient_viz_period"]

    def gradient_step(self):
        """
            TODO
        """
        if len(self.memory) > self.learn_from:
            x, a, r, y, d = self.memory.sample(self.batch_size)
            QYmax = self.model(y).max(1)[0].detach()
            update = torch.addcmul(r, self.gamma, 1-d, QYmax)
            QXA = self.model(x).gather(1, a.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            grad_dict = {k: v.grad.flatten() for k, v in self.model.named_parameters()}
            return loss.detach().item(), grad_dict
        else:
            return -1, {k: torch.zeros(size=v.shape).flatten() for k, v in self.model.named_parameters()}

    def train(self, env, max_episode):
        """
            TODO
        """
        episode_return = []
        epsilon = self.epsilon_max
        step = 0

        logging.info("Episode,  eps, return,      loss")
        main_loop = trange(max_episode)
        for episode in main_loop:
            state = env.reset(mode=self.patient_learning_mode, extras=self.patient_learning_extras)
            ep_cum_reward = 0
            ep_loss = 0
            grad_dict = {k: torch.zeros(size=v.shape).flatten().unsqueeze(0) for k, v in self.model.named_parameters()}
            for _ in trange(self.ep_length, leave=False):
                # update epsilon
                if step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

                # select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.nb_actions)
                else:
                    action = greedy_action(self.model, state)

                # step
                next_state, reward, done, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                ep_cum_reward += reward

                # update target every now and then
                if step % self.target_update == 0:
                    self.target.load_state_dict(self.model.state_dict())

                state = next_state
                step += 1

                # train
                for _ in range(self.nb_gradient_steps):
                    loss, gradients = self.gradient_step()
                    ep_loss += loss
                    for k in grad_dict:
                        grad_dict[k] = torch.cat((grad_dict[k], gradients[k].unsqueeze(0)), dim=0)

            ep_loss = ep_loss / (self.ep_length * self.nb_gradient_steps)
            logging.info(
                f"{episode:>7d}, {epsilon:>4.2f}, {ep_cum_reward:>6.1f}, {ep_loss:>.2E}",
            )
            episode_return.append(ep_cum_reward)
            if (episode+1) % self.evaluate_period == 0:
                states, rews, actions = self.evaluate(nb_eps=self.evaluation_nb_eps)
                plot_traj(states, actions)
            if (episode+1) % self.gradient_viz_period == 0:
                plot_gradients([grad_dict[k] for k in grad_dict])
                gradients = []
                # uncomment this to print gradient mean and std every now and then.
                # gradients = [grad_dict[k] for k in grad_dict]
                for grad in gradients:
                    print(torch.mean(grad, dim=0))
                    print(torch.std(grad, dim=0))

        return episode_return

    def evaluate(
            self,
            dur=750,
            nb_eps=10,
            patient_kwargs=dict(mode="unhealthy", extras="small-infection:immunity-failure")):
        """
            TODO
        """
        patient = HIVPatient(clipping=False, logscale=False)
        x = patient.reset(**patient_kwargs)
        dur = dur // 5
        states = [x]
        rewards = []
        actions = []
        eps = trange(nb_eps, desc="evaluation", leave=False)
        for ep in eps:
            t = trange(dur, desc=f"ep no {ep+1}", leave=False)
            for i in t:
                a = self.model(torch.from_numpy(x).float()).detach()
                actions.append(torch.argmax(a).item())
                t.set_description(desc=f"{a} ({np.argmax(a)})")
                y, r, d, _ = patient.step(actions[-1])
                states.append(x)
                rewards.append(r)
                x = y
            x = patient.reset(**patient_kwargs)
        return states, np.sum(rewards), actions


def main():
    """
        TODO
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = "hiv"
    train_nb_eps = 1000
    test_nb_steps = 100
    plot = True
    if env == "pong":
        env = gym.make("Pong-v4")
        env = WarpFrame(env)
        env = FrameStack(env, 4)
        model = AtariCNN(
            in_channels=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            device=device,
        )
        nb_actions = env.action_space.n
    elif env == "cartpole":
        env = gym.make("CartPole-v1")
        model = GymFF(
            state_dim=env.observation_space.shape[0],
            n_action=env.action_space.n,
            nb_neurons=24,
            device=device,
        )
        nb_actions = env.action_space.n
    elif env == "hiv":
        env = HIVPatient(clipping=True, logscale=True)
        model = GymFF(
            state_dim=6,
            n_action=4,
            nb_neurons=24,
            device=device,
        )
        nb_actions = 4
    else:
        print(f"Unknown option '{env}'.")
        exit()

    config = {
        "nb_actions": nb_actions,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "gamma": 0.98,
        "buffer_size": 256_000,
        "learn_from": 6_000,
        "epsilon_min": 0.15,
        "epsilon_max": 1.00,
        "epsilon_decay_period": 10_000,
        "epsilon_delay_decay": 20,
        "batch_size": 512,
        "criterion": torch.nn.SmoothL1Loss(),
        "optimizer": torch.optim.Adam,
        "nb_gradient_steps": 1,
        "target_update": 1024,
        "ep_length": 200,
        "gradient_viz_period": 300000,
        "evaluate_period": 30,
        "patient_learning_mode": "unhealthy",
        # "patient_learning_extras": "small-infection:immunity-failure",
        "patient_learning_extras": "",
        "evaluation_nb_eps": 1,
    }

    agent = DQN_agent(config, model, device=device)
    scores = agent.train(env, max_episode=train_nb_eps)
    if plot:
        plt.style.use("dark_background")
        plt.plot(scores)
        plt.show()

    if test_nb_steps > 0:
        x = env.reset()
        env.render()
        for _ in trange(test_nb_steps):
            a = greedy_action(model, x)
            y, _, d, _ = env.step(a)
            env.render()
            x = env.reset() if d else y

    env.close()


if __name__ == "__main__":
    plt.style.use("dark_background")
    logging.basicConfig(
        filename="out.log",
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    main()
