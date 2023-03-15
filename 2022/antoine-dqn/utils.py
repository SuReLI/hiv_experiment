#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from hiv_patient import HIVPatient

from tsne_torch import TorchTSNE as TSNE


def plot_traj(traj, actions):
    """
        TODO
    """
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    states = np.array(traj)
    axs[0, 0].semilogy(states[:, 0])
    axs[0, 0].set_title("T1")
    axs[0, 1].semilogy(states[:, 2])
    axs[0, 1].set_title("T2")
    axs[1, 0].semilogy(states[:, 1])
    axs[1, 0].set_title("T1*")
    axs[1, 1].semilogy(states[:, 3])
    axs[1, 1].set_title("T2*")
    axs[2, 0].semilogy(states[:, 4])
    axs[2, 0].set_title("V")
    axs[2, 1].semilogy(states[:, 5])
    axs[2, 1].set_title("E")
    RTI = [0.7 * (a > 1) for a in actions]
    axs[3, 0].plot(RTI)
    axs[3, 0].set_ylim([-0.1, 1.1])
    axs[3, 0].set_title("RTI")
    PI = [0.3 * (a % 2) for a in actions]
    axs[3, 1].set_ylim([-0.1, 1.1])
    axs[3, 1].plot(PI)
    axs[3, 1].set_title("PI")
    plt.show()


def plot_gradients(gradients):
    """
        TODO
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True)
    fig, axs = plt.subplots(2, 3, figsize=(15, 15))
    for grad, ax in zip(gradients, axs.flatten()):
        x_emb = tsne.fit_transform(grad)
        ax.scatter(x_emb[:, 0], x_emb[:, 1])
    plt.show()


def simulate(dur=400):
    """
        TODO
    """
    patient = HIVPatient(clipping=False, logscale=False)
    s = patient.reset(mode="healthy", extras="immunity-failure")
    dur = dur // 5
    states = [s]
    for i in trange(dur):
        s, r, d, _ = patient.step(0)
        states.append(s)

    return states


def evaluate(model, dur=750):
    """
        TODO
    """
    patient = HIVPatient(clipping=False, logscale=False)
    x = patient.reset(mode="unhealthy", extras="small-infection:immunity-failure")
    dur = dur // 5
    states = [x]
    rewards = []
    actions = []
    t = trange(dur, desc="evaluation", leave=False)
    for i in t:
        a = model([np.concatenate((x, a_)) for a_ in np.eye(4)])
        actions.append(np.argmax(a))
        t.set_description(desc=f"{a} ({np.argmax(a)})")
        y, r, d, _ = patient.step(actions[-1])
        states.append(x)
        rewards.append(r)
        x = y
    plt.hist(rewards)
    plt.show()
    return states, np.sum(rewards), actions


def greedy_action(model, state, device="cpu"):
    """
        TODO
    """
    with torch.no_grad():
        Q = model(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


def kernel_2d_size(in_kernel, kernel, padding, stride):
    """
        TODO
    """
    return (
        1 + (in_kernel[0] - kernel[0] + 2 * padding[0]) / stride[0],
        1 + (in_kernel[1] - kernel[1] + 2 * padding[1]) / stride[1],
    )


if __name__ == "__main__":
    model = lambda x: [np.random.random() for xi in x]
    evaluate(model, dur=10_000)
