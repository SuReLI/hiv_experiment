#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from hiv_patient import HIVPatient

from buffer import ReplayBuffer
from sklearn.ensemble import RandomForestRegressor

from utils import evaluate
from utils import plot_traj


def fqi(n, t, eval=10, eval_t=500, gamma=0.98, Q=0.1, R1=20000, R2=2000, S=1000):
    """
        TODO
    """
    dur = t // 5
    patient = HIVPatient(clipping=False, logscale=False)
    rf = None

    rewards = []

    main_loop = trange(n)
    buffer = ReplayBuffer(capacity=60_000)
    for i in main_loop:
        x = patient.reset(mode="unhealthy", extras="immunity-failure")

        for j in trange(dur, desc="Building dataset", leave=False):
            if np.random.random() < 0.15 or rf is None:
                a = np.random.random(size=(4,))
            else:
                a = rf.predict([np.concatenate((x, a_)) for a_ in np.eye(4)])
            y, r, d, _ = patient.step(np.argmax(a))
            buffer.append(x, a, r, y, d)
            x = y

        # sample minibatch
        b = buffer.sample(len(buffer), flatten=False)

        # extract elements of the minibatch
        x = np.stack(b[:, 0]).astype(float)
        a = np.stack(b[:, 1]).astype(float)
        r = np.stack(b[:, 2]).astype(float)
        y = np.stack(b[:, 3]).astype(float)

        # build the feature matrix
        X = np.concatenate((x, a), axis=1).astype(float)

        # build the target vector
        q_n = np.max(
            [
                rf.predict(np.concatenate((y, np.stack([a_]*y.shape[0])), axis=1).astype(float))
                for a_ in np.eye(4)
            ], axis=0) if rf is not None else np.zeros(dur)
        y = (r + gamma * q_n).astype(float)

        # fit a new random forest
        rf = RandomForestRegressor(n_estimators=10, max_depth=3, max_leaf_nodes=50)
        rf.fit(X, y)

        # evaluate every now and then
        if (i+1) % eval == 0:
            states, rews, actions = evaluate(rf, dur=eval_t)
            rewards.append(np.mean(rews))
            main_loop.set_description(desc=f"reward {i}: {rewards[-1]}")
            plot_traj(states, actions)

    plt.plot(rewards)
    plt.show()

    return rf
