import random

import numpy as np
from buffer import Experience, ReplayBuffer
from sklearn.ensemble import RandomForestRegressor

from dqn.hiv_patient import HIVPatient


def update_estimator(buffer, estimator):
    if estimator is None:
        estimator = RandomForestRegressor()
        x, y = buffer.to_array()
        estimator.fit(x, y)
    else:
        x, y = buffer.to_array()
        estimator.fit(x, y)
    return estimator


def train_fqi(
    num_steps,
    steps_per_update,
    patient: HIVPatient,
    buffer: ReplayBuffer,
    estimator=None,
    gamma: float = 0.98,
):

    experience = buffer.sample_experience()
    state, action, reward, _, _ = experience

    possible_actions = patient.one_hot_action_space()

    for _ in num_steps:

        if num_steps % steps_per_update == 0:
            estimator = update_estimator(buffer, estimator)

    if random.uniform(0, 1) < 0.15:
        action = np.zeros(4)
        action[random.randint(4)] = 1
        next_state, reward, done, _ = patient.step(action)
        experience = Experience(state, action, reward, done, next_state)
        buffer.append(experience)
        state = next_state
    else:
        state_action_pairs = np.array(
            [np.concatenate([state, action]) for action in possible_actions]
        )
        q_values = estimator.predict(state_action_pairs)
        next_action = np.zeros(4)
        next_action[np.argamx(q_values)] = 1
        experience = Experience(
            state,
            next_action,
            gamma * np.max(q_values),
            done,
            next_state,
        )
        buffer.append(experience)
