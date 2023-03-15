import torch
from dacite import from_dict

from dqn.buffer import ReplayBuffer
from dqn.hiv_patient import HIVPatient
from dqn.q_agent import Agent
from dqn.q_learning import Qlearner, QLearningCongfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


conf_dict = {
    "gamma": 0.99,
    "batch_size": 32,
    "epsilon": {"start": 0.95, "end": 0, "decay": 200},
    "learning_rate": 0.001,
    "num_episodes": 10,
    "steps_per_episode": 80,
    "warm_start": {"episodes": 10, "steps_per_episode": 10},
    "target_update_rate": 10,
}

if __name__ == "__main__":

    patient = HIVPatient()
    conf = from_dict(data_class=QLearningCongfig, data=conf_dict)
    memory = ReplayBuffer(100)
    agent = Agent(patient=patient, replay_buffer=memory)
    q_learner = Qlearner(
        memory=memory,
        conf=conf,
        patient=patient,
        agent=agent,
        device=DEVICE,
    )

    q_learner.train()
