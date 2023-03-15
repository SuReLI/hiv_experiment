from hiv_patient import HIVPatient
from collections import deque
import numpy as np



class Buffer: 

    def __init__(self, maxlen: int): 
        self.states = deque([], maxlen=maxlen)
        self.actions = deque([], maxlen=maxlen)
        self.rewards = deque([], maxlen=maxlen)
        self.next_states = deque([], maxlen=maxlen)
        self.done = deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.states)
        
    def append(self,state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done.append(done)
    
    def get(self):
        states = np.stack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        next_states = np.stack(self.next_states)
        done = np.array(self.done)
        return states, actions, rewards, next_states,done