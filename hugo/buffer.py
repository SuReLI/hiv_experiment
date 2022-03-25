from collections import deque
import numpy as np
class Buffer: 
    def __init__(self, maxlen: int): 
        self.states = deque([], maxlen=maxlen)
        self.actions = deque([], maxlen=maxlen)
        self.rewards = deque([], maxlen=maxlen)
        self.next_states = deque([], maxlen=maxlen)
    def __len__(self):
        return len(self.states)
    def append(self,state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)   
    def get(self):
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.next_states)