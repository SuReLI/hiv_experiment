import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Agent import Agent

class FQI(Agent):

    def __init__(self, npatients=5, buffer=None, device=None) -> None:
        super().__init__(npatients, buffer, device)
        self.Qn = RandomForestRegressor(50)
        self.action = np.eye(self.naction)

    
    def get_greedy_action(self,state):
        """
        return greedy action from specified state
        """
        sa = np.array([np.concatenate([state,self.action[an]]) for an in range(4)])
        a = np.argmax(self.Qn.predict(sa))
        return a

    def transform_action(self, a):
        """
        transform action depending on Qn implementation: onehot encoder, no transform (by default) ...
        """
        return self.action[a]


    def iteration(self,all_random=False):
        """
        iteration over one episode: generates needed samples and updates Q
        """
        self.generate_sp(all_random)
        x = np.concatenate([np.array(self.buffer.states) , np.array(self.buffer.actions)],axis=1)
        if all_random:
            y = np.array(self.buffer.rewards) # only cost function in first iteration
        else:
            n_sp = self.buffer.__len__()
            Qns = np.zeros((n_sp,4)) 
            for an in range(4):
                sa = np.concatenate([self.buffer.next_states,[self.action[an]]*n_sp],axis=1)
                Qns[:,an] = self.Qn.predict(sa)
            Qn_optim = np.max(Qns,axis=1)
            y = np.array(self.buffer.rewards) + self.gamma * Qn_optim
        self.Qn.fit(x,y)