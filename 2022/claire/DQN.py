import numpy as np
import torch
import torch.nn as nn
from Agent import Agent

class Net():
    def __init__(self,n_obs,n_act,device) -> None:
        self.state_dim = n_obs
        self.n_action = n_act
        self.nb_neurons = 50
        self.model = torch.nn.Sequential(nn.Linear(self.state_dim, self.nb_neurons),
                                nn.Relu(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.Relu(), 
                                nn.Linear(self.nb_neurons, self.n_action)).to(device) #output is Q(s,a) for every possible action (4 possibilities)


class DQN(Agent):

    def __init__(self, npatients=5, buffer=None, device=None) -> None:
        super().__init__(npatients, buffer, device)
        self.device = device
        self.Qn = Net(self.nobs,self.naction,self.device)
        self.batch_size = 512
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Qn.model.parameters(), lr=5*1e-3)
        self.losses = [] ### TO CHECK VALUES

    
    def get_greedy_action(self,s):
        """
        return greedy action from specified state
        """
        with torch.no_grad():
            a = torch.argmax(self.Qn.model(torch.Tensor(s).unsqueeze(0).to(self.device))).item()
            return a


    def gradient_step(self, nsamples):
        """
        update Qn with one gradient step
        """
        if self.buffer.__len__() > nsamples:
            S, A, R, S2, D = self.buffer.sample(nsamples)
            QS2max = self.Qn.model(S2).max(1)[0].detach()
            update = torch.addcmul(R,self.gamma,1-D,QS2max)
            QS = self.Qn.model(S) # get Q(s,a) for each possible action a 
            QSA = QS.gather(1, A.to(torch.long)[:,np.newaxis])
            loss = self.criterion(QSA, update.unsqueeze(1))
            self.losses.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 


    def iteration(self,all_random=False):
        """
        iteration over one episode: generates needed samples and updates Q
        """
        first_step = True
        for t in range(self.ncontrol):
            if t>0: first_step = False
            self.sp_one_step(first_step,all_random)
            self.gradient_step(self.batch_size)