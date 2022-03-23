import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from DQN import DQN, Net


class DDQN(DQN):

    def __init__(self, npatients=5, buffer=None, device=None) -> None:
        super().__init__(npatients, buffer, device)
        self.Qn_target = Net(self.nobs,self.naction,self.device)
        self.Qn_target.model = deepcopy(self.Qn.model).to(self.device)
        self.update_freq = 50
        self.nb_gradient_steps = 10
        self.criterion = nn.SmoothL1Loss() # loss for error clipping
        self.epsilon = .15
        self.gradients = [] ### TO CHECK VALUES
        self.q_values = [] ### TO CHECK VALUES


    def transform_reward(self,r):
        """
        applies transform to reward, none by default
        """
        return np.sign(r)*np.log10(abs(r)+1) 
    
    def gradient_step(self, nsamples):
        """
        update Qn with one gradient step
        """
        if self.buffer.__len__() > nsamples:
            S, A, R, S2, D = self.buffer.sample(nsamples)
            QS2max = self.Qn_target.model(S2).max(1)[0].detach()
            update = torch.addcmul(R,self.gamma,1-D,QS2max)
            QS = self.Qn.model(S)
            QSA = QS.gather(1, A.to(torch.long)[:,np.newaxis])
            loss = self.criterion(QSA, update.unsqueeze(1))
            self.losses.append(loss) ### TO CHECK VALUES
            self.optimizer.zero_grad()
            loss.backward()
            grad_value = nn.utils.clip_grad_norm_(self.Qn.model.parameters(), max_norm=1) # clip gradient
            self.gradients.append(grad_value) ### TO CHECK VALUES
            self.optimizer.step() 


    def iteration(self,all_random=False):
        """
        iteration over one episode: generates needed samples and updates Q
        """
        first_step = True
        for t in range(self.ncontrol):
            if t>0: first_step = False
            self.sp_one_step(first_step,all_random)

            # do several gradient steps
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step(self.batch_size)

            # update target network following update frequency
            if t % self.update_freq == 0:
                self.Qn_target.model.load_state_dict(self.Qn.model.state_dict())
