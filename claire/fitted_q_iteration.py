from collections import deque
from hiv_patient import HIVPatient
from random import randint, random
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
    
    # def get(self):
    #     # the floor is yours
    #     return

class FQI():

    def __init__(self) -> None:
        self.buffer = Buffer(60000)
        self.gamma = 0.98
        self.ncontrol = 1000//5 # episode = 1000 days and treatment is administered every 5 days
        self.niter = 10
        self.patients = [HIVPatient(False,False) for i in range(5)]
        self.Qn = RandomForestRegressor(50)
        self.action = np.eye(4)

    def generate_sp(self,all_random=False):
        for p in self.patients: #resetting to generate new trajectories
            p.reset()
        for t in range(self.ncontrol):
            for p in self.patients:
                s = p.state()
                if all_random:
                    a = randint(0,3)
                else:
                    if random()<.15:
                        a = randint(0,3)
                    else:
                        sa = np.array([np.concatenate([s,self.action[an]]) for an in range(4)])
                        a = np.argmax(self.Qn.predict(sa))
                s2,r,d,_ = p.step(a)
                self.buffer.append(s,self.action[a],r,s2,d)

    def iteration(self,all_random=False):
        self.generate_sp(all_random)
        # print('n sp ', self.buffer.__len__())
        x = np.concatenate([np.array(self.buffer.states) , np.array(self.buffer.actions)],axis=1)
        # print('x ',x.shape)
        if all_random:
            y = np.array(self.buffer.rewards) # only cost function in first iteration
        else:
            n_sp = self.buffer.__len__()
            Qns = np.zeros((n_sp,4)) 
            for an in range(4):
                sa = np.concatenate([self.buffer.next_states,[self.action[an]]*n_sp],axis=1)
                # print('sa ',sa.shape)
                Qns[:,an] = self.Qn.predict(sa)
            Qn_optim = np.max(Qns,axis=1)
            # print('Qn_optim ', Qn_optim.shape)
            y = np.array(self.buffer.rewards) + self.gamma * Qn_optim
        # print('y ',y.shape)
        self.Qn.fit(x,y)

    def test_agent(self):
        npatients = len(self.patients)
        rews = np.zeros(npatients)
        states = np.zeros((npatients,self.ncontrol,6))
        actions = np.zeros((self.ncontrol,4))
        for n_p in range(npatients):
            self.patients[n_p].reset()
            for t in range(self.ncontrol):
                s = self.patients[n_p].state()
                sa = np.array([np.concatenate([s,self.action[an]]) for an in range(4)])
                a = np.argmax(self.Qn.predict(sa)) # find best action
                s2,r,d,_ = self.patients[n_p].step(a)
                rews[n_p] += r
                states[n_p,t] = s
                if n_p == 0:
                    actions[t] = a
        return np.mean(rews), np.mean(states,axis=0), actions

    def learn(self):
        rews = []
        actions = []
        for iter in range(self.niter):
            if iter == 0:
                self.iteration(True)
            else:
                self.iteration(False)
            r, s, a = self.test_agent()
            rews.append(r)
            actions.append(a)
            print("mean reward over tested patients = ",rews[iter])
        return rews,s,actions


                
