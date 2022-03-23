from collections import deque
from hiv_patient import HIVPatient
from random import randint, random
import numpy as np
import random as rd
import torch


class Buffer(): 
    def __init__(self, maxlen: int, device=None): 
        self.states = deque([], maxlen=maxlen)
        self.actions = deque([], maxlen=maxlen)
        self.rewards = deque([], maxlen=maxlen)
        self.next_states = deque([], maxlen=maxlen)
        self.done = deque([], maxlen=maxlen)
        self.device = device

    def __len__(self):
        return len(self.states)

    def append(self,state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done.append(done)
    
    def sample(self,batch_size):
        batch_idx = rd.sample(list(range(self.__len__())),batch_size)
        batch = np.array(self.states)[batch_idx],np.array(self.actions)[batch_idx],np.array(self.rewards)[batch_idx],np.array(self.next_states)[batch_idx],np.array(self.done)[batch_idx]
        return list(map(lambda x:torch.Tensor(x).to(self.device), batch))


class Agent():

    def __init__(self,npatients=5,buffer=None,device=None) -> None:
        if buffer!=None:
            self.buffer = buffer
        else:
            self.buffer = Buffer(60000,device)
        self.gamma = 0.98
        self.ncontrol = 1000//5 # episode = 1000 days and treatment is administered every 5 days
        self.niter = 10
        self.patients = [HIVPatient(False,False) for _ in range(npatients)]
        self.naction = 4
        self.nobs = 6
        self.epsilon = .15

    def get_greedy_action(self,state):
        """
        return greedy action from specified state
        """

    def get_action(self,state,random_action=False):
        if random_action:
            a = randint(0,self.naction-1)
        else:
            if random()<self.epsilon:
                a = randint(0,self.naction-1)
            else:
                a = self.get_greedy_action(state)
        return a

    def transform_action(self,a):
        """
        transform action depending on Qn implementation: onehot encoder, no transform (by default) ...
        """
        return a

    def transform_reward(self,r):
        """
        applies transform to reward, none by default
        """
        return r

    def sp_one_step(self, first_step, all_random=False):
        """
        add nb of patients samples (only one step generated) to buffer
        inputs:
            all_random, is True if the samples must be generated with random actions only, 85% chance of a greedy action otherwise
        """
        for p in self.patients:
            if first_step:
                p.reset()
            s = p.state()
            a = self.get_action(s,all_random)
            s2,r,d,_ = p.step(a)
            a = self.transform_action(a)
            r = self.transform_reward(r)
            self.buffer.append(s,a,r,s2,d)

    def generate_sp(self,all_random=False):
        """
        generate one new trajectory per patient
        """
        first_step = True
        for t in range(self.ncontrol):
            if t>0: first_step=False
            self.sp_one_step(first_step,all_random)


    def iteration(self,all_random=False):
        """
        iteration over one episode: generates needed samples and updates Q
        """

    def test_agent(self, save=False):
        """
        calculates optimal trajectory with current Q function for all patients
        return:
            mean cumulated reward in trajectory
            mean states in trajectory
            actions of patient 0
        """
        npatients = len(self.patients)
        rews = np.zeros(npatients)
        states = np.zeros((npatients,self.ncontrol,self.nobs))
        actions = np.zeros((self.ncontrol,self.naction))
        for n_p in range(npatients):
            self.patients[n_p].reset()
            for t in range(self.ncontrol):
                s = self.patients[n_p].state()
                a = self.get_greedy_action(s)
                s2,r,d,_ = self.patients[n_p].step(a)
                rews[n_p] += r
                states[n_p,t] = s
                if n_p == 0:
                    actions[t] = a
        return np.mean(rews), np.mean(states,axis=0), actions
        

    def train(self):
        rews = []
        states = []
        actions = []
        for iter in range(self.niter):
            if iter == 0:
                self.iteration(True)
            else:
                self.iteration(False)
            r, s, a = self.test_agent()
            rews.append(r)
            states.append(s)
            actions.append(a)
            print("mean reward over tested patients = ",rews[iter])
        return rews,states,actions


                
