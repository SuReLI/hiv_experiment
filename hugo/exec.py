# -*- coding: utf-8 -*-
import numpy as np
import random 
from sklearn.ensemble import RandomForestRegressor
from buffer import Buffer
from hiv_patient import HIVPatient
import math
BATCH_SIZE=10000
STEPS_EPOCH=1000
memory = Buffer(BATCH_SIZE)

EPS_0 = 0.9
EPS_DECAY = 0.95
GAMMA = 0.99
patient = HIVPatient(clipping=False,logscale=False)
RFR = RandomForestRegressor(n_estimators=100, max_depth=3)
s0=np.array(patient.reset())
X0=list()
n_actions=len(patient.action_set)
Y0=[]
for i in range(n_actions):
    actions_vect=np.zeros(n_actions)
    actions_vect[i]=1
    
    state2, rew, _, _ = patient.step(i)
    Y0.append(rew)
    X0.append(np.concatenate((s0, actions_vect)))
Y0=np.array(Y0)
X0=np.array(X0)    
RFR.fit(np.array(X0), np.array(Y0))    
    


def action_policy(state):
        qvalues=list()
        for i in range(n_actions):
            actions_vect=np.zeros(n_actions)
            actions_vect[i]=1  

            qvalues.append(RFR.predict(np.concatenate((state, actions_vect)).reshape(1, -1))[0])
        return np.argmax(qvalues)    

def select_action(state, eps):

    sample = random.random()


    if sample > eps:


            return random.randint(0, 3)
    else:

        return action_policy(state)

def eval_episode():
        s = patient.reset()  
        avg_rew=0
        for i_time in range(time_steps):
            action = action_policy(s)
            state2, rew, _, _ = patient.step(action)  
            avg_rew+=rew
        return avg_rew/time_steps
num_patients = 1000
time_steps=200
for i_patients in range(num_patients):
        s = patient.reset() 
        eps=EPS_0*(EPS_DECAY)**i_patients
        for i_time in range(time_steps):
            action = select_action(s, eps)
            state2, rew, _, _ = patient.step(action) 
            memory.append(s, action, rew, state2)
            s=state2
            if i_time*i_patients % STEPS_EPOCH ==0 and i_time*i_patients>0:
                print('ok')
                state, action, rew, next_state = memory.get() 
                q_update_list=list()
                transitions_list=list()
                for t in range(len(memory)):
                    actions_vect=np.zeros(n_actions)
                    action_index=action_policy(next_state[t])
                    actions_vect[action_index]=1
                    transitions_list.append(np.concatenate((np.array(state[t]), np.array(actions_vect))))
                    q_update=rew[t]+GAMMA*RFR.predict([np.concatenate((next_state[t], actions_vect))])[0]
                    q_update_list.append(q_update)
                RFR.fit(np.array(transitions_list), np.array(q_update_list))
                print(eval_episode())
            else:
                pass

                    
                    
                       