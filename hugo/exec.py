# -*- coding: utf-8 -*-
import numpy as np
import random 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from buffer import Buffer
from hiv_patient import HIVPatient
import math
BATCH_SIZE=60000
STEPS_EPOCH=200
memory = Buffer(BATCH_SIZE)

EPS_0 = 1.0
EPS_MIN=0.1
GAMMA = 0.99
patient = HIVPatient(clipping=False,logscale=False)
RFR = ExtraTreesRegressor(n_estimators=100, max_depth=3)
s0=np.array(patient.reset())




X0=list()
n_actions=len(patient.action_set)
Y0=[]
_, q0, _, _ = patient.step(0)
patient.reset() 
for i in range(n_actions):
    actions_vect=np.zeros(n_actions)
    actions_vect[i]=1
    
    state2, rew, _, _ = patient.step(i)
    Y0.append(rew/q0)
    X0.append(np.concatenate((s0, actions_vect)))
Y0=np.array(Y0)
X0=np.array(X0)    
RFR.fit(X0, Y0)
 



    

def action_policy(state, RFR):
        qvalues=list()
        for i in range(n_actions):
            actions_vect=np.zeros(n_actions)
            actions_vect[i]=1  

            #qvalues.append(predict(state, s0, actions_vect, RFR))
            qvalues.append(RFR.predict(np.concatenate((np.array(state), np.array(actions_vect))).reshape(1, -1))[0])            
        return np.argmax(qvalues)    

def select_action(state, eps, RFR):

    sample = random.random()


    if sample < eps:


            return random.randint(0, 3)
    else:

        return action_policy(state, RFR)

def eval_episode(RFR):
        s0 = patient.reset() 
        s=s0
        avg_rew=0
        for i_time in range(time_steps):
            action = action_policy(s, RFR)
            state2, rew, _, _ = patient.step(action)  
            avg_rew+=rew
            s=state2
        return avg_rew/time_steps
num_patients = 300
time_steps=200
eps=EPS_0
for i_patients in range(num_patients):
        s0 = patient.reset()         
    
        if eps>EPS_MIN:
            eps=EPS_0-(EPS_0-EPS_MIN)*(1/num_patients)*i_patients
            
        s=s0    
        for i_time in range(time_steps):
            action = select_action(s, eps, RFR)
            state2, rew, _, _ = patient.step(action) 
            memory.append(s, action, rew, state2)
            s=state2
            if i_time*i_patients % STEPS_EPOCH ==0 and i_time*i_patients>0:
                state, action, rew, next_state = memory.get() 
                q_update_list=list()
                transitions_list=list()
                for t in range(len(memory)):
                    actions_vect=np.zeros(n_actions)
                    actions_vect[action[t]]=1
                    
                    
                    next_actions_vect=np.zeros(n_actions)
                    next_action_index=select_action(next_state[t], 0, RFR)
                    next_actions_vect[next_action_index]=1

                                                    
                    transitions_list.append(np.concatenate((np.array(state[t]), np.array(actions_vect))))
                    q_update=rew[t]+GAMMA*RFR.predict(np.concatenate((np.array(next_state[t]), np.array(next_actions_vect))).reshape(1, -1))[0]                    
                    q_update_list.append(q_update)
                RFR = ExtraTreesRegressor(n_estimators=100, max_depth=3)  
                RFR.fit(np.array(transitions_list), np.array(q_update_list))                  
                print(eval_episode(RFR))


                    
                   
                       