# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:14:50 2022

@author: USERS
"""

# -*- coding: utf-8 -*-
import numpy as np
import random 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from buffer import Buffer
from hiv_patient import HIVPatient
import math
BATCH_SIZE=60000
STEPS_EPOCH=6000
n_actions=4
memory = Buffer(BATCH_SIZE)

GAMMA = 0.98
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
    Y0.append(rew)
    X0.append(np.concatenate((s0, actions_vect)))
Y0=np.array(Y0)
X0=np.array(X0)    
RFR.fit(X0, Y0)
 



    

def action_policy(state, RFR):
        qvalues=list()
        actions_list=list()
        for i in range(n_actions):
            actions_vect=np.zeros(n_actions)
            actions_vect[i]=1  
            actions_list.append(actions_vect)
            #qvalues.append(predict(state, s0, actions_vect, RFR))
            qvalues.append(RFR.predict(np.concatenate((np.array(state), np.array(actions_vect))).reshape(1, -1))[0])            
        return actions_list[np.argmax(np.array(qvalues))], np.argmax(qvalues) 

def select_action(state, eps, RFR):

    sample = random.random()


    if sample < eps:
            actions_vect=np.zeros(n_actions)
            index=random.randint(0, 3)
            actions_vect[index]=1 

            return actions_vect, index
    else:

        return action_policy(state, RFR)

def eval_episode(RFR):
        s0 = patient.reset() 
        s=s0
        avg_rew=0
        for i_time in range(time_steps):
            action, index = action_policy(s, RFR)
            state2, rew, _, _ = patient.step(index)  
            avg_rew+=rew
            s=state2
        return avg_rew/time_steps
num_patients_epoch = 30
time_steps=200
epochs=10
q_iterations=400

for e in range(epochs):
    if e==0:
        eps=1.0
    else:
        eps=0.15
        
        
    for p in range(num_patients_epoch):
        s0 = patient.reset()
        s=s0
        for t in range(time_steps):
            action, index = select_action(s, eps, RFR)
            state2, rew, _, _ = patient.step(index) 
            
            memory.append(s, action, rew, state2)
            s=state2  
    state, action, rew, next_state = memory.get() 



                                                    
    transitions=np.concatenate((np.array(state), np.array(action)), axis=1)
                    
    transitions=np.array(transitions)
    a_1=list()
    a_2=list()
    a_3=list()
    a_4=list()
    
    for a in range(len(memory)):
        a_1.append(np.array([1,0,0,0]))
        a_2.append(np.array([0,1,0,0])) 
        a_3.append(np.array([0,0,1,0]))
        a_4.append(np.array([0,0,0,1]))   
    a_1=np.array(a_1)
    a_2=np.array(a_2)
    a_3=np.array(a_3)
    a_4=np.array(a_4)

               
    for i in range(q_iterations):
        q_update=list()
        q_1=RFR.predict(np.concatenate((np.array(next_state), a_1), axis=1))
        q_2=RFR.predict(np.concatenate((np.array(next_state), a_2), axis=1))  
        q_3=RFR.predict(np.concatenate((np.array(next_state), a_3), axis=1))  
        q_4=RFR.predict(np.concatenate((np.array(next_state), a_4), axis=1)) 
        print('iteration')         
        for a in range(len(memory)):
            if i==0:
                q_update.append(rew[a])
            else:    
                    q_update.append(rew[a]+GAMMA*np.max([q_1[a], q_2[a], q_3[a], q_4[a]]))                    


        RFR = ExtraTreesRegressor(n_estimators=100, max_depth=3)  
        RFR.fit(transitions, np.array(q_update))                  
        print(eval_episode(RFR))



