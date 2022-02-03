import numpy as np
import hiv_patient

class HIVPatientScaled(HIVPatient):
    """HIV patient simulator
    
    Implements the simulator defined in 'Dynamic Multidrug Therapies for HIV: Optimal and STI Control Approaches' by Adams et al. (2004).
    The transition() function allows to simulate continuous time dynamics and control.
    The step() function is tailored for the evaluation of Structured Treatment Interruptions.
    The state is log-scaled and adjusted according to the provided bounds
    """
    def __init__(self):
        HIVPatient.__init__()
        
        # bounds
        self.T1Upper     = 1e6
        self.T1starUpper = 5e4
        self.T2Upper     = 3200.
        self.T2starUpper = 80.
        self.VUpper      = 2.5e5
        self.EUpper      = 353200.
        self.upper       = np.array([self.T1Upper, self.T1starUpper, self.T2Upper, self.T2starUpper, self.VUpper, self.EUpper])
        self.T1Lower     = 0.
        self.T1starLower = 0.
        self.T2Lower     = 0.
        self.T2starLower = 0.
        self.VLower      = 0.
        self.ELower      = 0.
        self.lower       = np.array([self.T1Lower, self.T1starLower, self.T2Lower, self.T2starLower, self.VLower, self.ELower])
        return

    def step(self, a_index):
        state = np.array([self.T1, self.T1star, self.T2, self.T2star, self.V, self.E])
        action = self.action_set[a_index]
        state2 = self.transition(state,action,5)
        np.clip(state2, self.Lower, self.upper)
        rew = self.reward(state, action, state2)
        
        self.T1 = state2[0]
        self.T1star = state2[1]
        self.T2 = state2[2]
        self.T2star = state2[3]
        self.V = state2[4]
        self.E = state2[5]
        
        return state2, rew, False, None
