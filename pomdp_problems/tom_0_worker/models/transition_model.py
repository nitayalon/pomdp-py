"""Defines the TransitionModel for the Theory-Of-Mind level-0 worker domain;

Transition: non-deterministic
"""
import pomdp_py
import copy
from scipy.stats import norm, truncnorm
import numpy as np
from pomdp_problems.tom_0_worker.domain.state import *
from pomdp_problems.tom_0_worker.domain.observation import *
from pomdp_problems.tom_0_worker.domain.action import *


class TransitionModel(pomdp_py.TransitionModel):
    
    def __init__(self, loc, scale, states):
        self.loc = loc
        self.scale = scale
        self.states = states        

    def probability(self, next_state, state, action):
        """
        After each rejection the state moves accrording to the transition kernel
        """
        if(action.value > state.value and next_state.value > state.value): 
            probs = truncnorm(loc = self.loc, scale = self.scale, a = (state.value - self.loc) / self.scale, b = (np.inf - self.loc) / self.scale).pdf(next_state.value)            
        else:
            probs = 0 
        return probs

    def sample(self, state, action):
        if(action.value > state.value): 
            legal_states = self.states[self.states > state.value]
            probs = truncnorm(loc = self.loc, scale = self.scale, a = (state.value - self.loc) / self.scale, b = (np.inf - self.loc) / self.scale).pdf(legal_states)            
            next_state =  np.random.choice(legal_states, size = 1, p = probs / probs.sum())
        else:
            next_state = np.array([-1])
        return State(next_state)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [State(s) for s in self.states]

### Unit test ###
def unittest():    
    # observation model test
    loc = 35
    scale = 2.5
    n = 1000
    states = np.sort(norm(loc = loc, scale = scale).rvs(n))
    T = TransitionModel(loc, scale, states)
    next_state = T.sample(State(states[500]), Action(states[650]))
    assert next_state.value > states[500]
    deal = T.sample(State(states[500]), Action(states[490]))
    assert deal.value == -1

if __name__ == "__main__":
    unittest()
