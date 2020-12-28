"""Defines the ObservationModel for the the Theory-Of-Mind level-0 worker domain;

Observation: :math:`-1 \cup 0`.
    The obsevation is either accept if the opponent agrees to the deal or rejection if there's no deal

Observation Model

  The agent observes either acceptence or rejection.

"""

import pomdp_py
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pomdp_problems.tom_0_worker.domain.state import *
from pomdp_problems.tom_0_worker.domain.action import *
from pomdp_problems.tom_0_worker.domain.observation import *

#### Observation Models ####
class ToMZeroWorkerObservationModel(pomdp_py.ObservationModel): 
    
    def sample(self, next_state, action):
        # if the agent quits
        if action.value == 0:
            return Observation('quit')        
        # if next_state = -1 - that is action < previous state
        if next_state.value == -1:
            return Observation('accept')
        else:
            return Observation('reject')

### Unit test ###
def unittest():    
    # observation model test
    O = ToMZeroWorkerObservationModel()

    o_accept = O.sample(State(np.array([-1])), Action(np.array([36])))
    assert o_accept.value == 'accept'    

    o_quit = O.sample(State(np.array([35.2])), Action(np.array([0])))
    assert o_quit.value == 'quit'    

    o_reject = O.sample(State(np.array([35.2])), Action(np.array([36])))
    assert o_reject.value == 'reject'    
if __name__ == "__main__":
    unittest()
