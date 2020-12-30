import pomdp_py
import numpy as np
from pomdp_problems.tom_0_worker.domain.state import *
from pomdp_problems.tom_0_worker.domain.action import *
from pomdp_problems.tom_0_worker.domain.observation import *

class ToMZeroWorkerRewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action, fee, labor_costs):
        # According to the worker's model, if action < state - deal
        if action.value < state.value:            
            return action.value - labor_costs
        else:
            return -fee        

    def sample(self, state, action, next_state, fee, labor_costs):
        # deterministic
        return self._reward_func(state, action, fee, labor_costs)


### Unit test ###
def unittest():    
    # observation model test
    R = ToMZeroWorkerRewardModel()
    r_reject = R.sample(State(np.array([35.2])), Action(np.array([36])), State(np.array([35.03])), 2.8, 28)
    assert r_reject == -2.8
    r_deal = R.sample(State(np.array([35.2])), Action(np.array([34.8])), State(np.array([-1])), 2.8, 28)
    assert r_deal == 34.8-28

if __name__ == "__main__":
    unittest()

