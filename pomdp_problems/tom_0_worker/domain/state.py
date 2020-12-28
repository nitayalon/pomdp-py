"""Defines the State for the Theory-Of-Mind level-0 worker domain;

Description: Single agent labor market.

State space: 

    S = sample from :math:`X ~ F(x)` of size :math:`n`
"""

import pomdp_py

###### States ######
class State(pomdp_py.State):
    def __init__(self, value):
        if type(value.item()) not in [int,float]:
            raise ValueError(f'State must be either a float or an int')        
        self.value = value
    def __str__(self):
        return self.value
    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        if isinstance(other, State):
            return self.value == other.value
        return False    
    def __repr__(self):
        return "State(%s)" % self.value