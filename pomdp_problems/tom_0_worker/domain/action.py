"""
Defines the Action for the Theory-Of-Mind level-0 worker domain;

Action space: 

    Offer :math:`S`
    Quit :math:`0`    

* Offer actions: sample of actions from the parent distribution
* Quit action: terminates the process

Potentially wee can add the Accept action to the action set but this is only valid
for ToM bigger than 0
"""
import pomdp_py

class Action(pomdp_py.Action):
    def __init__(self, value):
        if type(value.item()) not in [int,float]:
            raise ValueError(f'Action must be either a float or an int')        
        self.value = value
    def __str__(self):
        return self.value
    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.value == other.value
        return False
    def __repr__(self):
        return "Action(%s)" % self.value
