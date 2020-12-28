"""
Defines the Observation for the Theory-Of-Mind level-0 worker domain;

Observation:

    :math:`-1 \cup 0`.
    The obsevation is either accept if the opponent agrees to the deal or rejection if there's no deal
"""
import pomdp_py

class Observation(pomdp_py.Observation):
    def __init__(self, value):
        self.value = value
    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.value == other.value
        return False
    def __str__(self):
        return self.value
    def __repr__(self):
        return "Observation(%s)" % self.value
