from pomdp_py.framework.basics cimport Action, Agent, Observation, State, Option,\
    TransitionModel, ObservationModel, RewardModel

cdef class Planner:

    cpdef public plan(self, Agent agent):    
        """The agent carries the information:
        Bt, ht, O,T,R/G, pi, necessary for planning"""
        raise NotImplementedError

    cpdef public update(self, Agent agent, Action real_action, Observation real_observation):
        """Updates the planner based on real action and observation.
        Updates the agent accordingly if necessary. If the agent's
        belief is also updated here, the `update_agent_belief`
        attribute should be set to True. By default, does nothing."""
        pass    

    def updates_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return False
