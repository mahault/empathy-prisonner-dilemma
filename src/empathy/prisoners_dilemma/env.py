import numpy as np

from typing import Union


class Environment:
    def __init__(self, K: int) -> None:
        
        self.K = K    # Number of agents
        
        # TODO: Need a better generalization when I move to more than K=2 agents
        self._action_observation_mapping = {
            (0, 0): [0, 0],  # [CC, CC]
            (0, 1): [1, 1],  # [CD, CD]
            (1, 0): [2, 2],  # [DC, DC]
            (1, 1): [3, 3]   # [DD, DD]
        }

    def step(self, t:int, actions: list) -> Union[list, "np.ndarray"]:
        """ 
        One environment step. Returns an array of observations indicating the combinations
        of actions (Cooperate (C), Defect (D)) for each agent. For two agents:
        
        - CC: 0   [Agent 0 Cooperates, Agent 1 Cooperates]
        - CD: 1   [Agent 0 Cooperates, Agent 1 Defects]
        - DC: 2   [Agent 0 Defects   , Agent 1 Cooperates]
        - DD: 3   [Agent 0 Defects   , Agent 1 Defects]
        
        This is encapsulated in self._action_observation_mapping where C = 0 and D = 1.
        
        Since each agent needs to receive this information, the values 0-3 are duplicated
        K times, one observation for each K agents. For K=2 agents with action combinations
        DC, the observation array will be [2, 2].
        
        Note that on time step t=0, no agent has acted yet so no observation is available.
        Instead, each agent generates expected observations using its model. Since the
        environment does not play a role here, the observation at t=0 is just [None] * K.
         
        """
        # When t=0, environment does not generate observations because the agents
        # generate their own observations
        if t==0:
            return [None] * self.K
        else: 
            return self._generate(actions=actions)
        
    def _generate(self, actions: list) -> "np.ndarray":
        """ 
        Maps actions to observations. Where Cooperate (C) = 0 and Defect (D) = 1.
        """
        return self._action_observation_mapping[tuple(actions)]
            