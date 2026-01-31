import numpy as np
import pymdp

from empathy.graph_navigation.agents.base import BaseAgent


class SearchAgent(BaseAgent):
    """
    An agent to search for rewards on a graph of locations.
    """

    def __init__(
        self,
        id,
        env,
        objects_of_interest=None,
        num_objects=None,
        not_here=True,
        planning_horizon=3,
        sophisticated=True,
    ):
        # particular objects this agent is interested in (i.e. has beliefs over)
        self.objects_of_interest = objects_of_interest

        # add a state dimension to reperesent the object being not here
        self.not_here = not_here

        # probability that we actually (expect to) see the object if we are at the correct location
        self.p_object_visible_on_target = 0.95

        # number of locations and objects
        self.num_locations = len(env.graph.nodes())
        self.num_object_locations = self.num_locations + int(self.not_here)
        if num_objects is not None:
            self.num_objects = num_objects
        else:
            self.num_objects = len(env.objects)

        # state factor for own location + each object location
        # observation modality for own location + each object visibility

        super().__init__(
            id,
            env,
            planning_horizon,
            sophisticated=sophisticated,
        )

    def generate_A(self):
        ## Likelihood ##

        A = pymdp.utils.obj_array(1 + self.num_objects)
        A_factor_list = []

        # Own location state factor -> location observation modality
        A[0] = np.eye(self.num_locations)
        A_factor_list.append([0])

        # Object location state factor -> object visibility observation modality
        p = self.p_object_visible_on_target
        for i in range(self.num_objects):
            A[i + 1] = np.zeros(
                (2, self.num_locations, self.num_object_locations)
            )

            for agent_loc in range(self.num_locations):
                for object_loc in range(self.num_locations):
                    if agent_loc == object_loc:
                        # object seen
                        A[i + 1][0, agent_loc, object_loc] = 1 - p
                        A[i + 1][1, agent_loc, object_loc] = p
                    else:
                        A[i + 1][0, agent_loc, object_loc] = p
                        A[i + 1][1, agent_loc, object_loc] = 1.0 - p

            if self.not_here:
                # object not here, we can't see it anywhere
                A[i + 1][0, :, -1] = 1.0
                A[i + 1][1, :, -1] = 0.0

            A_factor_list.append([0, i + 1])

        # Finally normalize
        A = pymdp.utils.norm_dist_obj_arr(A)
        return A, A_factor_list

    def generate_B(self):
        ## Transition ##
        B = pymdp.utils.obj_array(1 + self.num_objects)
        B_factor_list = []

        # Own location transitions, based on graph connectivity
        B[0] = np.zeros(
            (self.num_locations, self.num_locations, self.num_locations)
        )
        for action in range(self.num_locations):
            for from_loc in range(self.num_locations):
                for to_loc in range(self.num_locations):
                    if action == to_loc:
                        # we transition if connected in graph
                        if self.env.graph.has_edge(from_loc, to_loc):
                            B[0][to_loc, from_loc, action] = 1
                        else:
                            B[0][from_loc, from_loc, action] = 1

        B_factor_list.append([0])

        # Objects don't move
        for i in range(self.num_objects):
            B[i + 1] = np.zeros(
                (self.num_object_locations, self.num_object_locations, 1)
            )
            B[i + 1][:, :, 0] = np.eye(self.num_object_locations)
            B_factor_list.append([i + 1])

        B = pymdp.utils.norm_dist_obj_arr(B)
        return B, B_factor_list

    def generate_D(self):
        ## Prior ##
        states = [self.num_locations] + [
            self.num_object_locations
        ] * self.num_objects
        D = pymdp.utils.obj_array_ones(states) * 1e-3

        # we know our start location
        D[0][self.env.agents[self.id].location] = 1

        D = pymdp.utils.norm_dist_obj_arr(D)
        return D

    def generate_C(self):
        ## Preference ##
        observations = [self.num_locations] + [2] * self.num_objects
        C = pymdp.utils.obj_array_zeros(observations)
        if self.objects_of_interest is not None:
            for o in self.objects_of_interest:
                # prefer to see this object
                C[o + 1][1] = 1

        return C

    def valid_actions(self, qs):
        from_loc = np.argmax(qs[0])

        policies = []
        # append valid move actions to policy
        for to_loc in range(self.num_locations):
            if self.B[0][to_loc, from_loc, to_loc] > 0.5:
                action = np.zeros((2), dtype=int)
                action[0] = to_loc
                policies.append([action])

        return [np.asarray(p) for p in policies]
