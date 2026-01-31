import numpy as np
import pymdp

from empathy.graph_navigation.agents.searcher import SearchAgent


class ForageAgent(SearchAgent):
    """
    An agent that needs to forage for food to survive.
    """

    def __init__(
        self,
        id,
        env,
        objects_of_interest=None,
        num_objects=None,
        planning_horizon=3,
        sophisticated=True,
    ):
        super().__init__(
            id,
            env,
            objects_of_interest=objects_of_interest,
            num_objects=num_objects,
            not_here=True,
            planning_horizon=planning_horizon,
            sophisticated=sophisticated,
        )

    def generate_A(self):
        A, A_factor_list = super().generate_A()

        # add a state factor for energy level
        AA = pymdp.utils.obj_array(len(A) + 1)
        for i, a in enumerate(A):
            AA[i] = a

        AA[-1] = np.eye(10)
        A_factor_list.append([len(AA) - 1])

        AA = pymdp.utils.norm_dist_obj_arr(AA)
        return AA, A_factor_list

    def generate_B(self):
        B, B_factor_list = super().generate_B()

        BB = pymdp.utils.obj_array(len(B) + 1)
        for i, b in enumerate(B):
            BB[i] = b

        # add transition actions for consuming objects
        for i in range(self.num_objects):
            B_factor_list[i + 1] = [i + 1, 0]
            BB[i + 1] = np.zeros(
                [
                    self.num_object_locations,
                    self.num_object_locations,
                    self.num_locations,
                    2,
                ]
            )

            for l in range(self.num_locations):
                # we do nothing with object, so it stays on location
                BB[i + 1][:, :, l, 0] = np.eye(self.num_object_locations)

                # consume action respawns object (if agent at same location as object)
                for j in range(self.num_object_locations):
                    if j == l:
                        # object consumed and likely spawned somewhere else
                        # TODO this doesn't make the prior uniform unfortunately
                        # probably need a hierarchical model that resets D from level above?
                        # or hard code a reset in the infer_state?
                        BB[i + 1][:, j, l, 1] = 1.0
                        BB[i + 1][l, j, l, 1] = 0.1
                    else:
                        BB[i + 1][j, j, l, 1] = 1.0

        # add transition model for energy level
        BB[-1] = np.zeros(
            (10, 10, self.num_locations, self.num_object_locations, 2)
        )

        # energy decreases by 1 when moving
        BB[-1][0, 0, :, :, :] = 1
        for i in range(1, 10):
            BB[-1][i - 1, i, :, :, 0] = 1

        # energy level back to 10 when consuming food
        # but get sick (energy 0) when consuming without food
        BB[-1][0, :, :, :, 1] = 1
        for i in range(self.num_locations):
            BB[-1][:, :, i, i, 1] = 0
            # consume when object at location
            BB[-1][9, :, i, i, 1] = 1

        B_factor_list.append([len(BB) - 1, 0, 1])

        BB = pymdp.utils.norm_dist_obj_arr(BB)
        return BB, B_factor_list

    def generate_D(self):
        D = super().generate_D()

        DD = pymdp.utils.obj_array(len(D) + 1)
        for i, a in enumerate(D):
            DD[i] = a

        # add inventory dimension
        for i in range(self.num_objects):
            DD[i + 1] = np.ones([self.num_object_locations])

        # we start with full energy
        DD[-1] = np.zeros(10)
        DD[-1][-1] = 1

        DD = pymdp.utils.norm_dist_obj_arr(DD)
        return DD

    def generate_C(self):
        observations = [self.num_locations] + [2] * self.num_objects + [10]
        C = pymdp.utils.obj_array_zeros(observations)

        # you don't want low energy level
        C[-1][0] = -1

        return C

    def valid_actions(self, qs):
        from_loc = np.argmax(qs[0])

        policies = []
        # append valid move actions to policy
        for to_loc in range(self.num_locations):
            if self.B[0][to_loc, from_loc, to_loc] > 0.5:
                action = np.zeros((3), dtype=int)
                action[0] = to_loc
                policies.append([action])

        # append consume action to policy
        consume = np.zeros((3), dtype=int)
        consume[0] = from_loc
        consume[1] = 1
        consume[2] = 1
        policies.append([consume])

        return [np.asarray(p) for p in policies]


class PredatorAgent(ForageAgent):

    def __init__(
        self,
        id,
        env,
        planning_horizon=3,
        sophisticated=True,
    ):
        super().__init__(
            id,
            env,
            objects_of_interest=range(len(env.agents) - 1),
            num_objects=len(env.agents) - 1,
            planning_horizon=planning_horizon,
            sophisticated=sophisticated,
        )

    def generate_B(self):
        B, B_factor_list = super().generate_B()

        # add transition actions for consuming objects
        for i in range(self.num_objects):
            for l in range(self.num_locations):
                # we "objects" are other agents, so they can move
                for from_loc in range(self.num_locations):
                    for to_loc in range(self.num_locations):
                        # we transition if connected in graph
                        if self.env.graph.has_edge(from_loc, to_loc):
                            B[i + 1][to_loc, from_loc, l, 0] = 1

        B = pymdp.utils.norm_dist_obj_arr(B)
        return B, B_factor_list
