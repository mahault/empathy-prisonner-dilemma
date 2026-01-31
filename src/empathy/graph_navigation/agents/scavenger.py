import numpy as np
import pymdp

from empathy.graph_navigation.agents.forager import ForageAgent


class ScavengeAgent(ForageAgent):
    """
    An agent that can savenge for objects / food, i.e. pick and drop them,
    or consume them.
    """

    def __init__(
        self,
        id,
        env,
        objects_of_interest=None,
        planning_horizon=3,
        sophisticated=True,
    ):
        super().__init__(
            id,
            env,
            objects_of_interest=objects_of_interest,
            planning_horizon=planning_horizon,
            sophisticated=sophisticated,
        )

    def generate_A(self):
        A, A_factor_list = super().generate_A()

        # add an inventory state for each object, and an inventory observation per object
        for i in range(self.num_objects):
            A_obj = np.zeros(
                (3, self.num_locations, self.num_object_locations, 2)
            )
            # when not in inventory, same as forage/search agent
            A_obj[0:2, :, :, 0] = A[i + 1]
            # when object in inventory, observe 2
            A_obj[2, :, :, 1] = 1.0

            A[i + 1] = A_obj
            A_factor_list[i + 1] = [0, i + 1, 2 + self.num_objects + i]

        A = pymdp.utils.norm_dist_obj_arr(A)
        return A, A_factor_list

    def generate_B(self):
        B, B_factor_list = super().generate_B()

        BB = pymdp.utils.obj_array(len(B) + self.num_objects)
        for i, b in enumerate(B):
            BB[i] = b

        # add transition actions for object picking and dropping
        for i in range(self.num_objects):
            B_factor_list[i + 1] = [
                i + 1,
                0,
                self.num_objects + 2 + i,
            ]
            BB[i + 1] = np.zeros(
                [
                    self.num_object_locations,
                    self.num_object_locations,
                    self.num_locations,
                    2,  # inventory state
                    4,  # actions
                ]
            )

            # reuse forager transitions for moving/consuming
            for k in range(2):
                BB[i + 1][:, :, :, k, 0] = B[i + 1][:, :, :, 0]
                BB[i + 1][:, :, :, k, 1] = B[i + 1][:, :, :, 1]

            # and add pick/drop dynamics
            for l in range(self.num_locations):
                # pick object, no longer here
                for j in range(self.num_object_locations):
                    if j == l:
                        BB[i + 1][-1, j, l, :, 2] = 1.0
                    else:
                        BB[i + 1][j, j, l, :, 2] = 1.0

                # drop object, moves to location, if in inventory
                BB[i + 1][:, :, l, 0, 3] = np.eye(self.num_object_locations)
                BB[i + 1][l, :, l, 1, 3] = 1.0

        # add transition matrices for the inventory states
        for i in range(self.num_objects):
            BB[2 + self.num_objects + i] = np.zeros(
                [
                    2,
                    2,
                    self.num_object_locations,
                    self.num_locations,
                    4,
                ]
            )

            for k in range(self.num_object_locations):
                for l in range(self.num_locations):
                    # action 0 : stay
                    BB[2 + self.num_objects + i][:, :, k, l, 0] = np.eye(2)
                    # action 1: consume
                    BB[2 + self.num_objects + i][:, :, k, l, 1] = np.eye(2)
                    # action 2: pick
                    if k == l:
                        BB[2 + self.num_objects + i][1, :, k, l, 2] = 1.0
                    else:
                        BB[2 + self.num_objects + i][:, :, k, l, 2] = np.eye(2)
                    # action 3: drop
                    BB[2 + self.num_objects + i][0, :, k, l, 3] = 1.0

            B_factor_list.append([self.num_objects + 2 + i, i + 1, 0])

        BB = pymdp.utils.norm_dist_obj_arr(BB)
        return BB, B_factor_list

    def generate_D(self):
        D = super().generate_D()

        DD = pymdp.utils.obj_array(len(D) + self.num_objects)
        for i, a in enumerate(D):
            DD[i] = a

        # add object inventory dimension
        # start with nothing in inventory
        for i in range(self.num_objects):
            DD[2 + self.num_objects + i] = np.array([1.0, 0.0])

        DD = pymdp.utils.norm_dist_obj_arr(DD)
        return DD

    def generate_C(self):
        observations = [self.num_locations] + [3] * self.num_objects + [10]
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
                action = np.zeros((4), dtype=int)
                action[0] = to_loc
                policies.append([action])

        # append consume action to policy
        consume = np.zeros((4), dtype=int)
        consume[0] = from_loc
        consume[1] = 1
        consume[2] = 1
        consume[3] = 1
        policies.append([consume])

        # append pick
        pick = np.zeros((4), dtype=int)
        pick[0] = from_loc
        pick[1] = 2
        pick[2] = 0
        pick[3] = 2
        policies.append([pick])

        # append drop
        drop = np.zeros((4), dtype=int)
        drop[0] = from_loc
        drop[1] = 3
        drop[2] = 0
        drop[3] = 3
        policies.append([drop])

        return [np.asarray(p) for p in policies]
