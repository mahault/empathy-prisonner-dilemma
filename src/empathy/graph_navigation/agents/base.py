from abc import abstractmethod

import itertools
import numpy as np
import pymdp


class Agent:
    """
    Base agent class
    """

    @abstractmethod
    def imagine_futures(self, pi):
        pass

    @abstractmethod
    def infer_action(self):
        pass

    @abstractmethod
    def infer_state(self, observation):
        pass


class BaseAgent(Agent):
    """
    A base agent class that provides some basic functionality.
    """

    def __init__(
        self, id, env, planning_horizon=3, sophisticated=True, cache_plan=False
    ):
        self.id = id
        self.env = env
        self.planning_horizon = planning_horizon

        self.A, self.A_factor_list = self.generate_A()
        self.B, self.B_factor_list = self.generate_B()
        self.C = self.generate_C()
        self.D = self.generate_D()

        # modality datastructure for the inference method
        self.num_obs = [self.A[m].shape[0] for m in range(len(self.A))]
        self.num_modalities = len(self.num_obs)
        self.num_states = [self.B[f].shape[0] for f in range(len(self.B))]
        self.num_factors = len(self.num_states)
        self.num_controls = [
            self.B[f].shape[-1] for f in range(self.num_factors)
        ]

        A_modality_list = []
        for f in range(self.num_factors):
            A_modality_list.append(
                [
                    m
                    for m in range(self.num_modalities)
                    if f in self.A_factor_list[m]
                ]
            )

        self.mb_dict = {
            "A_factor_list": self.A_factor_list,
            "A_modality_list": A_modality_list,
        }

        self.action = None
        self.qs = self.D
        self.qL = pymdp.utils.obj_array_zeros([s.shape for s in self.qs])

        self.inference_params = {
            "num_iter": 10,
            "dF": 1.0,
            "dF_tol": 0.001,
            "compute_vfe": False,
        }

        self.sophisticated = sophisticated
        self.cache_plan = cache_plan
        self.current_plan = None

    @abstractmethod
    def generate_A(self):
        pass

    @abstractmethod
    def generate_B(self):
        pass

    @abstractmethod
    def generate_C(self):
        pass

    @abstractmethod
    def generate_D(self):
        pass

    @abstractmethod
    def valid_actions(self, qs):
        pass

    def generate_policies(self, qs, length=1):
        policies = self.valid_actions(qs)
        if length == 1:
            return policies

        t = 0
        result = []
        while t < length:
            for p in policies:
                # TODO do we need a valid qs here?
                # here we hack in the transition model to
                # generate sensible length > 1 navigation policies
                qs = [np.zeros(self.num_locations)]
                qs[0][p[-1][0]] = 1.0

                next_actions = self.valid_actions(qs)
                for a in next_actions:
                    result.append(list(p) + list(a))

            policies = result
            result = []
            t = t + 1

        return [np.asarray(p) for p in policies]

    def imagine_futures(self, qs, policies):
        qs_pi = []
        qo_pi = []

        # TODO move to jax and make this parallel?
        for idx, policy in enumerate(policies):
            qs_next = pymdp.control.get_expected_states_interactions(
                qs, self.B, self.B_factor_list, policy
            )
            qo_next = pymdp.control.get_expected_obs_factorized(
                qs_next, self.A, self.A_factor_list
            )
            qs_pi.append(qs_next)
            qo_pi.append(qo_next)

        return qs_pi, qo_pi

    def rollout(self, qs, length=1, gamma=32):
        policies = self.generate_policies(qs, length)
        qs, qo = self.imagine_futures(qs, policies)

        G = np.zeros(len(policies))
        EU = np.zeros(len(policies))
        EIG = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            # Expected Utility
            EU[idx] = pymdp.control.calc_expected_utility(qo[idx], self.C)

            # Expected state Info Gain
            EIG[idx] = pymdp.control.calc_states_info_gain_factorized(
                self.A, qs[idx], self.A_factor_list
            )

            G[idx] = -EU[idx] - EIG[idx]

        # Distribution over policies
        q_pi = pymdp.maths.softmax(-gamma * G)

        return (
            policies,
            q_pi,
            G,
            EU,
            EIG,
            qs,
            qo,
        )

    def update_node(self, node, gamma=32, prune_penalty=512):
        # update the G and q_pi of this node given changes in the children
        # EFE of this node
        G = np.zeros(len(node["children"]))
        q_pi = np.zeros(len(node["children"]))
        # + weighted avg of EFE of children
        for idx, n in enumerate(node["children"]):
            if "EU" in n.keys():
                G[idx] -= n["EU"]
            if "EIG" in n.keys():
                G[idx] -= n["EIG"]

            q_pi[idx] = n["q_pi"]

            # policy nodes
            if len(n["children"]) == 0:
                # add penalty?
                G[idx] += prune_penalty
            else:
                # sum over all likely observations
                for o in n["children"]:
                    prob = o["prob"]
                    G[idx] += o["G"] * prob

        ignore_update = False
        if "agent" in node["children"][0].keys():
            # these children are policies of _other_ agent, so don't adjust the
            # policy selection!
            ignore_update = True

        if not ignore_update:
            q_pi = pymdp.maths.softmax(-gamma * G)
            node["q_pi"] = q_pi

        node["G"] = np.dot(q_pi, G)

        if not ignore_update:
            for idx, n in enumerate(node["children"]):
                n["q_pi"] = q_pi[idx]
                n["G"] = G[idx]

        if node["parent"] is not None:
            self.update_node(node["parent"]["parent"], gamma, prune_penalty)

    def expand_node(
        self,
        node,
        tree,
        policy_prune_threshold=1 / 16,
        policy_prune_topk=-1,
        observation_prune_threshold=1 / 16,
        prune_penalty=512,
        gamma=32,
    ):
        # take a node with a (expected) posterior distribution, and then
        # expand to what might happen if I did action a, and what would
        # be the new posterior after observing o
        policies, q_pi, G, EU, EIG, qs_pi, qo_pi = self.rollout(
            node["qs"], gamma=gamma
        )
        node["policies"] = policies
        node["q_pi"] = q_pi
        node["qs_pi"] = qs_pi
        node["qo_pi"] = qo_pi
        node["G"] = np.dot(q_pi, G)

        # then update the tree with the new expected free energies

        node["children"] = []

        # expand the policies and observations of this node
        ordered = np.argsort(q_pi)[::-1]
        policies_to_consider = []
        for idx in ordered:
            if (
                policy_prune_topk > 0
                and len(policies_to_consider) >= policy_prune_topk
            ):
                break
            if q_pi[idx] >= policy_prune_threshold:
                policies_to_consider.append(idx)
            else:
                break

        for idx in range(len(policies)):
            policy_node = {
                "policy": policies[idx],
                "q_pi": q_pi[idx],
                "G": G[idx],
                "EU": EU[idx],
                "EIG": EIG[idx],
                "parent": node,
                "children": [],
                "n": node["n"] + 1,
            }
            node["children"].append(policy_node)
            tree.append(policy_node)

            if idx in policies_to_consider:
                # average over outcomes
                qo_next = qo_pi[idx][0]
                for k in itertools.product(
                    *[range(s.shape[0]) for s in qo_next]
                ):
                    prob = 1.0
                    for i in range(len(k)):
                        prob *= qo_pi[idx][0][i][k[i]]

                    # ignore low probability observations in the search tree
                    if prob < observation_prune_threshold:
                        continue

                    qo_one_hot = pymdp.utils.obj_array(len(qo_next))
                    for i in range(len(qo_one_hot)):
                        qo_one_hot[i] = pymdp.utils.onehot(
                            k[i], qo_next[i].shape[0]
                        )

                    qs_next = (
                        pymdp.inference.update_posterior_states_factorized(
                            self.A,
                            qo_one_hot,
                            self.num_obs,
                            self.num_states,
                            self.mb_dict,
                            qs_pi[idx][0],
                            **self.inference_params
                        )
                    )

                    observation_node = {
                        "observation": qo_one_hot,
                        "prob": prob,
                        "qs": qs_next,
                        "G": 1e-10,
                        "parent": policy_node,
                        "children": [],
                        "n": node["n"] + 1,
                    }
                    policy_node["children"].append(observation_node)
                    tree.append(observation_node)

        # now update (and recursively its parents)
        self.update_node(node, gamma, prune_penalty)

        return tree

    def sophisticated_search(
        self,
        qs,
        horizon=1,
        policy_prune_threshold=1 / 16,
        policy_prune_topk=-1,
        observation_prune_threshold=1 / 16,
        entropy_prune_threshold=0.5,
        prune_penalty=512,
    ):
        # check if we can cache the current plan
        tree = None
        if self.current_plan is not None:
            root_node = self.current_plan[0]
            self.current_plan = None
            for idx, p in enumerate(root_node["policies"]):
                if np.allclose(p[0], self.action):
                    policy_node = root_node["children"][idx]

                    for observation_node in policy_node["children"]:
                        predicted_qs = observation_node["qs"]
                        if all(
                            [
                                np.allclose(predicted_qs[i], qs[i])
                                for i in range(len(qs))
                            ]
                        ):
                            # we can reuse the tree - make a copy for the logging
                            new_root = {**observation_node}
                            new_root["n"] = new_root["n"] - 1
                            new_root["parent"] = None
                            tree = [new_root]
                            to_visit = [new_root]
                            while len(to_visit) > 0:
                                node = to_visit.pop()
                                children = []
                                for c in node["children"]:
                                    child = {**c}
                                    child["parent"] = node
                                    child["n"] = child["n"] - 1
                                    children.append(child)
                                    node["children"] = children
                                    to_visit.append(child)
                                    tree.append(child)
                            break

        # build root node
        if tree is None:
            root_node = {
                "qs": qs,
                "parent": None,
                "children": [],
                "n": 0,
            }
            tree = [root_node]
            h = 0
        else:
            h = max([n["n"] for n in tree])

        # TODO should we have a fixed horizon or rather track the entropy of q_pi
        # and stop when it is low enough?
        while h < horizon:
            if "q_pi" in tree[0].keys():
                q_pi = tree[0]["q_pi"]
                H = -np.dot(q_pi, np.log(q_pi + pymdp.maths.EPS_VAL))
                if H < entropy_prune_threshold:
                    break

            # expand all leaf nodes
            leaf_nodes = [
                n for n in tree if "qs" in n.keys() and len(n["children"]) == 0
            ]
            for node in leaf_nodes:
                tree = self.expand_node(
                    node,
                    tree,
                    policy_prune_threshold=policy_prune_threshold,
                    policy_prune_topk=policy_prune_topk,
                    observation_prune_threshold=observation_prune_threshold,
                    prune_penalty=prune_penalty,
                )
            h += 1

        return tree[0]["policies"], tree[0]["q_pi"], tree[0]["G"], tree

    def infer_action(self):
        if self.sophisticated:
            policies, q_pi, G, tree = self.sophisticated_search(
                self.qs,
                horizon=self.planning_horizon,
                policy_prune_threshold=1 / 16,
                policy_prune_topk=-1,
                observation_prune_threshold=1 / 16,
                prune_penalty=512,
            )
            if self.cache_plan:
                self.current_plan = tree
        else:
            policies, q_pi, G, EU, EIG, qs_pi, qo_pi = self.rollout(
                self.qs, self.planning_horizon
            )

        # Sample next action from policy distribution
        # idx = np.argmax(q_pi)
        sample_onehot = np.random.multinomial(1, q_pi)
        idx = np.where(sample_onehot == 1)[0][0]

        action = policies[idx][0]
        self.action = action

        if self.sophisticated:
            return action, {
                "policies": policies,
                "q_pi": q_pi,
                "G": G,
                "tree": tree,
            }
        else:
            return action, {
                "policies": policies,
                "q_pi": q_pi,
                "G": G,
                "EU": EU,
                "EIG": EIG,
                "qs_pi": qs_pi,
                "qo_pi": qo_pi,
            }

    def infer_state(self, observation):

        if self.action is not None:
            prior = pymdp.control.get_expected_states_interactions(
                self.qs,
                self.B,
                self.B_factor_list,
                self.action.reshape(1, -1),
            )[0]
        else:
            prior = self.D

        qs = pymdp.inference.update_posterior_states_factorized(
            self.A,
            observation,
            self.num_obs,
            self.num_states,
            self.mb_dict,
            prior,
            **self.inference_params
        )

        # calculate VFE here
        complexity = 0.0
        for modality, A_m in enumerate(self.A):
            complexity += pymdp.maths.kl_div(qs[modality], prior[modality])

        accuracy = 0.0
        for modality, A_m in enumerate(self.A):
            factor_idx = self.A_factor_list[modality]
            qo = pymdp.maths.spm_dot(A_m, qs[factor_idx])
            accuracy += np.log(qo[observation[modality]] + pymdp.maths.EPS_VAL)

        F = complexity - accuracy

        # set qs as new beliefs
        self.qs = qs

        # calculate log likelihood message from observation
        ln_prior = pymdp.maths.spm_log_obj_array(prior)
        ln_qs = pymdp.maths.spm_log_obj_array(qs)
        self.qL = ln_qs - ln_prior

        # calculate state entropy H[Q(s)]
        # and arousal as normalized entropy per state factor
        arousal = 0.0
        H = 0.0
        for modality, A_m in enumerate(self.A):
            entropy = -np.dot(qs[modality], ln_qs[modality])
            H += entropy
            arousal += entropy / np.log(qs[modality].shape[0])

        # calculate utility of the current observation
        # as well as what was expected from the prior
        # valence = utility - expected utility
        utility = 0.0
        expected = 0.0

        probC = pymdp.maths.softmax_obj_arr(self.C)
        lnC = pymdp.maths.spm_log_obj_array(probC)

        # calculate expected observation from prior
        qo = pymdp.utils.obj_array(len(self.A))
        for modality, A_m in enumerate(self.A):
            factor_idx = self.A_factor_list[modality]
            qo[modality] = pymdp.maths.spm_dot(A_m, prior[factor_idx])

        for modality, A_m in enumerate(self.A):
            ndim = A_m.shape[0]
            one_hot = pymdp.utils.onehot(observation[modality], ndim)
            utility += one_hot.dot(lnC[modality])
            expected += qo[modality].dot(lnC[modality])

        return qs, {
            "F": F,
            "H": H,
            "U": utility,
            "valence": utility - expected,
            "arousal": arousal,
        }

    def set_prior(self, D):
        D = pymdp.utils.norm_dist_obj_arr(D)
        self.D = D
        self.qs = D
