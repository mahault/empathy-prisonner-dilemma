import itertools
import numpy as np
import pymdp

from empathy.graph_navigation.agents.base import Agent


class ToMAgent(Agent):
    """
    An agent that takes into account other agent's beliefs in the ToM planning
    """

    def __init__(
        self,
        agent: Agent,
        others: list[Agent],
        self_states,  # which states are related to "self"
        observed_states,  # which states of others are "observed"
        shared_states,  # which states are belief-shared
    ):
        self.id = agent.id

        # TODO create agent pomdp copies for others as well?
        self.self_model = agent
        # we're not learning separate models of others, just reuse our "Agent"?
        # self.other_models = [copy.deepcopy(a) for a in others]

        # these are references to the other agents
        self.others = others

        self.self_states = self_states
        self.observed_states = observed_states
        self.shared_states = shared_states

        self.belief_share = True

        # ToM uses sophisticated search
        self.sophisticated = True
        # TODO implement caching of the plan in ToM as well?
        self.cache_plan = False

        # initialize agreggated qs
        self.qs = pymdp.utils.obj_array(
            len(self.self_model.qs) * (len(others) + 1)
        )
        k = 0
        for i in range(len(others) + 1):
            for q in self.self_model.qs:
                self.qs[k] = q
                k += 1

    def imagine_futures(self, qs, policies):
        return self.self_model.imagine_futures(qs, policies)

    def update_from_others(self, qs, other_qs, other_qL):
        # qs : prev aggregated posterior of self and others
        # other_qs : list of qs of other agents
        # other_qL : list of qL of other agents

        # TODO this is _very_ basic for now
        # ultimately we should properly infer the beliefs of the other agents
        # from whatever we can observe from them
        # for now assume we can observe their qs directly or get a shared likelihood message
        num_states = len(self.self_model.qs)
        qs_agg = pymdp.utils.obj_array(num_states * (len(other_qs) + 1))

        # first copy the current beliefs
        for i in range(num_states):
            qs_agg[i] = np.copy(qs[i])

        for k in range(num_states):
            # observed states: copy qs others into aggregated
            if k in self.observed_states:
                # just set the qs of the other agent as if we can access it
                for i in range(len(self.others)):
                    qs_agg[(1 + i) * num_states + k] = np.copy(other_qs[i][k])
            # shared states: merge the likelihood messages into own posterior
            elif k in self.shared_states and other_qL is not None:
                # use the likelihood messages to get a new shared posterior
                log_posterior = pymdp.maths.spm_log_single(qs[k])
                for i in range(len(self.others)):
                    log_posterior += other_qL[i][k]

                qs_agg[k] = pymdp.maths.softmax(log_posterior)

                # TODO which posterior do we attribute to the others?
                # for now assume we just have access to theirs?
                for i in range(len(self.others)):
                    qs_agg[(1 + i) * num_states + k] = np.copy(other_qs[i][k])
            else:
                # TODO what with other states?
                # TODO do we observe to others actions to use transition model on their beliefs?
                # for now also treat them as observed and get theirs directly
                for i in range(len(self.others)):
                    qs_agg[(1 + i) * num_states + k] = np.copy(other_qs[i][k])

        return qs_agg

    def expand_node_tom(
        self,
        agent_idx,
        node,
        tree,
        policy_prune_threshold=1 / 16,
        policy_prune_topk=-1,
        observation_prune_threshold=1 / 16,
        prune_penalty=512,
        gamma=32,
    ):
        # plan what the other agent might do given our estimate
        # of the other's posterior
        num_states = len(self.self_model.qs)
        other_qs = node["qs"][
            (agent_idx + 1) * num_states : (agent_idx + 2) * num_states
        ]

        # plan what the other agent might do
        policies, q_pi, G, other_plan = self.self_model.sophisticated_search(
            other_qs,
            horizon=self.self_model.planning_horizon,
            policy_prune_threshold=1 / 16,
            policy_prune_topk=-1,
            observation_prune_threshold=1 / 16,
            prune_penalty=512,
        )

        # and update the tree with the likely scenarios
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
                "agent": agent_idx,
                "policy": policies[idx],
                "q_pi": q_pi[idx],
                "parent": node,
                "children": [],
                "n": node["n"],
            }
            node["children"].append(policy_node)
            tree.append(policy_node)

            if idx in policies_to_consider:
                # imagine a new posterior for self given
                # what the other agent might do

                # to do this we first calculate the posterior
                # of the other given a potential outcome

                # and then update our own posterior by likelihood
                # sharing on the shared states

                # instead of calculating the probability of the outcome
                # form the other's perspective, we imagine using our
                # perspective (i.e. our qs of shared states)

                # create an expected outcome of the other given
                # our own posterior of shared states
                for i in range(len(other_qs)):
                    if i in self.shared_states:
                        other_qs[i] = node["qs"][i]
                qs_pi, qo_pi = self.imagine_futures(other_qs, policies)

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

                    # calculate the new posterior of the other agent
                    # and expected qL for the shared states
                    qs_prev = other_plan[0]["qs_pi"][idx][0]
                    qs_next = (
                        pymdp.inference.update_posterior_states_factorized(
                            self.self_model.A,
                            qo_one_hot,
                            self.self_model.num_obs,
                            self.self_model.num_states,
                            self.self_model.mb_dict,
                            qs_prev,
                            **self.self_model.inference_params
                        )
                    )
                    qL = pymdp.maths.spm_log_obj_array(
                        qs_next
                    ) - pymdp.maths.spm_log_obj_array(qs_prev)

                    # now create new aggregated qs
                    qs = self.update_from_others(node["qs"], [qs_next], [qL])

                    observation_node = {
                        "agent": agent_idx,
                        "observation": qo_one_hot,
                        "prob": prob,
                        "qs": qs,
                        "G": 1e-10,
                        "parent": policy_node,
                        "children": [],
                        "n": node["n"] + 1,
                    }
                    policy_node["children"].append(observation_node)
                    tree.append(observation_node)

        # now update (and recursively its parents)
        self.self_model.update_node(node, gamma, prune_penalty)

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

        # start a plan tree
        root_node = {
            "qs": qs,
            "parent": None,
            "children": [],
            "n": 0,
        }
        tree = [root_node]
        h = 0

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
                tree = self.self_model.expand_node(
                    node,
                    tree,
                    policy_prune_threshold=policy_prune_threshold,
                    policy_prune_topk=policy_prune_topk,
                    observation_prune_threshold=observation_prune_threshold,
                    prune_penalty=prune_penalty,
                )

            # then plan a move for all other agents
            # and update the predicted posterior of the other agent
            # but using OUR view of the world for the expected outcome(s)??
            for i, other in enumerate(self.others):
                leaf_nodes = [
                    n
                    for n in tree
                    if "qs" in n.keys() and len(n["children"]) == 0
                ]

                for node in leaf_nodes:
                    # we need to expand again to get the other agent's qs
                    parent_qs = node["parent"]["parent"]["qs"]
                    expanded_qs = pymdp.utils.obj_array(len(parent_qs))
                    for k in range(len(parent_qs)):
                        expanded_qs[k] = np.copy(parent_qs[k])
                    for k in range(len(node["qs"])):
                        expanded_qs[k] = node["qs"][k]
                    node["qs"] = expanded_qs

                    tree = self.expand_node_tom(
                        i,
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
        # first update the current qs with info from others
        # we do belief sharing here between the agents as well
        qL = None
        if self.belief_share:
            qL = [o.qL for o in self.others]
        self.qs = self.update_from_others(
            self.qs, [o.qs for o in self.others], qL
        )

        # also update self_model qs
        self.self_model.qs = self.qs[: len(self.self_model.qs)]

        policies, q_pi, G, tree = self.sophisticated_search(
            self.qs,
            horizon=self.self_model.planning_horizon,
            policy_prune_threshold=1 / 16,
            observation_prune_threshold=1 / 16,
            prune_penalty=512,
        )

        # Sample next action from policy distribution
        # idx = np.argmax(q_pi)
        sample_onehot = np.random.multinomial(1, q_pi)
        idx = np.where(sample_onehot == 1)[0][0]

        action = policies[idx][0]
        self.self_model.action = action

        return action, {
            "policies": policies,
            "q_pi": q_pi,
            "G": G,
            "tree": tree,
        }

    def infer_state(self, observation):
        # infer the new state of the focal agent
        qs, info = self.self_model.infer_state(observation)

        # make a copy of the full posterior of self and others aggregated
        posterior = pymdp.utils.obj_array(
            len(self.self_model.qs) * (len(self.others) + 1)
        )
        k = 0
        for _ in qs:
            posterior[k] = np.copy(qs[k])
            k += 1
        for _ in range(len(self.others)):
            for q in self.self_model.qs:
                posterior[k] = np.copy(self.qs[k])
                k += 1
        self.qs = posterior

        return self.qs, info


class BeliefShareAgent(Agent):
    """
    An agent that shares beliefs with other agents
    """

    def __init__(
        self,
        agent: Agent,
        others: list[Agent],
        shared_states,  # which states are belief-shared
    ):
        self.id = agent.id
        self.self_model = agent
        self.others = others
        self.shared_states = shared_states

        self.qs = self.self_model.qs

    def infer_action(self):
        # first update the current qs with info from others
        for k in range(len(self.qs)):
            if k in self.shared_states:
                log_posterior = pymdp.maths.spm_log_single(self.qs[k])
                for i in range(len(self.others)):
                    log_posterior += self.others[i].qL[k]

                self.qs[k] = pymdp.maths.softmax(log_posterior)

        self.self_model.qs = self.qs
        return self.self_model.infer_action()

    def infer_state(self, observation):
        # infer the new state of the focal agent
        qs, info = self.self_model.infer_state(observation)
        self.qs = qs
        return qs, info
