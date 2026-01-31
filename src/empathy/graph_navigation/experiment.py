import argparse
import networkx as nx
from empathy.graph_navigation import util
from empathy.graph_navigation.envs import GraphEnv
from empathy.graph_navigation.agents import *
from empathy.graph_navigation.visualize import *
import time
import numpy as np

np.random.seed(1)


def tick(agents, env, verbose=True):
    if not isinstance(agents, list):
        agents = [agents]

    # agents act in a random order
    order = np.arange(len(agents))
    np.random.shuffle(order)

    info = {a.id: {"agent": a.id} for a in agents}

    for idx in order:
        agent = agents[idx]
        action, plan_info = agent.infer_action()
        env_info = env.act(action, agent)

        info[agent.id].update({"action": action})
        info[agent.id].update(plan_info)
        info[agent.id].update(env_info)

    for idx in order:
        agent = agents[idx]
        observation = env.observe(agent)
        qs, state_info = agent.infer_state(observation)

        info[agent.id].update({"observation": observation, "qs": qs})
        info[agent.id].update(state_info)

    if verbose:
        for idx in order:
            agent = agents[idx]
            print(stringify(info[agent.id]))

    return info


def stringify(info):
    res = f"{info['agent']} moved to location {int(info['action'][0])}"
    if info["observation"][1][0] == 1:
        res += f" and observed object"

    if len(info["action"]) > 1:
        if info["action"][1] == 1:
            res = f"{info['agent']} eats"
        elif info["action"][1] == 2:
            res = f"{info['agent']} picks object"
        elif info["action"][1] == 3:
            res = f"{info['agent']} drops object"

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # generate graph
    graph, metadata = util.generate_line(5)
    # nx.draw(graph, with_labels=True, font_weight="bold")

    # define agents and their start locations
    agent_config = {
        "Impala": {"location": 0},
        "Lion": {"location": 4, "predator": True},
    }
    # define objects and their true locations
    object_config = {"Food": 2}

    # create environment
    env = GraphEnv(graph, agent_config, object_config, forage=True)

    # create agents
    agents = [
        ForageAgent(
            "Impala", env, [0], sophisticated=True, planning_horizon=2
        ),
        PredatorAgent("Lion", env, sophisticated=True, planning_horizon=2),
    ]

    # for agent in agents:
    #     D = agent.D
    #     D[1] = np.ones(D[1].shape) * 1e-3
    #     D[1][0] = 0.4
    #     D[1][4] = 0.6
    #     agent.set_prior(D)

    # add emotional inference
    # agents = [EmotionalAgent(a) for a in agents]

    # add Theory of mind
    # agents = [
    #     ToMAgent(
    #         a,
    #         others=[o for o in agents if o != a],
    #         self_states=[0, 3],  # own location is "self" state, and inventory
    #         observed_states=[
    #             0,
    #             2,
    #             3,
    #         ],  # observe other location and energy / inventory state?
    #         shared_states=[1],  # object location is "shared" state
    #     )
    #     for a in agents
    # ]

    # agents = [
    #     ToMAgent(
    #         a,
    #         others=[o for o in agents if o != a],
    #         self_states=[0],  # own location is "self" state, and inventory
    #         observed_states=[
    #             0,
    #             2,
    #         ],  # observe other location and energy / inventory state?
    #         shared_states=[1],  # object location is "shared" state
    #     )
    #     for a in agents
    # ]

    # add empathy
    # agents = [EmpathicToMAgent(a) for a in agents]

    result = {a: [] for a in agent_config.keys()}

    # simulate T timesteps

    T = 20
    for t in range(T):
        info = tick(agents, env)
        for agent, info in info.items():
            result[info["agent"]].append(info)

    plot_beliefs(result)
    # plot_vfe(result)
    # plot_efe(result)
    # plot_h(result)
    # plot_u(result)
