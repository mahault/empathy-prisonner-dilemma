import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt


def plot_beliefs(result):
    cols = len(result.keys())
    rows = len(result[next(iter(result))][0]["qs"]) - 1

    fig, ax = plt.subplots(rows, cols)
    if rows == 1:
        ax = [ax]

    if cols == 1:
        ax = [[a] for a in ax]

    for i in range(rows):
        for j, (agent, info) in enumerate(result.items()):
            qs_time = np.stack([info_t["qs"][i + 1] for info_t in info]).T

            ax[i][j].imshow(qs_time, cmap="gray_r", vmin=0.0, vmax=1.0)
            ax[i][j].set_xlabel("Time")
            ax[i][j].set_ylabel("Beliefs")
            if i == 0:
                ax[i][j].set_title(agent)

            ax[i][j].set_yticks(np.arange(qs_time.shape[0]))
            ax[i][j].set_yticklabels(np.arange(qs_time.shape[0]))
            ax[i][j].set_xticks(np.arange(qs_time.shape[1])[::5])
            ax[i][j].set_xticks(np.arange(qs_time.shape[1]), minor=True)
            ax[i][j].set_xticklabels(np.arange(qs_time.shape[1])[::5])

            if i == 0:
                locations = [np.argmax(info_t["qs"][0]) for info_t in info]
                ax[i][j].scatter(
                    np.arange(qs_time.shape[1]), locations, c="tab:blue"
                )

                object_locations = [info_t["object"] for info_t in info]
                ax[i][j].scatter(
                    np.arange(qs_time.shape[1]),
                    object_locations,
                    c="tab:red",
                    marker="x",
                    linewidths=1,
                )

    plt.show()


def plot_efe(result):
    fig, ax = plt.subplots(1, len(result.keys()))
    if len(result.keys()) == 1:
        ax = [ax]

    for i, (agent, info) in enumerate(result.items()):
        energy = np.array([np.min(info_t["G"]) for info_t in info])

        ax[i].plot(energy)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("EFE")
        ax[i].set_title(agent)

    plt.show()


def plot_vfe(result):
    fig, ax = plt.subplots(1, len(result.keys()))
    if len(result.keys()) == 1:
        ax = [ax]

    for i, (agent, info) in enumerate(result.items()):
        energy = np.array([np.min(info_t["F"]) for info_t in info])

        ax[i].plot(energy)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("VFE")
        ax[i].set_title(agent)

    plt.show()


def plot_h(result):
    fig, ax = plt.subplots(1, len(result.keys()))
    if len(result.keys()) == 1:
        ax = [ax]

    for i, (agent, info) in enumerate(result.items()):
        entropy = np.array([np.min(info_t["H"]) for info_t in info])

        ax[i].plot(entropy)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("H")
        ax[i].set_title(agent)

    plt.show()


def plot_u(result):
    fig, ax = plt.subplots(1, len(result.keys()))
    if len(result.keys()) == 1:
        ax = [ax]

    for i, (agent, info) in enumerate(result.items()):
        entropy = np.array([np.min(info_t["U"]) for info_t in info])

        ax[i].plot(entropy)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("U")
        ax[i].set_title(agent)

    plt.show()


def plot_plan_tree(
    root_node,
    policy_label=lambda x: "{:.2f}".format(x["G"]),
    observation_label=lambda x: "{:.2f}".format(x["G"]),
    font_size=12,
    depth=-1,
):
    # we can pass in a node or the whole tree list object
    if isinstance(root_node, list):
        root_node = root_node[0]

    # color observation / policy nodes for self / other
    colormap = [plt.cm.Blues, plt.cm.Purples]
    colormap_policy = [plt.cm.Reds, plt.cm.Greens]

    # plot depth relative to root node
    max_n = 5
    if depth > 0:
        max_n = root_node["n"] + depth

    # create graph
    count = 0
    G = nx.Graph()
    to_visit = [(root_node, 0)]
    labels = {}

    # label and color root node
    labels[0] = ""
    c = 1 if "agent" in root_node.keys() else 0
    if "policy" in root_node.keys():
        labels[count] = policy_label(root_node)
        r, g, b, a = colormap_policy[c](root_node["q_pi"])
    elif "observation" in root_node.keys():
        labels[count] = observation_label(root_node)
        r, g, b, a = colormap[c](root_node["prob"])
    else:
        r, g, b, a = colormap[c](0.5)

    node_color = [(r, g, b, a)]

    G.add_node(count)
    count += 1

    # visit children
    while len(to_visit) > 0:
        node, id = to_visit.pop()
        for child in node["children"]:
            c = 1 if "agent" in child.keys() else 0

            G.add_node(count)
            G.add_edge(id, count)
            if "policy" in child.keys():
                labels[count] = policy_label(child)
                r, g, b, a = colormap_policy[c](child["q_pi"])
            elif "observation" in child.keys():
                labels[count] = observation_label(child)
                r, g, b, a = colormap[c](child["prob"])

            node_color.append((r, g, b, a))

            if child["n"] <= max_n:
                to_visit.append((child, count))
            count += 1.0

    from networkx.drawing.nx_pydot import graphviz_layout

    pos = graphviz_layout(G, prog="dot")
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_color=node_color,
        font_size=font_size,
    )


def plot_circumplex(result):
    cols = len(result.keys())
    fig = plt.figure()

    for i, (agent, info) in enumerate(result.items()):

        xs = [info_t["valence"] for info_t in info]
        ys = [info_t["arousal"] * 2 - 1 for info_t in info]

        theta = [np.arctan2(y, x) for x, y in zip(xs, ys)]
        r = [np.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
        numbers = range(len(theta))

        ax = fig.add_subplot(polar=True)
        ax.scatter(theta, r, label="Agent 1")
        ax.plot(theta, r)
        for i, txt in enumerate(numbers):
            ax.annotate(txt, (theta[i], r[i]))
        ax.set_xticks(
            np.linspace(0, 2 * np.pi, 12, endpoint=False), minor=True
        )
        ax.set_xticklabels(
            [
                "$0\degree$ \n  Happy",
                "$45\degree$ \n     Excited",
                "$90\degree$ \n Alert",
                "$135\degree$ \n Angry   ",
                "$180\degree$ \n Sad  ",
                "$225\degree$ \n Depressed  ",
                " $270\degree$ \n Calm",
                "$315\degree$ \n Relaxed",
            ]
        )
