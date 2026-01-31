import networkx
import typing




def generate_interactive_inference_maze(
    *_,
) -> typing.Tuple[networkx.Graph, dict]:
    edges = [
        (0, 1),
        (0, 5),
        (1, 2),
        (2, 3),
        (2, 6),
        (3, 4),
        (4, 7),
        (5, 8),
        (6, 10),
        (7, 12),
        (8, 9),
        (8, 13),
        (9, 10),
        (10, 11),
        (10, 14),
        (11, 12),
        (12, 15),
        (13, 16),
        (14, 18),
        (15, 20),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    return graph, {"locations": list(range(21))}


def generate_chain(
    length: int, closed=False
) -> typing.Tuple[networkx.Graph, dict]:
    edges = [
        (_from, _to) for _from, _to in zip(range(length), range(1, length))
    ]
    if closed:
        edges.append((length - 1, 0))
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    return graph, {"locations": list(range(length))}


def generate_t_maze(arms=3) -> typing.Tuple[networkx.Graph, dict]:
    if arms < 3:
        raise RuntimeError("T mazes require at least three arms")
    edges = [(0, a) for a in range(1, arms + 1)]
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    return graph, {
        "locations": ["center"] + [f"arm {i}" for i in range(1, arms + 1)]
    }


def generate_hourglass(nodes=5) -> typing.Tuple[networkx.Graph, dict]:
    edges = [(i, j) for i, j in zip(range(nodes), range(1, nodes))]
    edges += [(i, j) for i, j in zip(range(0, nodes, 2), range(2, nodes, 2))]
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    locations = []
    for i in range(nodes):
        if i > 1 and i % 2 == 0:
            locations.append(f"knot {i}")
        else:
            locations.append(f"link {i}")
    return graph, {"locations": locations}


def generate_hallway_with_rooms(rooms=2) -> typing.Tuple[networkx.Graph, dict]:
    edges = [(i, j) for i, j in zip(range(rooms + 1), range(1, rooms + 1))]
    hallway_nodes = len(edges)
    edges += [
        (i, j)
        for i, j in zip(
            range(hallway_nodes),
            range(hallway_nodes + 1, hallway_nodes + rooms + 1),
        )
    ]
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    return graph, {
        "locations": [
            (
                f"hallway {i}"
                if len(list(graph.neighbors(loc))) == 2
                else f"room {i}"
            )
            for i, loc in enumerate(graph.nodes)
        ]
    }


def generate_connected_clusters(
    cluster_size=2, connections=2
) -> typing.Tuple[networkx.Graph, dict]:
    edges = []
    connecting_node = 0
    while connecting_node < connections * cluster_size:
        edges += [
            (connecting_node, a)
            for a in range(
                connecting_node + 1, connecting_node + cluster_size + 1
            )
        ]
        connecting_node = len(edges)
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    return graph, {
        "locations": [
            (
                f"hallway {i}"
                if len(list(graph.neighbors(loc))) > 1
                else f"room {i}"
            )
            for i, loc in enumerate(graph.nodes)
        ]
    }

def generate_line(length: int) -> typing.Tuple[networkx.Graph, dict]:
    edges = [(i, i + 1) for i in range(length - 1)]
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    return graph, {"locations": list(range(length))}
