import numpy as np
from empathy.graph_navigation import agents
from collections import OrderedDict

MAX_ENERGY = 9


class AgentData:

    def __init__(self, id, location, predator=False):
        self.id = id
        self.start_location = location
        self.location = location
        self.predator = predator
        self.inventory = []
        self.energy = MAX_ENERGY

    def in_inventory(self, object_idx):
        for i in range(len(self.inventory)):
            if self.inventory[i] == object_idx:
                return i
        return -1


class ObjectData:

    def __init__(self, id, location):
        self.id = id
        self.location = location


class GraphEnv:

    def __init__(self, graph, agents_config, objects_config, forage=False):
        # graph is a networkx graph
        self.graph = graph

        # agents_config is a dict of str -> node id
        self.agents = OrderedDict()
        for id, config in agents_config.items():
            if isinstance(config, int):
                self.agents[id] = AgentData(id, config)
            elif isinstance(config, dict):
                self.agents[id] = AgentData(
                    id,
                    config.get("location"),
                    config.get("predator", False),
                )

        # objects_config is a dict of str -> node id
        self.objects = []
        for id, config in objects_config.items():
            self.objects.append(ObjectData(id, config))

        self.forage = forage

    def act(self, action, agent):
        agent_location_idx = self.agents[agent.id].location
        target_idx = int(action[0])

        # move agent
        if self.graph.has_edge(agent_location_idx, target_idx):
            self.agents[agent.id].location = target_idx

        # interact with object
        if len(action) > 1:
            i = 1
            if self.agents[agent.id].predator:
                # interested in consuming other agents
                for k, other in self.agents.items():
                    if other.id != agent.id:
                        object_action = int(action[i])

                        if object_action == 1:
                            # CONSUME
                            if (
                                other.location
                                == self.agents[agent.id].location
                            ):
                                # respawn other agent at start location?
                                other.location = other.start_location
                                print("Agent consumed")
                                self.agents[agent.id].energy = MAX_ENERGY + 1
                        i += 1
            else:
                for o in self.objects:
                    object_action = int(action[i])

                    if object_action == 1:
                        # CONSUME
                        consumed = False
                        if o.location == self.agents[agent.id].location:
                            consumed = True
                        inventory_idx = self.agents[agent.id].in_inventory(
                            o.id
                        )
                        if inventory_idx >= 0:
                            del self.agents[agent.id].inventory[inventory_idx]
                            consumed = True

                        if consumed:
                            # respawn object somewhere else
                            o.location = np.random.randint(
                                len(self.graph.nodes)
                            )
                            print(
                                "Object consumed, respawned at location ",
                                o.location,
                            )
                            self.agents[agent.id].energy = MAX_ENERGY + 1
                        else:
                            # tried to consume something that is not edible
                            self.agents[agent.id].energy = 0

                    elif object_action == 2:
                        # PICK
                        if o.location == self.agents[agent.id].location:
                            o.location = -1
                            self.agents[agent.id].inventory.append(o.id)
                            print("Object picked")
                    elif object_action == 3:
                        # DROP
                        inventory_idx = self.agents[agent.id].in_inventory(
                            o.id
                        )
                        if inventory_idx >= 0:
                            o.location = self.agents[agent.id].location
                            del self.agents[agent.id].inventory[inventory_idx]
                            print("Object dropped")

                    i += 1

        # energy goes down one
        self.agents[agent.id].energy = max(0, self.agents[agent.id].energy - 1)

        return {
            "object": self.objects[0].location,
            "energy": self.agents[agent.id].energy,
        }

    def observe(self, agent):
        obs = []
        # location observation
        obs.append([self.agents[agent.id].location])

        # object observation
        # 0 = not visible , 1 = visible, 2 = in inventory, 3 = consumed
        agent_location_idx = self.agents[agent.id].location
        if self.agents[agent.id].predator:
            # predators observe other agents instead of objects
            for k, other in self.agents.items():
                if other.id != agent.id:
                    if other.location == agent_location_idx:
                        # visible
                        obs.append([1])
                    else:
                        # not visible
                        obs.append([0])
        else:
            for o in self.objects:
                if self.agents[agent.id].in_inventory(o.id) >= 0:
                    # in inventory
                    obs.append([2])
                elif o.location == agent_location_idx:
                    # visible
                    obs.append([1])
                else:
                    # not visible
                    obs.append([0])

        # energy observation
        if self.forage or isinstance(agent, agents.ForageAgent):
            obs.append([self.agents[agent.id].energy])

        return obs
