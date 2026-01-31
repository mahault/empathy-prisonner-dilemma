from empathy.graph_navigation.agents.base import Agent
from empathy.graph_navigation.agents.tom import ToMAgent
from empathy.graph_navigation.agents.emotional import EmotionalAgent


class EmpathicToMAgent(ToMAgent):
    """
    An agent that takes into account other agent's emotional states in the ToM planning
    """

    def __init__(self, agent: EmotionalAgent, others: list[Agent]):
        super().__init__(agent, others)

    def imagine_futures(self, policies):
        return super().imagine_futures(self, policies)

    def infer_action(self):
        # TODO same as ToM but now also score policies
        # based on other agent's estimated futureemotional states
        pass

    def infer_state(self, observation):
        return super().infer_state(self, observation)
