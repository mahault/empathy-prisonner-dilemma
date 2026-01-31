from empathy.graph_navigation.agents.base import Agent


class EmotionalAgent(Agent):
    """
    An agent that infers emotional states based on the (E)FE
    """

    def __init__(self, agent):
        self.agent = agent

    def imagine_futures(self, qs, policies):
        # TODO now also predict future emotional states and add to qs_pi
        pass

    def infer_action(self):
        pass

    def infer_state(self, observation):
        qs = self.agent.infer_state(observation)

        # TODO infer emotional state and add to qs

        return qs
