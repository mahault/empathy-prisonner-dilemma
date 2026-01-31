"""
Integration tests: trust, reliability, social Dirichlet updates, and agent Dirichlet integration.
"""

import numpy as np

from empathy.clean_up.agent.agent import Agent
from empathy.clean_up.agent.beliefs import DirichletBeliefs
from empathy.clean_up.agent.social.trust import compute_confidence, compute_reliability
from empathy.clean_up.agent.social.update import social_dirichlet_update


class TestSocialLearningIntegration:
    """Test social learning with Dirichlet beliefs."""

    def test_reliability_prevents_update_when_zero(self):
        """Zero reliability should prevent Dirichlet updates."""
        beliefs_self = DirichletBeliefs(n_contexts=5, initial_alpha=2.0)
        alpha_before = {c: beliefs_self.alpha[c].copy() for c in range(5)}
        alpha_other_dict = {
            c: np.array([10.0, 10.0]) for c in range(5)
        }
        contexts_observed = [0, 1, 2]

        social_dirichlet_update(
            beliefs_self=beliefs_self,
            alpha_other_dict=alpha_other_dict,
            reliability=0.0,
            trust=1.0,
            eta_0=0.3,
            contexts_observed=contexts_observed,
        )

        for c in range(5):
            assert np.allclose(beliefs_self.alpha[c], alpha_before[c]), (
                "beliefs_self should be unchanged when reliability=0"
            )

    def test_reliability_filters_uncertain_tom(self):
        """Uniform weights should yield low reliability."""
        weights = np.ones(5) / 5
        confidence = compute_confidence(weights, len(weights))
        reliability = compute_reliability(confidence, u_threshold=0.3, kappa=0.2)
        assert reliability < 0.5

    def test_trust_scales_update_magnitude(self):
        """Higher trust should move beliefs more toward alpha_other."""
        beliefs_self = DirichletBeliefs(n_contexts=5, initial_alpha=2.0)
        alpha_other_dict = {c: np.array([20.0, 20.0]) for c in range(5)}
        contexts_observed = [0]

        beliefs_low = beliefs_self.copy()
        social_dirichlet_update(
            beliefs_self=beliefs_low,
            alpha_other_dict=alpha_other_dict,
            reliability=1.0,
            trust=0.2,
            eta_0=0.3,
            contexts_observed=contexts_observed,
        )

        beliefs_high = beliefs_self.copy()
        social_dirichlet_update(
            beliefs_self=beliefs_high,
            alpha_other_dict=alpha_other_dict,
            reliability=1.0,
            trust=1.0,
            eta_0=0.3,
            contexts_observed=contexts_observed,
        )

        # After high trust, context 0 should be closer to [20, 20] than after low trust
        dist_low = np.linalg.norm(beliefs_low.alpha[0] - np.array([20.0, 20.0]))
        dist_high = np.linalg.norm(beliefs_high.alpha[0] - np.array([20.0, 20.0]))
        assert dist_high < dist_low


class TestAgentDirichletIntegration:
    """Test agent with Dirichlet beliefs (functional behavior)."""

    def test_learn_from_observation_updates_beliefs(self):
        """Beliefs should update after learn_from_observation (spawn at pollution 2)."""
        config = {"self_learning": True, "use_tom": False, "learning_rate": 1.0}
        agent = Agent(config=config, agent_id=0)
        initial_alpha = agent.beliefs_self.get_alpha(context=2).copy()

        # first observation: pollution=2, no apples (establishes baseline)
        # encoding: pollution = (obs_value // 81) % 5, apples = (obs_value // 405) % 8
        # obs=162 gives pollution=2 (162//81=2), apples=0 (162//405=0)
        agent.learn_from_observation((162, 0))
        
        # second observation: pollution=2, apple spawned in cell 0
        # obs=567 gives pollution=2 (567//81=7, 7%5=2), apples=1 (567//405=1)
        agent.learn_from_observation((567, 0))
        updated_alpha = agent.beliefs_self.get_alpha(context=2)

        # alpha[1] should increase due to spawn event at pollution=2
        assert not np.array_equal(initial_alpha, updated_alpha)
        assert updated_alpha[1] > initial_alpha[1]

    def test_act_returns_valid_action(self):
        """Agent.act() should return action in [0, n_actions)."""
        agent = Agent(config={"use_tom": False}, agent_id=0)
        action = agent.act()
        assert 0 <= action < agent.n_actions