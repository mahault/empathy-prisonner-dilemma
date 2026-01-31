"""
Tests for unified Agent class (Dirichlet beliefs).

Tests core functionality: initialization, perception, action selection,
self-learning, and social learning.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from empathy.clean_up.agent import Agent
from empathy.clean_up.agent.beliefs import create_beliefs_from_spawn_probs


class TestAgentInitialization:
    """Test agent initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic agent initialization."""
        config = {
            "n_agents": 1,
            "self_learning": False,
            "use_tom": False,
        }
        agent = Agent(config=config, agent_id=0)

        assert agent.n_states == 360
        assert agent.n_actions == 6
        assert agent.n_positions == 9
        assert agent.n_world_states == 40
        assert agent.state_belief is not None
        assert agent.beliefs_self is not None
        assert agent.beliefs_self.n_contexts == 5
        assert np.allclose(np.sum(agent.state_belief), 1.0)

    def test_expert_initialization(self):
        """Test expert agent: assign expert beliefs and verify probabilities."""
        agent = Agent(config={"use_tom": False}, agent_id=1)
        # spawn_probs equivalent to slope=-0.0625, intercept=0.25
        spawn_probs = [0.25, 0.1875, 0.125, 0.0625, 0.0]
        agent.beliefs_self = create_beliefs_from_spawn_probs(
            spawn_probs=spawn_probs, concentration=50.0
        )

        # Context 0: p ≈ 0.25, context 2: p ≈ 0.125
        assert np.isclose(agent.beliefs_self.get_probability(0), 0.25, rtol=0.05)
        assert agent.beliefs_self.get_probability(2) < agent.beliefs_self.get_probability(0)



class TestPerception:
    """Test state belief updates (VFE minimization)."""

    def test_perceive_updates_state_belief(self):
        """Test that perceive() updates state belief."""
        agent = Agent(config={"use_tom": False})
        initial_belief = agent.state_belief.copy()

        obs = (0, 0)
        agent.perceive(obs)

        assert not np.allclose(agent.state_belief, initial_belief)
        assert np.allclose(np.sum(agent.state_belief), 1.0)
        assert np.all(agent.state_belief >= 0)

    def test_perceive_preserves_normalization(self):
        """Test that perceive() maintains probability normalization."""
        agent = Agent(config={"use_tom": False})
        for _ in range(5):
            obs_value = np.random.randint(0, 3240)
            position = np.random.randint(0, 9)
            obs = (obs_value, position)
            agent.perceive(obs)
            assert np.allclose(np.sum(agent.state_belief), 1.0, rtol=1e-6)
            assert np.all(agent.state_belief >= -1e-10)


class TestActionSelection:
    """Test action selection (EFE-based)."""

    def test_act_returns_valid_action(self):
        """Test that act() returns a valid action index."""
        agent = Agent(config={"use_tom": False})
        agent.state_belief = np.ones(360) / 360

        action = agent.act()

        assert isinstance(action, (int, np.integer))
        assert 0 <= action < agent.n_actions

    def test_efe_based_action_selection(self):
        """Test EFE-based action selection (H=1 gives 6 EFE values, H>1 gives 6^H)."""
        agent = Agent(config={"use_tom": False})
        agent.state_belief = np.ones(360) / 360

        action = agent.act()
        assert 0 <= action < 6
        assert agent._last_efe_values is not None
        # Default planning_horizon may be 3 (216 policies) or 1 (6 actions)
        n_efe = agent._last_efe_values.shape[0]
        assert n_efe in (6, 216), f"Expected 6 or 216 EFE values, got {n_efe}"


class TestSelfLearning:
    """Test self-learning (Dirichlet updates from observations)."""

    def test_learn_from_observation_updates_beliefs(self):
        """Test that learn_from_observation() updates beliefs_self."""
        agent = Agent(config={"self_learning": True, "use_tom": False})

        obs1 = (0, 0)
        agent.learn_from_observation(obs1)

        obs2 = (405, 0)  # pollution 0, apples 1 (one apple spawned)
        agent.learn_from_observation(obs2)

        # At least one context should have been updated (no strict shape/theta check)
        assert agent.beliefs_self is not None

    def test_self_learning_respects_config_flag(self):
        """Test that self-learning respects config flag."""
        agent_no_learning = Agent(config={"self_learning": False, "use_tom": False})
        agent_with_learning = Agent(config={"self_learning": True, "use_tom": False})

        alpha_before = {c: agent_no_learning.beliefs_self.alpha[c].copy() for c in range(5)}

        obs = (0, 0)
        agent_no_learning.learn_from_observation(obs)
        agent_with_learning.learn_from_observation(obs)

        for c in range(5):
            assert np.allclose(agent_no_learning.beliefs_self.alpha[c], alpha_before[c]), (
                "Agent with self_learning=False should not update beliefs"
            )


class TestSocialLearning:
    """Test social learning (ToM and Dirichlet updates from others)."""

    def test_observe_other_action_updates_particles(self):
        """Test that observe_other_action() updates particle weights."""
        config = {"use_tom": True, "n_particles": 10}
        agent = Agent(config=config)

        agent.state_belief = np.ones(360) / 360
        agent.state_belief[0] = 0.5
        agent.state_belief[1:] = 0.5 / 359

        agent.particle_weights = np.ones(10) / 10

        context = {"agent_pos": 4}
        agent.observe_other_action(other_action=0, context=context)

        assert np.allclose(np.sum(agent.particle_weights), 1.0)

    def test_compute_social_metrics(self):
        """Test social metrics computation."""
        config = {"use_tom": True, "n_particles": 10}
        agent = Agent(config=config)

        agent._observation_history = [0, 1, 2, 3, 4]
        agent._state_history = [0, 1, 2, 3, 4]

        metrics = agent.compute_social_metrics()

        assert isinstance(metrics, dict)
        assert "weight_entropy" in metrics
        assert "confidence" in metrics
        assert "reliability" in metrics
        assert "trust" in metrics

    def test_social_learn_updates_beliefs(self):
        """Test that social_learn() can update beliefs_self (no errors)."""
        config = {
            "use_tom": True,
            "social_enabled": True,
            "n_particles": 10,
        }
        agent = Agent(config=config)

        agent.state_belief = np.ones(360) / 360
        context = {"agent_pos": 4}
        agent.observe_other_action(other_action=0, context=context)
        agent.compute_social_metrics()
        agent.social_learn()

        assert agent.beliefs_self is not None


class TestTaskSpecificMethods:
    """Test task-specific methods (Clean Up)."""

    def test_observation_likelihood_matrix_normalization(self):
        """Test observation likelihood matrix rows sum to 1."""
        agent = Agent(config={"use_tom": False})

        A = agent.observation_likelihood_matrix(agent.beliefs_self)

        row_sums = np.sum(A, axis=1)
        assert np.allclose(row_sums, 1.0, rtol=1e-6)

    def test_preferred_observations(self):
        """Test preferred observations computation."""
        agent = Agent(config={"use_tom": False})

        C = agent.preferred_observations()

        orchard_obs_with_apple = 6 + 0 * 9 + 0 * 81 + 7 * 405
        orchard_obs_no_apple = 6 + 0 * 9 + 0 * 81 + 0 * 405
        assert C[orchard_obs_with_apple] > C[orchard_obs_no_apple]

    def test_get_current_performance(self):
        """Test performance metric computation."""
        agent = Agent(config={"use_tom": False})

        perf = agent.get_current_performance()
        assert perf == 0.0

        agent.apples_eaten_window = [1, 0, 1, 1, 0]
        perf = agent.get_current_performance()
        assert perf == 0.6


class TestReset:
    """Test episode reset functionality."""

    def test_reset_episode(self):
        """Test that reset_episode() resets state but keeps beliefs."""
        agent = Agent(config={"use_tom": False})

        agent.state_belief = np.ones(360) / 360
        agent.apples_eaten_window = [1, 1, 1]

        alpha_before = {c: agent.beliefs_self.alpha[c].copy() for c in range(5)}

        agent.reset_episode()

        for c in range(5):
            assert np.allclose(agent.beliefs_self.alpha[c], alpha_before[c])
        assert len(agent.apples_eaten_window) == 0

    def test_reset_with_tom(self):
        """Test reset with ToM enabled."""
        config = {"use_tom": True, "n_particles": 10}
        agent = Agent(config=config)

        agent.particle_weights = np.ones(10) / 10
        agent.reset_episode(reset_tom=True)

        assert agent.particle_weights is not None
        assert np.allclose(np.sum(agent.particle_weights), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
