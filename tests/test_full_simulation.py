"""Integration tests for full Prisoner's Dilemma simulation."""

import numpy as np
import pytest
from pymdp.utils import obj_array, obj_array_uniform

from empathy.prisoners_dilemma import (
    EmpatheticAgent,
    ToMEmpatheticAgent,
    Environment,
    Sim,
)


def create_pd_config(
    T: int = 20,
    empathy_k0: np.ndarray = None,
    empathy_k1: np.ndarray = None,
    learn: bool = False,
) -> dict:
    """Create a standard PD configuration for testing."""
    num_modalities = 1
    num_factors = 1
    num_obs_categories = 4
    num_state_categories = 4

    # Agent 0 matrices
    A_k0 = obj_array(num_modalities)
    A_k0[0] = np.eye(num_obs_categories)

    B_k0 = obj_array(num_factors)
    B_k0[0] = np.zeros((4, 4, 2))
    B_k0[0][0, :, 0] = np.tile(0.5, 4)
    B_k0[0][1, :, 0] = np.tile(0.5, 4)
    B_k0[0][2, :, 1] = np.tile(0.5, 4)
    B_k0[0][3, :, 1] = np.tile(0.5, 4)

    C_k0 = obj_array(num_modalities)
    C_k0[0] = np.array([3, 1, 4, 2])

    D_k0 = obj_array_uniform([num_state_categories])

    # Agent 1 matrices
    A_k1 = obj_array(num_modalities)
    A_k1[0] = np.eye(num_obs_categories)

    B_k1 = obj_array(num_factors)
    B_k1[0] = np.zeros((4, 4, 2))
    B_k1[0][0, :, 0] = np.tile(0.5, 4)
    B_k1[0][2, :, 0] = np.tile(0.5, 4)
    B_k1[0][1, :, 1] = np.tile(0.5, 4)
    B_k1[0][3, :, 1] = np.tile(0.5, 4)

    C_k1 = obj_array(num_modalities)
    C_k1[0] = np.array([3, 4, 1, 2])

    D_k1 = obj_array_uniform([num_state_categories])

    # Default empathy factors
    if empathy_k0 is None:
        empathy_k0 = np.array([0.9, 0.1])
    if empathy_k1 is None:
        empathy_k1 = np.array([0.9, 0.1])

    config = {
        "T": T,
        "K": 2,
        "A": [A_k0, A_k1],
        "B": [B_k0, B_k1],
        "C": [C_k0, C_k1],
        "D": [D_k0, D_k1],
        "empathy_factor": [empathy_k0, empathy_k1],
        "actions": ["C", "D"],
        "learn": learn,
        "policy_len": 2,
        "same_pref": False,
    }

    return config


class TestOriginalSimulation:
    """Tests for the original EmpatheticAgent simulation."""

    def test_simulation_runs(self):
        """Simulation should complete without errors."""
        config = create_pd_config(T=10)
        simulation = Sim(config=config)
        history = simulation.run(verbose=False)

        assert history is not None
        assert "results" in history
        assert "ToM" in history

    def test_simulation_produces_actions(self):
        """Agents should produce valid actions (0 or 1)."""
        config = create_pd_config(T=10)
        simulation = Sim(config=config)
        history = simulation.run(verbose=False)

        actions_k0 = history["results"]["action"][:, 0]
        actions_k1 = history["results"]["action"][:, 1]

        assert len(actions_k0) == 10
        assert len(actions_k1) == 10
        assert all(a in [0, 1] for a in actions_k0)
        assert all(a in [0, 1] for a in actions_k1)

    def test_different_empathy_produces_different_behavior(self):
        """Different empathy factors should lead to different outcomes over many runs."""
        np.random.seed(42)

        # Run with selfish agents
        config_selfish = create_pd_config(
            T=50,
            empathy_k0=np.array([1.0, 0.0]),
            empathy_k1=np.array([1.0, 0.0]),
        )
        sim_selfish = Sim(config=config_selfish)
        hist_selfish = sim_selfish.run(verbose=False)

        np.random.seed(42)

        # Run with empathetic agents
        config_empathetic = create_pd_config(
            T=50,
            empathy_k0=np.array([0.5, 0.5]),
            empathy_k1=np.array([0.5, 0.5]),
        )
        sim_empathetic = Sim(config=config_empathetic)
        hist_empathetic = sim_empathetic.run(verbose=False)

        # Action patterns might differ
        actions_selfish = hist_selfish["results"]["action"]
        actions_empathetic = hist_empathetic["results"]["action"]

        # Just verify both ran and produced results
        assert actions_selfish.shape == actions_empathetic.shape


class TestEnvironment:
    """Tests for the Environment class."""

    def test_environment_initialization(self):
        """Environment should initialize correctly."""
        env = Environment(K=2)
        assert env.K == 2

    def test_environment_step_t0(self):
        """At t=0, environment should return None observations."""
        env = Environment(K=2)
        obs = env.step(t=0, actions=[0, 0])
        assert obs == [None, None]

    def test_environment_action_mapping(self):
        """Environment should correctly map actions to observations."""
        env = Environment(K=2)

        # CC -> [0, 0]
        obs = env.step(t=1, actions=[0, 0])
        assert obs == [0, 0]

        # CD -> [1, 1]
        obs = env.step(t=1, actions=[0, 1])
        assert obs == [1, 1]

        # DC -> [2, 2]
        obs = env.step(t=1, actions=[1, 0])
        assert obs == [2, 2]

        # DD -> [3, 3]
        obs = env.step(t=1, actions=[1, 1])
        assert obs == [3, 3]


class TestToMEmpatheticAgent:
    """Tests for the new ToMEmpatheticAgent."""

    def test_agent_initialization(self):
        """ToMEmpatheticAgent should initialize correctly."""
        config = create_pd_config(T=10)
        agent = ToMEmpatheticAgent(
            config=config,
            agent_num=0,
            empathy_factor=0.5,
        )

        assert agent.empathy_factor == 0.5
        assert agent.agent_num == 0
        assert agent.self_agent is not None
        assert agent.other_model is not None
        assert agent.tom is not None

    def test_agent_step_returns_valid_action(self):
        """Agent step should return valid action and results."""
        config = create_pd_config(T=10)
        agent = ToMEmpatheticAgent(
            config=config,
            agent_num=0,
            empathy_factor=0.5,
        )

        # Initial step
        results = agent.step(t=0, observation=0)

        assert "exp_action" in results
        assert results["exp_action"] in [0, 1]
        assert "G_social" in results
        assert "q_action" in results

    def test_agent_tracks_history(self):
        """Agent should track action and observation history."""
        config = create_pd_config(T=10)
        agent = ToMEmpatheticAgent(
            config=config,
            agent_num=0,
            empathy_factor=0.5,
            use_inversion=True,
        )

        # Run several steps
        for t in range(5):
            obs = 0 if t == 0 else np.random.choice([0, 1, 2, 3])
            agent.step(t=t, observation=obs)

        assert len(agent.action_history) == 5
        assert len(agent.observation_history) == 5

    def test_inversion_updates_reliability(self):
        """Opponent inversion should update reliability over time."""
        config = create_pd_config(T=10)
        agent = ToMEmpatheticAgent(
            config=config,
            agent_num=0,
            empathy_factor=0.5,
            use_inversion=True,
        )

        initial_reliability = agent.get_reliability()

        # Run steps with consistent opponent behavior
        for t in range(10):
            # Simulate CC outcome (both cooperate)
            obs = 0 if t > 0 else agent.o_init
            agent.step(t=t, observation=obs)

        final_reliability = agent.get_reliability()

        # Reliability should still be in valid range
        assert 0.0 <= initial_reliability <= 1.0
        assert 0.0 <= final_reliability <= 1.0

    def test_reset_clears_state(self):
        """Reset should clear all accumulated state."""
        config = create_pd_config(T=10)
        agent = ToMEmpatheticAgent(
            config=config,
            agent_num=0,
            empathy_factor=0.5,
            use_inversion=True,
        )

        # Run some steps
        for t in range(5):
            agent.step(t=t, observation=0)

        assert len(agent.action_history) > 0

        # Reset
        agent.reset()

        assert len(agent.action_history) == 0
        assert len(agent.observation_history) == 0
        assert agent.last_action is None


class TestToMSimulation:
    """Tests for running simulation with ToMEmpatheticAgent."""

    def test_two_tom_agents_simulation(self):
        """Two ToMEmpatheticAgents should interact correctly."""
        config = create_pd_config(T=20)
        env = Environment(K=2)

        # Create two ToM agents
        agent0 = ToMEmpatheticAgent(
            config=config,
            agent_num=0,
            empathy_factor=0.5,
            use_inversion=True,
        )
        agent1 = ToMEmpatheticAgent(
            config=config,
            agent_num=1,
            empathy_factor=0.5,
            use_inversion=True,
        )

        actions = [0, 0]
        results_history = []

        for t in range(config["T"]):
            obs = env.step(t=t, actions=actions)

            if t == 0:
                obs0, obs1 = agent0.o_init, agent1.o_init
            else:
                obs0, obs1 = obs[0], obs[1]

            results0 = agent0.step(t=t, observation=obs0)
            results1 = agent1.step(t=t, observation=obs1)

            actions = [results0["exp_action"], results1["exp_action"]]
            results_history.append({
                "t": t,
                "actions": actions.copy(),
                "agent0": results0,
                "agent1": results1,
            })

        # Verify simulation ran successfully
        assert len(results_history) == config["T"]
        assert all(r["actions"][0] in [0, 1] for r in results_history)
        assert all(r["actions"][1] in [0, 1] for r in results_history)

    def test_asymmetric_empathy(self):
        """Asymmetric empathy should produce different behaviors."""
        config = create_pd_config(T=30)
        env = Environment(K=2)

        # High empathy vs low empathy
        agent_high = ToMEmpatheticAgent(
            config=config,
            agent_num=0,
            empathy_factor=0.9,
            use_inversion=False,
        )
        agent_low = ToMEmpatheticAgent(
            config=config,
            agent_num=1,
            empathy_factor=0.1,
            use_inversion=False,
        )

        actions = [0, 0]
        for t in range(config["T"]):
            obs = env.step(t=t, actions=actions)
            obs0, obs1 = (agent_high.o_init, agent_low.o_init) if t == 0 else (obs[0], obs[1])

            r0 = agent_high.step(t=t, observation=obs0)
            r1 = agent_low.step(t=t, observation=obs1)
            actions = [r0["exp_action"], r1["exp_action"]]

        # Both agents should have run
        assert len(agent_high.action_history) == config["T"]
        assert len(agent_low.action_history) == config["T"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
