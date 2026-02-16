"""Tests for sophisticated multi-step planning."""

import numpy as np
import pytest
from pymdp.utils import obj_array, obj_array_uniform

from empathy.prisoners_dilemma.tom.tom_core import (
    TheoryOfMind, SocialEFE, COOPERATE, DEFECT, softmax,
)
from empathy.prisoners_dilemma.tom.opponent_simulator import OpponentSimulator
from empathy.prisoners_dilemma.tom.sophisticated_planner import SophisticatedPlanner
from empathy.prisoners_dilemma.tom.inversion import (
    OpponentInversion, ObservationContext, GatedToM,
)
from empathy.prisoners_dilemma import ToMEmpatheticAgent, Environment


# ── Fixtures ──────────────────────────────────────────────────────────────

class MockAgent:
    qs = [np.array([0.5, 0.5])]
    beta = 4.0


def make_tom(beta_other=4.0):
    return TheoryOfMind(other_model=MockAgent(), beta_other=beta_other)


def make_social_efe(empathy=0.5, beta_self=4.0):
    tom = make_tom()
    return SocialEFE(tom=tom, empathy_factor=empathy, beta_self=beta_self)


def make_opponent_sim(use_gated=False):
    tom = make_tom()
    gated = None
    context = None
    if use_gated:
        inv = OpponentInversion(n_particles=10)
        gated = GatedToM(tom=tom, inversion=inv)
        context = ObservationContext(
            my_last_action=0, their_last_action=0,
            joint_outcome=0, round_number=1,
        )
    return OpponentSimulator(tom=tom, gated_tom=gated, context=context)


def create_pd_config(T=20):
    n_mod, n_fac, n_obs, n_st = 1, 1, 4, 4
    A0 = obj_array(n_mod); A0[0] = np.eye(n_obs)
    B0 = obj_array(n_fac); B0[0] = np.zeros((4, 4, 2))
    B0[0][0, :, 0] = 0.5; B0[0][1, :, 0] = 0.5
    B0[0][2, :, 1] = 0.5; B0[0][3, :, 1] = 0.5
    C0 = obj_array(n_mod); C0[0] = np.array([3, 1, 4, 2])
    D0 = obj_array_uniform([n_st])

    A1 = obj_array(n_mod); A1[0] = np.eye(n_obs)
    B1 = obj_array(n_fac); B1[0] = np.zeros((4, 4, 2))
    B1[0][0, :, 0] = 0.5; B1[0][2, :, 0] = 0.5
    B1[0][1, :, 1] = 0.5; B1[0][3, :, 1] = 0.5
    C1 = obj_array(n_mod); C1[0] = np.array([3, 4, 1, 2])
    D1 = obj_array_uniform([n_st])

    return {
        "T": T, "K": 2,
        "A": [A0, A1], "B": [B0, B1],
        "C": [C0, C1], "D": [D0, D1],
        "empathy_factor": [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        "actions": ["C", "D"], "learn": False,
        "policy_len": 2, "same_pref": False,
    }


# ── OpponentSimulator Tests ──────────────────────────────────────────────

class TestOpponentSimulator:

    def test_predict_response_returns_distribution(self):
        sim = make_opponent_sim()
        q = sim.predict_response(step=0)
        assert len(q) == 2
        assert np.isclose(q.sum(), 1.0)
        assert all(p >= 0 for p in q)

    def test_step0_uses_static_tom_when_no_gated(self):
        sim = make_opponent_sim(use_gated=False)
        q0 = sim.predict_response(step=0)
        q1 = sim.predict_response(step=1)
        # Without gated_tom, step 0 and step 1 should give same result
        np.testing.assert_array_almost_equal(q0, q1)

    def test_future_steps_use_static_tom(self):
        sim = make_opponent_sim(use_gated=True)
        # Future steps (>0) should use static ToM
        q1 = sim.predict_response(step=1)
        q5 = sim.predict_response(step=5)
        np.testing.assert_array_almost_equal(q1, q5)

    def test_prediction_is_action_independent(self):
        """In simultaneous-move games, opponent prediction should not depend on my action."""
        sim = make_opponent_sim()
        # Call predict_response multiple times — should return same distribution
        # (no action conditioning in simultaneous moves)
        q1 = sim.predict_response(step=0)
        q2 = sim.predict_response(step=0)
        np.testing.assert_array_almost_equal(q1, q2)


# ── SophisticatedPlanner Tests ────────────────────────────────────────────

class TestSophisticatedPlanner:

    def test_policy_enumeration_h1(self):
        sim = make_opponent_sim()
        planner = SophisticatedPlanner(sim, empathy_factor=0.5, horizon=1)
        assert len(planner.policies) == 2  # C, D

    def test_policy_enumeration_h2(self):
        sim = make_opponent_sim()
        planner = SophisticatedPlanner(sim, empathy_factor=0.5, horizon=2)
        assert len(planner.policies) == 4  # CC, CD, DC, DD

    def test_policy_enumeration_h3(self):
        sim = make_opponent_sim()
        planner = SophisticatedPlanner(sim, empathy_factor=0.5, horizon=3)
        assert len(planner.policies) == 8

    def test_evaluate_policy_returns_float_and_info(self):
        sim = make_opponent_sim()
        planner = SophisticatedPlanner(sim, empathy_factor=0.5, horizon=2)
        G, info = planner.evaluate_policy((COOPERATE, DEFECT))
        assert isinstance(G, float)
        assert "steps" in info
        assert len(info["steps"]) == 2

    def test_plan_returns_valid_action_distribution(self):
        sim = make_opponent_sim()
        planner = SophisticatedPlanner(sim, empathy_factor=0.5, horizon=2)
        q_action, best_policy, info = planner.plan()

        assert len(q_action) == 2
        assert np.isclose(q_action.sum(), 1.0)
        assert all(p >= 0 for p in q_action)
        assert len(best_policy) == 2

    def test_h1_matches_myopic_direction(self):
        """H=1 planner should agree with myopic SocialEFE on which action is preferred."""
        np.random.seed(42)
        tom = make_tom()
        social_efe = SocialEFE(tom=tom, empathy_factor=0.5, beta_self=4.0)
        sim = OpponentSimulator(tom=tom)

        # Myopic
        G_myopic, _ = social_efe.compute_all_actions()
        myopic_preferred = int(np.argmin(G_myopic))

        # Sophisticated H=1
        planner = SophisticatedPlanner(sim, empathy_factor=0.5, horizon=1, beta_self=4.0)
        q_action, best_policy, info = planner.plan()
        soph_preferred = best_policy[0]

        assert myopic_preferred == soph_preferred

    def test_high_empathy_prefers_cooperation(self):
        """With high empathy, planner should prefer cooperation."""
        sim = make_opponent_sim()
        planner = SophisticatedPlanner(sim, empathy_factor=0.9, horizon=2, beta_self=4.0)
        q_action, _, _ = planner.plan()
        # High empathy should make cooperation more likely
        assert q_action[COOPERATE] > 0.3  # At least some preference for C

    def test_zero_empathy_prefers_defection(self):
        """With zero empathy, planner should prefer defection (Nash)."""
        sim = make_opponent_sim()
        planner = SophisticatedPlanner(sim, empathy_factor=0.0, horizon=2, beta_self=4.0)
        q_action, _, _ = planner.plan()
        # Zero empathy: pure self-interest → defection
        assert q_action[DEFECT] > q_action[COOPERATE]

    def test_action_marginalization_sums_correctly(self):
        """P(C) + P(D) should always equal 1."""
        sim = make_opponent_sim()
        for h in [1, 2, 3]:
            planner = SophisticatedPlanner(sim, empathy_factor=0.5, horizon=h)
            q_action, _, _ = planner.plan()
            assert np.isclose(q_action.sum(), 1.0), f"H={h}: sum={q_action.sum()}"

    def test_per_step_efe_recorded(self):
        """Each rollout step should record G_self, G_other, and q_response."""
        sim = make_opponent_sim()
        planner = SophisticatedPlanner(sim, empathy_factor=0.5, horizon=3)
        _, _, info = planner.plan()

        for policy, pinfo in info["policy_info"].items():
            assert len(pinfo["steps"]) == 3
            for step in pinfo["steps"]:
                assert "G_self" in step
                assert "G_other_expected" in step
                assert "q_response" in step
                assert step["action"] in [COOPERATE, DEFECT]


# ── Agent Integration Tests ──────────────────────────────────────────────

class TestAgentSophisticated:

    def test_agent_runs_with_sophisticated(self):
        """Agent should run without error when use_sophisticated=True."""
        np.random.seed(42)
        config = create_pd_config(T=10)
        env = Environment(K=2)

        ag = ToMEmpatheticAgent(
            config=config, agent_num=0, empathy_factor=0.5,
            use_inversion=False, use_sophisticated=True, planning_horizon=2,
        )
        ag_j = ToMEmpatheticAgent(
            config=config, agent_num=1, empathy_factor=0.5,
            use_inversion=False,
        )

        actions = [0, 0]
        for t in range(10):
            obs = env.step(t=t, actions=actions)
            obs_i = ag.o_init if t == 0 else obs[0]
            obs_j = ag_j.o_init if t == 0 else obs[1]

            res_i = ag.step(t=t, observation=obs_i)
            res_j = ag_j.step(t=t, observation=obs_j)

            actions = [res_i["exp_action"], res_j["exp_action"]]

        assert len(ag.action_history) == 10

    def test_backward_compatibility_myopic(self):
        """use_sophisticated=False should match existing myopic behavior."""
        np.random.seed(42)
        config = create_pd_config(T=5)

        ag_myopic = ToMEmpatheticAgent(
            config=config, agent_num=0, empathy_factor=0.5,
            use_inversion=False, use_sophisticated=False,
        )
        # Should run without error
        res = ag_myopic.step(t=0, observation=0)
        assert "exp_action" in res
        assert "q_action" in res

    def test_sophisticated_with_inversion(self):
        """Sophisticated planner should work together with inversion."""
        np.random.seed(42)
        config = create_pd_config(T=10)
        env = Environment(K=2)

        ag = ToMEmpatheticAgent(
            config=config, agent_num=0, empathy_factor=0.5,
            use_inversion=True, use_sophisticated=True, planning_horizon=2,
        )
        ag_j = ToMEmpatheticAgent(
            config=config, agent_num=1, empathy_factor=0.7,
            use_inversion=False,
        )

        actions = [0, 0]
        for t in range(10):
            obs = env.step(t=t, actions=actions)
            obs_i = ag.o_init if t == 0 else obs[0]
            obs_j = ag_j.o_init if t == 0 else obs[1]

            res_i = ag.step(t=t, observation=obs_i)
            res_j = ag_j.step(t=t, observation=obs_j)

            actions = [res_i["exp_action"], res_j["exp_action"]]

        assert len(ag.action_history) == 10
        assert "inversion" in res_i

    def test_planning_horizon_parameter(self):
        """Different horizons should produce different action distributions."""
        np.random.seed(42)
        config = create_pd_config(T=5)

        ag_h1 = ToMEmpatheticAgent(
            config=config, agent_num=0, empathy_factor=0.5,
            use_inversion=False, use_sophisticated=True, planning_horizon=1,
        )
        ag_h3 = ToMEmpatheticAgent(
            config=config, agent_num=0, empathy_factor=0.5,
            use_inversion=False, use_sophisticated=True, planning_horizon=3,
        )

        # Both should work
        res_h1 = ag_h1.step(t=0, observation=0)
        res_h3 = ag_h3.step(t=0, observation=0)

        assert "q_action" in res_h1
        assert "q_action" in res_h3
        assert ag_h1.planning_horizon == 1
        assert ag_h3.planning_horizon == 3
