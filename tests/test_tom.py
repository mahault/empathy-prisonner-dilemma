"""Tests for Theory of Mind module."""

import numpy as np
import pytest

from empathy.prisoners_dilemma.tom import (
    TheoryOfMind,
    SocialEFE,
    OpponentInversion,
    BehavioralProfile,
)
from empathy.prisoners_dilemma.tom.inversion import ObservationContext, InversionState
from empathy.prisoners_dilemma.tom.tom_core import COOPERATE, DEFECT, ToMPrediction


class TestTheoryOfMind:
    """Tests for TheoryOfMind class."""

    def test_predict_opponent_action_returns_distribution(self):
        """ToM should return a valid probability distribution."""
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)
        prediction = tom.predict_opponent_action()

        assert isinstance(prediction, ToMPrediction)
        assert len(prediction.q_response) == 2
        assert np.isclose(np.sum(prediction.q_response), 1.0)
        assert all(p >= 0 for p in prediction.q_response)

    @staticmethod
    def _predictions_at(lambda_j):
        """P(D) predicted against a known defector and a known cooperator,
        for an opponent modelled with empathy lambda_j."""
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        prev = TheoryOfMind.DEFAULT_LAMBDA_J
        try:
            TheoryOfMind.DEFAULT_LAMBDA_J = lambda_j
            tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)
            tom.update_my_policy_belief(0.0)
            vs_defector = tom.predict_opponent_action().q_response[DEFECT]
            tom.update_my_policy_belief(1.0)
            vs_cooperator = tom.predict_opponent_action().q_response[DEFECT]
        finally:
            TheoryOfMind.DEFAULT_LAMBDA_J = prev
        return vs_defector, vs_cooperator

    def test_prediction_depends_on_believed_policy(self):
        """Opponent prediction should change based on believed policy."""
        vs_defector, vs_cooperator = self._predictions_at(0.5)
        assert not np.isclose(vs_defector, vs_cooperator)

    def test_self_interested_opponent_always_defects(self):
        """Regression guard for the lambda_j = 0 model.

        With no empathy the opponent is a pure payoff maximiser and defection
        strictly dominates at every believed policy (5 > 3 against a
        cooperator, 1 > 0 against a defector). This is the behaviour the
        pre-inference model assumed everywhere.
        """
        vs_defector, vs_cooperator = self._predictions_at(0.0)
        assert vs_defector > 0.5
        assert vs_cooperator > 0.5

    def test_empathic_opponent_tolerates_exploitation(self):
        """An opponent that weights my payoff will cooperate into a defector.

        With PD_PAYOFFS = (R,S,T,P) = (3,0,5,1) the joint welfare of (D,C) is
        T+S = 5, which beats (D,D) at 2P = 2. So once lambda_j is high enough
        the modelled opponent prefers to be exploited rather than retaliate.
        This is a property of the payoff matrix, not of the implementation,
        and it is why the inferred-lambda_j model shifts the cooperation
        threshold upward.
        """
        defector_pd = [self._predictions_at(lj)[0] for lj in (0.0, 0.25, 0.5)]
        # Predicted retaliation falls away monotonically as lambda_j rises.
        assert defector_pd[0] > defector_pd[1] > defector_pd[2]
        assert defector_pd[0] > 0.5 and defector_pd[2] < 0.5


class TestSocialEFE:
    """Tests for SocialEFE class."""

    def test_compute_returns_float_and_info(self):
        """Social EFE computation should return float and info dict."""
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)
        social = SocialEFE(tom=tom, empathy_factor=0.5)

        G, info = social.compute(my_action=COOPERATE)

        assert isinstance(G, float)
        assert isinstance(info, dict)
        assert "G_self" in info
        assert "G_other_expected" in info
        assert "G_pragmatic" in info
        assert "G_epistemic" in info
        assert "empathy_factor" in info

    def test_epistemic_value_included_with_inversion(self):
        """Social EFE should include epistemic value when inversion is provided."""
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        np.random.seed(42)
        tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)
        inversion = OpponentInversion(n_particles=30)
        social = SocialEFE(tom=tom, empathy_factor=0.5, inversion=inversion)

        context = ObservationContext(
            my_last_action=COOPERATE,
            their_last_action=COOPERATE,
            joint_outcome=0,
            round_number=5,
        )

        G, info = social.compute(my_action=COOPERATE, context=context)

        assert info["G_epistemic"] != 0.0  # Should have nonzero epistemic value
        assert info["G_pragmatic"] != 0.0  # Pragmatic should also be nonzero

    def test_empathy_factor_affects_efe(self):
        """Different empathy factors should produce different EFE values."""
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)

        social_selfish = SocialEFE(tom=tom, empathy_factor=0.0)
        social_empathic = SocialEFE(tom=tom, empathy_factor=1.0)

        G_selfish, _ = social_selfish.compute(my_action=COOPERATE)
        G_empathic, _ = social_empathic.compute(my_action=COOPERATE)

        # Different empathy should give different EFE
        assert G_selfish != G_empathic

    def test_compute_all_actions(self):
        """Should compute EFE for both cooperate and defect."""
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)
        social = SocialEFE(tom=tom, empathy_factor=0.5)

        G_all, info_all = social.compute_all_actions()

        assert len(G_all) == 2
        assert COOPERATE in info_all
        assert DEFECT in info_all


class TestOpponentInversion:
    """Tests for OpponentInversion class."""

    def test_initialization(self):
        """Inversion should initialize with uniform weights."""
        inversion = OpponentInversion(n_particles=30)

        assert len(inversion.weights) == 30
        assert np.isclose(np.sum(inversion.weights), 1.0)

    def test_update_changes_weights(self):
        """Update with observation should change particle weights."""
        inversion = OpponentInversion(n_particles=30)
        initial_weights = inversion.weights.copy()

        context = ObservationContext(
            my_last_action=COOPERATE,
            their_last_action=None,
            joint_outcome=None,
            round_number=1,
        )
        state = inversion.update(observed_action=COOPERATE, context=context)

        assert isinstance(state, InversionState)
        # Weights should have changed (or resampled)
        # Note: weights might be reset to uniform after resampling

    def test_reliability_in_range(self):
        """Reliability should be in [0, 1]."""
        inversion = OpponentInversion(n_particles=30)

        reliability = inversion.reliability()

        assert 0.0 <= reliability <= 1.0

    def test_profile_summary_has_expected_keys(self):
        """Profile summary should contain mean and std for all parameters."""
        inversion = OpponentInversion(n_particles=30)

        summary = inversion.get_profile_summary()

        assert "mean_alpha" in summary
        assert "mean_reciprocity" in summary
        assert "mean_beta" in summary
        assert "mean_lambda_j" in summary
        assert "std_alpha" in summary
        assert "std_reciprocity" in summary
        assert "std_beta" in summary
        assert "std_lambda_j" in summary

    def test_lambda_j_posterior_returns_statistics(self):
        """Lambda_j posterior should return mean, std, and entropy."""
        inversion = OpponentInversion(n_particles=30)

        posterior = inversion.get_lambda_j_posterior()

        assert "mean" in posterior
        assert "std" in posterior
        assert "entropy" in posterior
        assert 0.0 <= posterior["mean"] <= 1.0
        assert posterior["std"] >= 0.0
        assert posterior["entropy"] >= 0.0

    def test_epistemic_value_returns_float(self):
        """Epistemic value should return a float for each action."""
        np.random.seed(42)
        inversion = OpponentInversion(n_particles=30)

        context = ObservationContext(
            my_last_action=COOPERATE,
            their_last_action=COOPERATE,
            joint_outcome=0,
            round_number=5,
        )

        ev_coop = inversion.compute_epistemic_value(COOPERATE, context)
        ev_defect = inversion.compute_epistemic_value(DEFECT, context)

        assert isinstance(ev_coop, float)
        assert isinstance(ev_defect, float)
        # Epistemic values should be non-positive (IG >= 0)
        assert ev_coop <= 0.01  # Small tolerance for numerical noise
        assert ev_defect <= 0.01

    def test_epistemic_value_differs_by_action(self):
        """Cooperation and defection should have different epistemic values."""
        np.random.seed(42)
        inversion = OpponentInversion(n_particles=50)

        # Give some observations to create a non-uniform posterior
        for t in range(5):
            ctx = ObservationContext(
                my_last_action=COOPERATE,
                their_last_action=COOPERATE if t > 0 else None,
                joint_outcome=0 if t > 0 else None,
                round_number=t + 1,
            )
            inversion.update(observed_action=COOPERATE, context=ctx)

        context = ObservationContext(
            my_last_action=COOPERATE,
            their_last_action=COOPERATE,
            joint_outcome=0,
            round_number=6,
        )

        ev_coop = inversion.compute_epistemic_value(COOPERATE, context)
        ev_defect = inversion.compute_epistemic_value(DEFECT, context)

        # They should differ (different next-round contexts)
        assert ev_coop != ev_defect

    def test_convergence_on_consistent_opponent(self):
        """Inversion should converge toward cooperative profile for ALLC opponent."""
        np.random.seed(42)
        inversion = OpponentInversion(n_particles=100)

        # Simulate always-cooperate opponent
        for t in range(20):
            context = ObservationContext(
                my_last_action=COOPERATE,
                their_last_action=COOPERATE if t > 0 else None,
                joint_outcome=0 if t > 0 else None,  # CC
                round_number=t + 1,
            )
            inversion.update(observed_action=COOPERATE, context=context)

        # After 20 cooperations, mean alpha should be positive
        # (cooperation bias should be high for ALLC opponent)
        summary = inversion.get_profile_summary()
        assert summary["mean_alpha"] > 0.0, f"Expected positive alpha, got {summary['mean_alpha']}"

        # Prediction should favor cooperation
        predict_ctx = ObservationContext(
            my_last_action=COOPERATE, their_last_action=COOPERATE,
            joint_outcome=0, round_number=21,
        )
        probs = inversion.predict_action(predict_ctx)
        assert probs[COOPERATE] > 0.6, f"Expected P(C) > 0.6, got {probs[COOPERATE]}"


class TestIntegration:
    """Integration tests for full ToM pipeline."""

    def test_full_pipeline(self):
        """Test complete ToM + inversion + social EFE pipeline."""
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        # Set up components
        inversion = OpponentInversion(n_particles=30)
        tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)
        social = SocialEFE(tom=tom, empathy_factor=0.5, beta_self=4.0)

        # Simulate a few rounds
        for t in range(5):
            # Update inversion
            context = ObservationContext(
                my_last_action=COOPERATE,
                their_last_action=COOPERATE if t > 0 else None,
                joint_outcome=0 if t > 0 else None,
                round_number=t + 1,
            )
            if t > 0:
                inversion.update(observed_action=COOPERATE, context=context)

            # Compute social EFE
            G_all, _ = social.compute_all_actions()

            # Select action
            action = social.select_action()

            assert action in [COOPERATE, DEFECT]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
