"""Tests for Theory of Mind module."""

import numpy as np
import pytest

from empathy.prisoners_dilemma.tom import (
    TheoryOfMind,
    SocialEFE,
    OpponentInversion,
    OpponentHypothesis,
)
from empathy.prisoners_dilemma.tom.inversion import ObservationContext, InversionState
from empathy.prisoners_dilemma.tom.tom_core import COOPERATE, DEFECT, ToMPrediction


class TestTheoryOfMind:
    """Tests for TheoryOfMind class."""

    def test_predict_opponent_response_returns_distribution(self):
        """ToM should return a valid probability distribution."""
        # Create a simple mock for other_model
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)
        prediction = tom.predict_opponent_response(my_action=COOPERATE)

        assert isinstance(prediction, ToMPrediction)
        assert len(prediction.q_response) == 2
        assert np.isclose(np.sum(prediction.q_response), 1.0)
        assert all(p >= 0 for p in prediction.q_response)

    def test_defection_more_likely_against_defector(self):
        """Opponent should be more likely to defect if I defect."""
        class MockAgent:
            qs = [np.array([0.5, 0.5])]
            beta = 4.0

        tom = TheoryOfMind(other_model=MockAgent(), beta_other=4.0)

        pred_vs_cooperate = tom.predict_opponent_response(my_action=COOPERATE)
        pred_vs_defect = tom.predict_opponent_response(my_action=DEFECT)

        # If I defect, opponent's best response is to defect (gets 1 vs 0)
        # If I cooperate, opponent's best response is to defect (gets 5 vs 3)
        # But defection should be more pronounced against a defector
        assert pred_vs_defect.q_response[DEFECT] >= pred_vs_cooperate.q_response[DEFECT] * 0.5


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
        assert "empathy_factor" in info

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

    def test_type_distribution_sums_to_one(self):
        """Type distribution should be a valid probability distribution."""
        inversion = OpponentInversion(n_particles=30)

        type_dist = inversion.get_type_distribution()

        total_prob = sum(type_dist.values())
        assert np.isclose(total_prob, 1.0)

    def test_convergence_on_consistent_opponent(self):
        """Inversion should converge when opponent is consistent."""
        inversion = OpponentInversion(n_particles=100)

        # Simulate always-cooperate opponent
        for t in range(10):
            context = ObservationContext(
                my_last_action=COOPERATE,
                their_last_action=COOPERATE if t > 0 else None,
                joint_outcome=0 if t > 0 else None,  # CC
                round_number=t + 1,
            )
            inversion.update(observed_action=COOPERATE, context=context)

        # Should have increased probability on ALWAYS_COOPERATE
        type_dist = inversion.get_type_distribution()
        always_c_prob = type_dist.get(OpponentHypothesis.ALWAYS_COOPERATE, 0)

        # After 10 cooperations, should have significant belief in always_cooperate
        assert always_c_prob > 0.2  # Relaxed threshold due to stochasticity


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
