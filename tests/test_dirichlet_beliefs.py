"""
Unit tests for DirichletBeliefs class.

Tests cover:
- Initialization
- Probability computation
- Belief updates
- Uncertainty computation
- Serialization/deserialization
- Edge cases and error handling
"""

import pytest
import numpy as np
from empathy.clean_up.agent.beliefs import DirichletBeliefs, create_beliefs_from_spawn_probs


class TestDirichletBeliefsInitialization:
    """Test initialization of DirichletBeliefs."""

    def test_default_initialization(self):
        """Test default initialization with uniform prior."""
        beliefs = DirichletBeliefs(n_contexts=5, initial_alpha=1.0)

        assert beliefs.n_contexts == 5
        assert len(beliefs.alpha) == 5

        for context in range(5):
            alpha = beliefs.alpha[context]
            assert alpha.shape == (2,)
            assert np.allclose(alpha, [1.0, 1.0])

    def test_custom_initial_alpha(self):
        """Test initialization with custom alpha value."""
        beliefs = DirichletBeliefs(n_contexts=3, initial_alpha=2.5)

        for context in range(3):
            alpha = beliefs.alpha[context]
            assert np.allclose(alpha, [2.5, 2.5])

    def test_initialization_with_alpha_dict(self):
        """Test initialization with pre-specified alpha values."""
        alpha_dict = {
            0: np.array([10.0, 5.0]),
            1: np.array([3.0, 7.0]),
            2: np.array([1.0, 1.0])
        }

        beliefs = DirichletBeliefs(n_contexts=3, alpha_dict=alpha_dict)

        assert beliefs.n_contexts == 3
        assert np.allclose(beliefs.alpha[0], [10.0, 5.0])
        assert np.allclose(beliefs.alpha[1], [3.0, 7.0])
        assert np.allclose(beliefs.alpha[2], [1.0, 1.0])


class TestProbabilityComputation:
    """Test probability computation from Dirichlet beliefs."""

    def test_uniform_prior_probability(self):
        """Test that uniform prior gives p=0.5."""
        beliefs = DirichletBeliefs(n_contexts=5, initial_alpha=1.0)

        for context in range(5):
            prob = beliefs.get_probability(context)
            assert np.isclose(prob, 0.5)

    def test_probability_with_observations(self):
        """Test probability computation after observations."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=1.0)

        # Update with 3 spawns and 1 no-spawn
        beliefs.update(context=0, outcome=True, learning_rate=1.0)   # spawn
        beliefs.update(context=0, outcome=True, learning_rate=1.0)   # spawn
        beliefs.update(context=0, outcome=True, learning_rate=1.0)   # spawn
        beliefs.update(context=0, outcome=False, learning_rate=1.0)  # no-spawn

        # Expected: alpha = [1+1, 1+3] = [2, 4]
        # p_spawn = 4 / (2 + 4) = 2/3
        prob = beliefs.get_probability(context=0)
        assert np.isclose(prob, 2.0 / 3.0)

    def test_probability_extreme_values(self):
        """Test probability computation with extreme alpha values."""
        alpha_dict = {
            0: np.array([100.0, 1.0]),    # Very low probability
            1: np.array([1.0, 100.0]),    # Very high probability
            2: np.array([50.0, 50.0])     # Equal probability
        }

        beliefs = DirichletBeliefs(n_contexts=3, alpha_dict=alpha_dict)

        assert np.isclose(beliefs.get_probability(0), 1.0 / 101.0)
        assert np.isclose(beliefs.get_probability(1), 100.0 / 101.0)
        assert np.isclose(beliefs.get_probability(2), 0.5)

    def test_get_all_probabilities(self):
        """Test getting all probabilities at once."""
        beliefs = DirichletBeliefs(n_contexts=3, initial_alpha=1.0)
        beliefs.update(context=0, outcome=True, learning_rate=2.0)
        beliefs.update(context=1, outcome=False, learning_rate=3.0)

        probs = beliefs.get_all_probabilities()

        assert len(probs) == 3
        assert np.isclose(probs[0], 3.0 / 4.0)  # [1+0, 1+2] = [1, 3]
        assert np.isclose(probs[1], 1.0 / 5.0)  # [1+3, 1+0] = [4, 1]
        assert np.isclose(probs[2], 0.5)        # [1, 1]

    def test_invalid_context_error(self):
        """Test that invalid context raises error."""
        beliefs = DirichletBeliefs(n_contexts=3)

        with pytest.raises(ValueError, match="Context 5 not in belief model"):
            beliefs.get_probability(context=5)


class TestBeliefUpdates:
    """Test belief update mechanics."""

    def test_update_spawn_outcome(self):
        """Test update with spawn outcome."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=1.0)

        beliefs.update(context=0, outcome=True, learning_rate=1.0)

        alpha = beliefs.alpha[0]
        assert np.allclose(alpha, [1.0, 2.0])

    def test_update_no_spawn_outcome(self):
        """Test update with no-spawn outcome."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=1.0)

        beliefs.update(context=0, outcome=False, learning_rate=1.0)

        alpha = beliefs.alpha[0]
        assert np.allclose(alpha, [2.0, 1.0])

    def test_update_custom_learning_rate(self):
        """Test update with custom learning rate."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=1.0)

        beliefs.update(context=0, outcome=True, learning_rate=2.5)

        alpha = beliefs.alpha[0]
        assert np.allclose(alpha, [1.0, 3.5])

    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=1.0)

        # Simulate 10 observations: 7 spawns, 3 no-spawns
        for _ in range(7):
            beliefs.update(context=0, outcome=True, learning_rate=1.0)
        for _ in range(3):
            beliefs.update(context=0, outcome=False, learning_rate=1.0)

        alpha = beliefs.alpha[0]
        assert np.allclose(alpha, [4.0, 8.0])

        prob = beliefs.get_probability(context=0)
        assert np.isclose(prob, 8.0 / 12.0)

    def test_update_invalid_learning_rate(self):
        """Test that negative learning rate raises error."""
        beliefs = DirichletBeliefs(n_contexts=1)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            beliefs.update(context=0, outcome=True, learning_rate=-1.0)

    def test_update_invalid_context(self):
        """Test that invalid context raises error."""
        beliefs = DirichletBeliefs(n_contexts=3)

        with pytest.raises(ValueError, match="Context 10 not in belief model"):
            beliefs.update(context=10, outcome=True, learning_rate=1.0)


class TestUncertaintyComputation:
    """Test uncertainty (concentration) computation."""

    def test_initial_uncertainty(self):
        """Test initial uncertainty with uniform prior."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=1.0)

        uncertainty = beliefs.get_uncertainty(context=0)

        # Total alpha = 2.0, uncertainty = 1/2.0 = 0.5
        assert np.isclose(uncertainty, 0.5)

    def test_uncertainty_decreases_with_observations(self):
        """Test that uncertainty decreases as more observations are made."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=1.0)

        uncertainty_initial = beliefs.get_uncertainty(context=0)

        # Add 10 observations
        for _ in range(10):
            beliefs.update(context=0, outcome=True, learning_rate=1.0)

        uncertainty_after = beliefs.get_uncertainty(context=0)

        # Uncertainty should decrease
        assert uncertainty_after < uncertainty_initial

        # Total alpha = 1 + 1 + 10 = 12, uncertainty = 1/12
        assert np.isclose(uncertainty_after, 1.0 / 12.0)

    def test_uncertainty_different_contexts(self):
        """Test that uncertainty can differ across contexts."""
        beliefs = DirichletBeliefs(n_contexts=3, initial_alpha=1.0)

        # Context 0: no additional observations
        # Context 1: 5 observations
        # Context 2: 20 observations

        for _ in range(5):
            beliefs.update(context=1, outcome=True, learning_rate=1.0)

        for _ in range(20):
            beliefs.update(context=2, outcome=False, learning_rate=1.0)

        u0 = beliefs.get_uncertainty(context=0)
        u1 = beliefs.get_uncertainty(context=1)
        u2 = beliefs.get_uncertainty(context=2)

        # More observations -> lower uncertainty
        assert u0 > u1 > u2


class TestCopyAndSerialization:
    """Test copying and serialization of beliefs."""

    def test_copy_creates_independent_instance(self):
        """Test that copy creates an independent instance."""
        beliefs = DirichletBeliefs(n_contexts=2, initial_alpha=1.0)
        beliefs.update(context=0, outcome=True, learning_rate=2.0)

        beliefs_copy = beliefs.copy()

        # Modify original
        beliefs.update(context=0, outcome=True, learning_rate=5.0)

        # Copy should not be affected
        assert not np.allclose(beliefs.alpha[0], beliefs_copy.alpha[0])
        assert np.allclose(beliefs_copy.alpha[0], [1.0, 3.0])  # Original state

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        beliefs = DirichletBeliefs(n_contexts=2, initial_alpha=1.0)
        beliefs.update(context=0, outcome=True, learning_rate=1.0)
        beliefs.update(context=1, outcome=False, learning_rate=2.0)

        data = beliefs.to_dict()

        assert data['n_contexts'] == 2
        assert len(data['alpha']) == 2
        assert np.allclose(data['alpha']['0'], [1.0, 2.0])
        assert np.allclose(data['alpha']['1'], [3.0, 1.0])

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            'n_contexts': 3,
            'alpha': {
                '0': [5.0, 10.0],
                '1': [2.0, 3.0],
                '2': [1.0, 1.0]
            }
        }

        beliefs = DirichletBeliefs.from_dict(data)

        assert beliefs.n_contexts == 3
        assert np.allclose(beliefs.alpha[0], [5.0, 10.0])
        assert np.allclose(beliefs.alpha[1], [2.0, 3.0])
        assert np.allclose(beliefs.alpha[2], [1.0, 1.0])

    def test_round_trip_serialization(self):
        """Test that serialize + deserialize preserves beliefs."""
        beliefs_original = DirichletBeliefs(n_contexts=5, initial_alpha=2.0)

        for context in range(5):
            for _ in range(context + 1):
                beliefs_original.update(context=context, outcome=True, learning_rate=1.0)

        # Serialize and deserialize
        data = beliefs_original.to_dict()
        beliefs_restored = DirichletBeliefs.from_dict(data)

        # Check all alphas match
        for context in range(5):
            assert np.allclose(
                beliefs_original.alpha[context],
                beliefs_restored.alpha[context]
            )


class TestBeliefCreation:
    """Test creation of beliefs from spawn probability arrays."""

    def test_create_beliefs_from_spawn_probs(self):
        """Test beliefs created from spawn probability array."""
        spawn_probs = [0.25, 0.15, 0.05, 0.0, 0.0]
        expert_beliefs = create_beliefs_from_spawn_probs(
            spawn_probs=spawn_probs,
            concentration=100.0
        )

        assert expert_beliefs.n_contexts == 5

        # Check probabilities match input spawn_probs
        for context in range(5):
            estimated_prob = expert_beliefs.get_probability(context)
            # Should be very close due to high concentration
            assert np.isclose(estimated_prob, spawn_probs[context], atol=0.01)

    def test_beliefs_high_concentration(self):
        """Test that beliefs with high concentration have low uncertainty."""
        spawn_probs = [0.25, 0.15, 0.05, 0.0, 0.0]
        expert_beliefs = create_beliefs_from_spawn_probs(
            spawn_probs=spawn_probs,
            concentration=100.0
        )

        for context in range(5):
            uncertainty = expert_beliefs.get_uncertainty(context)

            # Total concentration should be ~100, uncertainty = 1/100
            assert uncertainty < 0.02

    def test_beliefs_spawn_probabilities_decreasing(self):
        """Test that probabilities decrease with pollution."""
        spawn_probs = [0.25, 0.1875, 0.125, 0.0625, 0.0]
        expert_beliefs = create_beliefs_from_spawn_probs(
            spawn_probs=spawn_probs,
            concentration=100.0
        )

        probs = expert_beliefs.get_all_probabilities()

        # Probabilities should decrease with pollution
        for context in range(4):
            assert probs[context] >= probs[context + 1]


class TestStringRepresentations:
    """Test string representations of beliefs."""

    def test_repr(self):
        """Test __repr__ method."""
        beliefs = DirichletBeliefs(n_contexts=2, initial_alpha=1.0)

        repr_str = repr(beliefs)

        assert "DirichletBeliefs" in repr_str
        assert "0:0.500" in repr_str
        assert "1:0.500" in repr_str

    def test_str(self):
        """Test __str__ method."""
        beliefs = DirichletBeliefs(n_contexts=2, initial_alpha=1.0)
        beliefs.update(context=0, outcome=True, learning_rate=3.0)

        str_repr = str(beliefs)

        assert "DirichletBeliefs:" in str_repr
        assert "Context 0:" in str_repr
        assert "Context 1:" in str_repr
        assert "p=" in str_repr
        assert "alpha=" in str_repr
        assert "uncertainty=" in str_repr


class TestNumericalStability:
    """Test numerical stability of belief updates."""

    def test_probabilities_always_in_bounds(self):
        """Test that probabilities always stay in [0, 1]."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=0.001)

        # Many updates
        for _ in range(1000):
            outcome = np.random.rand() > 0.5
            beliefs.update(context=0, outcome=outcome, learning_rate=1.0)

            prob = beliefs.get_probability(context=0)
            assert 0.0 <= prob <= 1.0

    def test_alpha_always_positive(self):
        """Test that alpha concentrations always remain positive."""
        beliefs = DirichletBeliefs(n_contexts=5, initial_alpha=1.0)

        # Many random updates
        for _ in range(1000):
            context = np.random.randint(0, 5)
            outcome = np.random.rand() > 0.5
            learning_rate = np.random.uniform(0.1, 2.0)

            beliefs.update(context=context, outcome=outcome, learning_rate=learning_rate)

            # All alphas should remain positive
            for ctx in range(5):
                alpha = beliefs.alpha[ctx]
                assert np.all(alpha > 0)

    def test_no_nan_or_inf(self):
        """Test that no NaN or infinity values appear."""
        beliefs = DirichletBeliefs(n_contexts=3, initial_alpha=1.0)

        # Extreme updates
        for _ in range(100):
            context = np.random.randint(0, 3)
            outcome = True
            beliefs.update(context=context, outcome=outcome, learning_rate=10.0)

            # Check for NaN/inf
            for ctx in range(3):
                alpha = beliefs.alpha[ctx]
                assert np.all(np.isfinite(alpha))

                prob = beliefs.get_probability(ctx)
                assert np.isfinite(prob)

    def test_convergence_with_consistent_data(self):
        """Test convergence to correct probability with consistent data."""
        beliefs = DirichletBeliefs(n_contexts=1, initial_alpha=1.0)

        # 80% spawn rate
        n_observations = 1000
        n_spawns = 800

        for _ in range(n_spawns):
            beliefs.update(context=0, outcome=True, learning_rate=1.0)

        for _ in range(n_observations - n_spawns):
            beliefs.update(context=0, outcome=False, learning_rate=1.0)

        prob = beliefs.get_probability(context=0)

        # Should be very close to 0.8
        assert np.isclose(prob, 0.8, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
