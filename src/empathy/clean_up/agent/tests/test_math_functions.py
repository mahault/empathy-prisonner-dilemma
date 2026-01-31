"""
Mathematical Verification Tests.

Tests that core mathematical functions match their documented behavior.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from empathy.clean_up.agent.inference.state import (
    update_state_belief,
    compute_entropy,
    kl_divergence,
)
from empathy.clean_up.agent.planning.efe import (
    compute_efe_one_step,
    compute_information_gain,
)
from empathy.clean_up.agent.social.update import social_dirichlet_update
from empathy.clean_up.agent.social.trust import compute_trust
from empathy.clean_up.agent.beliefs import DirichletBeliefs
from empathy.clean_up.agent.social.utils import effective_particle_count


class TestVFEUpdate:
    """Test VFE-minimizing state belief update."""
    
    def test_update_state_belief_normalization(self):
        """Test that update_state_belief preserves normalization."""
        prior = np.array([0.3, 0.3, 0.4])
        theta = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        obs = 0
        
        posterior = update_state_belief(prior, obs, theta)
        
        assert np.allclose(np.sum(posterior), 1.0)
        assert np.all(posterior >= 0)
    
    def test_update_state_belief_math(self):
        """Test VFE update matches mathematical formula."""
        prior = np.array([0.5, 0.5])
        theta = np.array([[0.9, 0.1], [0.1, 0.9]])  # Strong observation model
        obs = 0
        precision = 1.0
        
        posterior = update_state_belief(prior, obs, theta, precision)
        
        # Manual calculation: q*(s) ∝ q(s) · p(o|s)
        # For state 0: 0.5 * 0.9 = 0.45
        # For state 1: 0.5 * 0.1 = 0.05
        # Normalized: [0.9, 0.1]
        expected = np.array([0.9, 0.1])
        assert_allclose(posterior, expected, rtol=1e-6)


class TestEntropy:
    """Test entropy computation."""
    
    def test_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = compute_entropy(uniform)
        
        # H = -Σ p log p = -4 * (0.25 * log(0.25)) = log(4)
        expected = np.log(4)
        assert_allclose(entropy, expected, rtol=1e-6)
    
    def test_entropy_deterministic(self):
        """Test entropy of deterministic distribution."""
        deterministic = np.array([1.0, 0.0, 0.0])
        entropy = compute_entropy(deterministic)
        
        # H = 0 (no uncertainty)
        assert_allclose(entropy, 0.0, rtol=1e-6)


class TestKLDivergence:
    """Test KL divergence computation."""
    
    def test_kl_divergence_identical(self):
        """Test KL divergence of identical distributions."""
        p = np.array([0.3, 0.3, 0.4])
        q = p.copy()
        
        kl = kl_divergence(p, q)
        
        # D_KL(p || p) = 0
        assert_allclose(kl, 0.0, rtol=1e-6)
    
    def test_kl_divergence_math(self):
        """Test KL divergence matches mathematical formula."""
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        
        kl = kl_divergence(p, q)
        
        # D_KL = 0.5 * log(0.5/0.9) + 0.5 * log(0.5/0.1)
        expected = 0.5 * np.log(0.5/0.9) + 0.5 * np.log(0.5/0.1)
        assert_allclose(kl, expected, rtol=1e-6)


class TestEFE:
    """Test Expected Free Energy computation."""
    
    def test_efe_components(self):
        """Test that EFE has pragmatic and epistemic components."""
        state_belief = np.ones(10) / 10
        theta = np.ones((10, 5)) / 5  # Uniform observation model
        transition = np.eye(10)  # Identity transition
        preferred_obs = np.ones(5) / 5  # Uniform preferences
        
        efe = compute_efe_one_step(
            0, state_belief, theta, transition, preferred_obs, lambda_epist=0.5
        )
        
        # EFE should be finite and non-negative
        assert np.isfinite(efe)
        assert efe >= 0
    
    def test_information_gain_non_negative(self):
        """Test that information gain is non-negative."""
        state_belief = np.ones(10) / 10
        transition = np.eye(10)
        obs_matrix = np.ones((10, 5)) / 5
        
        ig = compute_information_gain(state_belief, transition, obs_matrix)
        
        # IG = H[q(s')] - E[H(q(s'|o'))] >= 0 (entropy reduction)
        assert ig >= -1e-10  # Allow small numerical errors


class TestSocialLearning:
    """Test social learning mathematics (Dirichlet)."""

    def test_social_dirichlet_update_math(self):
        """Test social Dirichlet update: alpha_new = (1-eta_t)*alpha_self + eta_t*alpha_other."""
        beliefs_self = DirichletBeliefs(n_contexts=3, initial_alpha=2.0)
        # context 0: [2, 2] -> interpolate toward [10, 10]
        alpha_other_dict = {
            0: np.array([10.0, 10.0]),
            1: np.array([5.0, 15.0]),
            2: np.array([1.0, 1.0]),
        }
        social_dirichlet_update(
            beliefs_self=beliefs_self,
            alpha_other_dict=alpha_other_dict,
            reliability=1.0,
            trust=1.0,
            eta_0=0.2,
            contexts_observed=[0],
        )
        # eta_t = 0.2; alpha_new[0] = 0.8*[2,2] + 0.2*[10,10] = [1.6+2, 1.6+2] = [3.6, 3.6]
        expected_0 = 0.8 * np.array([2.0, 2.0]) + 0.2 * np.array([10.0, 10.0])
        assert_allclose(beliefs_self.alpha[0], expected_0, rtol=1e-6)
        # contexts 1, 2 unchanged (not in contexts_observed)
        assert np.allclose(beliefs_self.alpha[1], [2.0, 2.0])
        assert np.allclose(beliefs_self.alpha[2], [2.0, 2.0])

    def test_trust_computation(self):
        """Test trust = reliability * accuracy_gate."""
        reliability = 0.8
        tau_accuracy = 0.9
        
        trust = compute_trust(reliability, tau_accuracy)
        
        assert_allclose(trust, 0.8 * 0.9, rtol=1e-6)
    
    def test_effective_particle_count(self):
        """Test effective sample size computation."""
        # Uniform weights: N_eff = N_p
        uniform = np.ones(10) / 10
        n_eff = effective_particle_count(uniform)
        assert_allclose(n_eff, 10.0, rtol=1e-6)
        
        # Deterministic: N_eff = 1
        deterministic = np.zeros(10)
        deterministic[0] = 1.0
        n_eff = effective_particle_count(deterministic)
        assert_allclose(n_eff, 1.0, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
