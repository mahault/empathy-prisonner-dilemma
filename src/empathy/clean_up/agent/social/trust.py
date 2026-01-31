"""
Trust Computation: Reliability and Accuracy Gate.

This module provides functions for computing trust, which combines:
- Reliability (r_t): Quality of ToM inference (particle concentration)
- Accuracy Gate (τ_accuracy): Predictive accuracy comparison

Mathematical Foundation:
---------------------------------------------------
Reliability (ToM Confidence):
Even if Δ_t > 0, ToM inference might be unreliable (high uncertainty over
what parameters the other agent has). We measure confidence via particle
weight concentration.

Particle Weight Entropy (uncertainty measure):
    H(w) = -Σ_j w_j log w_j

where w_j are normalized particle weights.

Confidence Score (normalized):
    u_t = 1 - H(w) / log(N_p)

This is in [0, 1]:
- u_t = 0: uniform weights (maximum uncertainty)
- u_t = 1: all weight on one particle (maximum confidence)

Compute u_t BEFORE resampling, as resampling flattens weights
and creates spurious low-confidence signals.

Soft Reliability Factor:
    r_t = σ((u_t - u_0) / κ)

where:
- u_0 is confidence threshold (default: 0.05, matches agent/base/base.py DEFAULT_U_0)
- κ controls smoothness (default: 0.05, matches agent/base/base.py DEFAULT_KAPPA)

Accuracy Gate:
    τ_accuracy = σ(accuracy_advantage / T_a)
    accuracy_advantage = log p(o_{t-K:t} | θ_other) - log p(o_{t-K:t} | θ_self)

Trust Formula:
    trust = reliability × τ_accuracy

Effective Social Learning Rate:
    η_t = η_0 · trust = η_0 · reliability · τ_accuracy

where:
- η_0: Base learning rate (default: 0.1)
- reliability: ToM reliability (particle concentration)
- τ_accuracy: Accuracy gate (observation log-likelihood comparison)

Performance Notes:
-----------------
- Confidence and reliability computation is fast (< 0.3 ms)
- Main cost is entropy calculation
- Numba-compiled functions for performance-critical paths

Dependencies:
------------
- numpy: Array operations
- numba: JIT compilation (@njit decorator)
- agent.social.tom_core: compute_weight_entropy
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable
import numpy as np

from empathy.clean_up.agent.beliefs import DirichletBeliefs

# Optional numba acceleration can be enabled if available.
from numba import njit, float64, int64


# =============================================================================
# UTILITIES
# =============================================================================

def _sigmoid(x: float) -> float:
    """Sigmoid function σ(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# =============================================================================
# CONFIDENCE & RELIABILITY (ToM Quality)
# =============================================================================

@njit(float64(float64[:], int64), cache=True)
def compute_confidence(
    weights: np.ndarray,
    n_particles: int
) -> float:
    """
    Compute normalized confidence from particle weights.

    Mathematical Definition:
        u_t = 1 - H(w) / log(N_p)

    where H(w) = -Σ_j w_j log w_j is the weight entropy.

    Interpretation:
    - u_t = 0: Uniform weights, maximum uncertainty (don't trust ToM)
    - u_t = 1: All weight on one particle, maximum confidence
    - u_t ~ 0.5: Moderate concentration

    Args:
        weights: Normalized particle weights, shape [N_particles]
        n_particles: Total number of particles (for normalization)

    Returns:
        confidence: u_t in [0, 1]

    Implementation Notes:
    --------------------
    1. Compute weight entropy H(w)
    2. Normalize by maximum entropy log(N_p)
    3. Return 1 - normalized_entropy

    Must be called BEFORE resampling particles (resampling flattens weights).
    Numba Signature:
        @njit(float64(float64[:], int64))
    """
    if n_particles <= 1:
        return 1.0
    max_entropy = np.log(n_particles)
    entropy = 0.0
    for w in weights:
        if w > 1e-12:
            entropy -= w * np.log(w)
    if max_entropy <= 0.0:
        return 1.0
    confidence = 1.0 - entropy / max_entropy
    # Note: np.clip doesn't work with scalars in @njit, use max/min instead
    return float(max(0.0, min(confidence, 1.0)))


@njit(float64(float64, float64, float64), cache=True)
def compute_reliability(
    confidence: float,
    u_threshold: float = 0.05,
    kappa: float = 0.05
) -> float:
    """
    Compute soft reliability gate from confidence.

    Mathematical Definition:
        r_t = σ((u_t - u_0) / κ)

    where σ is the sigmoid function.

    Interpretation:
    - r_t ≈ 0: ToM inference uncertain, don't copy
    - r_t ≈ 1: ToM inference confident, okay to copy
    - Transition centered at u_t = u_0

    Confidence (u_t) must be computed from particle weights
    BEFORE resampling. Post-resampling weights are uniform and give
    false low confidence values.

    Args:
        confidence: Confidence score u_t in [0, 1] (computed before resampling!)
        u_threshold: Confidence threshold u_0 (default: 0.05)
        kappa: Smoothness parameter κ (default: 0.05)
               - Smaller κ: sharper transition
               - Larger κ: smoother transition

    Returns:
        reliability: r_t in [0, 1]

    Numba Signature:
        @njit(float64(float64, float64, float64))
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    x = (confidence - u_threshold) / kappa
    # Use inline sigmoid for Numba compatibility (can't call Python function from @njit)
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# ACCURACY GATE (Model Comparison)
# =============================================================================

def compute_accuracy_advantage(
    params_self: Any,  # DirichletBeliefs or dict
    params_other: Dict[str, Any],
    obs_history: np.ndarray,
    state_history: np.ndarray,
    observation_likelihood_fn: Callable,
    context: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute accuracy advantage: log-likelihood difference between models.

    Accuracy advantage = log p(o_{t-K:t} | params_other) - log p(o_{t-K:t} | params_self)

    Positive values indicate the other model predicts observations better.

    Args:
        params_self: Self model parameters (DirichletBeliefs or dict of alphas)
        params_other: Other model parameters (dict of alphas from ToM)
        obs_history: Observation history, shape [K]
        state_history: State index history, shape [K]
        observation_likelihood_fn: Function (params, context) -> obs_matrix [n_states, n_obs]
        context: Additional context for observation likelihood

    Returns:
        accuracy_advantage: Log-likelihood difference (positive = other better)
    """
    if len(obs_history) == 0 or len(obs_history) != len(state_history):
        return 0.0

    # convert params_self dict to DirichletBeliefs for type consistency
    if isinstance(params_self, dict) and not isinstance(params_self, DirichletBeliefs):
        n_contexts = len(params_self)
        beliefs_self_input = DirichletBeliefs(n_contexts=n_contexts, alpha_dict=params_self)
    else:
        beliefs_self_input = params_self

    # convert params_other dict to DirichletBeliefs for type consistency
    if isinstance(params_other, dict) and not isinstance(params_other, DirichletBeliefs):
        n_contexts = len(params_other)
        beliefs_other = DirichletBeliefs(n_contexts=n_contexts, alpha_dict=params_other)
    else:
        beliefs_other = params_other

    # get observation likelihood matrices for both models
    # note: for Clean Up, observation model is deterministic (doesn't actually use params)
    A_self = observation_likelihood_fn(beliefs_self_input, context or {})
    A_other = observation_likelihood_fn(beliefs_other, context or {})

    # Compute log-likelihood for each observation
    log_p_self = 0.0
    log_p_other = 0.0
    for obs, state_idx in zip(obs_history, state_history):
        log_p_self += np.log(A_self[state_idx, obs] + 1e-12)
        log_p_other += np.log(A_other[state_idx, obs] + 1e-12)

    # Accuracy advantage: positive = other model better
    return log_p_other - log_p_self


def compute_accuracy_gate(
    accuracy_advantage: float,
    T_a: float
) -> float:
    """
    Compute accuracy gate (tau_accuracy) from accuracy advantage.

    τ_accuracy = σ(accuracy_advantage / T_a)

    Args:
        accuracy_advantage: Log-likelihood difference (from compute_accuracy_advantage)
        T_a: Temperature parameter for sigmoid (default: 2.0)

    Returns:
        tau_accuracy: Accuracy gate value in [0, 1]
    """
    if not np.isfinite(accuracy_advantage):
        return 0.5  # Neutral if invalid

    return _sigmoid(accuracy_advantage / T_a)


# =============================================================================
# TRUST & LEARNING RATE
# =============================================================================

def compute_trust(
    reliability: float,
    tau_accuracy: float
) -> float:
    """
    Compute trust from reliability and accuracy gate.

    trust = reliability × τ_accuracy

    Args:
        reliability: ToM reliability r_t in [0, 1]
        tau_accuracy: Accuracy gate τ_accuracy in [0, 1]

    Returns:
        trust: Trust value in [0, 1]
    """
    return reliability * tau_accuracy


def compute_effective_learning_rate(
    eta_0: float,
    trust: float
) -> float:
    """
    Compute effective social learning rate.

    η_t = η_0 · trust

    Args:
        eta_0: Base social learning rate (default: 0.1)
        trust: Trust value in [0, 1]

    Returns:
        eta_t: Effective learning rate
    """
    return eta_0 * trust
