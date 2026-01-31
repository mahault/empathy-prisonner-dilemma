"""
State Inference Numba Kernels.

This module provides Numba-compiled functions for VFE-minimizing state belief updates,
the core perception loop in active inference.

Mathematical Foundation:
-----------------------------------------------------
State inference minimizes variational free energy (VFE) by updating state beliefs
based on observations. The VFE is defined as:

    F = -log p(o|s) + D_KL(q(s) || p(s))
        = accuracy term - complexity term

For discrete state spaces, the analytical solution that minimizes VFE is:

    q*(s_t) ∝ q(s_{t-1}) · p(o_t | s_t, θ)^precision

This is VFE minimization via analytical solution, where the posterior over hidden states
is computed by combining the prior belief with the observation likelihood.

Key Operations:
--------------
1. VFE Minimization: Update state beliefs to minimize variational free energy
2. Belief Propagation: Predict next state belief through transitions
3. Entropy Computation: Measure uncertainty in state beliefs

Performance Notes:
-----------------
- All functions use @njit for JIT compilation
- Explicit type signatures for maximum efficiency
- Minimize allocations in hot paths
- Target: < 1.5 ms per step for state inference

Dependencies:
------------
- numpy: Array operations
- numba: JIT compilation (@njit decorator)
"""

import numpy as np
from typing import Tuple

# Optional numba acceleration can be enabled if available.
from numba import njit, float64, int64

from empathy.clean_up.agent.inference.utils import gaussian_bin_prob, manhattan_distance_cell, max_manhattan_dist, normalize


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@njit(float64(float64[:], float64[:]))
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence D_KL(p || q).

    D_KL(p || q) = Σ p(x) log(p(x) / q(x))

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        kl: KL divergence ≥ 0
    """
    eps = 1e-12
    kl = 0.0
    for i in range(p.shape[0]):
        p_i = p[i]
        if p_i > 0.0:
            q_i = max(q[i], eps)
            kl += p_i * np.log(p_i / q_i)
    return kl


# =============================================================================
# VFE MINIMIZATION: STATE BELIEF UPDATE
# =============================================================================

@njit
def update_state_belief(
    prior: np.ndarray,
    observation: int,
    theta: np.ndarray,
    precision: float = 1.0
) -> np.ndarray:
    """
    VFE-minimizing state belief update given observation.

    This function implements the analytical solution to variational free energy (VFE)
    minimization for discrete state spaces. The VFE is:

        F = -log p(o|s) + D_KL(q(s) || p(s))

    Minimizing F with respect to q(s) yields the optimal posterior:

        q*(s_t) ∝ q(s_{t-1}) · p(o_t | s_t, θ)^precision

    The precision parameter is the sensory precision (inverse temperature) in active
    inference, controlling how strongly observations influence the posterior.

    Args:
        prior: Prior state belief q(s_{t-1}), shape [N_states]
        observation: Observed value (integer index)
        theta: Observation model parameters (task-specific)
        precision: Sensory precision (inverse temperature), default 1.0

    Returns:
        posterior: Normalized posterior q(s_t) that minimizes VFE, shape [N_states]

    Implementation Notes:
    --------------------
    1. Compute observation likelihood for each state
    2. Multiply prior by likelihood^precision
    3. Normalize to sum to 1
    4. Handle numerical stability (log-space for small values)

    Numba Signature:
        @njit(float64[:](float64[:], int64, float64[:], float64))
    """
    if theta.ndim == 1:
        if theta.shape[0] != prior.shape[0]:
            raise ValueError("theta length must match prior length.")
        likelihood = theta
    elif theta.ndim == 2:
        if theta.shape[0] != prior.shape[0]:
            raise ValueError("theta rows must match prior length.")
        if observation < 0 or observation >= theta.shape[1]:
            raise ValueError("observation index out of bounds for theta.")
        likelihood = theta[:, observation]
    else:
        raise ValueError("theta must be 1D (likelihood) or 2D (likelihood matrix).")

    unnorm = prior * np.power(likelihood, precision)
    return normalize(unnorm)


@njit(float64[:](float64[:], int64, float64[:, :]))
def belief_propagation_step(
    beliefs: np.ndarray,
    action: int,
    transition_matrix: np.ndarray
) -> np.ndarray:
    """
    Propagate beliefs through state transition model.

    Mathematical Update:
        q(s') = Σ_s p(s' | s, a) · q(s)

    This is the prediction step in active inference, used to predict future state
    beliefs before observing outcomes.

    Args:
        beliefs: Current state belief q(s), shape [N_states]
        action: Action taken
        transition_matrix: p(s' | s, a) for given action, shape [N_states, N_states]
                          where [i, j] = p(s'=j | s=i, a)

    Returns:
        predicted_beliefs: Predicted next-step belief q(s'), shape [N_states]

    Implementation Notes:
    --------------------
    1. Matrix-vector multiplication: transition_matrix.T @ beliefs
    2. Or equivalently: beliefs @ transition_matrix
    3. Result automatically normalized if transition_matrix rows sum to 1

    Numba Signature:
        @njit(float64[:](float64[:], int64, float64[:, :]))
    """
    beliefs_c = np.ascontiguousarray(beliefs)
    transition_c = np.ascontiguousarray(transition_matrix)
    predicted = beliefs_c @ transition_c
    return normalize(predicted)


@njit(float64(float64[:]))
def compute_entropy(belief: np.ndarray) -> float:
    """
    Compute the Shannon entropy of a discrete state belief distribution.

    Parameters:
        belief (np.ndarray): 1D array of state probabilities that sums to 1 (shape [N_states]).

    Returns:
        entropy (float): Shannon entropy (>= 0) of the distribution computed using the natural logarithm.
    """
    entropy = 0.0
    for p in belief:
        if p > 1e-12:
            entropy -= p * np.log(p)
    return entropy

