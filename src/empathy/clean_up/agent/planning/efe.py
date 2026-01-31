"""
Planning: Expected Free Energy Computation.

This module provides Numba-compiled functions for computing Expected Free Energy
(EFE), the core planning computation in active inference.

Mathematical Foundation:
-----------------------------------------------------
Action selection minimizes expected free energy:

    p(a) ∝ exp(-β · G(a))      (single-step, H=1)
    p(π) ∝ exp(-β · G(π))      (multi-step, H>1)

where G(a) is the one-step Expected Free Energy:

    G(a) = E_{s',o'} [-log p(o'_pref | s', θ)] - λ_epist · IG(a)

The EFE has two components:
1. Pragmatic (extrinsic): Expected negative log-likelihood of preferred observations
2. Epistemic (intrinsic): True information gain about hidden states

Multi-Step Planning:
--------------------
For a policy π = [a₀, a₁, ..., a_{H-1}] of length H, the multi-step EFE is:

    G(π, H) = G(a₀) + G(π_{1:}, H-1)

Where:
- G(a₀) is one-step EFE for the first action (marginalizes over observations)
- G(π_{1:}, H-1) is the EFE for remaining policy with predicted state belief
- Computation operates on belief distributions q(s'), not individual observations

Implementation uses standard active inference state-space marginalization.
Works in STATE space, not observation space:
- Observations are marginalized via matrix operations (A @ states)
- No explicit enumeration over observation space
- Complexity: O(n_states × H) vs. O(n_obs^H) for observation enumeration

Base case: When H=1, G(π, 1) = G(a₀) (one-step EFE)
Multi-step case: When H>1, predict next state belief and continue.

TRUE Information Gain:
-----------------------------------------
    IG(a) = H[q(s')] - E_{o' ~ p(o'|a)} [H(q(s' | o'))]

This is the expected reduction in entropy about the hidden state after
observing outcome o', computed as:
1. Compute prior entropy H[q(s')]
2. For each possible observation o' (parallelized):
   - Compute p(o'|a) by marginalizing over s': p(o'|a) = A @ q(s')
   - Update belief: q(s'|o') ∝ q(s') · p(o'|s', θ)
   - Compute posterior entropy H(q(s'|o'))
3. Take expectation over observation distribution

Note: Information gain computation uses parallel observation enumeration
within a single timestep, but multi-step planning avoids observation
enumeration by iterating directly on state beliefs.

Performance Notes:
-----------------
- All functions use @njit for JIT compilation
- EFE computation is the bottleneck: target < 3.5 ms per step
- Vectorize over actions where possible
- Multi-step planning uses pure Python for iteration (Numba-compatible helpers)

Dependencies:
------------
- numpy: Array operations
- numba: JIT compilation (@njit decorator)
"""

import numpy as np

from empathy.clean_up.agent.inference.state import compute_entropy
from empathy.clean_up.agent.inference.utils import softmax, normalize

# Optional numba acceleration can be enabled if available.
from numba import njit, float64, prange



# =============================================================================
# EXPECTED FREE ENERGY COMPUTATION
# =============================================================================

@njit(float64(float64[:], float64[:, :], float64[:]), cache=True)
def compute_pragmatic_value(
    predicted_belief: np.ndarray,
    theta: np.ndarray,
    preferred_obs: np.ndarray
) -> float:
    """
    Compute the pragmatic (extrinsic) component of expected free energy for a predicted state belief.
    
    Parameters:
        predicted_belief (np.ndarray): Predicted state belief q(s'), shape [N_states].
        theta (np.ndarray): Observation likelihood matrix p(o|s, θ), shape [N_states, N_obs].
        preferred_obs (np.ndarray): Distribution over preferred observations, shape [N_obs].
    
    Returns:
        float: Expected surprise of preferred observations (pragmatic value). Lower values indicate observations closer to the preferred distribution.
    
    Raises:
        ValueError: If `theta` is not a 2D observation likelihood matrix with shape [N_states, N_obs].
    """
    if theta.ndim != 2:
        raise ValueError("theta must be an observation likelihood matrix [N_states, N_obs].")

    predicted_c = np.ascontiguousarray(predicted_belief)
    theta_c = np.ascontiguousarray(theta)
    expected_obs = predicted_c @ theta_c
    expected_obs = normalize(expected_obs)

    pref = normalize(preferred_obs)
    eps = 1e-12
    return -float(np.sum(expected_obs * np.log(pref + eps)))


@njit(parallel=True, cache=True)
def compute_information_gain(
    state_belief: np.ndarray,
    transition_matrix: np.ndarray,
    obs_likelihood_matrix: np.ndarray
) -> float:
    """
    Compute the expected information gain about hidden states produced by an action.
    
    Returns the reduction in entropy of the predicted next-state belief after observing outcomes:
    IG(a) = H[q(s')] - E_{o' ~ p(o'|a)}[H(q(s'|o'))].
    
    Parameters:
        state_belief (np.ndarray): Current belief q(s), shape [N_states].
        transition_matrix (np.ndarray): Transition probabilities p(s'|s,a), shape [N_states, N_states].
        obs_likelihood_matrix (np.ndarray): Observation likelihoods p(o|s,θ), shape [N_states, N_obs].
    
    Returns:
        float: Information gain (IG) for the action; a scalar greater than or equal to 0.0.
    """
    state_c = np.ascontiguousarray(state_belief)
    transition_c = np.ascontiguousarray(transition_matrix)
    # B[s',s] = P(s'|s,a), so predicted P(s') = Σ_s P(s'|s,a) * P(s) = B @ belief
    predicted = transition_c @ state_c
    predicted = normalize(predicted)
    prior_entropy = compute_entropy(predicted)

    n_obs = obs_likelihood_matrix.shape[1]
    expected_posterior_entropy = 0.0
    eps = 1e-12
    
    # parallel reduction: compute weighted entropy contribution per observation
    # numba handles the reduction automatically with +=
    for o in prange(n_obs):
        likelihood = obs_likelihood_matrix[:, o]
        p_o = np.sum(predicted * likelihood)
        if p_o > eps:
            posterior = predicted * likelihood / p_o
            posterior_entropy = compute_entropy(posterior)
            expected_posterior_entropy += p_o * posterior_entropy

    info_gain = prior_entropy - expected_posterior_entropy
    return max(0.0, info_gain)


@njit(cache=True)
def compute_efe_one_step(
    action: int,
    state_belief: np.ndarray,
    theta: np.ndarray,
    transition_matrix: np.ndarray,
    preferred_obs: np.ndarray,
    lambda_epist: float = 0.5
) -> float:
    """
    Compute the one-step Expected Free Energy (EFE) for a given action.
    
    Parameters:
        action (int): Index of the action to evaluate.
        state_belief (np.ndarray): Current belief over hidden states q(s), shape [N_states].
        theta (np.ndarray): Observation model p(o|s,θ), shape [N_states, N_obs].
        transition_matrix (np.ndarray): State transition matrix p(s'|s,a) for the given action, shape [N_states, N_states].
        preferred_obs (np.ndarray): Preferred observation distribution, shape [N_obs].
        lambda_epist (float): Weight applied to the epistemic (information-gain) term.
    
    Returns:
        float: Scalar EFE value for the action; lower values indicate more favorable actions.
    
    Raises:
        ValueError: If `theta` is not a 2D observation likelihood matrix [N_states, N_obs].
    """
    if theta.ndim != 2:
        raise ValueError("theta must be an observation likelihood matrix [N_states, N_obs].")

    state_c = np.ascontiguousarray(state_belief)
    transition_c = np.ascontiguousarray(transition_matrix)
    # B[s',s] = P(s'|s,a), so predicted P(s') = Σ_s P(s'|s,a) * P(s) = B @ belief
    predicted_belief = transition_c @ state_c
    predicted_belief = normalize(predicted_belief)
    pragmatic = compute_pragmatic_value(predicted_belief, theta, preferred_obs)
    # compute information gain (uses parallel loop internally)
    # theta represents the observation likelihood matrix p(o|s,θ) used as obs_likelihood_matrix
    obs_likelihood_matrix = theta
    epistemic = compute_information_gain(state_belief, transition_matrix, obs_likelihood_matrix)
    return pragmatic - lambda_epist * epistemic


# =============================================================================
# MULTI-STEP EFE COMPUTATION
# =============================================================================

def compute_efe(
    policy: np.ndarray,
    planning_horizon: int,
    state_belief: np.ndarray,
    obs_matrices: np.ndarray,
    transition_matrices: np.ndarray,
    preferred_obs: np.ndarray,
    lambda_epist: float = 0.5,
) -> float:
    """
    Compute Expected Free Energy (EFE) for a policy over planning horizon H.

    Works in STATE space, not observation space - observations are marginalized via
    matrix operations, never explicitly enumerated.

    For a policy π = [a₀, a₁, ..., a_{H-1}] of length H, the multi-step EFE is:

        G(π, H) = Σ_{t=0}^{H-1} G(a_t | q(s_t))

    Where:
    - G(a_t) is one-step EFE for action t (marginalizes over observations)
    - q(s_t) is the predicted belief at timestep t
    - Computation operates on belief distributions, not individual observations

    When H=1: Returns G(a₀), the one-step EFE.
    When H>1: Sums current step EFE plus future EFE with predicted belief.

    Parameters:
        policy (np.ndarray): Action sequence [a₀, a₁, ..., a_{H-1}], shape [H].
        planning_horizon (int): Number of steps to plan ahead (H).
        state_belief (np.ndarray): Current belief q(s), shape [N_states].
        obs_matrices (np.ndarray): Observation models for each action, shape [N_actions, N_states, N_obs].
        transition_matrices (np.ndarray): Transition matrices for each action, shape [N_actions, N_states, N_states].
        preferred_obs (np.ndarray): Preferred observation distribution, shape [N_obs].
        lambda_epist (float): Epistemic weight (default 0.5).

    Returns:
        float: Total EFE for the policy over the planning horizon.

    Note:
        Computational complexity: O(n_states × H) vs. O(n_obs^H) in observation-space enumeration.
    """
    total_efe = 0.0
    current_belief = state_belief

    for t in range(planning_horizon):
        # get action for this timestep
        action = int(policy[t])

        # get observation and transition matrices for this action
        theta = obs_matrices[action]
        transition_matrix = transition_matrices[action]

        # compute one-step EFE for current action (marginalizes over observations internally)
        step_efe = float(compute_efe_one_step(
            action, current_belief, theta, transition_matrix, preferred_obs, lambda_epist
        ))
        total_efe += step_efe

        # if more steps remain, predict next state belief for next iteration
        if t < planning_horizon - 1:
            state_c = np.ascontiguousarray(current_belief)
            transition_c = np.ascontiguousarray(transition_matrix)
            current_belief = transition_c @ state_c
            current_belief = normalize(current_belief)

    return total_efe

# =============================================================================
# BATCH OPERATIONS (for efficiency)
# =============================================================================

@njit(parallel=True, cache=True)
def compute_batched_efe(
    state_beliefs: np.ndarray,
    obs_matrices: np.ndarray,
    transition_matrices: np.ndarray,
    preferred_obs: np.ndarray,
    lambda_epist: float = 0.5
) -> np.ndarray:
    """
    Compute one-step Expected Free Energy (EFE) for each item in a batch using the provided per-item observation and transition models.
    
    Parameters:
        state_beliefs (np.ndarray): Array of shape [Batch, N_states] containing current state beliefs q(s) for each batch item.
        obs_matrices (np.ndarray): Array of shape [Batch, N_states, N_obs] containing per-item observation models θ (p(o|s)).
        transition_matrices (np.ndarray): Array of shape [Batch, N_states, N_states] containing per-item transition matrices p(s'|s,a).
        preferred_obs (np.ndarray): Array of shape [Batch, N_obs] containing preferred observation distributions for each batch item.
        lambda_epist (float): Weighting factor for the epistemic (information-gain) term (default 0.5).
    
    Returns:
        np.ndarray: 1D array of shape [Batch] with the computed EFE value for each batch item.
    """
    batch_size = state_beliefs.shape[0]
    efe_values = np.zeros(batch_size, dtype=float)
    
    for i in prange(batch_size):
        efe_values[i] = compute_efe_one_step(
            0,  # Action index is dummy here as transition matrix is already specific
            state_beliefs[i],
            obs_matrices[i],
            transition_matrices[i],
            preferred_obs[i],
            lambda_epist=lambda_epist,
        )
    return efe_values


def compute_batched_efe_policies(
    policies: np.ndarray,
    planning_horizon: int,
    state_belief: np.ndarray,
    obs_matrices: np.ndarray,
    transition_matrices: np.ndarray,
    preferred_obs: np.ndarray,
    lambda_epist: float = 0.5,
) -> np.ndarray:
    """
    Compute Expected Free Energy (EFE) for a batch of policies using multi-step planning.
    
    This function generalizes EFE computation to work with policies (action sequences)
    when planning_horizon > 1. Each policy is evaluated using multi-step EFE computation.
    
    Parameters:
        policies (np.ndarray): Array of policies, shape [N_policies, H] where H is planning_horizon.
                              Each row is an action sequence [a₀, a₁, ..., a_{H-1}].
        planning_horizon (int): Number of steps to plan ahead (H). Must match policies.shape[1].
        state_belief (np.ndarray): Current belief q(s), shape [N_states].
        obs_matrices (np.ndarray): Observation models for each action, shape [N_actions, N_states, N_obs].
        transition_matrices (np.ndarray): Transition matrices for each action, shape [N_actions, N_states, N_states].
        preferred_obs (np.ndarray): Preferred observation distribution, shape [N_obs].
        lambda_epist (float): Epistemic weight (default 0.5).
    
    Returns:
        np.ndarray: 1D array of shape [N_policies] with EFE value for each policy.
    
    Note:
        When planning_horizon=1, this reduces to single-action EFE computation,
        making it backward compatible with the original batched EFE function.
    """
    n_policies = policies.shape[0]
    efe_values = np.zeros(n_policies, dtype=float)
    
    for i in range(n_policies):
        efe_values[i] = compute_efe(
            policies[i],
            planning_horizon,
            state_belief,
            obs_matrices,
            transition_matrices,
            preferred_obs,
            lambda_epist
        )
    
    return efe_values
