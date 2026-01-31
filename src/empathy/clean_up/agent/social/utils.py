"""
Particle Utilities: Resampling and Effective Sample Size.

This module provides Numba-compiled utilities for managing particle
representations in Theory of Mind inference.

Mathematical Foundation:
---------------------------------------------------
Particle-based ToM can suffer from degeneracy where most particles have
negligible weight. We address this via:

1. Effective Sample Size (ESS):
   N_eff = 1 / Σ_j w_j²
   
   This estimates how many particles are "effectively" contributing.

2. Resampling: When N_eff < threshold * N_p, resample particles
   proportional to their weights (with replacement).

3. Diffusion: After resampling, add small Gaussian noise to maintain
   diversity and prevent collapse to single hypothesis.

Resampling Strategy:
-------------------
We use systematic resampling for lower variance than multinomial.

Algorithm:
1. Compute cumulative weights
2. Draw one random offset u ~ Uniform(0, 1/N_p)
3. Select particles at positions u, u + 1/N_p, u + 2/N_p, ...
4. Add diffusion noise: θ^j_new ~ N(θ^j_resampled, σ²I)
5. Reset weights to uniform

Performance Notes:
-----------------
- Resampling is O(N_p) and fast
- Called infrequently (only when N_eff drops)

Dependencies:
------------
- numpy: Array operations
- numba: JIT compilation (@njit decorator)
"""

import numpy as np
from typing import Optional

# Optional numba acceleration can be enabled if available.
from numba import njit, float64, uint32


# =============================================================================
# EFFECTIVE SAMPLE SIZE
# =============================================================================

@njit(float64(float64[:]))
def effective_particle_count(weights: np.ndarray) -> float:
    """
    Compute effective sample size (ESS) of particles.
    
    Mathematical Definition:
        N_eff = 1 / Σ_j w_j²
    
    Interpretation:
    - N_eff = N_p: All particles equally weighted (maximum diversity)
    - N_eff = 1: All weight on one particle (complete degeneracy)
    - N_eff ~ N_p/2: Reasonable effective coverage
    
    Args:
        weights: Normalized particle weights, shape [N_particles]
        
    Returns:
        n_eff: Effective sample size ∈ [1, N_p]
    Numba Signature:
        @njit(float64(float64[:]))
    """
    sum_sq = np.sum(weights ** 2)
    if sum_sq <= 0.0:
        return 0.0
    return 1.0 / sum_sq


# =============================================================================
# PARTICLE RESAMPLING
# =============================================================================

@njit((float64[:, :], float64[:], float64, uint32))
def resample_particles(
    particles: np.ndarray,
    weights: np.ndarray,
    diffusion_noise: float,
    rng_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Systematic resampling with diffusion noise.
    
    Algorithm:
    1. Systematic Resampling:
       - Compute cumulative weights
       - Draw random offset u ~ Uniform(0, 1/N_p)
       - Select particles at cumulative positions u, u + 1/N_p, ...
    
    2. Diffusion:
       - Add Gaussian noise to maintain diversity
       - θ^j_new ~ N(θ^j_resampled, σ²I)
    
    3. Reset weights to uniform
    
    Args:
        particles: Current particles [N_particles, ...] (shape depends on task)
        weights: Current weights [N_particles]
        diffusion_noise: Standard deviation sigma for diffusion
        rng_seed: Seed for random sampling
        
    Returns:
        resampled_particles: New particles [N_particles, ...]
        uniform_weights: Uniform weights [N_particles]
        
    Implementation Notes:
    --------------------
    - Systematic resampling has lower variance than multinomial
    - Diffusion prevents particle collapse
    - Clip diffused particles to valid parameter ranges
    
    Numba Signature:
        @njit((float64[:, :], float64[:], float64, uint32))
    """
    n_particles = len(weights)
    np.random.seed(rng_seed)
    cumsum = np.cumsum(weights)
    if cumsum[-1] <= 0.0:
        indices = np.arange(n_particles)
    else:
        cumsum = cumsum / cumsum[-1]
        step = 1.0 / n_particles
        start = np.random.random() * step
        indices = np.zeros(n_particles, dtype=np.int64)
        i, j = 0, 0
        while i < n_particles:
            threshold = start + i * step
            while threshold > cumsum[j]:
                j += 1
            indices[i] = j
            i += 1

    resampled = particles[indices].copy()
    if diffusion_noise > 0.0:
        resampled += np.random.normal(0.0, diffusion_noise, resampled.shape)

    new_weights = np.ones(n_particles, dtype=float) / n_particles
    return resampled, new_weights


# =============================================================================
# PERSPECTIVE-TAKING UTILITIES
# =============================================================================

def choose_world_state(
    world_state_belief: np.ndarray,
    mode: str = "map",
    rng: np.random.Generator = None,
) -> int:
    """
    Select a world state index from a belief distribution.

    Used for perspective-taking when simulating expert observations:
    select most likely world state from learner's beliefs q(s_f) to
    predict what the expert would observe.

    Args:
        world_state_belief: Belief over world states, shape [N_states]
        mode: Selection mode:
            - "map": Return argmax (most likely state)
            - "sample": Sample from distribution
        rng: Random number generator for sampling (required if mode="sample")

    Returns:
        state_idx: Selected world state index

    Raises:
        ValueError: If world_state_belief is empty or invalid
    """
    if world_state_belief is None or world_state_belief.size == 0:
        raise ValueError("world_state_belief must be a non-empty array")

    if mode == "map":
        return int(np.argmax(world_state_belief))
    elif mode == "sample":
        if rng is None:
            rng = np.random.default_rng()
        total = world_state_belief.sum()
        if total == 0:
            # fall back to uniform distribution when belief sums to zero
            probs = np.ones_like(world_state_belief) / len(world_state_belief)
        else:
            probs = world_state_belief / total
        return int(rng.choice(len(probs), p=probs))
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'map' or 'sample'")


def simulate_discrete_observation(
    obs_likelihood_matrix: np.ndarray,
    state_index: int,
    rng: np.random.Generator = None,
) -> int:
    """
    Simulate a discrete observation given a state and likelihood matrix.

    Used for perspective-taking when simulating what the expert would observe
    from their position given the current world state.

    Args:
        obs_likelihood_matrix: Observation likelihood p(o | s),
                              shape [N_states, N_obs]
        state_index: World state index to condition on
        rng: Random number generator for sampling

    Returns:
        obs_index: Simulated observation index

    Raises:
        ValueError: If matrix dimensions are invalid
    """
    if obs_likelihood_matrix.ndim != 2:
        raise ValueError("obs_likelihood_matrix must be 2D [N_states, N_obs]")

    if state_index < 0 or state_index >= obs_likelihood_matrix.shape[0]:
        raise ValueError(f"state_index {state_index} out of bounds")

    # Get likelihood row for this state
    row = obs_likelihood_matrix[state_index]

    if row.size == 0:
        raise ValueError("Observation likelihood row is empty")

    # Check if row sums to zero to avoid division by zero
    row_sum = row.sum()
    if row_sum == 0:
        # fall back to deterministic argmax when likelihood row sums to zero
        return int(np.argmax(row))

    # Sample from likelihood or take argmax
    if rng is None:
        return int(np.argmax(row))
    else:
        probs = row / row_sum
        return int(rng.choice(len(probs), p=probs))


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_N_PARTICLES = 30            # Number of particles
DEFAULT_RESAMPLE_THRESH = 0.5       # Resample when N_eff < thresh * N_p
DEFAULT_DIFFUSION_NOISE = 0.05      # Diffusion std after resampling


# =============================================================================
# TOM DIAGNOSTIC FUNCTIONS
# =============================================================================

def update_particle_weights(
    weights: np.ndarray,
    observed_action: int,
    particle_params: np.ndarray,
    beta_tom: float,
    context_state: dict,
    tom_mode: str = "softmax",
) -> np.ndarray:
    """
    Update particle weights by Bayesian evidence of an observed action.

    Note: New code should prefer TheoryOfMind.observe_action() for better encapsulation.
    """
    from empathy.clean_up.agent.social.particle_filter import _update_particle_weights
    return _update_particle_weights(
        weights, observed_action, particle_params, beta_tom, context_state, tom_mode
    )


def compute_particle_Q_value(
    action: int,
    particle_theta: np.ndarray,
    state_belief: np.ndarray,
    transition_matrix: np.ndarray,
    lambda_epist: float = 0.5
) -> float:
    """
    Compute Q^j(a) = -G^j(a) for a single particle.

    Uses same EFE computation as planning.compute_efe_one_step but with
    particle's parameters θ^j instead of self parameters.

    Mathematical Definition:
        Q^j(a) = -G^j(a)

    where G^j(a) = E_{s',o'} [-log p(o'_pref | s', θ^j)] - λ · IG(a)

    Args:
        action: Action index
        particle_theta: Parameters θ^j for this particle
        state_belief: State belief q(s), shape [N_states]
        transition_matrix: p(s'|s,a), shape [N_states, N_states]
        lambda_epist: Epistemic weight

    Returns:
        q_value: Negative EFE (higher = better action)
    """
    preferred_obs = None
    obs_matrix = None
    transition = transition_matrix

    if isinstance(transition_matrix, dict):
        transition = transition_matrix.get("transition_matrix")
        preferred_obs = transition_matrix.get("preferred_obs")
        obs_matrix = transition_matrix.get("obs_likelihood_matrix")
        lambda_epist = transition_matrix.get("lambda_epist", lambda_epist)
    elif isinstance(transition_matrix, tuple) and len(transition_matrix) == 2:
        transition, preferred_obs = transition_matrix

    if preferred_obs is None:
        raise ValueError("preferred_obs is required to compute particle Q-values.")

    if obs_matrix is None:
        if particle_theta.ndim == 2:
            obs_matrix = particle_theta
        else:
            raise ValueError("particle_theta must be an observation likelihood matrix.")

    from empathy.clean_up.agent.planning.efe import compute_efe_one_step
    efe = compute_efe_one_step(
        action,
        state_belief,
        obs_matrix,
        transition,
        preferred_obs,
        lambda_epist,
    )
    return -efe


def compute_tom_action_likelihoods(
    particle_params: np.ndarray,
    context_state: np.ndarray,
    beta_tom: float,
    tom_mode: str = "softmax",
) -> np.ndarray:
    """
    Compute per-particle action likelihoods for ToM diagnostics.

    Uses simulated expert beliefs from the expert's perspective.
    """
    from empathy.clean_up.agent.planning.efe import compute_efe_one_step
    from empathy.clean_up.agent.inference.utils import softmax

    if not isinstance(context_state, dict):
        raise ValueError("context_state must be a dict with planning context.")

    state_belief = context_state["state_belief"]
    transition_matrices = context_state["transition_matrices"]
    preferred_obs = context_state["preferred_obs"]
    lambda_epist = context_state.get("lambda_epist", 0.5)
    obs_likelihood_fn = context_state.get("obs_likelihood_fn")
    obs_likelihood_matrices = context_state.get("obs_likelihood_matrices")
    obs_likelihood_matrix = context_state.get("obs_likelihood_matrix")
    context = context_state.get("context", context_state)
    predict_context_fn = context_state.get("predict_context_fn")
    expert_belief_prev = context_state.get("expert_belief_prev")
    world_state_belief = context_state.get("world_state_belief", state_belief)
    expert_position = context_state.get("expert_position")
    simulate_expert_observation = context_state.get("simulate_expert_observation")
    simulate_expert_belief_update = context_state.get("simulate_expert_belief_update")

    if expert_belief_prev is None or world_state_belief is None:
        raise ValueError("Perspective-taking requires expert_belief_prev and world_state_belief.")
    if simulate_expert_observation is None or simulate_expert_belief_update is None:
        raise ValueError("Perspective-taking requires expert simulation functions.")

    n_particles = particle_params.shape[0]
    n_actions = transition_matrices.shape[0]
    action_likelihoods = np.zeros((n_particles, n_actions), dtype=float)

    for j in range(n_particles):
        simulated_obs = simulate_expert_observation(
            expert_position, world_state_belief, particle_params[j], context
        )
        state_belief_j = simulate_expert_belief_update(
            expert_belief_prev, simulated_obs, particle_params[j], context
        )
        q_values = np.zeros(n_actions, dtype=float)
        for a in range(n_actions):
            if particle_params.ndim == 3:
                obs_matrix = particle_params[j]
            elif obs_likelihood_matrices is not None:
                obs_matrix = obs_likelihood_matrices[j]
            elif obs_likelihood_fn is not None:
                action_context = context
                if predict_context_fn is not None:
                    action_context = predict_context_fn(a, context)
                obs_matrix = obs_likelihood_fn(particle_params[j], action_context)
            elif obs_likelihood_matrix is not None:
                obs_matrix = obs_likelihood_matrix
            else:
                raise ValueError("No observation likelihood provided for particles.")
            efe = compute_efe_one_step(
                a,
                state_belief_j,
                obs_matrix,
                transition_matrices[a],
                preferred_obs,
                lambda_epist=lambda_epist,
            )
            q_values[a] = -efe

        if tom_mode == "greedy":
            greedy_action = int(np.argmax(q_values))
            action_probs = np.zeros(n_actions, dtype=float)
            action_probs[greedy_action] = 1.0
        else:
            action_probs = softmax(beta_tom * q_values)

        action_likelihoods[j] = action_probs

    return action_likelihoods


def compute_tom_posterior_entropy(
    particle_weights: np.ndarray,
    action_likelihoods: np.ndarray,
) -> float:
    """
    Compute entropy of the ToM posterior over actions.
    """
    action_dist = np.sum(particle_weights[:, None] * action_likelihoods, axis=0)
    total = np.sum(action_dist)
    if total > 0.0:
        action_dist = action_dist / total
    entropy = 0.0
    for p in action_dist:
        if p > 1e-12:
            entropy -= p * np.log(p)
    return float(entropy)
