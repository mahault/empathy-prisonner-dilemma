"""
Particle Filter: Particle-Based Inference for Theory of Mind.

This module provides the ParticleFilter class and core weight update logic
for representing and updating beliefs about other agents' parameters.

Mathematical Foundation:
---------------------------------------------------
q(θ_other) is represented as a weighted particle set:

    q(θ_other) ≈ Σ_j w_j δ_{θ^j}

where θ^j are parameter hypotheses and w_j are normalized weights.

Bayesian weight update from action evidence under perspective-taking:

    w_j^new ∝ w_j^old · p(a_other | θ^j, context)

Action likelihood uses STOCHASTIC ToM:

    p(a | θ^j, context) ∝ exp(β_ToM · Q^j(a | context))

where Q^j(a) = -G^j(a) is the negative EFE under particle θ^j.

Particle Resampling:
-------------------
To prevent particle degeneracy (all weight on one particle):
1. Compute effective sample size: N_eff = 1 / Σ_j w_j²
2. If N_eff < threshold · N_p, resample with replacement
3. Add diffusion noise to maintain diversity

Compute weight entropy H(w) BEFORE resampling for support calculation.

Performance Notes:
-----------------
- ToM update is the most expensive step: target < 5 ms per step
- Vectorize Q-value computation across particles
- Cache transition matrices and observation models

Dependencies:
------------
- numpy: Array operations
- numba: JIT compilation (@njit decorator)
"""

from __future__ import annotations

from itertools import product
from typing import Any
import numpy as np

from empathy.clean_up.agent.beliefs import DirichletBeliefs
from empathy.clean_up.agent.planning.efe import compute_batched_efe, compute_batched_efe_policies
from empathy.clean_up.agent.inference.utils import softmax
from empathy.clean_up.agent.social.utils import (
    choose_world_state,
    effective_particle_count,
    resample_particles,
)
from empathy.clean_up.agent.social.trust import (
    compute_confidence,
    compute_reliability,
)

# Optional numba acceleration can be enabled if available.
from numba import njit, float64


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _marginalize_policies_to_actions(policy_probs: np.ndarray, policies: np.ndarray, n_actions: int) -> np.ndarray:
    """
    Marginalize policy probabilities to get action probabilities.
    
    For multi-step planning, we compute P(policy) for each policy sequence.
    To get P(action), we sum over all policies that start with that action:
    
        P(action_a) = Σ_{π: π[0]=a} P(π)
    
    Args:
        policy_probs: Probability of each policy, shape [n_policies]
        policies: Policy sequences, shape [n_policies, planning_horizon]
        n_actions: Number of possible actions
        
    Returns:
        action_probs: Probability of each action, shape [n_actions]
    """
    action_probs = np.zeros(n_actions, dtype=np.float64)
    for i, policy in enumerate(policies):
        first_action = policy[0]
        action_probs[first_action] += policy_probs[i]
    return action_probs


def _get_obs_matrix(
    particle_idx: int,
    action: int | None,
    particle_beliefs: DirichletBeliefs,
    context: dict,
    context_independent: bool,
    obs_likelihood_matrices: np.ndarray | None,
    obs_likelihood_fn: Any | None,
    obs_likelihood_matrix: np.ndarray | None,
    predict_context_fn: Any | None,
) -> np.ndarray:
    """
    Get observation matrix for a particle, optionally action-dependent.
    
    Args:
        particle_idx: Index of the particle (j)
        action: Action index, or None if context-independent
        particle_beliefs: Beliefs for this particle
        context: Context dict for observation likelihood
        context_independent: If True, obs matrix doesn't depend on action
        obs_likelihood_matrices: Pre-computed matrices per particle [n_particles, n_states, n_obs]
        obs_likelihood_fn: Function (beliefs, context) -> obs_matrix
        obs_likelihood_matrix: Single shared obs matrix
        predict_context_fn: Function (action, context) -> action_context
        
    Returns:
        Observation matrix [n_states, n_obs]
    """
    # determine context (action-dependent if not context_independent)
    effective_context = context
    if not context_independent and action is not None and predict_context_fn is not None:
        effective_context = predict_context_fn(action, context)
    
    # priority: per-particle matrices > function > shared matrix
    if obs_likelihood_matrices is not None:
        return obs_likelihood_matrices[particle_idx]
    elif obs_likelihood_fn is not None:
        return obs_likelihood_fn(particle_beliefs, effective_context)
    elif obs_likelihood_matrix is not None:
        return obs_likelihood_matrix
    else:
        raise ValueError("No observation likelihood provided for particles.")


# =============================================================================
# CORE WEIGHT UPDATE FUNCTION
# =============================================================================

def _update_particle_weights(
    weights: np.ndarray,
    observed_action: int,
    particle_params: np.ndarray,
    beta_tom: float,
    context_state: dict,
    tom_mode: str = "softmax",
) -> np.ndarray:
    """
    Update particle weights by Bayesian evidence of an observed action under a perspective-taking Theory of Mind.

    This is a private helper function. Use TheoryOfMind.observe_action() instead.
    """
    if not isinstance(context_state, dict):
        raise TypeError("context_state must be a dict with planning context.")

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
    record_simulations = bool(context_state.get("record_tom_simulations", False))
    planning_horizon = int(context_state.get("planning_horizon", 1))

    if expert_belief_prev is None or world_state_belief is None:
        raise ValueError("Perspective-taking requires expert_belief_prev and world_state_belief.")
    if simulate_expert_observation is None or simulate_expert_belief_update is None:
        raise ValueError("Perspective-taking requires expert simulation functions.")

    n_particles = len(weights)
    n_actions = transition_matrices.shape[0]

    n_states = state_belief.shape[0]
    n_obs = preferred_obs.shape[0]
    batch_size = n_particles * n_actions

    batched_state_beliefs = np.empty((batch_size, n_states), dtype=np.float64)
    batched_obs_matrices = np.empty((batch_size, n_states, n_obs), dtype=np.float64)
    batched_transition_matrices = np.empty((batch_size, n_states, n_states), dtype=np.float64)
    batched_preferred_obs = np.tile(preferred_obs, (batch_size, 1))

    sim_obs_buffer = None
    if record_simulations:
        sim_obs_buffer = np.full(n_particles, -1, dtype=int)

    action_loglikes = np.zeros(n_particles, dtype=float)

    world_state_idx = int(choose_world_state(world_state_belief, mode="map"))
    context_with_world_state = {**(context or {}), "world_state_idx": world_state_idx}

    context_independent = predict_context_fn is None
    build_transition_matrices_fn = context_with_world_state.get("build_transition_matrices")
    use_per_particle_transitions = build_transition_matrices_fn is not None

    # =========================================================================
    # MULTI-STEP PLANNING (H>1) - Compute per-particle policy EFE
    # =========================================================================
    if planning_horizon > 1:
        # generate all policies (action sequences of length H)
        policies = np.array(list(product(range(n_actions), repeat=planning_horizon)), dtype=np.int64)
        n_policies = policies.shape[0]
        
        new_weights = np.zeros(n_particles, dtype=float)
        
        for j in range(n_particles):
            # convert particle params to DirichletBeliefs
            particle_beliefs = DirichletBeliefs.from_array(particle_params[j])
            
            # simulate expert's belief update
            simulated_obs = simulate_expert_observation(
                expert_position, world_state_belief, particle_params[j], context_with_world_state
            )
            state_belief_j = simulate_expert_belief_update(
                expert_belief_prev, simulated_obs, particle_params[j], context_with_world_state
            )
            
            if record_simulations and sim_obs_buffer is not None:
                sim_obs_buffer[j] = simulated_obs if simulated_obs is not None else -1
            
            # build particle-specific transition matrices
            if use_per_particle_transitions:
                particle_transitions = build_transition_matrices_fn(particle_beliefs)
            else:
                particle_transitions = transition_matrices
            
            # build observation matrices for each action
            obs_matrices_j = np.zeros((n_actions, n_states, n_obs), dtype=np.float64)
            for a in range(n_actions):
                obs_matrices_j[a] = _get_obs_matrix(
                    j, a if not context_independent else None, particle_beliefs, context,
                    context_independent, obs_likelihood_matrices, obs_likelihood_fn,
                    obs_likelihood_matrix, predict_context_fn
                )
            
            # compute EFE for all policies
            policy_efe = compute_batched_efe_policies(
                policies,
                planning_horizon,
                state_belief_j,
                obs_matrices_j,
                particle_transitions,
                preferred_obs,
                lambda_epist
            )
            
            # convert to policy probabilities via softmax
            policy_q_values = -policy_efe
            policy_probs = softmax(beta_tom * policy_q_values)
            
            # marginalize to get action probabilities
            action_probs = _marginalize_policies_to_actions(policy_probs, policies, n_actions)
            
            # compute likelihood of observed action
            if tom_mode == "greedy":
                greedy_action = int(np.argmax(action_probs))
                likelihood = 1.0 if observed_action == greedy_action else 0.0
            else:
                likelihood = action_probs[observed_action]
            
            action_loglike = np.log(likelihood + 1e-12)
            log_w = np.log(weights[j] + 1e-12) + action_loglike
            new_weights[j] = np.exp(log_w)
            action_loglikes[j] = action_loglike
    
    # =========================================================================
    # SINGLE-STEP PLANNING (H=1) - Original batched computation
    # =========================================================================
    else:
        for j in range(n_particles):
            # convert particle params to DirichletBeliefs once per particle
            # (needed for build_transition_matrices and observation_likelihood_matrix)
            particle_beliefs = DirichletBeliefs.from_array(particle_params[j])

            state_belief_j = state_belief
            simulated_obs = simulate_expert_observation(
                expert_position, world_state_belief, particle_params[j], context_with_world_state
            )
            state_belief_j = simulate_expert_belief_update(
                expert_belief_prev, simulated_obs, particle_params[j], context_with_world_state
            )

            if record_simulations and sim_obs_buffer is not None:
                sim_obs_buffer[j] = simulated_obs if simulated_obs is not None else -1

            # cache obs matrix if context-independent (same for all actions)
            obs_matrix_j = None
            if context_independent:
                obs_matrix_j = _get_obs_matrix(
                    j, None, particle_beliefs, context, context_independent,
                    obs_likelihood_matrices, obs_likelihood_fn, obs_likelihood_matrix, predict_context_fn
                )

            if use_per_particle_transitions:
                particle_transitions = build_transition_matrices_fn(particle_beliefs)

            for a in range(n_actions):
                idx = j * n_actions + a
                batched_state_beliefs[idx] = state_belief_j

                if use_per_particle_transitions:
                    batched_transition_matrices[idx] = particle_transitions[a, :, :]
                else:
                    batched_transition_matrices[idx] = transition_matrices[a]

                if context_independent:
                    obs_matrix = obs_matrix_j
                else:
                    obs_matrix = _get_obs_matrix(
                        j, a, particle_beliefs, context, context_independent,
                        obs_likelihood_matrices, obs_likelihood_fn, obs_likelihood_matrix, predict_context_fn
                    )

                batched_obs_matrices[idx] = obs_matrix

        batched_efe = compute_batched_efe(
            batched_state_beliefs,
            batched_obs_matrices,
            batched_transition_matrices,
            batched_preferred_obs,
            lambda_epist
        )

        new_weights = np.zeros(n_particles, dtype=float)
        all_q_values = -batched_efe.reshape((n_particles, n_actions))

        for j in range(n_particles):
            q_values = all_q_values[j]

            if tom_mode == "greedy":
                greedy_action = int(np.argmax(q_values))
                likelihood = 1.0 if observed_action == greedy_action else 0.0
            else:
                action_probs = softmax(beta_tom * q_values)
                likelihood = action_probs[observed_action]

            action_loglike = np.log(likelihood + 1e-12)
            log_w = np.log(weights[j] + 1e-12) + action_loglike
            new_weights[j] = np.exp(log_w)
            action_loglikes[j] = action_loglike

    total = np.sum(new_weights)
    if total <= 0.0:
        return np.ones_like(new_weights) / new_weights.size
    if isinstance(context_state, dict):
        context_state["tom_loglike_action_mean"] = float(np.mean(action_loglikes))
        if record_simulations:
            context_state["tom_action_loglikes"] = action_loglikes.copy()
            if sim_obs_buffer is not None:
                context_state["tom_simulated_obs"] = sim_obs_buffer
    return new_weights / total


# =============================================================================
# NUMBA-COMPILED HELPER FUNCTIONS
# =============================================================================

@njit(float64(float64[:]))
def compute_weight_entropy(weights: np.ndarray) -> float:
    """
    Compute entropy of particle weights.

    Mathematical Definition:
        H(w) = -Σ_j w_j log w_j

    Entropy measures uncertainty in ToM inference:
    - H(w) = 0: all weight on one particle (maximum confidence)
    - H(w) = log(N_p): uniform weights (maximum uncertainty)

    Must be computed BEFORE resampling, as resampling flattens
    weights and creates spurious low-confidence signals.

    Args:
        weights: Particle weights, shape [N_particles]

    Returns:
        entropy: Weight entropy ≥ 0
    Numba Signature:
        @njit(float64(float64[:]))
    """
    entropy = 0.0
    for w in weights:
        if w > 1e-12:
            entropy -= w * np.log(w)
    return entropy

# =============================================================================
# PARTICLE FILTER CLASS
# =============================================================================

class ParticleFilter:
    """
    Manages particle-based representation of q(θ_other).

    Encapsulates particle parameters, weights, and operations like
    weight updates, resampling, and statistics computation.
    """

    def __init__(
        self,
        particle_params: np.ndarray,
        particle_weights: np.ndarray,
        config: dict[str, Any],
        rng: np.random.Generator
    ):
        """
        Initialize particle filter.

        Args:
            particle_params: Parameter hypotheses [N_particles, ...]
                For clean_up: [N_particles, 5 contexts, 2 alphas]
            particle_weights: Normalized weights [N_particles]
            config: Configuration dictionary
            rng: Random number generator
        """
        self.particle_params = particle_params
        self.particle_weights = particle_weights
        self.config = config
        self.rng = rng
        self._reliability_computed_before_resample = False

    def update_weights(
        self,
        observed_action: int,
        beta_tom: float,
        context_state: dict,
        tom_mode: str = "softmax"
    ) -> None:
        """
        Update particle weights based on observed action.

        Args:
            observed_action: Action taken by other agent
            beta_tom: ToM inverse temperature
            context_state: Planning and simulation context
            tom_mode: "softmax" or "greedy"
        """
        self.particle_weights = _update_particle_weights(
            self.particle_weights,
            observed_action,
            self.particle_params,
            beta_tom,
            context_state,
            tom_mode,
        )

    def resample_if_needed(self) -> dict[str, float]:
        """
        Resample particles if effective sample size drops below threshold.

        Returns:
            stats: Dictionary with resampling statistics
        """
        n_eff_pre = effective_particle_count(self.particle_weights)
        n_particles = int(self.config.get("n_particles", 30))
        n_eff_post = n_eff_pre
        is_resampled = 0.0

        if not self._reliability_computed_before_resample:
            raise RuntimeError(
                "Reliability MUST be computed before resampling! "
                "Post-resampling weights are uniform and give false low confidence."
            )

        resample_thresh = float(self.config.get("resample_thresh", 0.5))
        if n_eff_pre < resample_thresh * n_particles:
            seed = int(self.rng.integers(0, 2**31 - 1))
            diffusion = float(self.config.get("diffusion_noise", 0.05))
            self.particle_params, self.particle_weights = resample_particles(
                self.particle_params, self.particle_weights, diffusion, seed
            )
            n_eff_post = effective_particle_count(self.particle_weights)
            is_resampled = 1.0
            self._reliability_computed_before_resample = False

        return {
            "n_eff_pre": float(n_eff_pre),
            "n_eff_post": float(n_eff_post),
            "is_resampled": float(is_resampled),
        }

    def get_expected_alpha(self, context: int) -> np.ndarray:
        """
        Get particle-weighted expected Dirichlet parameters for a context.

        Args:
            context: Pollution level (0-4)

        Returns:
            alpha_expected shape [2] = [alpha_no_spawn, alpha_spawn]
        """
        alpha_sum = np.zeros(2, dtype=float)
        for i, weight in enumerate(self.particle_weights):
            alpha_sum += weight * self.particle_params[i, context, :]
        return alpha_sum

    def get_spawn_probability(self, context: int) -> float:
        """
        Get expected spawn probability for context.

        Args:
            context: Pollution level (0-4)

        Returns:
            Expected spawn probability
        """
        alpha = self.get_expected_alpha(context)
        return float(alpha[1] / (alpha[0] + alpha[1]))

    def get_expected_parameters(self) -> dict:
        """
        Get expected Dirichlet parameters for all contexts.

        Returns:
            Dictionary mapping context -> expected alpha [2]
        """
        n_contexts = self.particle_params.shape[1]
        return {ctx: self.get_expected_alpha(ctx) for ctx in range(n_contexts)}

    def compute_statistics(
        self,
        u_threshold: float = 0.05,
        kappa: float = 0.05
    ) -> dict[str, float]:
        """
        Compute particle statistics: entropy, confidence, reliability.

        Must be called BEFORE resampling (resampling flattens weights).

        Args:
            u_threshold: Confidence threshold
            kappa: Smoothness parameter

        Returns:
            stats: Dictionary with weight_entropy, confidence, reliability
        """
        weight_entropy = compute_weight_entropy(self.particle_weights)
        n_particles = len(self.particle_weights)
        confidence = compute_confidence(self.particle_weights, n_particles)
        reliability = compute_reliability(confidence, u_threshold, kappa)

        self._reliability_computed_before_resample = True

        return {
            "weight_entropy": float(weight_entropy),
            "confidence": float(confidence),
            "reliability": float(reliability),
        }

    def reset(self, particle_params: np.ndarray, particle_weights: np.ndarray) -> None:
        """Reset particles and weights."""
        self.particle_params = particle_params
        self.particle_weights = particle_weights
        self._reliability_computed_before_resample = False
