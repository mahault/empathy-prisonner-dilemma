"""
Theory of Mind Core: Unified API for Particle-Based Inference.

This module provides the TheoryOfMind class, a unified interface that orchestrates
particle-based inference, perspective-taking, and trust computation for modeling
other agents.

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

Key Innovation: Stochastic ToM vs Deterministic ToM
---------------------------------------------------
- Deterministic (greedy) ToM: p(a) = 1 if a = argmax Q(a), else 0
- Stochastic (soft) ToM: p(a) ∝ exp(β · Q(a))

Stochastic ToM is crucial because:
1. Accounts for expert exploration (they don't always act optimally)
2. More robust to policy noise
3. Reduces false negatives (incorrectly concluding expert is wrong)

Particle Resampling:
-------------------
To prevent particle degeneracy (all weight on one particle):
1. Compute effective sample size: N_eff = 1 / Σ_j w_j²
2. If N_eff < threshold · N_p, resample with replacement
3. Add diffusion noise to maintain diversity

Perspective-taking mode:
------------------------
Particles are weighted using simulated expert observations and belief updates
(q(s_o)) from the expert's perspective.

Compute weight entropy H(w) BEFORE resampling for support calculation.

Performance Notes:
-----------------
- ToM update is the most expensive step: target < 5 ms per step
- Vectorize Q-value computation across particles
- Cache transition matrices and observation models

Module Structure:
----------------
This module now focuses solely on the high-level TheoryOfMind interface.
Implementation details are delegated to specialized modules:
- particle_filter: Particle weight management and updates
- perspective: Expert perspective tracking
- trust: Trust and reliability computation
- utils: Diagnostic and utility functions

Dependencies:
------------
- numpy: Array operations
- particle_filter: ParticleFilter class
- perspective: PerspectiveTracker class
- trust: Trust computation functions
- utils: Diagnostic functions
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable
import numpy as np

from empathy.clean_up.agent.social.particle_filter import ParticleFilter
from empathy.clean_up.agent.social.perspective import PerspectiveTracker
from empathy.clean_up.agent.social.trust import (
    compute_accuracy_advantage,
    compute_accuracy_gate,
    compute_trust,
    compute_effective_learning_rate,
)
from empathy.clean_up.agent.social.utils import (
    compute_tom_action_likelihoods,
    compute_tom_posterior_entropy,
)


# =============================================================================
# MAIN THEORY OF MIND CLASS
# =============================================================================

class TheoryOfMind:
    """
    Unified Theory of Mind inference for modeling other agents.

    Encapsulates particle-based parameter inference, perspective-taking,
    and trust computation in a single object-oriented interface.

    Usage:
        tom = TheoryOfMind(config, n_states, ...)
        tom.observe_action(action, context, ...)
        metrics = tom.compute_trust_metrics(...)
        expected_theta = tom.get_expected_parameters()
    
    Note: For clean_up task, particle_params has shape [N_particles, 5 contexts, 2 alphas]
    representing Dirichlet parameters for each pollution context's spawn probability.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        n_states: int,
        particle_params: np.ndarray,
        particle_weights: np.ndarray,
        observation_likelihood_fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
        get_transition_matrices_fn: Callable[[Optional[Dict[str, Any]]], np.ndarray],
        preferred_observations_fn: Callable[[], np.ndarray],
        predict_context_fn: Optional[Callable[[int, Dict[str, Any]], Dict[str, Any]]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize Theory of Mind inference.

        Args:
            config: Configuration dictionary
            n_states: Number of states
            particle_params: Initial particle parameters
                For clean_up: [N_particles, 5 contexts, 2 alphas] Dirichlet params
            particle_weights: Initial particle weights [N_particles]
            observation_likelihood_fn: Function (beliefs, context) -> obs_matrix
            get_transition_matrices_fn: Function (context) -> transition_matrices
            preferred_observations_fn: Function () -> preferred_obs
            predict_context_fn: Optional function (action, context) -> new_context
            rng: Random number generator
        """
        self.config = config
        self.n_states = n_states
        self.observation_likelihood_fn = observation_likelihood_fn
        self.get_transition_matrices_fn = get_transition_matrices_fn
        self.preferred_observations_fn = preferred_observations_fn
        self.predict_context_fn = predict_context_fn

        if rng is None:
            import numpy as np
            rng = np.random.default_rng()
        self.rng = rng

        # Initialize helper components
        self.particle_filter = ParticleFilter(
            particle_params,
            particle_weights,
            config,
            rng
        )
        self.perspective_tracker = PerspectiveTracker(n_states)

        # Track statistics
        self._last_resample_stats: Dict[str, float] = {}
        self._last_reliability_stats: Dict[str, float] = {}
        self._last_tom_diagnostics: Dict[str, Any] = {}

    def observe_action(
        self,
        observed_action: int,
        state_belief: np.ndarray,
        expert_position: int,
        context: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        """
        Main entry point: update particles and perspective based on observed action.

        Args:
            observed_action: Action taken by other agent
            state_belief: Learner's current state belief q(s_f)
            expert_position: Expert's position in environment
            context: Additional context information
            debug: Whether to compute diagnostic information
        """
        if context is None:
            context = {}

        # Build ToM context
        # Create wrapper functions that match expected signature for _update_particle_weights
        # But PerspectiveTracker.simulate_observation needs: (expert_pos, world_belief, theta_j, obs_likelihood_fn, context)
        def simulate_obs_wrapper(expert_pos, world_belief, theta_j, ctx):
            """Wrapper to match expected signature: (expert_pos, world_belief, theta_j, context)."""
            return self.perspective_tracker.simulate_observation(
                expert_pos,
                world_belief,
                theta_j,
                self.observation_likelihood_fn,
                ctx,
            )

        def simulate_belief_wrapper(prev_belief, sim_obs, theta_j, ctx):
            """Wrapper to match expected signature: (prev_belief, sim_obs, theta_j, context).
            
            CRITICAL FIX: Returns the predicted belief AFTER transition but BEFORE
            observation update. This preserves particle-specific differences:
            
            Why this matters:
            - Different particles have different spawn beliefs → different transition matrices
            - The transition T_j[a] @ belief produces different predicted beliefs per particle
            - But observation update is DETERMINISTIC (precision=1.0) → collapses to delta
            - If we apply obs update, all particles get same delta belief → same Q-values
            - By skipping obs update, we preserve the transition-induced differences
            
            The observation update serves a different purpose (state estimation from
            sensory data) than action prediction (which depends on expected dynamics).
            For ToM action likelihood, we want: "Given particle j's transition model,
            what belief would the expert have?" - this is the post-transition belief.
            """
            # Get build_transition_matrices from context (passed from agent.py)
            build_transition_fn = ctx.get("build_transition_matrices") if ctx else None
            last_action = ctx.get("last_expert_action") if ctx else None
            
            if build_transition_fn is not None and last_action is not None:
                # Convert particle params to DirichletBeliefs for transition model
                from empathy.clean_up.agent.beliefs import DirichletBeliefs
                particle_beliefs = DirichletBeliefs.from_array(theta_j)
                
                # Build particle-specific transition matrices
                # Each particle has different spawn beliefs → different transitions
                transition_matrices = build_transition_fn(particle_beliefs)
                
                # Predict belief after action: P(s'|a) = Σ_s P(s'|s,a) × P(s)
                # B[s',s] convention: columns sum to 1, so P(s') = B @ P(s)
                predicted_belief = transition_matrices[last_action] @ prev_belief
                
                # Normalize (may have numerical issues)
                predicted_sum = predicted_belief.sum()
                if predicted_sum > 1e-12:
                    predicted_belief = predicted_belief / predicted_sum
                else:
                    predicted_belief = np.ones_like(predicted_belief) / len(predicted_belief)
                
                # CRITICAL: Return predicted belief WITHOUT observation update
                # The obs update is deterministic and would collapse all particles
                # to the same delta belief, erasing the transition differences
                return predicted_belief
            else:
                # No transition model available, use prior directly (first timestep)
                # Fall back to observation update for first step when no action history
                old_belief = self.perspective_tracker.state_belief_other
                self.perspective_tracker.state_belief_other = prev_belief
                self.perspective_tracker.simulate_belief_update(
                    sim_obs,
                    theta_j,
                    self.observation_likelihood_fn,
                    ctx,
                )
                updated = self.perspective_tracker.state_belief_other.copy()
                if old_belief is None:
                    self.perspective_tracker.state_belief_other = old_belief
                return updated

        # Initialize expert belief if needed (first call)
        self.perspective_tracker.initialize_if_needed()

        tom_context = {
            "state_belief": state_belief,
            "transition_matrices": self.get_transition_matrices_fn(context),
            "preferred_obs": self.preferred_observations_fn(),
            "lambda_epist": float(self.config.get("lambda_epist", 0.5)),
            "obs_likelihood_fn": self.observation_likelihood_fn,
            "predict_context_fn": self.predict_context_fn,
            "context": context,
            "expert_position": expert_position,
            "expert_belief_prev": self.perspective_tracker.state_belief_other.copy(),
            "world_state_belief": state_belief,  # Use full state belief as world belief for Clean Up
            "simulate_expert_observation": simulate_obs_wrapper,
            "simulate_expert_belief_update": simulate_belief_wrapper,
            "record_tom_simulations": debug,
            "planning_horizon": int(self.config.get("planning_horizon", 3)),
        }

        # Update particle weights
        beta_tom = float(self.config.get("beta_tom", 2.0))
        tom_mode = self.config.get("tom_mode", "softmax")
        self.particle_filter.update_weights(
            observed_action,
            beta_tom,
            tom_context,
            tom_mode,
        )

        # Compute reliability BEFORE resampling
        u_threshold = float(self.config.get("u_0", 0.05))
        kappa = float(self.config.get("kappa", 0.05))
        reliability_stats_pre = self.particle_filter.compute_statistics(u_threshold, kappa)

        # Resample if needed
        resample_stats = self.particle_filter.resample_if_needed()
        self._last_resample_stats = resample_stats

        # Compute post-resample statistics
        reliability_stats_post = self.particle_filter.compute_statistics(u_threshold, kappa)

        self._last_reliability_stats = {
            "weight_entropy_pre": reliability_stats_pre["weight_entropy"],
            "weight_entropy_post": reliability_stats_post["weight_entropy"],
            "confidence_pre": reliability_stats_pre["confidence"],
            "confidence_post": reliability_stats_post["confidence"],
            "reliability_pre": reliability_stats_pre["reliability"],
            "reliability_post": reliability_stats_post["reliability"],
        }

        # Update perspective (expert state belief) using transition-predicted belief
        # This preserves uncertainty instead of collapsing to delta via observation update
        build_transition_fn = context.get("build_transition_matrices") if context else None
        self.perspective_tracker.track_state(
            state_belief,
            self.particle_filter.get_expected_parameters(),
            expert_position,
            self.observation_likelihood_fn,
            context,
            build_transition_fn=build_transition_fn,
            last_expert_action=observed_action,
        )

        # Compute diagnostics if requested
        if debug:
            action_likelihoods = compute_tom_action_likelihoods(
                self.particle_filter.particle_params,
                tom_context,
                beta_tom,
                tom_mode,
            )
            tom_entropy = compute_tom_posterior_entropy(
                self.particle_filter.particle_weights,
                action_likelihoods
            )
            action_dist = np.sum(
                self.particle_filter.particle_weights[:, None] * action_likelihoods,
                axis=0
            )
            total = np.sum(action_dist)
            if total > 0.0:
                action_dist = action_dist / total
            tom_hypotheses = float(np.sum(action_dist > 0.01))
            self._last_tom_diagnostics = {
                "tom_action_entropy": float(tom_entropy),
                "tom_action_hypotheses": tom_hypotheses,
                "tom_loglike_action_mean": float(
                    tom_context.get("tom_loglike_action_mean", np.nan)
                ),
                "tom_simulated_obs": tom_context.get("tom_simulated_obs"),
                "tom_action_loglikes": tom_context.get("tom_action_loglikes"),
                "tom_action_likelihoods": action_likelihoods,
            }
        else:
            self._last_tom_diagnostics = {
                "tom_action_entropy": np.nan,
                "tom_action_hypotheses": np.nan,
                "tom_loglike_action_mean": np.nan,
            }

    def compute_trust_metrics(
        self,
        params_self,  # DirichletBeliefs
        state_belief: np.ndarray,
        obs_history: np.ndarray,
        state_history: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute trust metrics: reliability, accuracy gate, trust, learning rate.

        Args:
            params_self: Self model parameters (DirichletBeliefs)
            state_belief: Learner's state belief
            obs_history: Observation history for accuracy gate
            state_history: State history for accuracy gate
            context: Additional context

        Returns:
            metrics: Dictionary with trust-related metrics
        """
        if context is None:
            context = {}

        # Get reliability from pre-resample statistics
        reliability_stats = self._last_reliability_stats
        weight_entropy = reliability_stats.get("weight_entropy_pre", np.nan)
        confidence = reliability_stats.get("confidence_pre", np.nan)
        reliability = reliability_stats.get("reliability_pre", np.nan)

        # Recompute if not available
        if not np.isfinite(weight_entropy):
            stats = self.particle_filter.compute_statistics(
                float(self.config.get("u_0", 0.05)),
                float(self.config.get("kappa", 0.05))
            )
            weight_entropy = stats["weight_entropy"]
            confidence = stats["confidence"]
            reliability = stats["reliability"]

        # Compute accuracy gate
        alpha_other_dict = self.particle_filter.get_expected_parameters()
        T_a = float(self.config.get("T_a", 2.0))

        accuracy_advantage = compute_accuracy_advantage(
            params_self,
            alpha_other_dict,
            obs_history,
            state_history,
            self.observation_likelihood_fn,
            context,
        )
        tau_accuracy = compute_accuracy_gate(accuracy_advantage, T_a)

        # Compute trust
        trust = compute_trust(reliability, tau_accuracy)

        # Effective learning rate
        eta_0 = float(self.config.get("eta_0", 0.1))
        eta_t = compute_effective_learning_rate(eta_0, trust)

        # Belief similarity
        belief_similarity = self.perspective_tracker.get_belief_similarity(state_belief)

        # Combine all metrics
        metrics = {
            "weight_entropy": float(weight_entropy),
            "confidence": float(confidence),
            "reliability": float(reliability),
            "tau_accuracy": float(tau_accuracy),
            "accuracy_advantage": float(accuracy_advantage),
            "trust": float(trust),
            "eta_t": float(eta_t),
            "alpha_other_expected": alpha_other_dict,  # Expected Dirichlet params per context
            "weight_entropy_pre": float(reliability_stats.get("weight_entropy_pre", np.nan)),
            "confidence_pre": float(reliability_stats.get("confidence_pre", np.nan)),
            "reliability_pre": float(reliability_stats.get("reliability_pre", np.nan)),
            "weight_entropy_post": float(reliability_stats.get("weight_entropy_post", np.nan)),
            "confidence_post": float(reliability_stats.get("confidence_post", np.nan)),
            "reliability_post": float(reliability_stats.get("reliability_post", np.nan)),
            "n_eff_pre": float(self._last_resample_stats.get("n_eff_pre", np.nan)),
            "n_eff_post": float(self._last_resample_stats.get("n_eff_post", np.nan)),
            "is_resampled": float(self._last_resample_stats.get("is_resampled", np.nan)),
            "tom_action_entropy": float(self._last_tom_diagnostics.get("tom_action_entropy", np.nan)),
            "tom_action_hypotheses": float(self._last_tom_diagnostics.get("tom_action_hypotheses", np.nan)),
            "tom_loglike_action_mean": float(
                self._last_tom_diagnostics.get("tom_loglike_action_mean", np.nan)
            ),
            "belief_similarity": belief_similarity,
        }

        return metrics

    def get_expected_parameters(self) -> dict:
        """Return expected Dirichlet parameters for all contexts."""
        return self.particle_filter.get_expected_parameters()

    def get_state_belief_other(self) -> Optional[np.ndarray]:
        """Return expert's state belief q(s_o)."""
        return self.perspective_tracker.state_belief_other

    def reset(self) -> None:
        """Reset all state."""
        # Reset is handled by reinitializing particles in base.py
        self.perspective_tracker.reset()
        self._last_resample_stats = {}
        self._last_reliability_stats = {}
        self._last_tom_diagnostics = {}
