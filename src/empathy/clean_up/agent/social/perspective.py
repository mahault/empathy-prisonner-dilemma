"""
Perspective Tracking: Expert State Belief via Perspective-Taking.

This module provides the PerspectiveTracker class for tracking and updating
the expert's state belief q(s_o) from the expert's perspective.

Mathematical Foundation:
---------------------------------------------------
The learner maintains beliefs about:
1. Their own state belief: q(s_f)
2. The expert's state belief: q(s_o)

Perspective-taking simulates how the expert updates their belief:

1. Simulate expert observation:
   - Select world state from learner's belief: s ~ q(s_f)
   - Generate observation from expert's position: o ~ p(o | s, pos_expert, θ_expert)

2. Simulate expert belief update:
   - Update expert belief using simulated observation
   - q(s_o)' ∝ q(s_o) · p(o | s, θ_expert)

This allows the learner to understand what the expert knows and how they
make decisions based on their own (potentially different) observations.

Key Features:
------------
- Track expert belief history (window of last 10 beliefs)
- Simulate observations from expert's perspective
- Update expert beliefs using observed actions
- Compute belief similarity (KL divergence)

Performance Notes:
-----------------
- Perspective tracking is fast (< 1 ms per step)
- Main cost is belief update computation
- History window prevents memory growth

Dependencies:
------------
- numpy: Array operations
- agent.inference.state: State belief updates
- agent.social.utils: Perspective simulation utilities
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, Any
import numpy as np

from empathy.clean_up.agent.inference.state import update_state_belief, kl_divergence
from empathy.clean_up.agent.social.utils import (
    choose_world_state,
    simulate_discrete_observation,
)


# =============================================================================
# PERSPECTIVE TRACKER CLASS
# =============================================================================

class PerspectiveTracker:
    """
    Manages expert's state belief q(s_o) via perspective-taking.

    Encapsulates expert state tracking, observation simulation, and
    belief updates from the expert's perspective.
    """

    def __init__(self, n_states: int):
        """
        Initialize perspective tracker.

        Args:
            n_states: Number of states

        Raises:
            ValueError: If n_states is not a positive integer
        """
        if not isinstance(n_states, int) or n_states <= 0:
            raise ValueError(f"n_states must be a positive integer, got {n_states}")
        self.n_states = n_states
        self.state_belief_other: Optional[np.ndarray] = None
        self.state_belief_other_window: list = []

    def initialize_if_needed(self) -> None:
        """Initialize expert belief to uniform if not set."""
        if self.state_belief_other is None:
            self.state_belief_other = np.ones(self.n_states, dtype=float) / self.n_states

    def track_state(
        self,
        state_belief_self: np.ndarray,
        params_other: dict,  # Changed from theta_other to params_other (dict of alphas)
        expert_position: int,
        observation_likelihood_fn: Callable,
        context: Optional[Dict[str, Any]] = None,
        build_transition_fn: Optional[Callable] = None,
        last_expert_action: Optional[int] = None,
    ) -> None:
        """
        Update expert's belief state q(s_o) via perspective-taking.
        
        CRITICAL: When transition model is available, we use the post-transition
        belief WITHOUT observation update. This preserves uncertainty that allows
        particles to discriminate based on their different transition models.
        
        The observation update (precision=1.0) is deterministic and collapses
        beliefs to deltas, erasing the differences between particles.

        Args:
            state_belief_self: Learner's belief q(s_f)
            params_other: Expert's parameters (dict of alpha values per context from ToM)
            expert_position: Expert's position
            observation_likelihood_fn: Function (params, context) -> obs_matrix
            context: Additional context
            build_transition_fn: Function to build transition matrices from beliefs
            last_expert_action: Expert's last action (for transition prediction)
        """
        self.initialize_if_needed()

        if build_transition_fn is not None and last_expert_action is not None:
            # Use transition-predicted belief WITHOUT observation collapse
            # This preserves uncertainty for particle discrimination
            from empathy.clean_up.agent.beliefs import DirichletBeliefs
            
            # Convert params_other (dict of alphas) to DirichletBeliefs
            # params_other has format {context: alpha_array} from get_expected_parameters()
            n_contexts = len(params_other)
            alpha_array = np.zeros((n_contexts, 2), dtype=float)
            for ctx, alpha in params_other.items():
                alpha_array[ctx] = alpha
            expert_beliefs = DirichletBeliefs.from_array(alpha_array)
            
            # Build transition matrices using expected expert parameters
            transition_matrices = build_transition_fn(expert_beliefs)
            
            # Predict belief after expert's action: P(s') = B[a] @ P(s)
            predicted_belief = transition_matrices[last_expert_action] @ self.state_belief_other
            
            # Normalize
            predicted_sum = predicted_belief.sum()
            if predicted_sum > 1e-12:
                self.state_belief_other = predicted_belief / predicted_sum
            else:
                self.state_belief_other = np.ones(self.n_states, dtype=float) / self.n_states
        else:
            # Fallback: use observation update (first timestep when no action history)
            # Simulate expert's observation
            simulated_obs = self.simulate_observation(
                expert_position,
                state_belief_self,
                params_other,
                observation_likelihood_fn,
                context,
            )

            # Update expert's belief
            self.simulate_belief_update(
                simulated_obs,
                params_other,
                observation_likelihood_fn,
                context,
            )

        # Track history
        self.state_belief_other_window.append(self.state_belief_other.copy())
        window_size = 10  # Could be configurable
        if len(self.state_belief_other_window) > window_size:
            self.state_belief_other_window.pop(0)

    def simulate_observation(
        self,
        expert_position: int,
        world_belief: np.ndarray,
        params_other: dict,
        observation_likelihood_fn: Callable,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Simulate the observation an expert would receive from their perspective.

        Args:
            expert_position: Expert's position
            world_belief: Learner's belief over world states
            params_other: Expert's parameters (dict of alphas)
            observation_likelihood_fn: Function to build obs likelihood matrix
            context: Additional context

        Returns:
            simulated_obs: Simulated observation index
        """
        state_idx = choose_world_state(world_belief, mode="map")
        return self.simulate_observation_from_state(
            expert_position,
            state_idx,
            params_other,
            observation_likelihood_fn,
            context,
        )

    def simulate_observation_from_state(
        self,
        expert_position: int,
        state_idx: int,
        params_other: dict,
        observation_likelihood_fn: Callable,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Simulate observation from pre-selected world state.

        Args:
            expert_position: Expert's position
            state_idx: Pre-selected world state index
            params_other: Expert's parameters (dict of alphas)
            observation_likelihood_fn: Function to build obs likelihood matrix
            context: Additional context

        Returns:
            simulated_obs: Simulated observation index
        """
        expert_context = {**(context or {})}
        expert_context["agent_pos"] = int(expert_position)
        # Note: observation_likelihood_fn expects params but may not use them (deterministic obs model)
        obs_matrix = observation_likelihood_fn(params_other, expert_context)
        simulated_obs = simulate_discrete_observation(obs_matrix, int(state_idx))
        return int(simulated_obs)

    def simulate_belief_update(
        self,
        simulated_obs: int,
        params_other: dict,
        observation_likelihood_fn: Callable,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update expert's belief q(s_o) using simulated observation.

        Args:
            simulated_obs: Simulated observation
            params_other: Expert's parameters (dict of alphas)
            observation_likelihood_fn: Function to build obs likelihood matrix
            context: Additional context
        """
        self.initialize_if_needed()
        obs_matrix = observation_likelihood_fn(params_other, context or {})
        self.state_belief_other = update_state_belief(
            self.state_belief_other,
            int(simulated_obs),
            obs_matrix,
            precision=1.0,
        )

    def get_belief_similarity(self, state_belief_self: np.ndarray) -> float:
        """
        Compute KL divergence between self and other beliefs.

        Args:
            state_belief_self: Learner's belief q(s_f)

        Returns:
            kl_div: D_KL(q_focal || q_other)
        """
        if self.state_belief_other is None:
            return np.nan
        return float(kl_divergence(state_belief_self, self.state_belief_other))

    def reset(self) -> None:
        """Reset expert belief to uniform."""
        self.state_belief_other = np.ones(self.n_states, dtype=float) / self.n_states
        self.state_belief_other_window.clear()
