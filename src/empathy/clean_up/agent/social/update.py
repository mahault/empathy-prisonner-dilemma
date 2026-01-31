"""
Social Parameter Update: The Core Social Learning Mechanism.

This module provides Numba-compiled functions for the social parameter
update - how the learner incorporates inferred knowledge from the other agent.

Mathematical Foundation (Accuracy Gate Trust):
---------------------------------------------------
The social learning update moves self-parameters toward the expected
parameters under the other model:

    θ_self^new = θ_self^old + η_0 * r_t * τ_accuracy * (E[θ_other] - θ_self^old)

where:
- η_0: Base social learning rate (default: 0.1)
- r_t: Reliability (ToM confidence, from particle weight concentration)
- τ_accuracy: Accuracy gate (sigmoid of log-likelihood advantage)
- E[θ_other] = Σ_j w_j θ^j: Expected other parameters

The update is continuously gated by trust = r_t · τ_accuracy:
- Low trust → small update step (η_t = η_0 · trust)
- High trust → larger update step

Roles (Clear Separation):
-------------------------
- Reliability (r_t): Is our ToM inference confident enough?
- Accuracy gate (τ_accuracy): Does the expert predict observations better?
- Trust = r_t · τ_accuracy: Combined gating factor
- Base rate (η_0): Maximum step size per timestep

Why Parameter Space:
-------------------
We update in parameter space (not count space) because:
1. Avoids issues with negative pseudo-counts
2. More interpretable (direct interpolation of beliefs)
3. Bounded and well-defined for our low-dimensional θ

Effective Social Influence:
--------------------------
We track the combined influence term:
    η_t = η_0 · r_t · τ_t

This is the effective "social learning rate" at each timestep.

Performance Notes:
-----------------
- Social update is fast (< 0.5 ms)
- Main computation already done by evidence and support modules

Dependencies:
------------
- numpy: Array operations
- numba: JIT compilation (@njit decorator)
"""

from __future__ import annotations

from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from empathy.clean_up.agent.beliefs import DirichletBeliefs

import numpy as np

# Optional numba acceleration can be enabled if available.
from numba import njit, float64


# =============================================================================
# SOCIAL PARAMETER UPDATE
# =============================================================================

def social_dirichlet_update(
    beliefs_self: DirichletBeliefs,
    alpha_other_dict: Dict[int, Any],
    reliability: float,
    trust: float,
    eta_0: float,
    contexts_observed: List[int],
) -> None:
    """
    Apply social parameter update for Dirichlet beliefs.

    Mathematical Update (per context):
        alpha_new[c] = (1 - η_t) · alpha_self[c] + η_t · E[alpha_other[c]]

    Where η_t = η_0 * r_t * τ_accuracy (reliability * trust)

    Args:
        beliefs_self: Learner's DirichletBeliefs (modified in-place)
        alpha_other_dict: Expected alpha parameters from particle filter {context: [alpha_0, alpha_1]}
        reliability: Reliability r_t in [0, 1] (ToM confidence filter)
        trust: Trust τ_accuracy in [0, 1] (accuracy gate)
        eta_0: Base social learning rate
        contexts_observed: List of contexts to update (pollution levels observed)
    """
    # Compute effective learning rate
    eta_t = eta_0 * reliability * trust
    eta_t = max(0.0, min(eta_t, eta_0))

    if eta_t == 0 or not contexts_observed:
        return

    # Update each observed context
    for context in contexts_observed:
        if context not in alpha_other_dict:
            continue

        alpha_other = alpha_other_dict[context]
        alpha_self = beliefs_self.alpha[context]

        # Weighted averaging: interpolate toward expert's beliefs
        alpha_new = (1.0 - eta_t) * alpha_self + eta_t * alpha_other
        beliefs_self.alpha[context] = alpha_new


@njit(float64(float64, float64, float64), cache=True)
def compute_effective_influence(
    reliability: float,
    trust: float,
    eta_0: float = 0.1
) -> float:
    """
    Compute effective social learning rate η_t.

    Mathematical Definition:
        η_t = η_0 · τ_t

    where η_t = η_0 * r_t * τ_accuracy (reliability * trust).

    This combined term represents the total "social learning rate":
    - η_t ≈ 0: No social learning (low reliability or low accuracy)
    - η_t ≈ η_0: Maximum social learning

    Logged for analysis and regime classification.

    Args:
        reliability: Reliability r_t (ToM confidence filter)
        trust: Trust τ_accuracy (accuracy gate)
        eta_0: Base learning rate (default: 0.1)

    Returns:
        eta_t: Effective learning rate ≥ 0

    Numba Signature:
        @njit(float64(float64, float64, float64))
    """
    # eta_t = eta_0 * reliability * trust = eta_0 * r_t * tau_accuracy
    return eta_0 * reliability * trust


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_ETA_0 = 0.1                 # Base social learning rate η_0
DEFAULT_PERFORMANCE_WINDOW = 20    # Window for accuracy gate observation history
