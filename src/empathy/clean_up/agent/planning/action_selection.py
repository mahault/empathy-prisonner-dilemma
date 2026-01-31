"""
Action Selection: Softmax Policy from EFE Values.

This module provides action selection based on Expected Free Energy (EFE) values
using a softmax policy.

Supports both single-step (H=1) and multi-step (H>1) planning:
- Single-step: select_action_softmax() samples from softmax over action EFEs
- Multi-step: select_action_softmax_policies() samples policy, returns first action
"""

import numpy as np
from numba import njit, float64, int64, uint32

from empathy.clean_up.agent.inference.utils import softmax


@njit(int64(float64[:], float64, uint32), cache=True)
def select_action_softmax(
    efe_values: np.ndarray,
    beta: float,
    rng_seed: int
) -> int:
    """
    Select action via softmax policy over EFE values.
    
    Implements: p(a) ∝ exp(-β · G(a))
    
    where:
        G(a) = EFE[a] (lower is better)
        β = inverse temperature (policy precision)
           - β → 0: random action selection
           - β → ∞: greedy selection (argmin)
    
    Args:
        efe_values: EFE for each action, shape [N_actions]
        beta: Inverse temperature (policy precision)
        rng_seed: Seed for random sampling (for Numba RNG)
        
    Returns:
        action: Sampled action index
        
    Implementation Notes:
    --------------------
    1. Subtract max for numerical stability before exp
    2. Compute softmax probabilities
    3. Sample from categorical using uniform random
    
    Numba Signature:
        @njit(int64(float64[:], float64, uint32))
    """
    logits = -beta * efe_values
    probs = softmax(logits)
    np.random.seed(rng_seed)
    u = np.random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if u < cumsum:
            return i
    return len(probs) - 1


def select_action_softmax_policies(
    efe_values: np.ndarray,
    policies: np.ndarray,
    beta: float,
    rng_seed: int,
) -> int:
    """
    Select first action from policy via softmax over policy EFE values.
    
    For multi-step planning (H>1), this function:
    1. Applies softmax over policies: p(π) ∝ exp(-β · G(π))
    2. Samples a policy index from the distribution
    3. Returns the first action from the sampled policy
    
    This enables multi-step planning while maintaining the same return type
    (single action) as single-step planning for seamless integration.
    
    Args:
        efe_values: EFE for each policy, shape [N_policies]
        policies: Array of policies, shape [N_policies, H] where each row is
                 an action sequence [a₀, a₁, ..., a_{H-1}]
        beta: Inverse temperature (policy precision)
        rng_seed: Seed for random sampling
        
    Returns:
        action: First action (a₀) from the sampled policy
        
    Example:
        >>> policies = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])  # 3 policies, H=3
        >>> efe_values = np.array([1.0, 0.5, 2.0])  # EFE for each policy
        >>> action = select_action_softmax_policies(efe_values, policies, beta=1.0, rng_seed=42)
        >>> # Returns first action from sampled policy (likely policies[1, 0] = 1 due to lowest EFE)
    
    Note:
        When H=1, policies has shape [N_actions, 1] and this function is equivalent
        to select_action_softmax(), maintaining backward compatibility.
    """
    # Sample policy index using softmax
    policy_idx = select_action_softmax(efe_values, beta, rng_seed)
    
    # Return first action from sampled policy
    return int(policies[policy_idx, 0])
