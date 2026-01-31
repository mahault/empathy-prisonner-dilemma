"""Planning Module: EFE computation and action selection.

Supports both single-step (H=1) and multi-step (H>1) planning:
- Single-step: Use compute_batched_efe() with actions
- Multi-step: Use compute_batched_efe_policies() with policies (action sequences)
- Policy EFE: compute_efe() for individual policy evaluation
"""

from empathy.clean_up.agent.planning.efe import (
    compute_efe_one_step,
    compute_pragmatic_value,
    compute_information_gain,
    compute_batched_efe,
    compute_efe,
    compute_batched_efe_policies,
)
from empathy.clean_up.agent.planning.action_selection import (
    select_action_softmax,
    select_action_softmax_policies,
)

__all__ = [
    "compute_batched_efe",
    "compute_batched_efe_policies",
    "compute_efe",
    "compute_efe_one_step",
    "compute_information_gain",
    "compute_pragmatic_value",
    "select_action_softmax",
    "select_action_softmax_policies",
]
