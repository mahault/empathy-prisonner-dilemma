"""Inference Module: State belief updates and Bayesian learning."""

from empathy.clean_up.agent.inference.state import (
    update_state_belief,
    belief_propagation_step,
    compute_entropy,
    kl_divergence,
)
from empathy.clean_up.agent.inference.utils import (
    normalize,
    softmax,
    gaussian_bin_prob,
    discretize_to_bin,
)

__all__ = [  # noqa: RUF022
    # State Inference
    "update_state_belief",
    "belief_propagation_step",
    "compute_entropy",
    "kl_divergence",
    # Utilities
    "normalize",
    "softmax",
    "gaussian_bin_prob",
    "discretize_to_bin",
]
