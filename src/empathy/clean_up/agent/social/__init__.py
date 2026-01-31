"""Social Learning Module: Theory of Mind, trust, and social updates."""

from empathy.clean_up.agent.social.tom import TheoryOfMind
from empathy.clean_up.agent.social.particle_filter import (
    ParticleFilter,
    compute_weight_entropy,
)
from empathy.clean_up.agent.social.perspective import PerspectiveTracker
from empathy.clean_up.agent.social.utils import (
    # Particle utilities
    effective_particle_count,
    resample_particles,
    # ToM diagnostic functions
    update_particle_weights,
    compute_tom_action_likelihoods,
    compute_tom_posterior_entropy,
)
from empathy.clean_up.agent.social.trust import (
    compute_confidence,
    compute_reliability,
    compute_accuracy_advantage,
    compute_accuracy_gate,
    compute_trust,
    compute_effective_learning_rate,
)
from empathy.clean_up.agent.social.update import (
    social_dirichlet_update,
    compute_effective_influence,
)

__all__ = [
    # Main ToM Class
    "TheoryOfMind",
    "ParticleFilter",
    "PerspectiveTracker",
    # Standalone ToM Functions
    "update_particle_weights",
    "compute_weight_entropy",
    "compute_tom_action_likelihoods",
    "compute_tom_posterior_entropy",
    # Trust
    "compute_confidence",
    "compute_reliability",
    "compute_accuracy_advantage",
    "compute_accuracy_gate",
    "compute_trust",
    "compute_effective_learning_rate",
    # Social Update
    "social_dirichlet_update",
    "compute_effective_influence",
    # Utilities
    "effective_particle_count",
    "resample_particles",
]
