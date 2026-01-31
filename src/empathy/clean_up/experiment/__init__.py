"""
Clean Up Experiment Module.

Experiment orchestration, configuration, logging, and analysis for the clean_up task.
"""

from empathy.clean_up.experiment.runner import ExperimentRunner, run_single_episode, EpisodeResult
from empathy.clean_up.experiment.config import (
    EXPERIMENT_CONFIGS,
    get_experiment_config,
    create_experiment_grid,
    compute_expected_episodes,
)
from empathy.clean_up.experiment.logger import ExperimentLogger, MetricTracker
from empathy.clean_up.experiment.metrics import (
    compute_final_error,
    compute_drift_metrics,
    compute_detection_time,
    compute_stability_metrics,
    compute_regime_label,
)

__all__ = [
    # Runner
    "ExperimentRunner",
    "run_single_episode",
    "EpisodeResult",
    # Config
    "EXPERIMENT_CONFIGS",
    "get_experiment_config",
    "create_experiment_grid",
    "compute_expected_episodes",
    # Logger
    "ExperimentLogger",
    "MetricTracker",
    # Metrics
    "compute_final_error",
    "compute_drift_metrics",
    "compute_detection_time",
    "compute_stability_metrics",
    "compute_regime_label",
]
