"""
Unit Tests for Experiment Module.
"""

import numpy as np

from empathy.clean_up.experiment.config import get_experiment_config, create_experiment_grid, validate_episode_count
from empathy.clean_up.experiment.metrics import compute_regime_label
from empathy.clean_up.experiment.logger import ExperimentLogger
from empathy.clean_up.experiment.runner import (
    run_single_episode,
    create_agents_for_config,
    create_environment_for_config,
    ExperimentRunner,
)


def test_create_experiment_grid_counts():
    config = get_experiment_config("smoke")
    grid = create_experiment_grid(config)
    assert len(grid) == config["expected_episodes"]
    assert validate_episode_count(config)


def test_logger_records_timestep():
    from empathy.clean_up.agent.beliefs import DirichletBeliefs

    logger = ExperimentLogger()
    dummy_agent = type("Dummy", (), {})()
    dummy_agent.state_belief = np.array([0.5, 0.5])
    dummy_agent.beliefs_self = DirichletBeliefs(n_contexts=5, initial_alpha=1.0)
    dummy_agent.particle_params = np.array([[[1.0, 1.0]] * 5])  # shape (1, 5, 2)
    dummy_agent.particle_weights = np.array([1.0])
    dummy_agent.trust = 0.6

    social_metrics = {
        "weight_entropy": 0.0,
        "confidence": 1.0,
        "reliability": 0.9,
        "tau_accuracy": 0.7,
        "accuracy_advantage": 1.0,
        "trust": 0.63,
        "eta_t": 0.05,
    }
    logger.log_timestep(0, dummy_agent, social_metrics, env_info={"progress": 0.0})
    results = logger.get_results()
    assert results["metadata"]["n_steps"] == 1
    assert len(results["time_series"]["trust"]) == 1


def test_run_single_episode_basic():
    """
    Verifies that running a single episode for a small clean_up configuration produces time series data and completes within the configured step limit.
    """
    config = {
        "scenario": "default_truth",
        "seed": 0,
        "max_steps": 5,
        "n_agents": 2,
        "agents": [],
        "social_enabled": False,
    }
    learner, expert = create_agents_for_config(config)
    env = create_environment_for_config(config)
    results = run_single_episode(learner, expert, env, config)
    assert "time_series" in results
    assert results["outcomes"]["time_to_success"] <= config["max_steps"]


def test_experiment_runner_small():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        override = {
            "tasks": [
                {"task": "clean_up", "scenario": "default_truth"},
            ],
            "n_seeds": 1,
            "expected_episodes": 1,
            "max_steps": 20,  # Shorter test to avoid timeout
        }
        runner = ExperimentRunner("smoke", config_overrides=override)
        results = runner.run(n_workers=1, verbose=False, output_dir=Path(tmpdir))
        assert isinstance(results, list)