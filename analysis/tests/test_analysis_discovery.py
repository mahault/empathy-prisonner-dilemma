"""
Tests for discovery and echo-chamber analysis utilities.
"""

import numpy as np
import pytest
from typing import Any

from analysis import (
    compute_discovery_success_rate,
    compute_tom_accuracy_vs_belief_similarity,
    plot_discovery_trajectories,
)


def _make_result(task: str = "clean_up") -> dict[str, Any]:
    """
    Create a synthetic result dict used in tests for discovery and echo-chamber analysis.
    
    Parameters:
        task (str): Task name to include in the result's config (defaults to "clean_up").
    
    Returns:
        result (dict): A dictionary with keys:
            - config (dict): Contains 'task' and 'beliefs_expert' with spawn_probs (5 values).
            - time_series (dict): Contains time-series fields:
                - delta_t (list of float)
                - param_self (list of numpy.ndarray) - 5 spawn probs per timestep
                - param_other_mean (list of numpy.ndarray) - 5 spawn probs per timestep
                - belief_similarity (list of float)
            - outcomes (dict): Empty outcomes dictionary.
    """
    # Expert spawn probabilities: decreasing with pollution level
    expert_spawn_probs = [0.8, 0.6, 0.4, 0.2, 0.1]
    config = {
        "task": task,
        "beliefs_expert": {"spawn_probs": expert_spawn_probs},
    }
    time_series = {
        "delta_t": [1.0, -1.0, 1.0],
        # Learner spawn probs converging toward expert
        "param_self": [
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),  # Initial uniform
            np.array([0.75, 0.55, 0.42, 0.25, 0.15]),  # Closer to expert
        ],
        "param_other_mean": [
            np.array(expert_spawn_probs),
            np.array(expert_spawn_probs),
        ],
        "belief_similarity": [0.2, 0.1],
    }
    return {"config": config, "time_series": time_series, "outcomes": {}}

def test_compute_discovery_success_rate():
    results = [_make_result()]
    stats = compute_discovery_success_rate(results, convergence_threshold=0.3)
    assert stats["total_count"] == 1
    assert stats["success_count"] == 1


def test_compute_tom_accuracy_vs_belief_similarity():
    results = [_make_result()]
    stats = compute_tom_accuracy_vs_belief_similarity(results)
    assert "correlation" in stats
    assert np.isfinite(stats["mean_tom_accuracy"])


def test_plot_discovery_trajectories(tmp_path):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")
    output_path = tmp_path / "discovery_plot.png"
    result_path = plot_discovery_trajectories([_make_result()], str(output_path))
    assert result_path is not None
    assert output_path.exists()