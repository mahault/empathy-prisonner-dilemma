"""
Discovery analysis functions for experiment results.

Naming convention: This module uses 'spawn_probs' terminology throughout.
Spawn probs are 5-dimensional vectors (one probability per pollution context 0-4)
representing P(apple_spawns | pollution_context).

Config keys use 'spawn_probs_true', 'beliefs_expert', 'beliefs_self_init'.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from empathy.clean_up.experiment.metrics import (
    extract_result_fields,
    normalize_spawn_probs,
    get_spawn_prob_series,
    compute_error,
)


def compute_discovery_success_rate(
    results: list[Any],
    ground_truth_params: dict[str, dict[str, float]] | None = None,
    convergence_threshold: float = 0.1,
) -> dict[str, Any]:
    success_count = 0
    total_count = 0
    convergence_times: list[float] = []
    final_param_error: list[float] = []

    for result in results:
        config, time_series, _ = extract_result_fields(result)
        task = config.get("task", "clean_up")
        spawn_probs_expert = None
        if ground_truth_params and task in ground_truth_params:
            spawn_probs_expert = normalize_spawn_probs(ground_truth_params[task])
        if spawn_probs_expert is None:
            spawn_probs_expert = normalize_spawn_probs(config.get("beliefs_expert"))
        spawn_probs_series = get_spawn_prob_series(time_series)
        if spawn_probs_expert is None or spawn_probs_series is None:
            continue
        # validate that spawn_probs_series is not empty before processing
        if len(spawn_probs_series) == 0:
            continue
        total_count += 1
        errors = []
        for spawn_probs_self in spawn_probs_series:
            errors.append(compute_error(spawn_probs_self, spawn_probs_expert))
        errors = np.array(errors, dtype=float)
        final_param_error.append(float(errors[-1]))
        hit_indices = np.where(errors <= convergence_threshold)[0]
        if hit_indices.size > 0:
            success_count += 1
            convergence_times.append(float(hit_indices[0]))

    discovery_rate = float(success_count / total_count) if total_count else float("nan")
    return {
        "discovery_success_rate": discovery_rate,
        "convergence_time": convergence_times,
        "final_param_error": final_param_error,
        "success_count": success_count,
        "total_count": total_count,
    }


def plot_discovery_trajectories(
    results: list[Any],
    output_path: str,
    title: str = "Discovery Dynamics",
) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not results:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for result in results[:3]:
        _, time_series, _ = extract_result_fields(result)
        delta_t = time_series.get("delta_t", [])
        axes[0].plot(delta_t, alpha=0.7, label="delta_t")
        spawn_probs_self_series = get_spawn_prob_series(time_series)
        if spawn_probs_self_series is not None and spawn_probs_self_series.size > 0:
            # Dirichlet format: 5 spawn probabilities per timestep
            if spawn_probs_self_series.ndim == 2 and spawn_probs_self_series.shape[1] == 5:
                # Plot mean spawn probability across contexts
                axes[1].plot(np.mean(spawn_probs_self_series, axis=1), alpha=0.7)
            elif spawn_probs_self_series.ndim == 1:
                axes[1].plot(spawn_probs_self_series, alpha=0.7)
        raw_other = time_series.get("param_other_mean", [])
        # normalize: support list and ndarray; filter None (and NaN), then convert to float array
        if isinstance(raw_other, list):
            filtered = [p for p in raw_other if p is not None]
            spawn_probs_other_mean = np.array(filtered, dtype=float)
        else:
            spawn_probs_other_mean = np.asarray(raw_other, dtype=float)
        if spawn_probs_other_mean.size > 0:
            # drop rows with any NaN (optional filter)
            if spawn_probs_other_mean.ndim == 2:
                spawn_probs_other_mean = spawn_probs_other_mean[np.isfinite(spawn_probs_other_mean).all(axis=1)]
            elif spawn_probs_other_mean.ndim == 1:
                spawn_probs_other_mean = spawn_probs_other_mean[np.isfinite(spawn_probs_other_mean)]
        if spawn_probs_other_mean.size > 0:
            # Dirichlet format: 5 spawn probabilities per timestep
            if spawn_probs_other_mean.ndim == 2 and spawn_probs_other_mean.shape[1] == 5:
                # Plot mean spawn probability across contexts
                axes[2].plot(np.mean(spawn_probs_other_mean, axis=1), alpha=0.7)
            elif spawn_probs_other_mean.ndim == 1:
                axes[2].plot(spawn_probs_other_mean, alpha=0.7)
        belief_similarity = time_series.get("belief_similarity", [])
        axes[3].plot(belief_similarity, alpha=0.7)

    axes[0].set_title("evidence")
    axes[1].set_title("spawn_prob (mean)")
    axes[2].set_title("ToM spawn_prob (mean)")
    axes[3].set_title("belief_similarity")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
