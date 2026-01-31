"""
Social learning and Theory of Mind (ToM) analysis functions.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any

from empathy.clean_up.experiment.metrics import (
    extract_result_fields,
    normalize_spawn_probs,
    compute_error,
)

logger = logging.getLogger(__name__)


def compute_tom_accuracy_vs_belief_similarity(
    results: list[Any],
    expert_true_params: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    tom_accuracy_vals: list[float] = []
    belief_similarity_vals: list[float] = []

    for result in results:
        config, time_series, _ = extract_result_fields(result)
        task = config.get("task", "clean_up")
        spawn_probs_expert = None
        # check expert_true_params before calling normalize_spawn_probs
        if expert_true_params and task in expert_true_params:
            spawn_probs_expert = normalize_spawn_probs(expert_true_params[task])
        # check config.get("beliefs_expert") before calling normalize_spawn_probs
        if spawn_probs_expert is None:
            spawn_probs_expert_raw = config.get("beliefs_expert")
            if spawn_probs_expert_raw is not None:
                spawn_probs_expert = normalize_spawn_probs(spawn_probs_expert_raw)
        # ensure spawn_probs_expert is truthy before proceeding
        if spawn_probs_expert is None:
            continue
        spawn_probs_other_mean = time_series.get("param_other_mean")
        belief_similarity = np.array(time_series.get("belief_similarity", []), dtype=float)
        if spawn_probs_other_mean is None or belief_similarity.size == 0:
            continue
        # filter None from spawn_probs_other_mean
        if isinstance(spawn_probs_other_mean, list):
            spawn_probs_other_mean = [p for p in spawn_probs_other_mean if p is not None]
        elif isinstance(spawn_probs_other_mean, np.ndarray) and spawn_probs_other_mean.dtype == object:
            spawn_probs_other_mean = [p for p in spawn_probs_other_mean if p is not None]
        # bail out if spawn_probs_other_mean is empty after filtering
        if isinstance(spawn_probs_other_mean, list):
            if len(spawn_probs_other_mean) == 0:
                continue
            # normalize spawn_probs_other_mean with np.atleast_2d
            spawn_probs_other_mean = np.atleast_2d(np.array(spawn_probs_other_mean, dtype=float))
        else:
            # already a numpy array (not object dtype)
            if spawn_probs_other_mean.size == 0:
                continue
            # normalize spawn_probs_other_mean with np.atleast_2d
            spawn_probs_other_mean = np.atleast_2d(spawn_probs_other_mean.astype(float))
        
        # require matching shapes (both should be Dirichlet format with 5 spawn probs)
        if spawn_probs_expert.shape != spawn_probs_other_mean.shape[1:]:
            logger.warning(
                "spawn_probs shape mismatch (expert vs param_other_mean): expert %s vs param_other_mean %s",
                spawn_probs_expert.shape,
                spawn_probs_other_mean.shape[1:],
            )
            continue
        
        n = min(spawn_probs_other_mean.shape[0], belief_similarity.size)
        for i in range(n):
            tom_accuracy_vals.append(compute_error(spawn_probs_other_mean[i], spawn_probs_expert))
            belief_similarity_vals.append(float(belief_similarity[i]))

    correlation = float("nan")
    if len(tom_accuracy_vals) >= 2:
        if np.std(tom_accuracy_vals) > 0.0 and np.std(belief_similarity_vals) > 0.0:
            correlation = float(
                np.corrcoef(tom_accuracy_vals, belief_similarity_vals)[0, 1]
            )

    mean_tom = float(np.nanmean(tom_accuracy_vals)) if tom_accuracy_vals else float("nan")
    mean_belief = float(np.nanmean(belief_similarity_vals)) if belief_similarity_vals else float("nan")

    return {
        "correlation": correlation,
        "tom_accuracy_trajectory": tom_accuracy_vals,
        "belief_similarity_trajectory": belief_similarity_vals,
        "mean_tom_accuracy": mean_tom,
        "mean_belief_similarity": mean_belief,
        "echo_chamber_detected": bool(np.isfinite(correlation) and correlation > 0.5),
    }
