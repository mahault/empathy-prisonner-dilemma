"""
Basic metrics for experiment results.

These are fundamental metrics used by the experiment runner and for basic analysis.
For deeper analysis functions, see the analysis module.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from empathy.clean_up.experiment.constants import spawn_probs_from_dict

__all__ = [
    "extract_result_fields",
    "normalize_spawn_probs",
    "get_spawn_prob_series",
    "compute_error",
    "compute_final_error",
    "compute_drift_metrics",
    "compute_regime_label",
    "compute_detection_time",
    "compute_stability_metrics",
]


def extract_result_fields(result: Any) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Return (config, time_series, outcomes) for EpisodeResult or dict payloads.
    
    For dict payloads, falls back to metadata if config is not present (for backward compatibility).
    """
    if hasattr(result, "config") and hasattr(result, "time_series"):
        return result.config, result.time_series, result.outcomes
    if isinstance(result, dict):
        config = result.get("config")
        if config is None or not config:
            # Fall back to metadata for backward compatibility with old episode JSON files
            config = result.get("metadata", {})
        return config, result.get("time_series", {}), result.get("outcomes", {})
    raise ValueError("Unsupported result format for analysis.")


def normalize_spawn_probs(spawn_prob_value: Any) -> Optional[np.ndarray]:
    """Normalize spawn probability value to numpy array."""
    if spawn_prob_value is None:
        return None
    if isinstance(spawn_prob_value, dict):
        return spawn_probs_from_dict(spawn_prob_value)
    return np.array(spawn_prob_value, dtype=float)


def get_spawn_prob_series(time_series: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extract param_self time series as numpy array.
    
    Format: Each timestep has 5 spawn probabilities (one per pollution context).
    
    Returns:
        np.ndarray of shape (T, 5) where T is timesteps and 5 is the number of contexts.
        Returns None if param_self is not present or invalid.
    """
    if "param_self" not in time_series:
        return None
    param_self = time_series["param_self"]
    if param_self is None:
        return None
    
    # Handle string representation (from JSON serialization)
    if isinstance(param_self, str):
        try:
            # Parse numpy array string representation
            # Format: "[[val1 val2 ...]\n [val3 val4 ...]\n ...]"
            # Remove outer brackets and split by newlines to get rows
            cleaned = param_self.strip().strip('[]')
            rows = []
            for line in cleaned.split('\n'):
                line = line.strip().strip('[]')
                if not line:
                    continue
                # Parse space-separated values in this row
                values = [float(x) for x in line.split() if x.strip()]
                if len(values) > 0:
                    rows.append(values)
            if rows:
                return np.array(rows, dtype=float)
        except (ValueError, SyntaxError, TypeError):
            return None
    
    # Handle numpy array (from logger.to_array with dtype=object)
    if isinstance(param_self, np.ndarray):
        if param_self.dtype == object:
            # Filter out None values and convert list of arrays to 2D array
            valid_params = [p for p in param_self if p is not None]
            if not valid_params:
                return None
            return np.array(valid_params, dtype=float)
        else:
            # Already a numeric array
            if np.any(np.isnan(param_self)):
                return None
            return param_self
    
    # Handle list
    if isinstance(param_self, list):
        # Filter out None values
        valid_params = [p for p in param_self if p is not None]
        if not valid_params:
            return None
        # Convert list of arrays to 2D array
        return np.array(valid_params, dtype=float)
    
    return None


def compute_error(
    spawn_probs_self: Optional[np.ndarray],
    spawn_probs_true: Optional[np.ndarray],
) -> float:
    """
    Compute MAE (mean absolute error) between spawn probability arrays.
    
    Args:
        spawn_probs_self: Agent's learned spawn probs [5] (one per pollution context)
        spawn_probs_true: Ground truth spawn probs [5]
    
    Returns:
        MAE of (spawn_probs_self - spawn_probs_true), or nan if inputs invalid.
    """
    if spawn_probs_self is None or spawn_probs_true is None:
        return float("nan")
    
    spawn_probs_self = np.atleast_1d(spawn_probs_self)
    spawn_probs_true = np.atleast_1d(spawn_probs_true)
    
    # Handle shape mismatch
    if spawn_probs_self.shape != spawn_probs_true.shape:
        return float("nan")
    
    # Spawn probabilities are in [0, 1], no normalization needed
    return float(np.mean(np.abs(spawn_probs_self - spawn_probs_true)))


def compute_final_error(result: Any) -> float:
    """Compute L2 error between final learned spawn probs and ground truth."""
    config, time_series, _ = extract_result_fields(result)
    task = config.get("task", "clean_up")
    spawn_probs_true = normalize_spawn_probs(config.get("spawn_probs_true"))
    spawn_probs_series = get_spawn_prob_series(time_series)
    if spawn_probs_true is None or spawn_probs_series is None:
        return float("nan")
    spawn_probs_final = spawn_probs_series[-1]
    return compute_error(spawn_probs_final, spawn_probs_true)


def compute_drift_metrics(result: Any) -> Dict[str, Any]:
    """Compute distance to expert/true spawn probs and drift direction."""
    config, time_series, _ = extract_result_fields(result)
    spawn_probs_true = normalize_spawn_probs(config.get("spawn_probs_true"))
    spawn_probs_expert = normalize_spawn_probs(config.get("beliefs_expert"))
    spawn_probs_series = get_spawn_prob_series(time_series)
    if spawn_probs_true is None or spawn_probs_expert is None or spawn_probs_series is None:
        return {
            "dist_to_expert_initial": float("nan"),
            "dist_to_expert_final": float("nan"),
            "dist_to_true_initial": float("nan"),
            "dist_to_true_final": float("nan"),
            "drift_toward_expert": False,
        }
    spawn_probs_init = spawn_probs_series[0]
    spawn_probs_final = spawn_probs_series[-1]

    dist_to_expert_initial = compute_error(spawn_probs_init, spawn_probs_expert)
    dist_to_expert_final = compute_error(spawn_probs_final, spawn_probs_expert)
    dist_to_true_initial = compute_error(spawn_probs_init, spawn_probs_true)
    dist_to_true_final = compute_error(spawn_probs_final, spawn_probs_true)

    return {
        "dist_to_expert_initial": dist_to_expert_initial,
        "dist_to_expert_final": dist_to_expert_final,
        "dist_to_true_initial": dist_to_true_initial,
        "dist_to_true_final": dist_to_true_final,
        "drift_toward_expert": dist_to_expert_final < dist_to_expert_initial,
    }


def compute_regime_label(result: Any, baseline_result: Optional[Any] = None) -> str:
    """
    Classify behavior into one of four regimes based on error and drift.

    Regimes (total and mutually exclusive):
    - Independent: error improves AND drift_toward_expert is False
    - Beneficial: error improves AND drift_toward_expert is True
    - Overcopying: error worsens AND drift_toward_expert is True
    - Harmful influence: error worsens AND drift_toward_expert is False
    """
    config, time_series, _ = extract_result_fields(result)
    if config.get("social_enabled") is False:
        return "Independent"
    spawn_probs_true = normalize_spawn_probs(config.get("spawn_probs_true"))
    spawn_probs_series = get_spawn_prob_series(time_series)
    if spawn_probs_true is None or spawn_probs_series is None or spawn_probs_series.size == 0:
        return "Unknown"

    final_error = compute_error(spawn_probs_series[-1], spawn_probs_true)
    if baseline_result is not None:
        baseline_error = compute_final_error(baseline_result)
        error_improves = final_error < baseline_error
    else:
        initial_error = compute_error(spawn_probs_series[0], spawn_probs_true)
        error_improves = final_error < initial_error

    drift_metrics = compute_drift_metrics(result)
    drift_toward_expert = bool(drift_metrics.get("drift_toward_expert", False))

    if error_improves and not drift_toward_expert:
        return "Independent"
    if error_improves and drift_toward_expert:
        return "Beneficial"
    if (not error_improves) and drift_toward_expert:
        return "Overcopying"
    return "Harmful influence"


def compute_detection_time(
    result: Any,
    threshold: float = 0.5,
    consecutive: int = 5,
    warmup: int = 10,
) -> Optional[int]:
    """Return 0-indexed start timestep for consecutive tau_accuracy < threshold after warmup."""
    _, time_series, _ = extract_result_fields(result)
    tau_accuracy = np.array(time_series.get("tau_accuracy", []), dtype=float)
    if tau_accuracy.size == 0:
        return None
    start_idx = max(warmup, 0)
    count = 0
    for idx in range(start_idx, tau_accuracy.size):
        if tau_accuracy[idx] < threshold:
            count += 1
            if count >= consecutive:
                return idx - consecutive + 1
        else:
            count = 0
    return None


def compute_stability_metrics(
    result: Any,
    n_eff_collapse_frac: float = 0.1,
) -> Dict[str, Any]:
    """Compute reliability variance and n_eff collapse."""
    config, time_series, _ = extract_result_fields(result)
    reliability = np.array(time_series.get("reliability", []), dtype=float)
    n_eff = np.array(time_series.get("n_eff", []), dtype=float)
    n_particles = int(config.get("n_particles", 0))
    reliability_variance = float(np.nanvar(reliability)) if reliability.size else float("nan")
    n_eff_min = float(np.nanmin(n_eff)) if n_eff.size else float("nan")
    collapse_thresh = n_eff_collapse_frac * n_particles if n_particles else float("nan")
    n_eff_collapse = bool(n_eff_min < collapse_thresh) if n_particles else False
    return {
        "reliability_variance": reliability_variance,
        "n_eff_min": n_eff_min,
        "n_eff_collapse": n_eff_collapse,
    }
