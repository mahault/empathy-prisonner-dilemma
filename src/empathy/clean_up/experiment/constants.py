"""
Experiment-wide constants and deterministic helpers.

Params use Dirichlet-Categorical format:
- param_self: List of 5 spawn probabilities, one per pollution context (0-4)
- param_other_mean: Expected spawn probabilities from ToM particle filter
"""

import numpy as np


# Number of pollution contexts for Dirichlet beliefs
N_POLLUTION_CONTEXTS = 5

# Spawn probability ranges (one per pollution context)
# These are probabilities so range is [0, 1] for each context
SPAWN_PROB_RANGES = {
    "clean_up": {
        "min": np.zeros(N_POLLUTION_CONTEXTS),  # [0, 0, 0, 0, 0]
        "max": np.ones(N_POLLUTION_CONTEXTS),   # [1, 1, 1, 1, 1]
    },
}


def spawn_probs_from_dict(params: dict[str, float]) -> np.ndarray:
    """
    Convert config dict to spawn probability array.
    
    Format: {"spawn_probs": [...]} -> array of 5 spawn probabilities
    """
    if "spawn_probs" not in params:
        raise KeyError(f"Missing 'spawn_probs' in params dict. Got keys: {list(params.keys())}")
    return np.array(params["spawn_probs"], dtype=float)


def get_spawn_prob_ranges(task: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the minimum and maximum spawn probability arrays for the specified task.
    
    Parameters:
        task (str): Name of the task whose parameter ranges to retrieve.
    
    Returns:
        tuple: A pair (min_array, max_array) of numpy arrays, both shape (5,).
    
    Raises:
        ValueError: If the task is not present in the ranges dict.
    """
    if task not in SPAWN_PROB_RANGES:
        raise ValueError(f"Unknown task for spawn prob ranges: {task}")
    ranges = SPAWN_PROB_RANGES[task]
    return ranges["min"].copy(), ranges["max"].copy()