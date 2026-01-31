"""
Inference Utility Functions.

Shared utilities for state inference, planning, and learning modules.
"""

import math
import numpy as np
from typing import Tuple

# Optional numba acceleration can be enabled if available.
from numba import njit, float64, int64


@njit(float64[:](float64[:]))
def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to sum to 1.

    Handles edge case where sum is zero by returning uniform distribution.

    Args:
        arr: Input array

    Returns:
        normalized: Array summing to 1
    """
    total = np.sum(arr)
    if total <= 0.0:
        return np.ones_like(arr) / arr.size
    return arr / total


@njit(float64[:](float64[:]))
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    Args:
        x: Input logits

    Returns:
        prob: Softmax probabilities
    """
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def create_transition_matrix(grid_size: int, action: int) -> np.ndarray:
    """
    Create a deterministic state-to-state transition matrix for a grid action.

    The matrix maps each grid cell (flattened row-major index) to the single next cell resulting from applying the action; movement is clamped at grid boundaries. Action codes: 0=up, 1=down, 2=left, 3=right, 4=stay.

    Parameters:
        grid_size (int): Side length of the square grid (N). Number of states is N*N.
        action (int): Action code indicating movement direction (see above).

    Returns:
        np.ndarray: Transition matrix of shape (N_states, N_states) where each row is a one-hot distribution (1.0 at the deterministic next-state column).
    """
    if action < 0 or action > 4:
        raise ValueError("action must be in {0,1,2,3,4}.")

    n_states = grid_size * grid_size
    transition = np.zeros((n_states, n_states), dtype=float)
    for idx in range(n_states):
        row = idx // grid_size
        col = idx % grid_size
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(grid_size - 1, col + 1)
        else:
            pass

        next_idx = row * grid_size + col
        transition[idx, next_idx] = 1.0

    return transition


@njit(float64(float64, float64, float64))
def gaussian_cdf(x: float, mean: float, std: float) -> float:
    """
    Gaussian CDF using math.erf (Numba-compatible).

    CDF(x) = 0.5 * (1 + erf((x - mean) / (std * sqrt(2))))
    """
    if std <= 0.0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


@njit(float64(int64, int64, float64, float64))
def gaussian_bin_prob(bin_idx: int, n_bins: int, mean: float, std: float) -> float:
    """
    Integrate Gaussian over bin interval with infinite tails for edge bins.

    Canonical helper name: gaussian_bin_prob.
    - bin 0: (-inf, 1/n_bins)
    - bin n_bins-1: ((n_bins-1)/n_bins, +inf)
    - middle bins: [i/n_bins, (i+1)/n_bins)
    """
    if n_bins <= 0:
        return 0.0
    if bin_idx <= 0:
        bin_high = 1.0 / n_bins
        return gaussian_cdf(bin_high, mean, std)
    if bin_idx >= n_bins - 1:
        bin_low = (n_bins - 1.0) / n_bins
        return 1.0 - gaussian_cdf(bin_low, mean, std)
    bin_low = bin_idx / n_bins
    bin_high = (bin_idx + 1.0) / n_bins
    return gaussian_cdf(bin_high, mean, std) - gaussian_cdf(bin_low, mean, std)


@njit(int64(float64, int64))
def discretize_to_bin(value: float, n_bins: int) -> int:
    """
    Discretize a continuous value into a bin index with edge clamping.

    Rules:
    - value < 0 -> bin 0
    - value > 1 -> bin n_bins-1
    - value in [0,1] -> int(value * n_bins), clamped
    """
    if n_bins <= 1:
        return 0
    if value < 0.0:
        return 0
    if value > 1.0:
        return n_bins - 1
    idx = int(value * n_bins)
    if idx < 0:
        return 0
    if idx >= n_bins:
        return n_bins - 1
    return idx


@njit(int64(int64, int64, int64))
def manhattan_distance_cell(cell1: int, cell2: int, grid_size: int) -> int:
    """
    Manhattan distance between two cell indices on a grid.
    """
    r1 = cell1 // grid_size
    c1 = cell1 % grid_size
    r2 = cell2 // grid_size
    c2 = cell2 % grid_size
    dr = r1 - r2
    if dr < 0:
        dr = -dr
    dc = c1 - c2
    if dc < 0:
        dc = -dc
    return dr + dc


@njit(int64(int64))
def max_manhattan_dist(grid_size: int) -> int:
    """
    Maximum Manhattan distance on an N x N grid: 2 * (N - 1).
    """
    assert grid_size >= 2, "grid_size must be >= 2"
    return 2 * (grid_size - 1)


# Alias for clarity
integrate_gaussian_over_bin = gaussian_bin_prob