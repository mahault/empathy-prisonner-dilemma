"""Shared mathematical utilities for active inference."""
import numpy as np
from typing import Optional


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature parameter.

    Args:
        x: Input array of log-probabilities or scores
        temperature: Temperature parameter (higher = more uniform)

    Returns:
        Probability distribution that sums to 1
    """
    x_scaled = x / temperature
    exp_x = np.exp(x_scaled - np.max(x_scaled))
    return exp_x / np.sum(exp_x)


def entropy(p: np.ndarray, eps: float = 1e-16) -> float:
    """Shannon entropy of probability distribution.

    Args:
        p: Probability distribution
        eps: Small constant for numerical stability

    Returns:
        Entropy value (non-negative)
    """
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p_safe))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-16) -> float:
    """KL divergence D(p || q).

    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Small constant for numerical stability

    Returns:
        KL divergence (non-negative)
    """
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p_safe) - np.log(q_safe)))


def normalize(x: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    """Normalize array to sum to 1.

    Args:
        x: Input array
        eps: Small constant for numerical stability

    Returns:
        Normalized probability distribution
    """
    return x / (np.sum(x) + eps)
