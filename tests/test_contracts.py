"""
Contract tests for discretized tasks and experiment hygiene.
"""

from __future__ import annotations

import numpy as np

from empathy.clean_up.agent.inference.utils import discretize_to_bin, gaussian_bin_prob


def test_discretization_contract():
    """
    Contract test that validates discretization behavior and consistency with Gaussian bin probabilities.
    
    Verifies:
    - Boundary mapping and monotonicity: specific test values map to the first and last bins and discretization is non-decreasing.
    - Distributional agreement: empirical bin frequencies from 50,000 samples drawn from N(0.5, 0.1^2) match the per-bin probabilities returned by gaussian_bin_prob for 16 bins within total variation distance < 0.02.
    - Repeats the distributional agreement check for 8 bins with the same tolerance.
    """
    values = [-0.5, 0.0, 0.1, 0.5, 0.9, 1.0, 1.5]
    bins = [int(discretize_to_bin(v, 16)) for v in values]
    assert bins[0] == 0
    assert bins[-1] == 15
    assert all(b1 <= b2 for b1, b2 in zip(bins, bins[1:]))

    rng = np.random.default_rng(0)
    mean = 0.5
    std = 0.1
    n_samples = 50000
    samples = rng.normal(mean, std, size=n_samples)
    empirical = np.zeros(16, dtype=float)
    for sample in samples:
        empirical[int(discretize_to_bin(sample, 16))] += 1.0
    empirical /= float(n_samples)

    model = np.array([gaussian_bin_prob(i, 16, mean, std) for i in range(16)])
    tv = 0.5 * np.sum(np.abs(empirical - model))
    assert tv < 0.02

    samples8 = rng.normal(mean, std, size=n_samples)
    empirical8 = np.zeros(8, dtype=float)
    for sample in samples8:
        empirical8[int(discretize_to_bin(sample, 8))] += 1.0
    empirical8 /= float(n_samples)
    model8 = np.array([gaussian_bin_prob(i, 8, mean, std) for i in range(8)])
    tv8 = 0.5 * np.sum(np.abs(empirical8 - model8))
    assert tv8 < 0.02



