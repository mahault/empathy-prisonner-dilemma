"""
Baseline comparison utilities for experiment results.

Naming convention: This module uses 'spawn_probs' terminology for internal variables.
Spawn probs are 5-dimensional vectors (one probability per pollution context 0-4).

Config keys used: 'spawn_probs_true' for ground truth, 'beliefs_expert' for expert beliefs.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from empathy.clean_up.experiment.metrics import (
    extract_result_fields,
    normalize_spawn_probs,
)


def _canonical_spawn_probs_hash(spawn_probs: Any) -> str:
    """
    Compute canonical hash for spawn_probs array.
    
    Canonicalizes dtype and memory layout before hashing to ensure
    consistent hashes regardless of input array properties.
    """
    canonical = np.ascontiguousarray(spawn_probs, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def build_join_key(result: Any) -> dict[str, Any]:
    """Build canonical join key for baseline alignment."""
    config, _, _ = extract_result_fields(result)
    task = config.get("task", "clean_up")
    spawn_probs_true = normalize_spawn_probs(config.get("spawn_probs_true"))
    spawn_probs_expert = normalize_spawn_probs(config.get("beliefs_expert"))
    spawn_probs_true_hash = None
    spawn_probs_expert_hash = None
    if spawn_probs_true is not None:
        spawn_probs_true_hash = _canonical_spawn_probs_hash(spawn_probs_true)
    if spawn_probs_expert is not None:
        spawn_probs_expert_hash = _canonical_spawn_probs_hash(spawn_probs_expert)
    return {
        "experiment_name": config.get("experiment_name", config.get("name")),
        "task": task,
        "scenario": config.get("scenario"),
        "seed": config.get("seed"),
        "spawn_probs_true_hash": spawn_probs_true_hash,
        "beliefs_expert_hash": spawn_probs_expert_hash,
    }


def index_by_join_key(results: list[Any]) -> dict[tuple[Any, ...], Any]:
    """
    Index results by join key for fast lookup.
    
    Raises ValueError if duplicate join keys are found, including
    the duplicate key and conflicting results.
    """
    indexed: dict[tuple[Any, ...], Any] = {}
    for result in results:
        key_dict = build_join_key(result)
        key = (
            key_dict["experiment_name"],
            key_dict["task"],
            key_dict["scenario"],
            key_dict["seed"],
            key_dict["spawn_probs_true_hash"],
            key_dict["beliefs_expert_hash"],
        )
        if key in indexed:
            existing_result = indexed[key]
            raise ValueError(
                f"Duplicate join key found: {key}\n"
                f"Existing result: {existing_result}\n"
                f"Conflicting result: {result}"
            )
        indexed[key] = result
    return indexed


def join_baseline_results(results: list[Any], baseline_results: list[Any]) -> list[dict[str, Any]]:
    """Join baseline results using canonical join keys."""
    baseline_index = index_by_join_key(baseline_results)
    joined: list[dict[str, Any]] = []
    for result in results:
        key_dict = build_join_key(result)
        key = (
            key_dict["experiment_name"],
            key_dict["task"],
            key_dict["scenario"],
            key_dict["seed"],
            key_dict["spawn_probs_true_hash"],
            key_dict["beliefs_expert_hash"],
        )
        baseline = baseline_index.get(key)
        joined.append({
            "result": result,
            "baseline": baseline,
        })
    return joined


def _build_baseline_key(result: Any) -> tuple[Any, ...]:
    """
    Build key for matching baseline to social runs.
    
    Matches based on seed and spawn_probs_true to find corresponding baseline and social runs.
    """
    config, _, _ = extract_result_fields(result)
    task = config.get("task", "clean_up")
    spawn_probs_true = normalize_spawn_probs(config.get("spawn_probs_true"))
    spawn_probs_true_hash = None
    if spawn_probs_true is not None:
        spawn_probs_true_hash = _canonical_spawn_probs_hash(spawn_probs_true)
    return (
        config.get("experiment_name", config.get("name")),
        task,
        config.get("scenario"),
        config.get("seed"),
        spawn_probs_true_hash,
    )
