"""
Regime analysis functions for experiment results.
"""

from __future__ import annotations

from typing import Any
from collections import defaultdict

from empathy.clean_up.experiment.metrics import (
    extract_result_fields,
    compute_regime_label,
)
from analysis.comparison import _build_baseline_key


def _get_result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def analyze_regime_labels(results: list[Any]) -> dict[str, Any]:
    """Compute baseline-aware regime labels and summary counts."""
    baseline_results = []
    for result in results:
        config, _, _ = extract_result_fields(result)
        if config.get("social_enabled") is False:
            baseline_results.append(result)

    baseline_index = {_build_baseline_key(r): r for r in baseline_results}
    regime_counts: dict[str, int] = defaultdict(int)
    per_episode: list[dict[str, Any]] = []
    baseline_found = 0
    baseline_missing = 0

    for result in results:
        config, _, _ = extract_result_fields(result)
        baseline = None
        if config.get("social_enabled") is not False:
            baseline = baseline_index.get(_build_baseline_key(result))
            if baseline is None:
                baseline_missing += 1
            else:
                baseline_found += 1

        regime = compute_regime_label(result, baseline_result=baseline)
        regime_counts[regime] += 1

        per_episode.append(
            {
                "episode_file": _get_result_value(result, "episode_file"),
                "episode_path": _get_result_value(result, "episode_path"),
                "seed": config.get("seed"),
                "episode_idx": config.get("episode_idx"),
                "learning_condition": config.get("learning_condition"),
                "social_enabled": config.get("social_enabled"),
                "regime_label": regime,
                "baseline_found": baseline is not None,
            }
        )

    return {
        "regimes": dict(regime_counts),
        "baseline_found": baseline_found,
        "baseline_missing": baseline_missing,
        "per_episode": per_episode,
    }
