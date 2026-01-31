"""
Standardized plotting utilities for experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple
import numpy as np

from collections import Counter

logger = logging.getLogger(__name__)
from empathy.clean_up.agent.agent import DEFAULT_T_A
from empathy.clean_up.experiment.metrics import compute_final_error, compute_regime_label


def _extract_time_series(result: Any) -> Dict[str, Any]:
    if hasattr(result, "time_series"):
        return result.time_series
    return result.get("time_series", {})


def _extract_config(result: Any) -> Dict[str, Any]:
    if hasattr(result, "config"):
        return result.config
    return result.get("config", {})


def plot_timeseries_panel(
    results: List[Any],
    seeds_to_plot: int = 3,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not results:
        return None

    selected = results[:max(1, seeds_to_plot)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for result in selected:
        series = _extract_time_series(result)
        axes[0].plot(series.get("trust", []), alpha=0.8)
        axes[1].plot(series.get("tau_accuracy", []), alpha=0.8)
        axes[2].plot(series.get("reliability", []), alpha=0.8)
        spawn_probs_self_series = series.get("param_self", [])
        if len(spawn_probs_self_series) > 0:
            spawn_probs_self_series = np.array(spawn_probs_self_series)
            ndim, shape = spawn_probs_self_series.ndim, spawn_probs_self_series.shape
            if ndim == 2 and shape[1] == 5:
                axes[3].plot(np.mean(spawn_probs_self_series, axis=1), alpha=0.8, label="mean")
            elif ndim == 1:
                axes[3].plot(spawn_probs_self_series, alpha=0.8)
            elif ndim == 2:
                logger.warning("spawn_probs_self_series unexpected shape (2d, not 5 cols): %s", shape)
                axes[3].plot(np.mean(spawn_probs_self_series, axis=1), alpha=0.8)
            elif ndim > 2:
                logger.warning("spawn_probs_self_series unexpected shape (ndim>2): %s", shape)
                axes[3].plot(spawn_probs_self_series.ravel(), alpha=0.8)
            elif ndim == 0:
                # scalar case: wrap as 1d array with single value
                axes[3].plot(np.atleast_1d(spawn_probs_self_series), alpha=0.8)

    axes[0].set_title("trust")
    axes[1].set_title("tau_accuracy")
    axes[2].set_title("reliability")
    axes[3].set_title("spawn_prob (mean)")
    fig.tight_layout()

    if output_dir is None:
        return None
    output_path = Path(output_dir) / "timeseries_panel.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_aggregate_panel(
    results: List[Any],
    factor_x: str,
    factor_y: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not results:
        return None

    data = []
    for result in results:
        config = _extract_config(result)
        data.append(
            (
                config.get(factor_x),
                config.get(factor_y) if factor_y else None,
                compute_final_error(result),
            )
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    if factor_y is None:
        xs = [d[0] for d in data]
        ys = [d[2] for d in data]
        ax.scatter(xs, ys, alpha=0.7)
        ax.set_xlabel(factor_x)
        ax.set_ylabel("final_error")
    else:
        xs = sorted(set(d[0] for d in data))
        ys = sorted(set(d[1] for d in data))
        heat = np.full((len(ys), len(xs)), np.nan)
        for x_val in xs:
            for y_val in ys:
                vals = [d[2] for d in data if d[0] == x_val and d[1] == y_val]
                if vals:
                    heat[ys.index(y_val), xs.index(x_val)] = float(np.nanmean(vals))
        im = ax.imshow(heat, origin="lower", aspect="auto")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys)
        ax.set_xlabel(factor_x)
        ax.set_ylabel(factor_y)
        fig.colorbar(im, ax=ax, label="final_error")

    fig.tight_layout()
    if output_dir is None:
        return None
    output_path = Path(output_dir) / "aggregate_panel.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_regime_heatmap(
    results: List[Any],
    x_axis: str,
    y_axis: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        return None

    if not results:
        return None

    data = []
    for result in results:
        config = _extract_config(result)
        outcomes = result.outcomes if hasattr(result, "outcomes") else result.get("outcomes", {})
        label = outcomes.get("regime_label") or compute_regime_label(result)
        data.append((config.get(x_axis), config.get(y_axis), label))

    xs = sorted(set(d[0] for d in data))
    ys = sorted(set(d[1] for d in data))
    if not xs or not ys:
        return None

    categories = ["Independent", "Beneficial", "Overcopying", "Harmful influence", "Unknown"]
    label_to_idx = {label: idx for idx, label in enumerate(categories)}
    heat = np.full((len(ys), len(xs)), label_to_idx["Unknown"], dtype=int)

    for x_val in xs:
        for y_val in ys:
            labels = [d[2] for d in data if d[0] == x_val and d[1] == y_val]
            if labels:
                most_common = Counter(labels).most_common(1)[0][0]
                heat[ys.index(y_val), xs.index(x_val)] = label_to_idx.get(
                    most_common, label_to_idx["Unknown"]
                )

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = ListedColormap(["#c7c7c7", "#4daf4a", "#ff7f00", "#e41a1c", "#9e9e9e"])
    norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, cmap.N)
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title("regime_heatmap")
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(categories)))
    cbar.ax.set_yticklabels(categories)
    fig.tight_layout()
    if output_dir is None:
        return None
    output_path = Path(output_dir) / "regime_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_error_heatmap(
    results: List[Any],
    x_axis: str,
    y_axis: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    return plot_aggregate_panel(results, factor_x=x_axis, factor_y=y_axis, output_dir=output_dir)


def plot_error_vs_trust_gate_comparison(
    results: List[Any],
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return None

    if not results:
        return None

    # Group results by accuracy gate temperature
    data = {}
    for result in results:
        config = _extract_config(result)
        T_a = float(
            config.get("T_a", DEFAULT_T_A)
        )
        data.setdefault(T_a, []).append(compute_final_error(result))

    threshold_values = sorted(data.keys())
    box_data = [data[threshold] for threshold in threshold_values]

    fig, ax = plt.subplots(figsize=(8, 4))
    if box_data:
        boxes = ax.boxplot(
            box_data,
            positions=range(len(threshold_values)),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor("#1b9e77")

    ax.set_xticks(range(len(threshold_values)))
    ax.set_xticklabels([f"{val:.2f}" for val in threshold_values])
    ax.set_xlabel("T_a (accuracy gate temperature)")
    ax.set_ylabel("Final Error")
    ax.set_title("Error vs Accuracy Gate Temperature")
    fig.tight_layout()

    if output_dir is None:
        return None
    output_path = Path(output_dir) / "error_vs_trust.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_tom_reliability_analysis(
    results: List[Any],
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not results:
        return None

    grouped: Dict[Tuple[str, float], Dict[str, List[float]]] = {}
    for result in results:
        config = _extract_config(result)
        tom_mode = str(config.get("tom_mode"))
        beta_tom = float(config.get("beta_tom", 0.0))
        scenario = config.get("scenario")
        key = (tom_mode, beta_tom)
        grouped.setdefault(key, {"reliability_variance": [], "false_negative": []})

        series = _extract_time_series(result)
        reliability = np.array(series.get("reliability", []), dtype=float)
        if reliability.size:
            grouped[key]["reliability_variance"].append(float(np.nanvar(reliability)))

        if scenario != "matched":
            tau_accuracy = np.array(series.get("tau_accuracy", []), dtype=float)
            if tau_accuracy.size:
                T_a = float(
                    config.get("T_a", DEFAULT_T_A)
                )
                # False negative: tau_accuracy stayed high when it should have been low
                # Using 0.5 as threshold (neutral accuracy)
                false_negative = float(np.nanmin(tau_accuracy)) >= 0.5
                grouped[key]["false_negative"].append(float(false_negative))

    labels = []
    false_negative_rates = []
    reliability_variances = []
    for (tom_mode, beta_tom), stats in sorted(grouped.items()):
        labels.append(f"{tom_mode}:{beta_tom:g}")
        false_negative_rates.append(
            float(np.mean(stats["false_negative"])) if stats["false_negative"] else float("nan")
        )
        reliability_variances.append(
            float(np.mean(stats["reliability_variance"])) if stats["reliability_variance"] else float("nan")
        )

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(x, false_negative_rates, color="#d95f02", alpha=0.7, label="false_negative_rate")
    ax1.set_ylabel("false_negative_rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, reliability_variances, color="#1b9e77", marker="o", label="reliability_variance")
    ax2.set_ylabel("Reliability Variance")

    ax1.set_title("ToM Reliability Analysis")
    fig.tight_layout()

    if output_dir is None:
        return None
    output_path = Path(output_dir) / "tom_reliability_analysis.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
