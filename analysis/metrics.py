"""
Metric computation functions for experiment analysis.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List
from collections import defaultdict

from empathy.clean_up.experiment.metrics import (
    compute_final_error,
    compute_drift_metrics,
    compute_stability_metrics,
)


def _get_metrics_block(result: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result.get("metrics", {})
    return {}


def _get_final_error(result: Dict[str, Any]) -> float:
    metrics = _get_metrics_block(result)
    if "final_error" in metrics:
        return metrics.get("final_error")
    return compute_final_error(result)


def _get_drift_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _get_metrics_block(result)
    drift_keys = [
        "dist_to_expert_initial",
        "dist_to_expert_final",
        "dist_to_true_initial",
        "dist_to_true_final",
        "drift_toward_expert",
    ]
    if any(key in metrics for key in drift_keys):
        return {key: metrics.get(key, float("nan")) for key in drift_keys}
    return compute_drift_metrics(result)


def _get_stability_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _get_metrics_block(result)
    stability_keys = [
        "reliability_variance",
        "n_eff_min",
    ]
    if any(key in metrics for key in stability_keys):
        return {key: metrics.get(key, float("nan")) for key in stability_keys}
    return compute_stability_metrics(result)


def compute_statistics(errors: List[float]) -> Dict[str, float]:
    """Compute error statistics."""
    if not errors:
        return {}
    errors_array = np.array(errors)
    return {
        "mean": float(np.mean(errors_array)),
        "std": float(np.std(errors_array)),
        "min": float(np.min(errors_array)),
        "max": float(np.max(errors_array)),
        "median": float(np.median(errors_array)),
        "q25": float(np.percentile(errors_array, 25)),
        "q75": float(np.percentile(errors_array, 75)),
        "n": len(errors),
    }


def analyze_basic_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze basic metrics: errors, regimes, drift, stability."""
    analysis = {
        "errors": [],
        "regimes": defaultdict(int),
        "drift_metrics": [],
        "stability_metrics": [],
    }
    
    for result in results:
        try:
            error = _get_final_error(result)
            if not np.isnan(error):
                analysis["errors"].append(error)
            
            regime = result.get("outcomes", {}).get("regime_label") or "Unknown"
            analysis["regimes"][regime] += 1
            
            drift = _get_drift_metrics(result)
            analysis["drift_metrics"].append(drift)
            
            stability = _get_stability_metrics(result)
            analysis["stability_metrics"].append(stability)
            
        except Exception as e:
            print(f"Warning: Failed to analyze episode: {e}")
            continue
    
    # Compute summary statistics
    errors = [e for e in analysis["errors"] if not np.isnan(e)]
    if errors:
        analysis["error_summary"] = compute_statistics(errors)
    else:
        analysis["error_summary"] = None
    
    return analysis


def analyze_parameter_breakdowns(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze breakdowns by task, ToM, eta_0, lambda_mix, u_0, kappa, severity."""
    breakdowns = {
        "task_breakdown": defaultdict(list),
        "use_tom_breakdown": defaultdict(list),
        "eta_0_breakdown": defaultdict(list),
        "lambda_mix_breakdown": defaultdict(list),
        "u_0_breakdown": defaultdict(list),
        "kappa_breakdown": defaultdict(list),
        "severity_breakdown": defaultdict(list),
    }
    
    for result in results:
        metadata = result.get("metadata", {})
        config = metadata.get("config", result.get("config", {}))
        
        # Compute error
        try:
            error = _get_final_error(result)
            if np.isnan(error):
                continue
        except Exception:
            continue
        
        # Extract parameters
        task = config.get("task") or metadata.get("task")
        if task:
            breakdowns["task_breakdown"][task].append(error)
        
        # Infer use_tom from tom_mode
        tom_mode = config.get("tom_mode")
        if tom_mode is not None:
            use_tom = (tom_mode != "off")
            breakdowns["use_tom_breakdown"][use_tom].append(error)
        
        eta_0 = config.get("eta_0") or metadata.get("eta_0")
        if eta_0 is not None:
            breakdowns["eta_0_breakdown"][eta_0].append(error)
        
        lambda_mix = config.get("lambda_mix") or metadata.get("lambda_mix")
        if lambda_mix is not None:
            breakdowns["lambda_mix_breakdown"][lambda_mix].append(error)
        
        u_0 = config.get("u_0") or metadata.get("u_0")
        if u_0 is not None:
            breakdowns["u_0_breakdown"][u_0].append(error)
        
        kappa = config.get("kappa") or metadata.get("kappa")
        if kappa is not None:
            breakdowns["kappa_breakdown"][kappa].append(error)
        
        severity = config.get("mismatch_severity") or metadata.get("mismatch_severity")
        if severity:
            breakdowns["severity_breakdown"][severity].append(error)
    
    # Compute summary statistics for each breakdown
    for breakdown_name, breakdown_data in breakdowns.items():
        for key, errors in breakdown_data.items():
            errors_clean = [e for e in errors if not np.isnan(e)]
            if errors_clean:
                breakdowns[breakdown_name][key] = compute_statistics(errors_clean)
            else:
                breakdowns[breakdown_name][key] = None
    
    return breakdowns


def analyze_social_learning(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze social learning stats: reliability, trust, effective influence."""
    stats = {
        "reliability_mean": [],
        "reliability_std": [],
        "trust_mean": [],
        "trust_std": [],
        "effective_influence_mean": [],
    }
    
    for result in results:
        time_series = result.get("time_series", {})
        
        if "reliability" in time_series:
            reliability = time_series["reliability"]
            if isinstance(reliability, str):
                reliability = eval(reliability)
            reliability = np.array(reliability)
            if reliability.size > 0:
                stats["reliability_mean"].append(float(np.nanmean(reliability)))
                stats["reliability_std"].append(float(np.nanstd(reliability)))
        
        if "trust" in time_series:
            trust = time_series["trust"]
            if isinstance(trust, str):
                trust = eval(trust)
            trust = np.array(trust)
            if trust.size > 0:
                stats["trust_mean"].append(float(np.nanmean(trust)))
                stats["trust_std"].append(float(np.nanstd(trust)))
        
        if "effective_influence" in time_series:
            eff_inf = time_series["effective_influence"]
            if isinstance(eff_inf, str):
                eff_inf = eval(eff_inf)
            eff_inf = np.array(eff_inf)
            if eff_inf.size > 0:
                stats["effective_influence_mean"].append(float(np.nanmean(eff_inf)))
    
    # Compute summary statistics
    summary = {}
    for key, values in stats.items():
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
        else:
            summary[key] = None
    
    return summary
