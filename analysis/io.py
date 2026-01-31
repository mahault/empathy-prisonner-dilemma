"""
Data loading functions for experiment results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List


def load_experiment_results(run_dir: Path) -> List[Dict[str, Any]]:
    """Load timeseries.json from a run directory and convert to expected format."""
    timeseries_file = run_dir / "timeseries.json"
    if not timeseries_file.exists():
        return []
    
    try:
        with open(timeseries_file, "r") as f:
            data = json.load(f)
        
        # Convert new structure (metadata + timesteps) to expected format (config + time_series + outcomes)
        metadata = data.get("metadata", {})
        timesteps = data.get("timesteps", [])
        
        if not timesteps:
            return []
        
        # Extract config from metadata
        config = {
            "experiment_name": metadata.get("experiment_name"),
            "task": metadata.get("task"),
            "scenario": metadata.get("scenario"),
            "seed": metadata.get("seed"),
            "run_name": metadata.get("run_name"),
            "run_idx": metadata.get("run_idx"),
            "tom_mode": metadata.get("tom_mode"),
            "beta_tom": metadata.get("beta_tom"),
            "eta_0": metadata.get("eta_0"),
            "lambda_mix": metadata.get("lambda_mix"),
            "self_learning": metadata.get("self_learning"),
            "social_enabled": metadata.get("social_enabled"),
            "spawn_probs_true": metadata.get("spawn_probs_true"),
            "beliefs_expert": metadata.get("beliefs_expert"),
            "beliefs_self_init": metadata.get("beliefs_self_init"),
        }
        
        # Convert timesteps list to time_series dict by transposing
        # Each key in timestep dicts becomes a key in time_series with array of values
        time_series: Dict[str, Any] = {}
        if timesteps:
            # Get all keys from first timestep (excluding 'step' which is just an index)
            keys = [k for k in timesteps[0].keys() if k != "step"]
            for key in keys:
                values = [ts.get(key) for ts in timesteps]
                time_series[key] = values
        
        # Extract outcomes from metadata
        outcomes = metadata.get("outcomes", {})
        
        # Build result in expected format
        result = {
            "config": config,
            "time_series": time_series,
            "outcomes": outcomes,
            "metadata": metadata,  # Keep original metadata for reference
            "episode_path": str(timeseries_file),  # For compatibility
        }
        
        # Return as list for compatibility with existing code
        return [result]
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {timeseries_file}: {e}")
        return []


def scan_results_directory(results_dir: Path) -> List[Path]:
    """Find all result run directories (checks data/ subfolder first, then falls back to old structure)."""
    if not results_dir.exists():
        return []
    
    run_dirs = []
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check in data/ subfolder first (new structure)
        data_dir = exp_dir / "data"
        if data_dir.exists() and data_dir.is_dir():
            for run_dir in data_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "timeseries.json").exists():
                    run_dirs.append(run_dir)
        else:
            # Fallback to old structure (runs directly under experiment)
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "timeseries.json").exists():
                    run_dirs.append(run_dir)
    
    return run_dirs


def load_all_experiments(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all experiment results from results directory (checks data/ subfolder first)."""
    all_results = {}
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        exp_results = []
        
        # Check in data/ subfolder first (new structure)
        data_dir = exp_dir / "data"
        if data_dir.exists() and data_dir.is_dir():
            search_dir = data_dir
        else:
            # Fallback to old structure (runs directly under experiment)
            search_dir = exp_dir
        
        for run_dir in search_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            timeseries_file = run_dir / "timeseries.json"
            if not timeseries_file.exists():
                continue
            
            try:
                # Use load_experiment_results to get converted format
                results = load_experiment_results(run_dir)
                if results:
                    result = results[0]
                    result["run_dir"] = str(run_dir)
                    exp_results.append(result)
            except Exception as e:
                print(f"Warning: Failed to load {timeseries_file}: {e}")
        
        if exp_results:
            all_results[exp_name] = exp_results
    
    return all_results
