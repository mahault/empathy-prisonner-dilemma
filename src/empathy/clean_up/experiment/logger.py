"""
Experiment Logger: Metric Tracking and Logging.

This module provides logging infrastructure for tracking metrics during
episode execution.

Logged Metrics:
--------------
Time series:
- weight_entropy: H(w) particle weight entropy
- confidence: u_t = 1 - H(w) / log(N_p)
- reliability: r_t = σ((u_t - u_0) / κ) (ToM confidence filter)
- tau_accuracy: Accuracy gate (sigmoid of log-likelihood advantage)
- accuracy_advantage: Log-likelihood difference (positive = expert better)
- trust: τ_t = r_t · τ_accuracy (reliability × accuracy gate)
- eta_t: Effective social learning rate η_t = η_0 · trust
- efe_values: Per-action EFE values used in action selection [n_actions]
- entropy_hidden: H[q(s)]
- param_self: θ_self values
- param_other_mean: E[θ_other]
- n_eff: Effective particle count

Outcome metrics:
- is_success: Binary
- is_timeout: Binary
- time_to_success: Convergence speed
- final_progress: Task progress at end

Diagnostics:
- tom_action_ll: Log-likelihood of observed actions under ToM

Dependencies:
------------
- numpy: Array operations
- pandas: Result aggregation (optional)
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
from dataclasses import dataclass, field
import os
import json
from pathlib import Path

from empathy.clean_up.agent.inference.state import compute_entropy
from empathy.clean_up.agent.social.utils import effective_particle_count
from empathy.clean_up.experiment.constants import spawn_probs_from_dict


@dataclass
class MetricTracker:
    """
    Tracks a single time series metric.
    
    Attributes:
        name: Metric name
        values: List of values over time
        dtype: Data type (for array conversion)
    """
    name: str
    values: List[Any] = field(default_factory=list)
    dtype: type = float
    
    def append(self, value: Any) -> None:
        """Append new value."""
        self.values.append(value)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.values, dtype=self.dtype)
    
    def reset(self) -> None:
        """Clear values."""
        self.values = []


class ExperimentLogger:
    """
    Logs metrics during episode execution.
    
    Usage:
        logger = ExperimentLogger()
        for t in range(max_steps):
            # ... episode loop ...
            logger.log_timestep(t, agent, social_metrics)
        results = logger.get_results()
    
    Provides structured logging for episode metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the logger's configuration, create MetricTracker instances for the standard time-series metrics, and initialize outcome and metadata containers.
        
        Parameters:
            config (Optional[Dict[str, Any]]): Optional configuration dictionary. Recognized keys and defaults:
                - "debug" (bool): enable debug logging; default False
                - "show_progress" (bool): enable progress printing; default True
                - "experiment_name" (str): name used in progress/debug output; default "unknown"
                - "episode_seed" or "seed" (int): seed associated with the episode; default 0
                - "episode_idx" (int): episode index for labeling; default 0
                - "worker_id" (Any): optional worker identifier used in multi-worker output; default None
        
        Notes:
            - Builds self.time_series with MetricTracker objects for trust, theory-of-mind (ToM) diagnostics,
              social-update diagnostics, reliability/influence metrics, state-inference entropy, expected
              free energy (EFE) metrics, parameter summaries, and progress.
            - Initializes self.outcomes and self.metadata for recording episode outcomes and timing/step info.
        """
        self.config = config or {}
        
        # Progress and debug logging configuration
        self.debug_enabled = bool(self.config.get("debug", False))
        self.debug_file = None  # Lazy open on first use
        self.show_progress = bool(self.config.get("show_progress", False))
        self.experiment_name = self.config.get("experiment_name", "unknown")
        self.episode_seed = self.config.get("episode_seed", self.config.get("seed", 0))
        self.episode_idx = self.config.get("episode_idx", 0)
        self.worker_id = self.config.get("worker_id", None)
        self.run_name = self.config.get("run_name", None)
        self.run_idx = self.config.get("run_idx", None)
        self.n_runs = self.config.get("n_runs", None)
        self.save_batch_size = self.config.get("save_batch_size", 10)
        self.run_dir = self.config.get("run_dir", None)
        self.pending_timesteps = []  # Accumulate timesteps for batched saving
        
        # Time series metrics (from Section 4.5)
        self.time_series: Dict[str, MetricTracker] = {
            # Trust metrics (accuracy gate)
            "tau_accuracy": MetricTracker("tau_accuracy"),
            "accuracy_advantage": MetricTracker("accuracy_advantage"),
            "trust": MetricTracker("trust"),
            "eta_t": MetricTracker("eta_t"),
            
            # ToM metrics
            "weight_entropy": MetricTracker("weight_entropy"),
            "confidence": MetricTracker("confidence"),
            "n_eff": MetricTracker("n_eff"),
            "n_eff_pre": MetricTracker("n_eff_pre", dtype=float),
            "n_eff_post": MetricTracker("n_eff_post", dtype=float),
            "is_resampled": MetricTracker("is_resampled", dtype=float),
            "weight_entropy_pre": MetricTracker("weight_entropy_pre", dtype=float),
            "weight_entropy_post": MetricTracker("weight_entropy_post", dtype=float),
            "confidence_pre": MetricTracker("confidence_pre", dtype=float),
            "confidence_post": MetricTracker("confidence_post", dtype=float),
            "reliability_pre": MetricTracker("reliability_pre", dtype=float),
            "reliability_post": MetricTracker("reliability_post", dtype=float),
            "tom_action_entropy": MetricTracker("tom_action_entropy", dtype=float),
            "tom_action_hypotheses": MetricTracker("tom_action_hypotheses", dtype=float),
            "tom_loglike_action_mean": MetricTracker("tom_loglike_action_mean", dtype=float),

            # Social update diagnostics
            "social_step": MetricTracker("social_step", dtype=float),
            "social_update_norm": MetricTracker("social_update_norm", dtype=float),
            "social_update_dot": MetricTracker("social_update_dot", dtype=float),
            "social_update_diff_norm": MetricTracker("social_update_diff_norm", dtype=float),
            "social_update_clipping_norm": MetricTracker("social_update_clipping_norm", dtype=float),

            # Reliability and trust
            "reliability": MetricTracker("reliability"),
            "belief_similarity": MetricTracker("belief_similarity"),
            
            # State inference
            "entropy_hidden": MetricTracker("entropy_hidden"),
            
            # Expected Free Energy (EFE) metrics
            "efe_values": MetricTracker("efe_values", dtype=object),
            
            # Parameters (multi-dimensional, will need special handling)
            "param_self": MetricTracker("param_self", dtype=object),
            "param_other_mean": MetricTracker("param_other_mean", dtype=object),
            "progress": MetricTracker("progress"),
            
            # Self-learning diagnostics
            "self_learning_count": MetricTracker("self_learning_count", dtype=object),
            "self_learning_contexts": MetricTracker("self_learning_contexts", dtype=object),
            "self_learning_outcomes": MetricTracker("self_learning_outcomes", dtype=object),
            "self_learning_pollution": MetricTracker("self_learning_pollution", dtype=object),
            
            # Social learning deltas
            "social_learning_delta": MetricTracker("social_learning_delta", dtype=object),
            "social_learning_contexts_updated": MetricTracker("social_learning_contexts_updated", dtype=object),
        }
        
        # Outcome metrics
        self.outcomes: Dict[str, Any] = {
            "is_success": None,
            "is_timeout": None,
            "time_to_success": None,
            "final_progress": None,
        }
        
        # Metadata
        self.metadata: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "n_steps": 0,
        }
    
    def log_timestep(
        self,
        t: int,
        agent: Any,  # ActiveInferenceAgent
        social_metrics: Dict[str, float],
        env_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log metrics for a single timestep.
        
        Args:
            t: Timestep index
            agent: Learner agent
            social_metrics: Dict from empathy.clean_up.agent.Agent.compute_social_metrics()
            
        """
        if social_metrics:
            self.time_series["weight_entropy"].append(social_metrics.get("weight_entropy"))
            self.time_series["confidence"].append(social_metrics.get("confidence"))
            self.time_series["reliability"].append(social_metrics.get("reliability"))
            self.time_series["tau_accuracy"].append(social_metrics.get("tau_accuracy", np.nan))
            self.time_series["accuracy_advantage"].append(social_metrics.get("accuracy_advantage", np.nan))
            self.time_series["trust"].append(social_metrics.get("trust"))
            self.time_series["eta_t"].append(social_metrics.get("eta_t", np.nan))
            self.time_series["belief_similarity"].append(
                social_metrics.get("belief_similarity", np.nan)
            )
            self.time_series["n_eff_pre"].append(social_metrics.get("n_eff_pre"))
            self.time_series["n_eff_post"].append(social_metrics.get("n_eff_post"))
            self.time_series["is_resampled"].append(social_metrics.get("is_resampled"))
            self.time_series["weight_entropy_pre"].append(
                social_metrics.get("weight_entropy_pre")
            )
            self.time_series["weight_entropy_post"].append(
                social_metrics.get("weight_entropy_post")
            )
            self.time_series["confidence_pre"].append(social_metrics.get("confidence_pre"))
            self.time_series["confidence_post"].append(social_metrics.get("confidence_post"))
            self.time_series["reliability_pre"].append(social_metrics.get("reliability_pre"))
            self.time_series["reliability_post"].append(social_metrics.get("reliability_post"))
            self.time_series["tom_action_entropy"].append(
                social_metrics.get("tom_action_entropy")
            )
            self.time_series["tom_action_hypotheses"].append(
                social_metrics.get("tom_action_hypotheses")
            )
            self.time_series["tom_loglike_action_mean"].append(
                social_metrics.get("tom_loglike_action_mean")
            )
            self.time_series["social_step"].append(social_metrics.get("social_step"))
            self.time_series["social_update_norm"].append(
                social_metrics.get("social_update_norm")
            )
            self.time_series["social_update_dot"].append(
                social_metrics.get("social_update_dot")
            )
            self.time_series["social_update_diff_norm"].append(
                social_metrics.get("social_update_diff_norm")
            )
            self.time_series["social_update_clipping_norm"].append(
                social_metrics.get("social_update_clipping_norm")
            )
        else:
            for name in [
                "weight_entropy",
                "confidence",
                "reliability",
                "effective_influence",
                "n_eff_pre",
                "n_eff_post",
                "is_resampled",
                "weight_entropy_pre",
                "weight_entropy_post",
                "confidence_pre",
                "confidence_post",
                "reliability_pre",
                "reliability_post",
                "tom_action_entropy",
                "tom_action_hypotheses",
                "tom_loglike_action_mean",
                "social_step",
                "social_update_norm",
                "social_update_dot",
                "social_update_diff_norm",
                "social_update_clipping_norm",
            ]:
                self.time_series[name].append(np.nan)
            # Also append NaN for trust components when no social metrics
            for name in ["tau_accuracy", "accuracy_advantage", "trust", "eta_t", "belief_similarity"]:
                self.time_series[name].append(np.nan)

        if env_info is not None and "progress" in env_info:
            self.time_series["progress"].append(env_info.get("progress"))
        else:
            self.time_series["progress"].append(np.nan)

        if agent is not None:
            if agent.state_belief is not None:
                self.time_series["entropy_hidden"].append(compute_entropy(agent.state_belief))
            else:
                self.time_series["entropy_hidden"].append(np.nan)

            # Log beliefs as spawn probabilities per context
            if hasattr(agent, "beliefs_self") and agent.beliefs_self is not None:
                num_contexts = agent.beliefs_self.n_contexts
                spawn_probs = [agent.beliefs_self.get_probability(ctx) for ctx in range(num_contexts)]
                self.time_series["param_self"].append(spawn_probs)
            else:
                self.time_series["param_self"].append(None)
            
            # Log expected other parameters (from particles)
            if agent.particle_params is not None and agent.particle_weights is not None:
                # Particles are now [N_particles, n_contexts, 2 alphas]
                # Compute expected spawn probabilities per context
                if agent.particle_params.ndim == 3:
                    # New Dirichlet particle format: compute expected spawn probability per context
                    expected_probs = []
                    num_contexts = agent.particle_params.shape[1]
                    for ctx in range(num_contexts):
                        # Weight particles by their weights
                        alpha_sum = np.sum(
                            agent.particle_weights[:, None] * agent.particle_params[:, ctx, :],
                            axis=0
                        )
                        # Convert to spawn probability (guard against zero division)
                        denom = alpha_sum[0] + alpha_sum[1]
                        denom = max(denom, 1e-12)
                        p_spawn = alpha_sum[1] / denom
                        expected_probs.append(float(p_spawn))
                    self.time_series["param_other_mean"].append(expected_probs)
                else:
                    raise ValueError(
                        f"Unexpected particle_params shape: {agent.particle_params.shape}. "
                        f"Expected [N_particles, 5 contexts, 2 alphas]."
                    )
                self.time_series["n_eff"].append(
                    effective_particle_count(agent.particle_weights)
                )
            else:
                self.time_series["param_other_mean"].append(None)
                self.time_series["n_eff"].append(np.nan)
            
            # Log EFE values from action selection
            if hasattr(agent, "_last_efe_values") and agent._last_efe_values is not None:
                self.time_series["efe_values"].append(agent._last_efe_values.copy())
            else:
                self.time_series["efe_values"].append(None)
            
            # Log self-learning stats
            self_stats = getattr(agent, "_last_self_learning_stats", {})
            self.time_series["self_learning_count"].append(self_stats.get("self_learning_count", 0))
            self.time_series["self_learning_contexts"].append(self_stats.get("self_learning_contexts", []))
            self.time_series["self_learning_outcomes"].append(self_stats.get("self_learning_outcomes", []))
            self.time_series["self_learning_pollution"].append(self_stats.get("self_learning_pollution"))
            
            # Log social learning deltas
            social_stats = getattr(agent, "_last_social_update_stats", {})
            self.time_series["social_learning_delta"].append(social_stats.get("social_learning_delta", []))
            self.time_series["social_learning_contexts_updated"].append(social_stats.get("social_learning_contexts_updated", []))

        # Trust and reliability validation (if debug enabled)
        debug_trust = bool(self.config.get("debug_trust", False))
        if not debug_trust and agent is not None:
            debug_trust = bool(getattr(agent, "config", {}).get("debug_trust", False))
        if debug_trust and social_metrics:
            trust_values = self.time_series["trust"].values
            reliability_values = self.time_series["reliability"].values
            if len(trust_values) > 1 and len(reliability_values) > 1:
                curr_trust = social_metrics.get("trust")
                curr_reliability = social_metrics.get("reliability")
                if (
                    curr_trust is not None
                    and curr_reliability is not None
                    and np.isfinite(curr_trust)
                    and np.isfinite(curr_reliability)
                ):
                    if not (0.0 <= curr_trust <= 1.0):
                        print(f"WARNING: Trust out of bounds: {curr_trust}")
                    if not (0.0 <= curr_reliability <= 1.0):
                        print(f"WARNING: Reliability out of bounds: {curr_reliability}")
                        print(
                            f"  tau_accuracy: {social_metrics.get('tau_accuracy')}, "
                            f"accuracy_advantage: {social_metrics.get('accuracy_advantage')}"
                        )

        self.metadata["n_steps"] = t + 1
        
        # Convert current timestep to dictionary and add to pending batch
        timestep_dict = self._current_timestep_to_dict(t)
        if timestep_dict is not None:
            self.pending_timesteps.append(timestep_dict)
            
            # Check if we should save a batch
            if (self.run_dir is not None and 
                self.save_batch_size > 0 and 
                len(self.pending_timesteps) >= self.save_batch_size):
                self._save_pending_timesteps()
    
    def _current_timestep_to_dict(self, step: int) -> Optional[Dict[str, Any]]:
        """
        Convert current timestep data to dictionary format for saving.
        
        Args:
            step: Current step index
            
        Returns:
            Dictionary with all timestep metrics, or None if conversion fails
        """
        try:
            timestep = {"step": step}
            
            # Extract last value from each MetricTracker
            for key, tracker in self.time_series.items():
                if hasattr(tracker, 'values') and len(tracker.values) > 0:
                    val = tracker.values[-1]  # Get last value
                    
                    # Convert numpy arrays to lists
                    if isinstance(val, np.ndarray):
                        timestep[key] = val.tolist()
                    # Convert numpy scalars to Python types
                    elif isinstance(val, (np.integer, np.floating)):
                        timestep[key] = float(val) if np.issubdtype(type(val), np.floating) else int(val)
                    # Handle None and other types
                    elif val is None:
                        timestep[key] = None
                    else:
                        timestep[key] = val
                else:
                    # No data yet for this metric
                    timestep[key] = None
            
            return timestep
        except Exception as e:
            # If conversion fails, return None (don't crash)
            return None
    
    def _save_pending_timesteps(self) -> None:
        """
        Save pending timesteps to disk in batches.
        
        Uses atomic write to prevent corruption. Preserves pending_timesteps
        if save fails so data isn't lost.
        """
        if not self.pending_timesteps or self.run_dir is None:
            return
        
        try:
            run_dir = Path(self.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            timeseries_file = run_dir / "timeseries.json"
            
            # Load existing timesteps if file exists
            existing_timesteps = []
            metadata = {}
            if timeseries_file.exists():
                try:
                    existing_data = json.loads(timeseries_file.read_text())
                    existing_timesteps = existing_data.get("timesteps", [])
                    metadata = existing_data.get("metadata", {})
                except (json.JSONDecodeError, IOError):
                    # If file is corrupted, start fresh
                    existing_timesteps = []
            
            # Append new timesteps
            all_timesteps = existing_timesteps + self.pending_timesteps
            
            # Update metadata if not set
            if not metadata:
                task = self.config.get("task")
                spawn_probs_true = self._normalize_spawn_probs(task, self.config.get("spawn_probs_true"))
                spawn_probs_expert = self._normalize_spawn_probs(task, self.config.get("beliefs_expert"))
                spawn_probs_init = self._normalize_spawn_probs(task, self.config.get("beliefs_self_init"))
                
                metadata = {
                    "experiment_name": self.config.get("experiment_name", "unknown"),
                    "task": task,
                    "scenario": self.config.get("scenario"),
                    "seed": self.config.get("seed"),
                    "run_name": self.config.get("run_name"),
                    "run_idx": self.config.get("run_idx"),
                    "tom_mode": self.config.get("tom_mode"),
                    "beta_tom": self.config.get("beta_tom"),
                    "eta_0": self.config.get("eta_0"),
                    "T_a": self.config.get("T_a"),
                    "self_learning": self.config.get("self_learning"),
                    "social_enabled": self.config.get("social_enabled"),
                    # Spawn probability configs
                    "spawn_probs_true": spawn_probs_true.tolist() if spawn_probs_true is not None else None,
                    "beliefs_expert": spawn_probs_expert.tolist() if spawn_probs_expert is not None else None,
                    "beliefs_self_init": spawn_probs_init.tolist() if spawn_probs_init is not None else None,
                }
            
            # Write updated data
            payload = {
                "metadata": metadata,
                "timesteps": all_timesteps,
            }
            
            # Atomic write
            temp_file = timeseries_file.with_suffix('.json.tmp')
            temp_file.write_text(json.dumps(payload, indent=2, default=str))
            temp_file.replace(timeseries_file)
            
            # Clear pending timesteps only after successful save
            self.pending_timesteps = []
            
        except Exception as e:
            # Log warning but don't crash - preserve pending_timesteps
            import warnings
            warnings.warn(f"Failed to save timestep batch: {e}. Data preserved in memory.", UserWarning)
    
    def _normalize_spawn_probs(self, task: Optional[str], spawn_probs_value: Any) -> Optional[np.ndarray]:
        """
        Normalize spawn probs value to numpy array.
        
        Helper method to avoid importing from runner (circular import).
        """
        if spawn_probs_value is None:
            return None
        if isinstance(spawn_probs_value, dict):
            if "spawn_probs" not in spawn_probs_value:
                return None
            return spawn_probs_from_dict(spawn_probs_value)
        return np.array(spawn_probs_value, dtype=float)
    
    def flush_pending_timesteps(self) -> None:
        """
        Save any remaining pending timesteps to disk.
        
        Called at the end of a run to ensure no data is lost.
        """
        if self.pending_timesteps:
            self._save_pending_timesteps()
    
    def log_outcome(
        self,
        is_success: bool,
        is_timeout: bool,
        time_to_success: int
    ) -> None:
        """
        Log episode outcome metrics.
        
        Args:
            is_success: Whether task was completed
            is_timeout: Whether episode timed out
            time_to_success: Steps to success (or max_steps)
        """
        self.outcomes["is_success"] = is_success
        self.outcomes["is_timeout"] = is_timeout
        self.outcomes["time_to_success"] = time_to_success
    
    def get_results(self) -> Dict[str, Any]:
        """
        Return aggregated logging results for the episode.
        
        Returns:
            dict: A dictionary with three keys:
                - time_series: mapping of metric name to a NumPy array of recorded values.
                - outcomes: recorded outcome values for the episode.
                - metadata: episode metadata such as start/end times and step counts.
        """
        return {
            "time_series": {
                name: tracker.to_array() 
                for name, tracker in self.time_series.items()
            },
            "outcomes": self.outcomes.copy(),
            "metadata": self.metadata.copy(),
        }
    
    def log_progress(self, step: int, max_steps: int, n_workers: int = 1) -> None:
        """
        Log a concise progress line for the current episode including experiment and optional worker identifiers.
        
        This prints a single-line progress message like:
        "[Experiment: {name}] [Worker-{id}] Episode {episode_idx} (seed {episode_seed}), Step {step}/{max_steps}"
        and does nothing if progress display is disabled.
        
        Parameters:
            step (int): Current step number (1-indexed).
            max_steps (int): Total number of steps in the episode.
            n_workers (int): Number of parallel workers; when greater than 1, output is throttled to reduce spam.
        """
        if not self.show_progress:
            return
        
        # Format: [Experiment: baseline_vs_social] Run 1/6: baseline, Step 5/200
        prefix = f"[Experiment: {self.experiment_name}]"
        
        # Add run info if available
        if self.run_name is not None and self.run_idx is not None and self.n_runs is not None:
            prefix += f" Run {self.run_idx}/{self.n_runs}: {self.run_name}"
        else:
            # Fallback to old format
            prefix += f" Run {self.episode_idx}"
            if self.episode_seed is not None:
                prefix += f" (seed {self.episode_seed})"
        
        if n_workers > 1 and self.worker_id is not None:
            prefix += f" [Worker-{self.worker_id}]"
        
        # Throttling logic: show more frequently with fewer workers, and always show first/last steps
        if n_workers == 1:
            # Single worker: show every step
            pass  # Show all steps
        elif n_workers > 1:
            # Adaptive throttling based on worker count
            if n_workers <= 3:
                throttle_interval = 5  # Every 5 steps for 2-3 workers
            elif n_workers <= 6:
                throttle_interval = 10  # Every 10 steps for 4-6 workers
            else:
                throttle_interval = 20  # Every 20 steps for 7+ workers
            
            # Always show first step, last step, and steps at throttle_interval
            if step != 1 and step != max_steps and step % throttle_interval != 0:
                return
        
        print(f"{prefix}, Step {step}/{max_steps}", flush=True)
    
    def debug_log(self, message: str, lazy_value: Optional[Callable[[], str]] = None) -> None:
        """
        Write a debug message to a lazily opened per-experiment debug file; does nothing when debug logging is disabled.
        
        Parameters:
        	message (str): Message to record in the debug log.
        	lazy_value (Optional[Callable[[], str]]): Optional callable producing a string to include with the message; evaluated only when debug logging is enabled.
        
        Description:
        	The log entry is prefixed with the experiment name, episode index, seed, and optional worker id. When provided, the `lazy_value` result is appended as "message = value". The debug file is opened on first use and flushed after each write to ensure real-time visibility.
        """
        if not self.debug_enabled:
            return  # Early exit, no computation
        
        # Lazy open file on first use
        if self.debug_file is None:
            pid = os.getpid()
            debug_filename = f"debug_{self.experiment_name}_ep{self.episode_idx}_worker{pid}.log"
            self.debug_file = open(debug_filename, "w", buffering=1)  # Line buffered
        
        # Write header with experiment/episode info
        header = f"[{self.experiment_name}] Episode {self.episode_idx} (seed {self.episode_seed})"
        if self.worker_id is not None:
            header += f" [Worker-{self.worker_id}]"
        
        # Evaluate lazy value only when logging
        if lazy_value is not None:
            value_str = lazy_value()
            self.debug_file.write(f"{header}: {message} = {value_str}\n")
        else:
            self.debug_file.write(f"{header}: {message}\n")
        self.debug_file.flush()  # Ensure real-time visibility
    
    def close_debug(self) -> None:
        """
        Close the lazily-opened debug log file and clear the internal file handle.
        
        This is a no-op if no debug file is currently open.
        """
        if self.debug_file is not None:
            self.debug_file.close()
            self.debug_file = None
    
    def reset(self) -> None:
        """
        Prepare the logger for a new episode by closing any open debug file and clearing collected data.
        
        Resets all time-series metric trackers, sets recorded outcomes to None, and resets metadata keys "start_time" and "end_time" to None and "n_steps" to 0.
        """
        self.close_debug()  # Close debug file between episodes
        for tracker in self.time_series.values():
            tracker.reset()
        self.outcomes = {k: None for k in self.outcomes}
        self.metadata = {"start_time": None, "end_time": None, "n_steps": 0}