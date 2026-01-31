"""
Experiment runner: orchestration and parallelization.

This runner supports clean_up.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import glob
import json
import multiprocessing as mp
import os
import shutil
import time
import warnings

import numpy as np

# tqdm removed - progress bar only updated when runs complete, not suitable for long-running experiments

from empathy.clean_up.agent import CleanUpAgent
from empathy.clean_up.agent.beliefs import DirichletBeliefs
from clean_up import CleanUpEnvironment
from empathy.clean_up.experiment.logger import ExperimentLogger
from empathy.clean_up.experiment.config import get_experiment_config, create_experiment_grid
from empathy.clean_up.experiment.constants import spawn_probs_from_dict, get_spawn_prob_ranges
from empathy.clean_up.experiment.metrics import (
    compute_final_error,
    compute_detection_time,
    compute_drift_metrics,
    compute_stability_metrics,
)


@dataclass
class EpisodeResult:
    seed: int
    config: Dict[str, Any]
    time_series: Dict[str, np.ndarray]
    outcomes: Dict[str, Any]


def _normalize_spawn_probs(spawn_probs_value: Any) -> Optional[np.ndarray]:
    """Normalize spawn probs value to numpy array for clean_up task."""
    if spawn_probs_value is None:
        return None
    if isinstance(spawn_probs_value, dict):
        if "spawn_probs" not in spawn_probs_value:
            return None
        return spawn_probs_from_dict(spawn_probs_value)
    return np.array(spawn_probs_value, dtype=float)


def _compute_error(spawn_probs_self: Optional[np.ndarray], spawn_probs_true: Optional[np.ndarray]) -> float:
    """
    Compute L2 error between spawn probability arrays.
    
    Since spawn probabilities are in [0, 1], computes normalized error using
    the spawn probability ranges as the denominator.
    """
    if spawn_probs_self is None or spawn_probs_true is None:
        return float("nan")
    spawn_probs_min, spawn_probs_max = get_spawn_prob_ranges("clean_up")
    denom = np.maximum(spawn_probs_max - spawn_probs_min, 1e-12)
    return float(np.linalg.norm((spawn_probs_self - spawn_probs_true) / denom))


def _beliefs_dict_to_dirichlet(
    beliefs_dict: Optional[Dict[str, Any]],
    default_concentration: float = 1.0
) -> Optional[DirichletBeliefs]:
    """
    Convert beliefs config dict to DirichletBeliefs.
    
    Args:
        beliefs_dict: Config dict with 'spawn_probs' and optional 'concentration' keys
            - spawn_probs: List of 5 spawn probabilities, one per pollution context (0-4)
            - concentration: Total concentration (pseudo-observations)
        default_concentration: Fallback concentration if not in beliefs_dict (default: 1.0)
            - Low values (1.0-2.0): Weak prior, high uncertainty
            - High values (50.0-100.0): Strong beliefs, low uncertainty
    
    Returns:
        DirichletBeliefs or None if beliefs_dict is None or doesn't have spawn_probs
        
    Note:
        Returns None for legacy k/sigma format used by phase_diagram experiments.
        These must be handled separately or converted to spawn_probs format.
    """
    if beliefs_dict is None:
        return None
    
    # get spawn probabilities from config
    spawn_probs = beliefs_dict.get("spawn_probs")
    if spawn_probs is None:
        # return None for legacy formats (e.g., k/sigma for phase_diagram)
        # the caller should handle this case appropriately
        return None
    
    spawn_probs = np.array(spawn_probs, dtype=float)
    n_contexts = len(spawn_probs)
    
    # get concentration from config or use default
    if "concentration" in beliefs_dict:
        concentration = float(beliefs_dict["concentration"])
    else:
        concentration = default_concentration
    if not (np.isfinite(concentration) and concentration > 0):
        concentration = default_concentration

    # Build alpha_dict from spawn probabilities
    alpha_dict = {}
    epsilon = 1e-6  # Prevent exactly-zero Dirichlet parameters for numerical stability
    for context in range(n_contexts):
        p_spawn = np.clip(spawn_probs[context], 0.0, 1.0)
        alpha_1 = max(epsilon, concentration * p_spawn)
        alpha_0 = max(epsilon, concentration * (1 - p_spawn))
        alpha_dict[context] = np.array([alpha_0, alpha_1], dtype=np.float64)
    
    return DirichletBeliefs(n_contexts=n_contexts, alpha_dict=alpha_dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_run_name(config: Dict[str, Any]) -> str:
    """
    Get a human-readable name for the current run based on configuration.
    
    Parameters:
        config: Episode configuration dictionary
        
    Returns:
        str: Run name like "baseline", "social+expert", "social+mismatched", etc.
    """
    social_enabled = config.get("social_enabled", False)
    
    if not social_enabled:
        return "baseline"
    
    # Check if expert params match true params
    spawn_probs_true = config.get("spawn_probs_true")
    beliefs_expert = config.get("beliefs_expert")
    
    if beliefs_expert is None:
        return "social"
    
    # Simple comparison: if expert == true, it's expert; otherwise mismatched
    if spawn_probs_true is not None and beliefs_expert is not None:
        # Convert to comparable format (handle dict vs dict)
        if isinstance(spawn_probs_true, dict) and isinstance(beliefs_expert, dict):
            if spawn_probs_true == beliefs_expert:
                return "social+expert"
            else:
                return "social+mismatched"
    
    return "social"


# =============================================================================
# SINGLE RUN EXECUTION
# =============================================================================


def run_single_episode(
    agent_learner: Any,
    agent_expert: Any,
    environment: Any,
    config: Dict[str, Any],
    learn: bool = True,
    update_tom: bool = True,
) -> Dict[str, Any]:
    """
    Run a single continuous learning trajectory (run) with the provided learner and expert agents.
    
    Runs a step loop up to the environment or config max_steps, driving agents' perceive/act/update cycle, optionally updating the learner's theory-of-mind and learning parameters, and logs per-timestep time series and final outcomes.
    
    Parameters:
        agent_learner: The learning agent instance; must implement reset_episode, perceive, act, observe_other_action, learn_from_observation, compute_social_metrics, social_learn, and related attributes used by logging.
        agent_expert: The expert agent instance; must implement reset_episode, perceive, act, and may provide a `_last_context` attribute for ToM updates.
        environment: The environment instance; must implement reset(seed) returning an observation dict with keys "learner" and "expert", and step(actions) returning (obs, reward, done, info).
        config (dict): Run configuration. Useful keys:
            - "episode_seed" or "seed": seed passed to environment.reset.
            - "episode_idx": run index for logging.
            - "worker_id": worker identifier for logging.
            - "n_workers": number of workers used (affects progress logging).
            - "max_steps": maximum timestep count.
            - "experiment_name": name included in run metadata.
        learn (bool): If True, perform learner updates (learn_from_observation and social_learn) during the run.
        update_tom (bool): If True, pass expert context to the learner via observe_other_action to allow ToM updates.
    
    Returns:
        dict: Run results produced by the ExperimentLogger, including logged time_series and an outcomes dictionary with keys such as `is_success`, `is_timeout`, and `time_to_success`.
    """
    max_steps = getattr(environment, "max_steps", None) or config.get("max_steps", 80)
    # Get run info for better logging
    run_name = config.get("run_name")
    if run_name is None:
        run_name = _get_run_name(config)
    
    logger = ExperimentLogger(config={
        **config,
        "experiment_name": config.get("experiment_name", "unknown"),
        "episode_seed": config.get("episode_seed", config.get("seed", 0)),
        "episode_idx": config.get("episode_idx", 0),
        "worker_id": config.get("worker_id"),  # Set by worker
        "show_progress": True,  # Always show progress (throttled based on worker count)
        "verbose": config.get("verbose", False),  # Store verbose flag for potential future use
        "run_name": run_name,
        "run_idx": config.get("run_idx"),
        "n_runs": config.get("n_runs"),
        "save_batch_size": config.get("save_batch_size", 10),
        "run_dir": config.get("run_dir"),
    })

    # No reset needed - single continuous run
    agent_learner.reset_episode(reset_tom=False)
    agent_expert.reset_episode(reset_tom=False)

    episode_seed = config.get("episode_seed", config.get("seed"))
    obs = environment.reset(seed=episode_seed)

    task_success = False
    is_timeout = False
    time_to_success = int(max_steps)

    for t in range(int(max_steps)):
        # Log progress at start of each step
        logger.log_progress(t + 1, max_steps, n_workers=config.get("n_workers", 1))
        agent_learner.perceive(obs["learner"])
        agent_expert.perceive(obs["expert"])

        expert_action = agent_expert.act()
        if update_tom:
            tom_context = getattr(agent_expert, "_last_context", None)
            if not isinstance(tom_context, dict):
                tom_context = None
            agent_learner.observe_other_action(expert_action, tom_context)

        learner_action = agent_learner.act()
        actions = {"learner": learner_action, "expert": expert_action}
        obs, rewards, done, info = environment.step(actions)

        # Performance history is updated during social_learn() if social learning is enabled

        if learn:
            agent_learner.learn_from_observation(obs["learner"])

        social_metrics = agent_learner.compute_social_metrics()
        if learn:
            agent_learner.social_learn()
            update_stats = getattr(agent_learner, "_last_social_update_stats", {})
            if social_metrics is not None:
                social_metrics.update(update_stats)

        logger.log_timestep(t, agent_learner, social_metrics, env_info=info)

        if done:
            task_success = bool(info.get("is_success", False))
            is_timeout = bool(info.get("is_timeout", False))
            time_to_success = t + 1
            break

    # If we completed all max_steps without done=True, mark as timeout
    if time_to_success >= max_steps and not task_success:
        is_timeout = True

    logger.log_outcome(
        is_success=task_success,
        is_timeout=is_timeout,
        time_to_success=time_to_success,
    )

    # Flush any remaining pending timesteps before closing
    logger.flush_pending_timesteps()

    logger.close_debug()  # Close debug file at run end
    results = logger.get_results()
    return results


# =============================================================================
# AGENT/ENVIRONMENT FACTORIES
# =============================================================================


def create_agents_for_config(config: Dict[str, Any]) -> Any:
    """
    Create agent instances configured for clean_up experiment.

    Both agents are identical Active Inference agents that use EFE-based planning.
    Differences between agents are controlled purely through per-agent configuration
    in the 'agents' list (beliefs_init, learning rates, etc.).

    Parameters:
        config (dict): Experiment configuration. May include:
            - "agents" (list): per-agent configuration dicts. Each agent config can have:
                - "beliefs_init": dict with 'spawn_probs' and optional 'concentration'
                - Other agent-specific settings (self_learning, use_tom, etc.)
            - "n_agents" (int): number of agents (defaults to 2).
            - "beliefs_self_init", "beliefs_expert" - will be converted to
              per-agent beliefs_init if agents don't have their own.

    Returns:
        tuple: (agent_learner, agent_expert)
            Both are CleanUpAgent instances with different configs.
            agent_expert may be the same object as agent_learner in solo mode (n_agents == 1).
            beliefs_init dicts are converted to DirichletBeliefs and assigned to
            `beliefs_self` attributes.
    """
    agents_config = config.get("agents", [])
    n_agents = config.get("n_agents", 2)

    # Index-based approach: Agent 0 is learner, Agent 1 is expert (by convention)
    # These are just naming conventions - both are identical Active Inference agents
    learner_config = agents_config[0].copy() if len(agents_config) > 0 else {}
    expert_config = agents_config[1].copy() if len(agents_config) > 1 else {}

    # Propagate experiment-level config to agent configs (agents can override)
    # Important parameters that should be shared unless explicitly overridden:
    for key in ["planning_horizon", "beta", "lambda_epist", "beta_tom", "tom_mode",
                "eta_0", "T_a", "self_learning", "social_enabled"]:
        if key in config:
            if key not in learner_config:
                learner_config[key] = config[key]
            if key not in expert_config:
                expert_config[key] = config[key]

    # Create learner agent (agent_id=0)
    learner_config["n_agents"] = n_agents
    agent_learner = CleanUpAgent(config=learner_config, agent_id=0)

    # Create expert agent (agent_id=1) or duplicate learner for solo mode
    if n_agents == 1:
        # Solo mode: expert is same as learner
        agent_expert = agent_learner
    else:
        # Multi-agent mode: create second agent with its own config
        # Note: expert_config already has experiment-level params from above
        expert_config["n_agents"] = n_agents
        agent_expert = CleanUpAgent(config=expert_config, agent_id=1)

    # Initialize beliefs from per-agent config (converts beliefs_init to DirichletBeliefs)
    # Each agent's beliefs_init should specify spawn_probs and concentration
    # Default concentrations (can be overridden by "concentration" key in beliefs_init):
    #   - Agent 0: weak prior (2.0) for learning
    #   - Agent 1: strong beliefs (100.0) for established knowledge
    
    # Agent 0 (learner) beliefs initialization
    learner_beliefs_init = learner_config.get("beliefs_init")
    if learner_beliefs_init is None:
        # Fall back to top-level beliefs_self_init
        learner_beliefs_init = config.get("beliefs_self_init")
    beliefs_self_init = _beliefs_dict_to_dirichlet(
        learner_beliefs_init,
        default_concentration=2.0  # Weak prior - can be overridden in config
    )
    if beliefs_self_init is not None:
        agent_learner.beliefs_self = beliefs_self_init

    # Agent 1 (expert) beliefs initialization (only if multi-agent mode)
    if n_agents > 1:
        expert_beliefs_init = expert_config.get("beliefs_init")
        if expert_beliefs_init is None:
            # Fall back to top-level beliefs_expert
            expert_beliefs_init = config.get("beliefs_expert")
        beliefs_expert = _beliefs_dict_to_dirichlet(
            expert_beliefs_init,
            default_concentration=100.0  # Strong beliefs - can be overridden in config
        )
        if beliefs_expert is not None:
            agent_expert.beliefs_self = beliefs_expert

    return agent_learner, agent_expert


def create_environment_for_config(config: Dict[str, Any]) -> Any:
    """
    Create and return a CleanUpEnvironment instance configured for the experiment.
    
    Parameters:
        config (dict): Experiment configuration. Recognized keys:
            - seed (int, optional): Random seed for the environment.
            - max_steps (int, optional): Maximum steps per run (defaults to 80).
            - spawn_probs_true (optional): Ground-truth spawn probabilities to pass to the environment.
            - n_agents (int, optional): Number of agents (defaults to 2).
            - environment (dict, optional): Environment parameters:
                - grid_size (int, default 3)
                - pollution_rate (float, default 0.15)
                - clean_power (float, default 0.8)
                - spawn_rate (float, default 0.25)
                - apple_obs_mode (str, default "full")
                - pollution_obs_mode (str, default "full")
    
    Returns:
        CleanUpEnvironment: Configured environment instance.
    """
    clean_up_config = config.get("environment", {})
    spawn_probs_true = _normalize_spawn_probs(config.get("spawn_probs_true"))
    return CleanUpEnvironment(
        n_agents=config.get("n_agents", 2),
        grid_size=clean_up_config.get("grid_size", 3),
        pollution_rate=clean_up_config.get("pollution_rate", 0.15),
        clean_power=clean_up_config.get("clean_power", 0.8),
        spawn_rate=clean_up_config.get("spawn_rate", 0.25),
        apple_obs_mode=clean_up_config.get("apple_obs_mode", "full"),
        pollution_obs_mode=clean_up_config.get("pollution_obs_mode", "full"),
        seed=config.get("seed"),
        max_steps=config.get("max_steps", 80),
        spawn_probs=spawn_probs_true,
    )


# =============================================================================
# RUN DIRECTORY NAMING
# =============================================================================


def _get_run_dir_name(config: Dict[str, Any], run_idx: int) -> str:
    """
    Generate run directory name using condition and seed.
    
    Format: run_{idx}_{condition}_seed{seed}
    """
    run_name = _get_run_name(config)
    seed = config.get("seed", 0)
    return f"run_{run_idx}_{run_name}_seed{seed}"


def _save_final_run_results(result: EpisodeResult, run_dir: Path, config: Dict[str, Any]) -> None:
    """
    Add computed metrics to existing timeseries.json file.
    
    The logger already saves timesteps incrementally during the run. This function
    loads the existing file, computes final metrics, adds them to metadata, and saves back.
    
    Parameters:
        result: EpisodeResult with time_series and outcomes
        run_dir: Directory where timeseries.json is saved
        config: Run configuration dictionary
    """
    timeseries_file = run_dir / "timeseries.json"
    
    if not timeseries_file.exists():
        # If file doesn't exist, logger hasn't saved anything yet - skip
        return
    
    # Load existing data
    try:
        existing_data = json.loads(timeseries_file.read_text())
        metadata = existing_data.get("metadata", {})
        timesteps = existing_data.get("timesteps", [])
    except (json.JSONDecodeError, IOError):
        # If file is corrupted, skip adding metrics
        return
    
    # Compute metrics from result
    metrics = compute_final_error(result)
    drift_metrics = compute_drift_metrics(result)
    stability_metrics = compute_stability_metrics(result)
    
    # Compute error_init and error_final
    spawn_probs_true = _normalize_spawn_probs(config.get("spawn_probs_true"))
    spawn_probs_init = _normalize_spawn_probs(config.get("beliefs_self_init"))
    
    param_self = result.time_series.get("param_self", [])
    error_init = None
    error_final = None
    if spawn_probs_true is not None and spawn_probs_init is not None:
        error_init = _compute_error(spawn_probs_init, spawn_probs_true)
    if spawn_probs_true is not None and len(param_self) > 0:
        spawn_probs_final = np.array(param_self[-1], dtype=float)
        error_final = _compute_error(spawn_probs_final, spawn_probs_true)
    
    # Update metadata with outcomes and metrics
    metadata["outcomes"] = result.outcomes
    metadata["metrics"] = {
        "final_error": metrics,
        "error_init": error_init,
        "error_final": error_final,
        "error_delta_vs_init": error_final - error_init if error_init is not None and error_final is not None else None,
        **drift_metrics,
        **stability_metrics,
    }
    
    # Write updated data
    payload = {
        "metadata": metadata,
        "timesteps": timesteps,
    }
    
    # Atomic write
    temp_file = timeseries_file.with_suffix('.json.tmp')
    try:
        temp_file.write_text(json.dumps(payload, indent=2, default=str))
        temp_file.replace(timeseries_file)
    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================


def _run_episode_worker(config: Dict[str, Any]) -> EpisodeResult:
    """
    Run a single continuous learning trajectory (run).
    
    Parameters:
        config (dict): Experiment and run settings. Recognized keys:
            - "seed" (int): Seed for run RNG.
            - "n_workers" (int): Total number of parallel workers (propagated into run config).
            - "run_dir" (str): Directory path for saving timesteps incrementally.
            - "experiment_name" (str): Name of the experiment.
            - "save_batch_size" (int): Save timesteps every N steps (default 10).
            - any other keys required by create_agents_for_config, create_environment_for_config, or run_single_episode.
    
    Returns:
        EpisodeResult: EpisodeResult object with seed, config, time_series, and outcomes.
    """
    worker_id = os.getpid()  # Use PID as worker ID
    
    base_seed = int(config.get("seed", 0))
    run_dir = Path(config.get("run_dir")) if config.get("run_dir") else None
    save_batch_size = int(config.get("save_batch_size", 10))

    agent_learner, agent_expert = create_agents_for_config(config)

    # Get stable episode_idx from config (set during run_configs construction)
    episode_idx = config.get("episode_idx", 0)
    
    run_config = {
        **config,
        "episode_seed": base_seed,
        "episode_idx": episode_idx,
        "worker_id": worker_id,  # Pass worker ID
        "n_workers": config.get("n_workers", 1),  # Pass worker count
        "run_dir": str(run_dir) if run_dir else None,  # Pass run_dir for batched saving
        "save_batch_size": save_batch_size,  # Pass batch size for saving
    }

    environment = create_environment_for_config(run_config)
    results = run_single_episode(agent_learner, agent_expert, environment, run_config)
    outcome = results["outcomes"]

    episode_result = EpisodeResult(
        seed=base_seed,
        config=run_config,
        time_series=results["time_series"],
        outcomes=outcome,
    )
    
    # Final save of remaining timesteps (if any) and metadata
    if run_dir is not None:
        try:
            _save_final_run_results(episode_result, run_dir, config)
        except Exception as e:
            # Log error but continue - don't fail entire run if save fails
            warnings.warn(f"Failed to save final run results: {e}", UserWarning)

    return episode_result


class ExperimentRunner:
    """Manages experiment configuration, execution, and result aggregation."""

    def __init__(
        self,
        experiment_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an ExperimentRunner for orchestration of experiment runs.

        Sets up the experiment configuration and derived run grid, applies any provided configuration overrides, and prepares internal containers for results and metadata.

        Parameters:
            experiment_name (str): Name of the experiment whose base configuration will be loaded.
            config_overrides (Optional[Dict[str, Any]]): Values to override in the loaded experiment configuration.
        """
        self.experiment_name = experiment_name
        self.config_overrides = config_overrides or {}
        self.config = get_experiment_config(experiment_name)
        self.config.update(self.config_overrides)
        self.episode_grid = create_experiment_grid(self.config)
        self.results: List[EpisodeResult] = []
        self.metadata: Dict[str, Any] = {}
        self.run_dir: Optional[Path] = None  # Set during run() execution

    def run(self, n_workers: int = 8, verbose: bool = True, output_dir: Optional[Path] = None) -> List[EpisodeResult]:
        """
        Execute the experiment grid and collect run results.
        
        Runs the remaining run configurations using up to `n_workers` parallel workers (or sequentially when `n_workers` <= 1). Completed runs (those with existing timeseries.json files) are automatically skipped. Updates the runner's `results` and `metadata` fields to reflect the run summary.
        
        Returns:
            List[EpisodeResult]: List of run results.
        """
        start_time = time.time()
        self.results = []
        
        # Determine output directory root
        results_root = Path(output_dir) if output_dir else Path("results")

        # Runs are automatically skipped if their timeseries.json exists
        # (handled in run_configs building below)

        # Build run configs and filter out completed runs
        experiment_dir = results_root / self.experiment_name
        data_dir = experiment_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        run_configs: List[Dict[str, Any]] = []
        n_runs = len(self.episode_grid)
        for grid_idx, episode_cfg in enumerate(self.episode_grid):
            # Get run name for this config
            run_name = _get_run_name(episode_cfg)
            run_idx = grid_idx + 1  # 1-indexed run number
            
            # Create individual run directory for this run in data/ subfolder
            run_dir_name = _get_run_dir_name(episode_cfg, grid_idx)
            run_dir = data_dir / run_dir_name
            
            # Check if run is already completed (has timeseries.json)
            if run_dir.exists() and (run_dir / "timeseries.json").exists():
                # Skip completed runs
                continue
            
            run_cfg = {
                **episode_cfg,
                "experiment_name": self.experiment_name,  # Add experiment name
                "n_workers": n_workers,  # Pass worker count
                "verbose": verbose,  # Pass verbose flag for progress logging
                "run_dir": str(run_dir),  # Pass run directory for batched saving
                "run_name": run_name,  # Human-readable run name
                "run_idx": run_idx,  # 1-indexed run number
                "n_runs": n_runs,  # Total number of runs
                "episode_idx": grid_idx,  # Use grid position as episode_idx
            }
            
            run_configs.append(run_cfg)

        if verbose:
            print(f"Running {len(run_configs)} runs", flush=True)

        # Run remaining runs
        new_results: List[EpisodeResult] = []
        if len(run_configs) > 0:
            if n_workers <= 1:
                for run_cfg in run_configs:
                    new_results.append(_run_episode_worker(run_cfg))
                    if verbose:
                        print(f"Completed {len(new_results)}/{len(run_configs)} run(s)", flush=True)
            else:
                with mp.Pool(processes=n_workers) as pool:
                    results_iter = pool.imap(_run_episode_worker, run_configs)
                    for result in results_iter:
                        new_results.append(result)
                        if verbose:
                            print(f"Completed {len(new_results)}/{len(run_configs)} run(s)", flush=True)

        # Store results
        self.results = new_results

        # Set run_dir to experiment directory (for save_results compatibility)
        self.run_dir = experiment_dir
        
        self.metadata = {
            "experiment": self.experiment_name,
            "n_runs": len(self.results),
            "n_new_runs": len(new_results),
            "duration_sec": time.time() - start_time,
            "config": self.config,
        }
        return self.results

    def save_results(self, output_dir: Optional[str] = None) -> Path:
        """
        Save experiment metadata and aggregates to disk.

        Note: Runs are saved incrementally during execution, so this method only
        writes metadata.json at the experiment level.

        Parameters:
            output_dir (Optional[str]): Base directory for results; defaults to "./results" when not provided.

        Returns:
            Path: Filesystem path to the experiment directory where results were written.

        Details:
            - Uses the experiment directory created during run() execution.
            - Writes `metadata.json` at the experiment level.
            - Individual runs are saved to their own directories with timeseries.json files.
            - Aggregated metrics are computed by analysis scripts, not here.
        """
        # Use the experiment directory created during run()
        if not hasattr(self, 'run_dir') or self.run_dir is None:
            # Fallback: create directory if run() wasn't called (shouldn't happen in normal flow)
            results_root = Path(output_dir) if output_dir else Path("results")
            self.run_dir = results_root / self.experiment_name
            self.run_dir.mkdir(parents=True, exist_ok=True)

        experiment_dir = self.run_dir

        # Write metadata
        (experiment_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str))

        # Copy debug files to results directory if debug was enabled
        if self.config.get("debug", False):
            debug_files = glob.glob(f"debug_{self.experiment_name}_worker*.log")
            for debug_file in debug_files:
                if Path(debug_file).exists():
                    shutil.copy2(debug_file, experiment_dir / Path(debug_file).name)
                    Path(debug_file).unlink()  # Clean up temp file

        return experiment_dir