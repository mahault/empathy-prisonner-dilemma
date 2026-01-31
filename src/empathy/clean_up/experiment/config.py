"""
Configuration loading and grid building for clean_up experiments.

Loads experiment configurations from JSON files and builds parameter grids.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import copy
import json


# =============================================================================
# CONFIG REGISTRY
# =============================================================================

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _load_json_config(path: Path) -> Dict[str, Any]:
    """Load a single JSON config file."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    payload.setdefault("name", path.stem)
    return payload


def _load_experiment_configs() -> Dict[str, Dict[str, Any]]:
    """Load all experiment configs from clean_up/configs/ directory."""
    if not CONFIGS_DIR.exists():
        return {}
    configs: Dict[str, Dict[str, Any]] = {}
    name_to_path: Dict[str, Path] = {}
    for path in sorted(CONFIGS_DIR.glob("*.json")):
        config = _load_json_config(path)
        name = config.get("name", path.stem)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Config name must be a non-empty string: {path}")
        if name in configs:
            previous_path = name_to_path[name]
            raise ValueError(
                f"Duplicate experiment name '{name}' found in multiple config files: "
                f"{previous_path} and {path}"
            )
        configs[name] = config
        name_to_path[name] = path
    return configs


EXPERIMENT_CONFIGS = _load_experiment_configs()


def get_experiment_config(name: str) -> Dict[str, Any]:
    """
    Return configuration dict for a named experiment.
    
    Args:
        name: Experiment name (must exist in EXPERIMENT_CONFIGS)
        
    Returns:
        Copy of the experiment configuration dictionary
        
    Raises:
        ValueError: If no configs found or name doesn't exist
    """
    if not EXPERIMENT_CONFIGS:
        raise ValueError(
            f"No configs found in {CONFIGS_DIR}. Add JSON configs before running."
        )
    if name not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment: {name}. Available: {list(EXPERIMENT_CONFIGS.keys())}"
        )
    return copy.deepcopy(EXPERIMENT_CONFIGS[name])


# =============================================================================
# GRID BUILDING
# =============================================================================


def _resolve_seeds(config: Dict[str, Any]) -> List[int]:
    """Resolve seed list from config (either explicit seeds or n_seeds)."""
    seeds = config.get("seeds")
    if seeds is None:
        return list(range(int(config["n_seeds"])))
    return [int(seed) for seed in seeds]


def _base_episode_config(
    config: Dict[str, Any],
    seed: int,
    *,
    task: str,
    scenario: str,
    experiment_name: Optional[str] = None,
    tom_mode: Optional[str] = None,
    beta_tom: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build base episode configuration with common fields.

    Uses experiment_family if available (for grouping related experiments),
    otherwise falls back to experiment name.
    
    Note: tom_mode is required for all episode configs. Grid builders should
    always pass a non-None tom_mode value (e.g., 'off', 'softmax', 'greedy').
    This ensures proper validation and correct analysis of experiment results.
    
    Automatically propagates common agent/planning parameters from experiment config
    to episode configs so they don't use defaults.
    """
    episode_config = {
        "experiment_name": experiment_name
        or config.get("experiment_family", config.get("name", "")),
        "task": task,
        "scenario": scenario,
        "seed": seed,
    }
    if tom_mode is not None:
        episode_config["tom_mode"] = tom_mode
    if beta_tom is not None:
        episode_config["beta_tom"] = float(beta_tom)
    
    # Propagate common parameters from experiment config to episode config
    # These ensure agents use experiment-specified values instead of defaults
    common_params = ["planning_horizon", "beta", "lambda_epist", "eta_0", "T_a",
                     "self_learning", "social_enabled", "max_steps"]
    for param in common_params:
        if param in config:
            episode_config[param] = config[param]
    
    return episode_config


def _build_smoke_grid(config: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    """Build grid for smoke test experiments."""
    grid: List[Dict[str, Any]] = []
    for task_cfg in config["tasks"]:
        for seed in seeds:
            episode_config = _base_episode_config(
                config,
                seed,
                task=task_cfg["task"],
                scenario=task_cfg["scenario"],
                experiment_name=config.get("name", "smoke"),
                tom_mode=config.get("tom_mode", "softmax"),
                beta_tom=config.get("beta_tom", 2.0),
            )
            grid.append(episode_config)
    return grid


def _build_killer_test_grid(config: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    """Build grid for killer test experiments."""
    grid: List[Dict[str, Any]] = []
    # Get parameter sweeps (defaults if not specified)
    social_enabled_values = config.get("social_enabled", [True])
    if not isinstance(social_enabled_values, list):
        social_enabled_values = [social_enabled_values]
    eta_0_values = config.get("eta_0", [0.1])
    if not isinstance(eta_0_values, list):
        eta_0_values = [eta_0_values]
    
    for task_cfg in config["tasks"]:
        task = task_cfg["task"]
        scenario = task_cfg["scenario"]
        truth = config.get("scenario_truth_params", {}).get(task, {}).get(scenario, {})
        if not truth:
            raise ValueError(
                f"Missing truth params for task={task} scenario={scenario}. "
                f"Available tasks: {list(config.get('scenario_truth_params', {}).keys())}"
            )
        expert = config.get("expert_mismatch_params", {}).get(task, {})
        if not expert:
            raise ValueError(
                f"Missing expert mismatch params for task={task}. "
                f"Available tasks: {list(config.get('expert_mismatch_params', {}).keys())}"
            )
        for social_enabled in social_enabled_values:
            # Set tom_mode based on social_enabled
            if social_enabled:
                tom_mode = config.get("tom_mode", "softmax")
            else:
                tom_mode = "off"
            
            for eta_0 in eta_0_values:
                for seed in seeds:
                    episode_config = _base_episode_config(
                        config,
                        seed,
                        task=task,
                        scenario=scenario,
                        experiment_name=config.get("name", "killer_test"),
                        tom_mode=tom_mode,
                        beta_tom=config.get("beta_tom", 2.0),
                    )
                    episode_config.update(
                        {
                            "mismatch": True,
                            "spawn_probs_true": truth,
                            "social_enabled": social_enabled,
                            "eta_0": float(eta_0),
                            "n_agents": 2,
                            # Per-agent configurations with beliefs_init
                            "agents": [
                                {
                                    # Agent 0: Learner
                                    "agent_id": 0,
                                    "self_learning": True,
                                    "use_tom": social_enabled,
                                    "social_enabled": social_enabled,
                                    "beliefs_init": {**dict(truth), "concentration": 2.0},
                                    "eta_0": float(eta_0),
                                },
                                {
                                    # Agent 1: Expert with mismatched params
                                    "agent_id": 1,
                                    "self_learning": False,
                                    "use_tom": False,
                                    "social_enabled": False,
                                    "beliefs_init": {**dict(expert), "concentration": expert.get("concentration", 100.0)},
                                },
                            ],
                        }
                    )
                    grid.append(episode_config)
    return grid


def _build_baseline_vs_social_grid(config: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    """Build grid for baseline vs social learning comparison experiments."""
    grid: List[Dict[str, Any]] = []
    
    task = config.get("task", "clean_up")
    scenario = config.get("scenario", "default_truth")
    spawn_probs_true = config.get("spawn_probs_true", {})
    beliefs_self_init = config.get("beliefs_self_init", spawn_probs_true)  # Use beliefs_self_init if provided, else spawn_probs_true
    beliefs_expert_mismatched = config.get("beliefs_expert_mismatched", {})
    debug_tom = config.get("debug_tom", False)
    
    # Common agent hyperparameters
    eta_0 = config.get("eta_0", 0.1)
    T_a = config.get("T_a", 2.0)
    u_0 = config.get("u_0", 0.05)
    kappa = config.get("kappa", 0.05)
    beta_tom = config.get("beta_tom", 2.0)
    
    # Condition 1: Baseline (self-learning only, no social learning)
    # Learner: learns from own observations, no ToM
    # Expert: just acts with true params, no learning
    for seed in seeds:
        episode_config = _base_episode_config(
            config,
            seed,
            task=task,
            scenario=scenario,
            experiment_name=config.get("name", "baseline_vs_social"),
            tom_mode="off",
            beta_tom=beta_tom,
        )
        episode_config.update(
            {
                "spawn_probs_true": dict(spawn_probs_true),
                "social_enabled": False,
                "self_learning": True,
                "debug_tom": debug_tom,
                "n_agents": config.get("n_agents", 2),
                "max_steps": config.get("max_steps", 80),
                "save_batch_size": config.get("save_batch_size", 10),
                "eta_0": eta_0,
                "T_a": T_a,
                "u_0": u_0,
                "kappa": kappa,
                # Per-agent configurations with beliefs_init
                "agents": [
                    {
                        # Agent 0: Learner - learns from own observations only
                        "agent_id": 0,
                        "self_learning": True,
                        "use_tom": False,
                        "social_enabled": False,
                        "beliefs_init": dict(beliefs_self_init),
                        "eta_0": eta_0,
                        "T_a": T_a,
                    },
                    {
                        # Agent 1: Expert - just acts, no learning
                        "agent_id": 1,
                        "self_learning": False,
                        "use_tom": False,
                        "social_enabled": False,
                        "beliefs_init": {**dict(spawn_probs_true), "concentration": 100.0},
                    },
                ],
            }
        )
        grid.append(episode_config)
    
    # Condition 2: Social + Expert (beliefs_expert = spawn_probs_true)
    # Learner: learns from own observations + social learning from expert
    # Expert: just acts with true params, no learning
    for seed in seeds:
        episode_config = _base_episode_config(
            config,
            seed,
            task=task,
            scenario=scenario,
            experiment_name=config.get("name", "baseline_vs_social"),
            tom_mode=config.get("tom_mode", "softmax"),
            beta_tom=beta_tom,
        )
        episode_config.update(
            {
                "spawn_probs_true": dict(spawn_probs_true),
                "social_enabled": True,
                "self_learning": True,
                "debug_tom": debug_tom,
                "n_agents": config.get("n_agents", 2),
                "max_steps": config.get("max_steps", 80),
                "save_batch_size": config.get("save_batch_size", 10),
                "eta_0": eta_0,
                "T_a": T_a,
                "u_0": u_0,
                "kappa": kappa,
                # Per-agent configurations with beliefs_init
                "agents": [
                    {
                        # Agent 0: Learner - learns from self + social
                        "agent_id": 0,
                        "self_learning": True,
                        "use_tom": True,
                        "social_enabled": True,
                        "beliefs_init": dict(beliefs_self_init),
                        "eta_0": eta_0,
                        "T_a": T_a,
                        "u_0": u_0,
                        "kappa": kappa,
                        "tom_mode": config.get("tom_mode", "softmax"),
                        "beta_tom": beta_tom,
                    },
                    {
                        # Agent 1: Expert - just acts with perfect params, no learning
                        "agent_id": 1,
                        "self_learning": False,
                        "use_tom": False,
                        "social_enabled": False,
                        "beliefs_init": {**dict(spawn_probs_true), "concentration": 100.0},
                    },
                ],
            }
        )
        grid.append(episode_config)
    
    # Condition 3: Social + Mismatched (beliefs_expert != spawn_probs_true)
    # Learner: learns from own observations + social learning from mismatched expert
    # Expert: just acts with mismatched params, no learning
    for seed in seeds:
        episode_config = _base_episode_config(
            config,
            seed,
            task=task,
            scenario=scenario,
            experiment_name=config.get("name", "baseline_vs_social"),
            tom_mode=config.get("tom_mode", "softmax"),
            beta_tom=beta_tom,
        )
        episode_config.update(
            {
                "spawn_probs_true": dict(spawn_probs_true),
                "social_enabled": True,
                "self_learning": True,
                "debug_tom": debug_tom,
                "n_agents": config.get("n_agents", 2),
                "max_steps": config.get("max_steps", 80),
                "save_batch_size": config.get("save_batch_size", 10),
                "eta_0": eta_0,
                "T_a": T_a,
                "u_0": u_0,
                "kappa": kappa,
                # Per-agent configurations with beliefs_init
                "agents": [
                    {
                        # Agent 0: Learner - learns from self + social
                        "agent_id": 0,
                        "self_learning": True,
                        "use_tom": True,
                        "social_enabled": True,
                        "beliefs_init": dict(beliefs_self_init),
                        "eta_0": eta_0,
                        "T_a": T_a,
                        "u_0": u_0,
                        "kappa": kappa,
                        "tom_mode": config.get("tom_mode", "softmax"),
                        "beta_tom": beta_tom,
                    },
                    {
                        # Agent 1: Expert - just acts with mismatched params, no learning
                        "agent_id": 1,
                        "self_learning": False,
                        "use_tom": False,
                        "social_enabled": False,
                        "beliefs_init": {**dict(beliefs_expert_mismatched), "concentration": beliefs_expert_mismatched.get("concentration", 100.0)},
                    },
                ],
            }
        )
        grid.append(episode_config)
    
    return grid


def _build_phase_diagram_grid(config: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    """Build grid for phase diagram experiments."""
    grid: List[Dict[str, Any]] = []
    # Get eta_0 sweep (default if not specified)
    eta_0_values = config.get("eta_0", [0.1])
    if not isinstance(eta_0_values, list):
        eta_0_values = [eta_0_values]
    
    beliefs_self_init = config.get("beliefs_self_init", config["spawn_probs_true"])
    
    for mismatch_k in config["mismatch_k"]:
        for eta_0 in eta_0_values:
            for seed in seeds:
                episode_config = _base_episode_config(
                    config,
                    seed,
                    task=config["task"],
                    scenario=config["scenario"],
                    tom_mode="softmax",
                    beta_tom=2.0,
                )
                episode_config.update(
                    {
                        "mismatch_k": mismatch_k,
                        "spawn_probs_true": dict(config["spawn_probs_true"]),
                        "eta_0": float(eta_0),
                        "n_agents": 2,
                        "social_enabled": True,
                        # Per-agent configurations with beliefs_init
                        "agents": [
                            {
                                # Agent 0: Learner
                                "agent_id": 0,
                                "self_learning": True,
                                "use_tom": True,
                                "social_enabled": True,
                                "beliefs_init": dict(beliefs_self_init),
                                "eta_0": float(eta_0),
                            },
                            {
                                # Agent 1: Expert (params derived from mismatch_k and sigma)
                                "agent_id": 1,
                                "self_learning": False,
                                "use_tom": False,
                                "social_enabled": False,
                                # k/sigma format for phase diagram mismatch computation
                                "beliefs_init": {
                                    "k": mismatch_k,
                                    "sigma": float(config["expert_sigma"]),
                                },
                            },
                        ],
                    }
                )
                grid.append(episode_config)
    return grid


def _build_phase_diagram_baseline_grid(
    config: Dict[str, Any], seeds: List[int]
) -> List[Dict[str, Any]]:
    """
    Builds the phase-diagram baseline grid where each episode runs with individual learning only (social learning disabled).
    
    Parameters:
        config (Dict[str, Any]): Experiment configuration. Must contain:
            - "mismatch_k": iterable of mismatch magnitudes to sweep.
            - "task": task identifier.
            - "scenario": scenario identifier.
            - "spawn_probs_true": base true parameters for the task.
            - "beliefs_self_init": initial self parameters.
            - "expert_sigma": numeric expert noise parameter.
        seeds (List[int]): List of integer seeds to expand the grid over.
    
    Returns:
        List[Dict[str, Any]]: A list of episode configuration dictionaries with per-agent beliefs_init.
    """
    grid: List[Dict[str, Any]] = []
    beliefs_self_init = config.get("beliefs_self_init", config["spawn_probs_true"])
    
    for mismatch_k in config["mismatch_k"]:
        for seed in seeds:
            episode_config = _base_episode_config(
                config,
                seed,
                task=config["task"],
                scenario=config["scenario"],
                tom_mode="off",
            )
            episode_config.update(
                {
                    "mismatch_k": mismatch_k,
                    "spawn_probs_true": dict(config["spawn_probs_true"]),
                    "social_enabled": False,
                    "self_learning": True,  # Always enabled
                    "n_agents": 2,
                    # Per-agent configurations with beliefs_init
                    "agents": [
                        {
                            # Agent 0: Learner (self-learning only)
                            "agent_id": 0,
                            "self_learning": True,
                            "use_tom": False,
                            "social_enabled": False,
                            "beliefs_init": dict(beliefs_self_init),
                        },
                        {
                            # Agent 1: Expert (params derived from mismatch_k and sigma)
                            "agent_id": 1,
                            "self_learning": False,
                            "use_tom": False,
                            "social_enabled": False,
                            # k/sigma format for phase diagram mismatch computation
                            "beliefs_init": {
                                "k": mismatch_k,
                                "sigma": float(config["expert_sigma"]),
                            },
                        },
                    ],
                }
            )
            grid.append(episode_config)
    return grid


def _build_mega_sweep_grid(config: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    """
    Build a grid of episode configurations sweeping combinations of weight and mismatch parameters.
    
    Generates one episode configuration per combination of:
    - task and scenario from config["tasks"],
    - mismatch severity entries from config["mismatch_severities"],
    - values for social_enabled, eta_0, T_a, u_0, kappa,
    - each seed in `seeds`.
    
    Parameters:
        config (Dict[str, Any]): Experiment configuration. Expected keys used:
            - "tasks": list of {"task": str, "scenario": str}
            - "scenario_truth_params": mapping task -> scenario -> truth params
            - "mismatch_severities": list of dicts; each dict may contain per-task expert params and a "name"
            - Optional sweeps (each may be a single value or list): "social_enabled", "eta_0", "T_a", "u_0", "kappa"
            - Optional: "name" (experiment name), "tom_mode", "beta_tom"
        seeds (List[int]): Seed values to include in the sweep.
    
    Returns:
        List[Dict[str, Any]]: A list of episode configuration dictionaries, each containing base fields
        (task, scenario, seed, experiment_name, tom_mode, beta_tom) plus sweep-specific fields such as
        "mismatch", "mismatch_severity", "spawn_probs_true", "beliefs_expert", "beliefs_self_init",
        "social_enabled", "eta_0", "T_a", "u_0", and "kappa".
    """
    grid: List[Dict[str, Any]] = []
    
    # Get parameter sweeps (defaults if not specified)
    social_enabled_values = config.get("social_enabled", [True, False])
    if not isinstance(social_enabled_values, list):
        social_enabled_values = [social_enabled_values]
    
    eta_0_values = config.get("eta_0", [0.1])
    if not isinstance(eta_0_values, list):
        eta_0_values = [eta_0_values]
    
    T_a_values = config.get("T_a", [2.0])
    if not isinstance(T_a_values, list):
        T_a_values = [T_a_values]
    
    u_0_values = config.get("u_0", [0.05])
    if not isinstance(u_0_values, list):
        u_0_values = [u_0_values]
    
    kappa_values = config.get("kappa", [0.05])
    if not isinstance(kappa_values, list):
        kappa_values = [kappa_values]
    
    mismatch_severities = config.get("mismatch_severities", [])
    if not mismatch_severities:
        # Fallback to single mismatch if not specified
        mismatch_severities = [{"name": "default", "clean_up": {}}]
    
    for task_cfg in config["tasks"]:
        task = task_cfg["task"]
        scenario = task_cfg["scenario"]
        truth = config.get("scenario_truth_params", {}).get(task, {}).get(scenario, {})
        if not truth:
            raise ValueError(
                f"Missing truth params for task={task} scenario={scenario}. "
                f"Available tasks: {list(config.get('scenario_truth_params', {}).keys())}"
            )
        
        for mismatch_sev in mismatch_severities:
            expert = mismatch_sev.get(task, {})
            if not expert:
                # Skip if mismatch not defined for this task
                continue
            
            for social_enabled in social_enabled_values:
                # Set tom_mode based on social_enabled
                if social_enabled:
                    tom_mode = config.get("tom_mode", "softmax")
                else:
                    tom_mode = "off"
                
                for eta_0 in eta_0_values:
                    for T_a in T_a_values:
                        for u_0 in u_0_values:
                            for kappa in kappa_values:
                                for seed in seeds:
                                    episode_config = _base_episode_config(
                                        config,
                                        seed,
                                        task=task,
                                        scenario=scenario,
                                        experiment_name=config.get("name", "mega_sweep"),
                                        tom_mode=tom_mode,
                                        beta_tom=config.get("beta_tom", 2.0),
                                    )
                                    episode_config.update(
                                        {
                                            "mismatch": True,
                                            "mismatch_severity": mismatch_sev.get("name", "default"),
                                            "spawn_probs_true": truth,
                                            "social_enabled": social_enabled,
                                            "eta_0": float(eta_0),
                                            "T_a": float(T_a),
                                            "u_0": float(u_0),
                                            "kappa": float(kappa),
                                            "n_agents": 2,
                                            # Per-agent configurations with beliefs_init
                                            "agents": [
                                                {
                                                    # Agent 0: Learner
                                                    "agent_id": 0,
                                                    "self_learning": True,
                                                    "use_tom": social_enabled,
                                                    "social_enabled": social_enabled,
                                                    "beliefs_init": {**dict(truth), "concentration": 2.0},
                                                    "eta_0": float(eta_0),
                                                    "T_a": float(T_a),
                                                    "u_0": float(u_0),
                                                    "kappa": float(kappa),
                                                },
                                                {
                                                    # Agent 1: Expert with mismatched params
                                                    "agent_id": 1,
                                                    "self_learning": False,
                                                    "use_tom": False,
                                                    "social_enabled": False,
                                                    "beliefs_init": {**dict(expert), "concentration": expert.get("concentration", 100.0)},
                                                },
                                            ],
                                        }
                                    )
                                    grid.append(episode_config)
    return grid


def _build_learning_conditions_sweep_grid(config: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    """
    Builds a grid of episode configurations that sweep learning conditions (no learning, self-only, social-only, both) and relevant social-learning parameters.
    
    Parameters:
        config (Dict[str, Any]): Experiment configuration. Expected keys:
            - "tasks": list of {"task": str, "scenario": str}.
            - "scenario_truth_params": mapping task -> scenario -> truth params (required).
            - Optional sweeps/overrides: "learning_conditions", "eta_0", "T_a", "u_0", "kappa", "mismatch_severities".
            - Optional defaults: "name", "tom_mode", "beta_tom".
        seeds (List[int]): List of integer seeds to generate episode variants for each parameter combination.
    
    Returns:
        List[Dict[str, Any]]: List of episode configuration dictionaries, each containing base fields plus:
            - "mismatch", "mismatch_severity", "learning_condition",
            - "spawn_probs_true", "beliefs_expert", "beliefs_self_init",
            - "self_learning", "social_enabled",
            - and social hyperparameters ("eta_0", "T_a", "u_0", "kappa") when applicable.
    
    Raises:
        ValueError: If truth parameters for any task/scenario listed in config["tasks"] are missing.
    """
    grid: List[Dict[str, Any]] = []
    
    # Get learning conditions
    learning_conditions = config.get("learning_conditions", [])
    if not learning_conditions:
        # Default: all four conditions
        learning_conditions = [
            {"name": "no_learning", "self_learning": False, "social_enabled": False},
            {"name": "self_only", "self_learning": True, "social_enabled": False},
            {"name": "social_only", "self_learning": False, "social_enabled": True},
            {"name": "both", "self_learning": True, "social_enabled": True},
        ]
    
    # Get parameter sweeps (only relevant when social learning is enabled)
    eta_0_values = config.get("eta_0", [0.1])
    if not isinstance(eta_0_values, list):
        eta_0_values = [eta_0_values]
    
    T_a_values = config.get("T_a", [2.0])
    if not isinstance(T_a_values, list):
        T_a_values = [T_a_values]
    
    u_0_values = config.get("u_0", [0.05])
    if not isinstance(u_0_values, list):
        u_0_values = [u_0_values]
    
    kappa_values = config.get("kappa", [0.05])
    if not isinstance(kappa_values, list):
        kappa_values = [kappa_values]
    
    mismatch_severities = config.get("mismatch_severities", [])
    if not mismatch_severities:
        # Fallback to single mismatch if not specified
        mismatch_severities = [{"name": "default", "clean_up": {}}]
    
    for task_cfg in config["tasks"]:
        task = task_cfg["task"]
        scenario = task_cfg["scenario"]
        truth = config.get("scenario_truth_params", {}).get(task, {}).get(scenario, {})
        if not truth:
            raise ValueError(
                f"Missing truth params for task={task} scenario={scenario}. "
                f"Available tasks: {list(config.get('scenario_truth_params', {}).keys())}"
            )
        
        for mismatch_sev in mismatch_severities:
            expert = mismatch_sev.get(task, {})
            if not expert:
                # Skip if mismatch not defined for this task
                continue
            
            for learning_cond in learning_conditions:
                self_learning = learning_cond.get("self_learning", True)
                social_enabled = learning_cond.get("social_enabled", False)
                cond_name = learning_cond.get("name", "unknown")
                
                # If social learning is disabled, only one parameter combination (no social params matter)
                if not social_enabled:
                    for seed in seeds:
                        episode_config = _base_episode_config(
                            config,
                            seed,
                            task=task,
                            scenario=scenario,
                            experiment_name=config.get("name", "learning_conditions_sweep"),
                            tom_mode="off",
                            beta_tom=config.get("beta_tom", 2.0),
                        )
                        episode_config.update(
                            {
                                "mismatch": True,
                                "mismatch_severity": mismatch_sev.get("name", "default"),
                                "learning_condition": cond_name,
                                "spawn_probs_true": truth,
                                "self_learning": self_learning,
                                "social_enabled": social_enabled,
                                # Social params don't matter when social is disabled, but set defaults
                                "eta_0": 0.1,
                                "T_a": 2.0,
                                "u_0": 0.05,
                                "kappa": 0.05,
                                "n_agents": 2,
                                # Per-agent configurations with beliefs_init
                                "agents": [
                                    {
                                        "agent_id": 0,
                                        "self_learning": self_learning,
                                        "use_tom": False,
                                        "social_enabled": False,
                                        "beliefs_init": {**dict(truth), "concentration": 2.0},
                                    },
                                    {
                                        "agent_id": 1,
                                        "self_learning": False,
                                        "use_tom": False,
                                        "social_enabled": False,
                                        "beliefs_init": {**dict(expert), "concentration": expert.get("concentration", 100.0)},
                                    },
                                ],
                            }
                        )
                        grid.append(episode_config)
                else:
                    # Social learning enabled: sweep over social parameters
                    for eta_0 in eta_0_values:
                        for T_a in T_a_values:
                            for u_0 in u_0_values:
                                for kappa in kappa_values:
                                    for seed in seeds:
                                        episode_config = _base_episode_config(
                                            config,
                                            seed,
                                            task=task,
                                            scenario=scenario,
                                            experiment_name=config.get("name", "learning_conditions_sweep"),
                                            tom_mode=config.get("tom_mode", "softmax"),
                                            beta_tom=config.get("beta_tom", 2.0),
                                        )
                                        episode_config.update(
                                            {
                                                "mismatch": True,
                                                "mismatch_severity": mismatch_sev.get("name", "default"),
                                                "learning_condition": cond_name,
                                                "spawn_probs_true": truth,
                                                "self_learning": self_learning,
                                                "social_enabled": social_enabled,
                                                "eta_0": float(eta_0),
                                                "T_a": float(T_a),
                                                "u_0": float(u_0),
                                                "kappa": float(kappa),
                                                "n_agents": 2,
                                                # Per-agent configurations with beliefs_init
                                                "agents": [
                                                    {
                                                        "agent_id": 0,
                                                        "self_learning": self_learning,
                                                        "use_tom": social_enabled,
                                                        "social_enabled": social_enabled,
                                                        "beliefs_init": {**dict(truth), "concentration": 2.0},
                                                        "eta_0": float(eta_0),
                                                        "T_a": float(T_a),
                                                        "u_0": float(u_0),
                                                        "kappa": float(kappa),
                                                    },
                                                    {
                                                        "agent_id": 1,
                                                        "self_learning": False,
                                                        "use_tom": False,
                                                        "social_enabled": False,
                                                        "beliefs_init": {**dict(expert), "concentration": expert.get("concentration", 100.0)},
                                                    },
                                                ],
                                            }
                                        )
                                        grid.append(episode_config)
    return grid


_GRID_BUILDERS = {
    "smoke": _build_smoke_grid,
    "killer_test": _build_killer_test_grid,
    "phase_diagram": _build_phase_diagram_grid,
    "phase_diagram_baseline": _build_phase_diagram_baseline_grid,
    "mega_sweep": _build_mega_sweep_grid,
    "baseline_vs_social": _build_baseline_vs_social_grid,
    "quick_test": _build_baseline_vs_social_grid,  # use same grid builder
    "test_self_learning": _build_baseline_vs_social_grid,  # use same grid builder for simple test
}


def validate_episode_configs(grid: List[Dict[str, Any]]) -> None:
    """
    Validate that all episode configs have required fields.
    
    Ensures that tom_mode is always present in episode configurations,
    as it's required for proper inference of use_tom in analysis scripts.
    
    Args:
        grid: List of episode configuration dictionaries
        
    Raises:
        ValueError: If any episode config is missing tom_mode
    """
    for i, ep_config in enumerate(grid):
        if "tom_mode" not in ep_config:
            raise ValueError(
                f"Episode config at index {i} is missing required field 'tom_mode'. "
                f"All episode configs must have tom_mode set (e.g., 'off', 'softmax', 'greedy'). "
                f"Config keys: {list(ep_config.keys())}"
            )


def create_experiment_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create parameter grid from experiment configuration.
    
    Each experiment type has its own grid builder function that generates
    all combinations of parameters (tasks, scenarios, seeds, etc.)
    as individual episode configurations.
    
    All episode configs are validated to ensure tom_mode is present.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        List of episode configuration dictionaries (all validated)
        
    Raises:
        ValueError: If experiment name has no grid builder, or if validation fails
    """
    name = config.get("name", "")
    seeds = _resolve_seeds(config)
    builder = _GRID_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown experiment config: {name}")
    grid = builder(config, seeds)
    validate_episode_configs(grid)
    return grid


def compute_expected_episodes(config: Dict[str, Any]) -> int:
    """Compute expected episode count for a config."""
    return len(create_experiment_grid(config))


def validate_episode_count(config: Dict[str, Any]) -> bool:
    """
    Validate that computed episode count matches expected_episodes.

    Returns:
        True if counts match, raises ValueError if mismatch.
    """
    expected = config.get("expected_episodes")
    if expected is None:
        return True

    computed = compute_expected_episodes(config)
    if computed != expected:
        raise ValueError(
            f"Episode count mismatch for {config.get('name', 'unknown')}: "
            f"expected={expected}, computed={computed}"
        )
    return True
