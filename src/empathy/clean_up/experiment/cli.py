"""
Simple CLI for running clean_up experiments.

Merged from arguments.py and executor.py with deprecated flags removed.
"""

import argparse
import copy
from pathlib import Path
from empathy.clean_up.experiment.config import get_experiment_config
from empathy.clean_up.experiment.runner import ExperimentRunner


CORE_EXPERIMENTS = [
    "smoke",
    "killer_test",
    "phase_diagram",
    "phase_diagram_baseline",
]


def _load_available_experiments() -> list[str]:
    """Load available experiment names from configs."""
    from empathy.clean_up.experiment.config import EXPERIMENT_CONFIGS
    return sorted(EXPERIMENT_CONFIGS.keys())


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the experiment runner.
    
    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            experiment (str): Experiment name or "all".
            n_workers (int): Number of parallel workers.
            output (str|None): Output directory or None to use the default pattern.
            seeds (list[int]|None): List of seed integers to override config, or None.
            verbose (bool): Whether to show detailed progress.
            dry_run (bool): Whether to print the configuration without running.
            debug (bool): Whether to enable debug logging.
            numba_threads (int|None): Number of threads for Numba parallelization.
    """
    available_experiments = _load_available_experiments()
    
    parser = argparse.ArgumentParser(description="Run clean_up experiments")

    parser.add_argument(
        "--experiment",
        type=str,
        default="smoke",
        choices=available_experiments + ["all"],
        help="Experiment to run (default: smoke; use 'all' for full suite)",
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/<experiment>/)",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override seeds (default: use experiment config)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to debug.log files",
    )

    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="Number of threads for Numba parallelization (default: auto-detect from CPU count, or NUMBA_NUM_THREADS env var)",
    )

    return parser.parse_args()


def _resolve_experiment_names(experiment: str) -> list[str]:
    """Resolve experiment name to list of experiment names."""
    return CORE_EXPERIMENTS if experiment == "all" else [experiment]


def _build_config_overrides(args: argparse.Namespace) -> dict:
    """
    Assemble configuration overrides derived from parsed CLI arguments.
    
    Parameters:
        args (argparse.Namespace): Parsed command-line arguments. Recognized fields:
            - seeds: an optional iterable of integer seeds to override experiment seeds.
            - debug: a boolean flag to enable debug mode.
    
    Returns:
        dict: A mapping of override keys to values:
            - "n_seeds": integer count of provided seeds (present if `seeds` was given).
            - "seeds": list of provided seed integers (present if `seeds` was given).
            - "debug": `True` if the debug flag was set, omitted otherwise.
    """
    overrides: dict = {}
    if args.seeds is not None:
        overrides["n_seeds"] = len(args.seeds)
        overrides["seeds"] = list(args.seeds)
    if args.debug:
        overrides["debug"] = True
    return overrides


def print_experiment_info(args: argparse.Namespace, project_root: Path | None = None) -> None:
    """Print experiment information header."""
    print("=" * 60)
    print("Clean Up Experiment Runner")
    print("=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Workers: {args.n_workers}")
    if project_root:
        output_path = args.output or (project_root / 'results')
    else:
        output_path = args.output or 'results'
    print(f"Output: {output_path}")
    print()


def run_experiment_from_cli(args: argparse.Namespace) -> int:
    """
    Execute experiments specified by command-line arguments.
    
    Handles dry-run mode (prints resolved experiment configurations without executing),
    determines the output directory (defaults to "results" when not provided),
    runs each experiment with configured overrides, saves results, and prints a completion summary.
    
    Parameters:
        args (argparse.Namespace): Parsed CLI arguments. Expected attributes:
            - experiment: experiment name or pattern to run
            - dry_run: if True, only print resolved configurations
            - seeds: optional iterable of seed values
            - output: optional output directory path (string)
            - n_workers: number of worker processes to use
            - verbose: verbosity flag passed to the runner
    
    Returns:
        exit_code (int): 0 on success.
    """
    exp_names = _resolve_experiment_names(args.experiment)

    if args.dry_run:
        print("[DRY RUN] Would run experiment with above configuration")
        for exp_name in exp_names:
            config = copy.deepcopy(get_experiment_config(exp_name))
            if args.seeds is not None:
                config["n_seeds"] = len(args.seeds)
                config["seeds"] = list(args.seeds)
            print(f"- {exp_name}: {config}")
        return 0

    # Determine output directory
    output_dir = Path(args.output) if args.output else Path("results")

    for exp_name in exp_names:
        config_overrides = _build_config_overrides(args)
        
        runner = ExperimentRunner(
            exp_name,
            config_overrides=config_overrides,
        )
        
        results = runner.run(
            n_workers=args.n_workers,
            verbose=args.verbose,
            output_dir=output_dir,
        )
        
        run_dir = runner.save_results(str(output_dir))
        print(f"Completed {exp_name}: {len(results)} episodes -> {run_dir}")
    return 0


def main(project_root: Path | None = None) -> int:
    """Main entry point for CLI."""
    args = parse_arguments()
    print_experiment_info(args, project_root=project_root)
    return run_experiment_from_cli(args)
