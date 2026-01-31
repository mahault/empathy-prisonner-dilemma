#!/usr/bin/env python3
"""
Unified analysis script for experiment results.

Scans the results/ directory and generates analysis outputs in analysis/ subfolders.
Supports CLI flags to enable/disable different analysis types.
"""

from __future__ import annotations

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis import (
    load_experiment_results,
    load_all_experiments,
    scan_results_directory,
    analyze_basic_metrics,
    analyze_parameter_breakdowns,
    analyze_social_learning,
    analyze_regime_labels,
    compute_discovery_success_rate,
    compute_tom_accuracy_vs_belief_similarity,
    plot_discovery_trajectories,
)


def save_analysis_output(
    analysis_dir: Path,
    filename: str,
    data: Any,
    is_json: bool = True,
) -> None:
    """Save analysis output to file."""
    if is_json:
        output_path = analysis_dir / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  ✓ Saved {filename}")
    else:
        output_path = analysis_dir / filename
        with open(output_path, "w") as f:
            f.write(data)
        print(f"  ✓ Saved {filename}")


def analyze_single_run(
    run_dir: Path,
    enabled_analyses: Dict[str, bool],
) -> Dict[str, Any]:
    """Analyze a single run directory."""
    print(f"\nAnalyzing: {run_dir.relative_to(PROJECT_ROOT)}")

    # Load results
    results = load_experiment_results(run_dir)
    if not results:
        print(f"  Warning: No run results found")
        return {}

    print(f"  Loaded {len(results)} run(s)")

    # Determine experiment directory (parent of data/ or parent of run_dir)
    # If run_dir is in data/, go up two levels; otherwise up one level
    if run_dir.parent.name == "data":
        experiment_dir = run_dir.parent.parent
    else:
        # Backward compatibility: if run_dir is directly under experiment, go up one level
        experiment_dir = run_dir.parent

    # Create analysis directory structure: analysis/{run_name}/
    run_name = run_dir.name
    analysis_dir = experiment_dir / "analysis" / run_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = {}

    # Basic metrics - read from timeseries.json metadata instead of recomputing
    if enabled_analyses.get("basic", False):
        print("  Loading basic metrics from timeseries.json...")
        try:
            # Read metrics directly from timeseries.json metadata
            timeseries_file = run_dir / "timeseries.json"
            if timeseries_file.exists():
                with open(timeseries_file) as f:
                    data = json.load(f)
                metadata = data.get("metadata", {})
                metrics = metadata.get("metrics", {})
                outcomes = metadata.get("outcomes", {})

                # Build basic_metrics from existing metadata
                basic_metrics = {
                    "errors": [metrics.get("final_error")] if metrics.get("final_error") is not None else [],
                    "regimes": {outcomes.get("regime_label", "Unknown"): 1},
                    "drift_metrics": [{
                        "dist_to_expert_initial": metrics.get("dist_to_expert_initial"),
                        "dist_to_expert_final": metrics.get("dist_to_expert_final"),
                        "dist_to_true_initial": metrics.get("dist_to_true_initial"),
                        "dist_to_true_final": metrics.get("dist_to_true_final"),
                        "drift_toward_expert": metrics.get("drift_toward_expert", False),
                    }],
                    "stability_metrics": [{
                        "reliability_variance": metrics.get("reliability_variance"),
                        "n_eff_min": metrics.get("n_eff_min"),
                        "n_eff_collapse": metrics.get("n_eff_collapse", False),
                    }],
                    "error_summary": None,
                }

                # Compute error_summary if we have errors
                errors = [e for e in basic_metrics["errors"] if e is not None and not np.isnan(e)]
                if errors:
                    from analysis import compute_statistics
                    basic_metrics["error_summary"] = compute_statistics(errors)
            else:
                # Fallback to computing if timeseries.json doesn't exist
                basic_metrics = analyze_basic_metrics(results)

            all_outputs["basic_metrics"] = basic_metrics
            save_analysis_output(analysis_dir, "basic_metrics.json", basic_metrics)
        except Exception as e:
            print(f"  Warning: Failed to load basic metrics: {e}")

    # Parameter breakdowns
    if enabled_analyses.get("parameters", False):
        print("  Computing parameter breakdowns...")
        try:
            param_breakdowns = analyze_parameter_breakdowns(results)
            all_outputs["parameter_breakdowns"] = param_breakdowns
            save_analysis_output(analysis_dir, "parameter_breakdowns.json", param_breakdowns)
        except Exception as e:
            print(f"  Warning: Failed to compute parameter breakdowns: {e}")

    # Social learning stats
    if enabled_analyses.get("social", False):
        print("  Computing social learning stats...")
        try:
            social_stats = analyze_social_learning(results)
            all_outputs["social_learning_stats"] = social_stats
            save_analysis_output(analysis_dir, "social_learning_stats.json", social_stats)
        except Exception as e:
            print(f"  Warning: Failed to compute social learning stats: {e}")

    # Regime analysis (baseline-aware)
    if enabled_analyses.get("regimes", False):
        print("  Computing regime analysis...")
        try:
            regime_data = analyze_regime_labels(results)
            per_episode = regime_data.pop("per_episode", [])
            all_outputs["regimes"] = regime_data
            save_analysis_output(analysis_dir, "regimes.json", regime_data)
            save_analysis_output(analysis_dir, "per_episode_regimes.json", per_episode)
            per_episode_by_path = {
                entry.get("episode_path"): entry.get("regime_label")
                for entry in per_episode
                if entry.get("episode_path") is not None
            }
            for result in results:
                episode_path = result.get("episode_path")
                if episode_path in per_episode_by_path:
                    result.setdefault("outcomes", {})["regime_label"] = per_episode_by_path[episode_path]
        except Exception as e:
            print(f"  Warning: Failed to compute regime analysis: {e}")

    # Discovery metrics
    if enabled_analyses.get("discovery", False):
        print("  Computing discovery metrics...")
        try:
            discovery_metrics = compute_discovery_success_rate(results)
            all_outputs["discovery_metrics"] = discovery_metrics
            save_analysis_output(analysis_dir, "discovery_metrics.json", discovery_metrics)

            # Generate plot in plots subfolder
            plots_dir = analysis_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            plot_path = plots_dir / "discovery_trajectories.png"
            plot_discovery_trajectories(results, str(plot_path))
            if plot_path.exists():
                print(f"  ✓ Saved discovery_trajectories.png")
        except Exception as e:
            print(f"  Warning: Failed to compute discovery metrics: {e}")

    # ToM echo chamber analysis
    if enabled_analyses.get("tom_echo", False):
        print("  Computing ToM echo chamber analysis...")
        try:
            tom_analysis = compute_tom_accuracy_vs_belief_similarity(results)
            # Convert numpy arrays to lists for JSON serialization
            tom_data = tom_analysis.copy()
            for key in ["tom_accuracy_trajectory", "belief_similarity_trajectory"]:
                if key in tom_data and isinstance(tom_data[key], list):
                    tom_data[key] = [float(x) if hasattr(x, "__float__") else x for x in tom_data[key]]
            all_outputs["tom_echo_chamber"] = tom_data
            save_analysis_output(analysis_dir, "tom_echo_chamber.json", tom_data)
        except Exception as e:
            print(f"  Warning: Failed to compute ToM echo chamber analysis: {e}")

    # Generate summary
    summary = {
        "run_directory": str(run_dir),
        "analysis_timestamp": datetime.now().isoformat(),
        "n_runs": len(results),
    }

    # Add summaries from analyses
    if "basic_metrics" in all_outputs:
        basic = all_outputs["basic_metrics"]
        if basic.get("error_summary"):
            summary["error_summary"] = basic["error_summary"]
        if basic.get("regimes"):
            summary["regimes"] = dict(basic["regimes"])

    if "regimes" in all_outputs:
        summary["regimes"] = dict(all_outputs["regimes"].get("regimes", {}))

    if "discovery_metrics" in all_outputs:
        summary["discovery_success_rate"] = all_outputs["discovery_metrics"].get("discovery_success_rate")

    if "tom_echo_chamber" in all_outputs:
        summary["echo_chamber_detected"] = all_outputs["tom_echo_chamber"].get("echo_chamber_detected")
        summary["tom_correlation"] = all_outputs["tom_echo_chamber"].get("correlation")

    save_analysis_output(analysis_dir, "summary.json", summary)

    if enabled_analyses.get("plots", False):
        print("  Generating plots...")
        try:
            from analysis.visuals import (
                plot_timeseries_panel,
                plot_aggregate_panel,
                plot_regime_heatmap,
                plot_error_heatmap,
                plot_error_vs_trust_gate_comparison,
            )
        except Exception as e:
            print(f"  Warning: Failed to import plotting utilities: {e}")
        else:
            # Per-run plots go in analysis/{run_name}/plots/
            plots_dir = analysis_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            plot_timeseries_panel(results, seeds_to_plot=3, output_dir=plots_dir)

            sample_config = results[0].get("config", {}) if results else {}
            if "mismatch_k" in sample_config:
                # Note: tau_comp_threshold removed, using T_a instead
                plot_aggregate_panel(
                    results, factor_x="T_a", factor_y="mismatch_k", output_dir=plots_dir
                )
                plot_regime_heatmap(
                    results, x_axis="T_a", y_axis="mismatch_k", output_dir=plots_dir
                )
                plot_error_heatmap(
                    results, x_axis="T_a", y_axis="mismatch_k", output_dir=plots_dir
                )

            if run_dir.parent.name == "killer_test":
                plot_error_vs_trust_gate_comparison(results, output_dir=plots_dir)

    return all_outputs


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses on all experiments
  python scripts/run_analysis.py --all

  # Run only basic metrics on specific experiment
  python scripts/run_analysis.py --basic --experiment smoke

  # Run discovery and ToM analysis on latest runs
  python scripts/run_analysis.py --discovery --tom-echo --latest
        """,
    )

    # Analysis type flags
    parser.add_argument("--basic", action="store_true", help="Basic metrics (errors, regimes, stability)")
    parser.add_argument("--parameters", action="store_true", help="Parameter breakdowns")
    parser.add_argument("--social", action="store_true", help="Social learning stats")
    parser.add_argument("--regimes", action="store_true", help="Regime analysis (baseline-aware)")
    parser.add_argument("--discovery", action="store_true", help="Discovery success rate and trajectories")
    parser.add_argument("--tom-echo", action="store_true", help="ToM echo chamber detection")
    parser.add_argument("--plots", action="store_true", help="Generate plots in analysis/plots/")
    parser.add_argument("--findings", action="store_true", help="Comprehensive findings report")
    parser.add_argument("--cross-exp", action="store_true", help="Cross-experiment comparisons")
    parser.add_argument("--all", action="store_true", help="Run all analyses (default if no flags specified)")

    # Target selection
    parser.add_argument("--experiment", type=str, help="Analyze specific experiment")
    parser.add_argument("--run-id", type=str, help="Analyze specific run ID")
    parser.add_argument("--all-experiments", action="store_true", help="Analyze all experiments (default)")
    parser.add_argument("--latest", action="store_true", help="Analyze most recent run per experiment")

    args = parser.parse_args()

    # Determine enabled analyses
    enabled_analyses = {
        "basic": args.basic or args.all,
        "parameters": args.parameters or args.all,
        "social": args.social or args.all,
        "regimes": args.regimes or args.all,
        "discovery": args.discovery or args.all,
        "tom_echo": args.tom_echo or args.all,
        "plots": args.plots,
    }

    # If no analysis flags specified, enable all
    if not any(enabled_analyses.values()):
        enabled_analyses = {k: True for k in enabled_analyses}
        print("No analysis flags specified, running all analyses")

    # Determine target runs
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    target_runs: List[Path] = []

    if args.run_id and args.experiment:
        # Specific run - check in data/ subfolder first, then fallback to old structure
        exp_dir = results_dir / args.experiment
        data_dir = exp_dir / "data"
        run_dir = data_dir / args.run_id if data_dir.exists() else exp_dir / args.run_id
        if run_dir.exists() and (run_dir / "timeseries.json").exists():
            target_runs.append(run_dir)
        else:
            print(f"Error: Run directory not found: {run_dir}")
            return 1
    elif args.experiment:
        # All runs for specific experiment
        exp_dir = results_dir / args.experiment
        if not exp_dir.exists():
            print(f"Error: Experiment directory not found: {exp_dir}")
            return 1

        # Check in data/ subfolder first, fallback to old structure
        data_dir = exp_dir / "data"
        search_dir = data_dir if data_dir.exists() else exp_dir

        if args.latest:
            # Most recent run
            run_dirs = [d for d in search_dir.iterdir() if d.is_dir() and (d / "timeseries.json").exists()]
            if run_dirs:
                target_runs.append(max(run_dirs, key=lambda p: p.stat().st_mtime))
        else:
            # All runs
            for run_dir in search_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "timeseries.json").exists():
                    target_runs.append(run_dir)
    else:
        # All experiments
        if args.latest:
            # Most recent run per experiment
            for exp_dir in results_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                # Check in data/ subfolder first, fallback to old structure
                data_dir = exp_dir / "data"
                search_dir = data_dir if data_dir.exists() else exp_dir
                run_dirs = [d for d in search_dir.iterdir() if d.is_dir() and (d / "timeseries.json").exists()]
                if run_dirs:
                    target_runs.append(max(run_dirs, key=lambda p: p.stat().st_mtime))
        else:
            # All runs
            target_runs = scan_results_directory(results_dir)

    if not target_runs:
        print("No target runs found")
        return 1

    print(f"Found {len(target_runs)} run(s) to analyze")

    # Group runs by experiment for collective analysis
    runs_by_experiment: Dict[str, List[Path]] = {}
    for run_dir in target_runs:
        # Extract experiment name (handle both new structure with data/ and old structure)
        if run_dir.parent.name == "data":
            exp_name = run_dir.parent.parent.name
            exp_dir = run_dir.parent.parent
        else:
            exp_name = run_dir.parent.name
            exp_dir = run_dir.parent

        if exp_name not in runs_by_experiment:
            runs_by_experiment[exp_name] = []
        runs_by_experiment[exp_name].append((run_dir, exp_dir))

    # Analyze each run
    all_analyses: Dict[str, Dict[str, Any]] = {}

    for exp_name, run_list in runs_by_experiment.items():
        exp_dir = run_list[0][1]  # Get experiment directory from first run
        all_run_results = []

        for run_dir, _ in run_list:
            outputs = analyze_single_run(run_dir, enabled_analyses)

            # Store for cross-experiment analysis
            if exp_name not in all_analyses:
                all_analyses[exp_name] = {
                    "n_runs": 0,
                    "error_summary": None,
                    "regimes": {},
                }

            # Merge outputs into experiment-level analysis
            if "basic_metrics" in outputs:
                basic = outputs["basic_metrics"]
                all_analyses[exp_name]["n_runs"] += 1
                if basic.get("error_summary"):
                    # Merge error summaries (simplified)
                    if all_analyses[exp_name]["error_summary"] is None:
                        all_analyses[exp_name]["error_summary"] = basic["error_summary"]
                if basic.get("regimes"):
                    for regime, count in basic["regimes"].items():
                        all_analyses[exp_name]["regimes"][regime] = all_analyses[exp_name]["regimes"].get(regime, 0) + count

            # Collect results for collective analysis
            run_results = load_experiment_results(run_dir)
            if run_results:
                all_run_results.extend(run_results)

        # Generate collective analysis for this experiment
        if all_run_results and any(enabled_analyses.values()):
            print(f"\nGenerating collective analysis for {exp_name}...")
            collective_analysis_dir = exp_dir / "analysis"
            collective_analysis_dir.mkdir(parents=True, exist_ok=True)

            # Collective basic metrics
            if enabled_analyses.get("basic", False):
                try:
                    collective_basic = analyze_basic_metrics(all_run_results)
                    save_analysis_output(collective_analysis_dir, "aggregated_metrics.json", collective_basic)
                except Exception as e:
                    print(f"  Warning: Failed to compute collective basic metrics: {e}")

            # Collective parameter breakdowns
            if enabled_analyses.get("parameters", False):
                try:
                    collective_param = analyze_parameter_breakdowns(all_run_results)
                    save_analysis_output(collective_analysis_dir, "aggregated_parameter_breakdowns.json", collective_param)
                except Exception as e:
                    print(f"  Warning: Failed to compute collective parameter breakdowns: {e}")

            # Collective social learning stats
            if enabled_analyses.get("social", False):
                try:
                    collective_social = analyze_social_learning(all_run_results)
                    save_analysis_output(collective_analysis_dir, "aggregated_social_learning.json", collective_social)
                except Exception as e:
                    print(f"  Warning: Failed to compute collective social learning stats: {e}")

            # Collective plots
            if enabled_analyses.get("plots", False):
                try:
                    from analysis.visuals import (
                        plot_aggregate_panel,
                        plot_regime_heatmap,
                        plot_error_heatmap,
                    )
                    collective_plots_dir = collective_analysis_dir / "plots"
                    collective_plots_dir.mkdir(exist_ok=True)

                    sample_config = all_run_results[0].get("config", {}) if all_run_results else {}
                    if "mismatch_k" in sample_config:
                        # Note: tau_comp_threshold removed, using T_a instead
                        plot_aggregate_panel(
                            all_run_results, factor_x="T_a", factor_y="mismatch_k", output_dir=collective_plots_dir
                        )
                        plot_regime_heatmap(
                            all_run_results, x_axis="T_a", y_axis="mismatch_k", output_dir=collective_plots_dir
                        )
                        plot_error_heatmap(
                            all_run_results, x_axis="T_a", y_axis="mismatch_k", output_dir=collective_plots_dir
                        )
                except Exception as e:
                    print(f"  Warning: Failed to generate collective plots: {e}")

    # Generate cross-experiment reports if requested
    if args.findings or args.cross_exp:
        print("\n" + "=" * 60)
        print("Generating cross-experiment reports...")
        print("=" * 60)

        # Load all experiments for comprehensive analysis
        all_results = load_all_experiments(results_dir)

        # Build comprehensive analysis structure
        comprehensive_analyses = {}
        for exp_name, results in all_results.items():
            if results:
                basic = analyze_basic_metrics(results)
                param = analyze_parameter_breakdowns(results)
                social = analyze_social_learning(results)

                comprehensive_analyses[exp_name] = {
                    "n_runs": len(results),
                    "error_summary": basic.get("error_summary"),
                    "regimes": dict(basic.get("regimes", {})),
                    "task_breakdown": param.get("task_breakdown", {}),
                    "use_tom_breakdown": param.get("use_tom_breakdown", {}),
                    "eta_0_breakdown": param.get("eta_0_breakdown", {}),
                    "social_learning_stats": social,
                    "stability_metrics": basic.get("stability_metrics", []),
                }

        if args.findings:
            print("  Generating findings report...")
            try:
                findings_report = generate_findings_report(comprehensive_analyses)
                findings_path = results_dir / "findings_report.md"
                with open(findings_path, "w") as f:
                    f.write(findings_report)
                print(f"  ✓ Saved findings report to {findings_path}")
            except (IOError, OSError) as e:
                print(f"  Warning: Failed to write findings report file: {e}")
                print(traceback.format_exc())
            except (KeyError, ValueError) as e:
                print(f"  Warning: Failed to generate findings report due to data issue: {e}")
                print(traceback.format_exc())
            except Exception as e:
                print(f"  Warning: Failed to generate findings report: {e}")
                print(traceback.format_exc())

        if args.cross_exp:
            print("  Generating cross-experiment comparison...")
            try:
                cross_exp_report = generate_cross_experiment_comparison(comprehensive_analyses)
                cross_exp_path = results_dir / "cross_experiment_comparison.md"
                with open(cross_exp_path, "w") as f:
                    f.write(cross_exp_report)
                print(f"  ✓ Saved cross-experiment comparison to {cross_exp_path}")
            except (IOError, OSError) as e:
                print(f"  Warning: Failed to write cross-experiment comparison file: {e}")
                print(traceback.format_exc())
            except (KeyError, ValueError) as e:
                print(f"  Warning: Failed to generate cross-experiment comparison due to data issue: {e}")
                print(traceback.format_exc())
            except Exception as e:
                print(f"  Warning: Failed to generate cross-experiment comparison: {e}")
                print(traceback.format_exc())

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
