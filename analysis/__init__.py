"""
Analysis module for experiment results.

Raw functions for data processing and analysis (no execution logic).
"""

from __future__ import annotations

# io - data loading
from analysis.io import (
    load_experiment_results,
    load_all_experiments,
    scan_results_directory,
)

# metrics - metric computation
from analysis.metrics import (
    analyze_basic_metrics,
    analyze_parameter_breakdowns,
    analyze_social_learning,
    compute_statistics,
)

# discovery - discovery analysis
from analysis.discovery import (
    compute_discovery_success_rate,
    plot_discovery_trajectories,
)

# social - social learning & ToM analysis
from analysis.social import (
    compute_tom_accuracy_vs_belief_similarity,
)

# comparison - baseline comparison utilities
from analysis.comparison import (
    build_join_key,
    index_by_join_key,
    join_baseline_results,
)

# regimes - regime analysis
from analysis.regimes import (
    analyze_regime_labels,
)

# reports - report generation
from analysis.reports import (
    format_analysis_summary,
    generate_cross_experiment_comparison,
    generate_findings_report,
)

# visuals - plotting utilities
from analysis.visuals import (
    plot_timeseries_panel,
    plot_aggregate_panel,
    plot_regime_heatmap,
    plot_error_heatmap,
    plot_error_vs_trust_gate_comparison,
    plot_tom_reliability_analysis,
)

__all__ = [
    # io - loaders
    "load_experiment_results",
    "load_all_experiments",
    "scan_results_directory",
    # metrics
    "analyze_basic_metrics",
    "analyze_parameter_breakdowns",
    "analyze_social_learning",
    "compute_statistics",
    # discovery
    "compute_discovery_success_rate",
    "plot_discovery_trajectories",
    # social/ToM
    "compute_tom_accuracy_vs_belief_similarity",
    # comparison
    "build_join_key",
    "index_by_join_key",
    "join_baseline_results",
    # regimes
    "analyze_regime_labels",
    # reports
    "format_analysis_summary",
    "generate_cross_experiment_comparison",
    "generate_findings_report",
    # visuals
    "plot_timeseries_panel",
    "plot_aggregate_panel",
    "plot_regime_heatmap",
    "plot_error_heatmap",
    "plot_error_vs_trust_gate_comparison",
    "plot_tom_reliability_analysis",
]
