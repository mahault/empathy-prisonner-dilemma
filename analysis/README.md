# Analysis Module

Deep analysis utilities for experiment results.

This module focuses on deeper analysis functions such as **discovery metrics**,
**echo chamber detection**, and **baseline comparisons**. For basic metrics like
error computation and regime labeling, see `experiment.metrics`. For
plotting utilities, see `analysis.visuals`.

**Theoretical Background:** See [`project_plan.txt`](../project_plan.txt) Section 7.1-7.2 (Analysis Plan) for discovery capability validation and echo chamber detection methodology.

## Structure

The analysis module is organized into submodules grouped by type of analysis/operation:

```
analysis/
├── __init__.py              # Main exports (backward compatible)
├── README.md                # This file
├── io/                      # Data loading (input/output)
│   ├── __init__.py
│   └── loaders.py           # Result loading utilities
├── metrics/                 # Metric computation
│   ├── __init__.py
│   └── computation.py       # Advanced metric computation
├── discovery/               # Discovery analysis
│   ├── __init__.py
│   └── analysis.py          # Discovery metrics and trajectory analysis
├── social/                  # Social learning & ToM analysis
│   ├── __init__.py
│   └── analysis.py          # Theory of Mind echo chamber analysis
├── comparison/              # Baseline comparison utilities
│   ├── __init__.py
│   └── baseline.py          # Baseline alignment and joining
├── regimes/                 # Regime analysis
│   ├── __init__.py
│   └── analysis.py          # Baseline-aware regime labeling
├── reports/                 # Report generation
│   ├── __init__.py
│   └── generation.py        # Report generation functions
└── visuals/                 # Plotting utilities
    ├── __init__.py
    ├── plotting.py
    └── README.md
```

## Submodules

### `io/` - Data Loading

Functions for loading experiment results from disk:
- `load_experiment_results(run_dir)` - Load timeseries.json from a run directory
- `scan_results_directory(results_dir)` - Find all result run directories
- `load_all_experiments(results_dir)` - Load all experiment results

### `metrics/` - Metric Computation

Functions for computing analysis metrics:
- `compute_statistics(errors)` - Compute error statistics (mean, std, min, max, etc.)
- `analyze_basic_metrics(results)` - Analyze basic metrics: errors, regimes, drift, stability
- `analyze_parameter_breakdowns(results)` - Analyze breakdowns by task, ToM, eta_0, etc.
- `analyze_social_learning(results)` - Analyze social learning stats: reliability, trust, influence

### `discovery/` - Discovery Analysis

Functions for analyzing discovery capabilities:
- `compute_discovery_success_rate(results, ground_truth_params=None, convergence_threshold=0.1)` - Compute discovery success metrics
- `plot_discovery_trajectories(results, output_path, title="Discovery Dynamics")` - Create 2×2 panel plot of discovery dynamics

### `social/` - Social Learning & ToM Analysis

Functions for analyzing social learning and Theory of Mind:
- `compute_tom_accuracy_vs_belief_similarity(results, expert_true_params=None)` - Analyze correlation between ToM accuracy and belief similarity, detects echo chamber effects

### `comparison/` - Baseline Comparison

Utilities for comparing results with baseline runs:
- `build_join_key(result)` - Build canonical join key for baseline alignment
- `index_by_join_key(results)` - Index results by join key for fast lookup
- `join_baseline_results(results, baseline_results)` - Join results with baseline results

### `regimes/` - Regime Analysis

Functions for analyzing behavioral regimes:
- `analyze_regime_labels(results)` - Compute baseline-aware regime labels and summary counts

### `reports/` - Report Generation

Functions for generating analysis reports:
- `format_analysis_summary(analysis)` - Format a single experiment analysis summary as markdown
- `generate_cross_experiment_comparison(all_analyses)` - Generate cross-experiment comparison report
- `generate_findings_report(all_analyses)` - Generate comprehensive markdown findings report

### `visuals/` - Plotting

Plotting utilities (see [`analysis/visuals/README.md`](visuals/README.md) for details):
- `plot_timeseries_panel()` - Plot time series panels
- `plot_aggregate_panel()` - Plot aggregate analysis panels
- `plot_regime_heatmap()` - Plot regime distribution heatmaps
- `plot_error_heatmap()` - Plot error heatmaps
- `plot_error_vs_trust_gate_comparison()` - Compare error vs trust gate
- `plot_tom_reliability_analysis()` - Analyze ToM reliability

## Usage

### Direct Import (Recommended)

All functions are available through the main `analysis` module for backward compatibility:

```python
from analysis import (
    # Loaders
    load_experiment_results,
    load_all_experiments,
    scan_results_directory,
    # Metrics
    analyze_basic_metrics,
    analyze_parameter_breakdowns,
    analyze_social_learning,
    compute_statistics,
    # Discovery
    compute_discovery_success_rate,
    plot_discovery_trajectories,
    # ToM Analysis
    compute_tom_accuracy_vs_belief_similarity,
    analyze_regime_labels,
    # Utilities
    build_join_key,
    index_by_join_key,
    join_baseline_results,
    # Visualization
    plot_timeseries_panel,
    plot_aggregate_panel,
    plot_regime_heatmap,
    plot_error_heatmap,
    plot_error_vs_trust_gate_comparison,
    plot_tom_reliability_analysis,
)

# Analyze discovery
discovery_metrics = compute_discovery_success_rate(results)

# Detect echo chambers
tom_analysis = compute_tom_accuracy_vs_belief_similarity(results)
if tom_analysis["echo_chamber_detected"]:
    print("Echo chamber detected!")

# Join with baseline
joined = join_baseline_results(results, baseline_results)
```

### Submodule Import (Alternative)

You can also import directly from submodules:

```python
from analysis.discovery import compute_discovery_success_rate
from analysis.social import compute_tom_accuracy_vs_belief_similarity
from analysis.comparison import join_baseline_results
```

### Using the Analysis Script

Run the interactive analysis script:

```bash
python scripts/run_analysis.py
```

This will:
1. List available results from `results/` directory
2. Allow you to select a result by number or path
3. Generate deep analysis outputs in `results/<experiment>/<run_id>/analysis/`

## Dependencies

- `experiment.metrics`: For helper functions (`extract_result_fields`, `normalize_spawn_probs`, etc.)
- `experiment.constants`: For spawn probability normalization and parameter ranges
- `numpy`: Array operations
- `matplotlib`: Plotting (optional, functions return None if not available)

## Integration with Experiment Runner

The experiment runner uses basic metrics from `experiment.metrics` for:
- Computing final error and regime labels

Deep analysis functions and plotting utilities in this module are typically run post-hoc using
`scripts/run_analysis.py` or imported directly for custom analysis workflows.

## References

**Theoretical Framework:**
- Project plan Section 1.4 (The Gap: Problems with Observation-Based Social Learning) - Echo chamber risk
- Project plan Section 2.3 (Hypotheses) - H1: Perspective-taking enables discovery of novel structure
- Project plan Section 7.1 (Primary Analyses) - Discovery capability validation
- Project plan Section 7.2 (Secondary Analyses) - Echo chamber detection

**Key Concepts:**
- **Discovery:** Novel-but-better expert models that observation-based approaches would reject (project plan Section 1.4)
- **Echo Chamber:** Social learning rewards agreement with current beliefs rather than discovery (project plan Section 1.4)

See [`project_plan.txt`](../project_plan.txt) for full theoretical framework.
