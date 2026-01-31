# Visualization Module

Plotting utilities for experiment results.

## Functions

### `plot_timeseries_panel(results, seeds_to_plot=3, output_dir=None)`
Create 2×2 panel of time series for selected seeds.
- Plots: `reliability`, `trust`, `tau_accuracy`, `theta_self[0]`
- Returns: `Path | None` (output path)

### `plot_aggregate_panel(results, factor_x, factor_y=None, output_dir=None)`
Create aggregate statistics panel.
- Scatter plot (if `factor_y=None`) or heatmap (if `factor_y` provided)
- Returns: `Path | None`

### `plot_regime_heatmap(results, x_axis, y_axis, output_dir=None)`
Create heatmap of regime classifications.
- Colors: Independent (gray), Beneficial (green), Overcopying (orange), Harmful (red)
- Returns: `Path | None`

### `plot_error_heatmap(results, x_axis, y_axis, output_dir=None)`
Create error heatmap (wrapper around `plot_aggregate_panel`).

### `plot_error_vs_trust_gate_comparison(results, output_dir=None)`
Box plot comparing error across trust gate values (reliability and accuracy gate thresholds).

### `plot_tom_reliability_analysis(results, output_dir=None)`
Analyze ToM mode and reliability variance.
- Plots false negative rate and reliability variance

## Usage

```python
from analysis.visuals import plot_timeseries_panel, plot_regime_heatmap

# Generate plots
plot_timeseries_panel(results, output_dir=Path("plots"))
plot_regime_heatmap(results, x_axis="task", y_axis="scenario", output_dir=Path("plots"))
```

## Dependencies

- `clean_up.experiment.metrics`: For `compute_final_error` and `compute_regime_label`
- `matplotlib`: Plotting (optional, functions return None if not available)
- `numpy`: Array operations
