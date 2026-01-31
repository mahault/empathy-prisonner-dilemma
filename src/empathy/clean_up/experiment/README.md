# Clean Up Experiment Module

Experiment orchestration, configuration, and analysis for the clean_up task.


## Structure

```
clean_up/experiment/
├── __init__.py          # Module exports
├── runner.py            # Parallel experiment execution + results contract
├── config.py            # Configuration loading and grid building (merged)
├── cli.py               # Simple CLI (merged arguments + executor)
├── metrics.py           # Basic metrics (error, drift, regime, etc.)
├── logger.py            # Metric tracking
└── constants.py         # Clean_up-specific constants
```

**Note:** Experiment JSON configs are stored in `clean_up/configs/`

**Note:**
- Basic metrics are in `clean_up.experiment.metrics` (error, drift, stability)
- Plotting utilities are in `analysis/visuals/` and invoked from `scripts/run_analysis.py`
- Deep analysis functions (discovery, echo chamber detection, regime analysis) are in the `analysis/` module
- See [`analysis/README.md`](../../analysis/README.md) for deep analysis details

## Experiments (`clean_up/configs/*.json`)

| Experiment | Episodes | Purpose |
|------------|----------|---------|
| **Smoke** | 2 | Sanity check + diagnostics |
| **Killer Test** | 40 | Mismatch protection for clean_up task |
| **Phase Diagram** | 162 | Trust × mismatch severity regimes |
| **Phase Diagram Baseline** | 27 | Baseline without social learning |

## Runner (`runner.py`)

```python
from clean_up.experiment import ExperimentRunner

runner = ExperimentRunner("killer_test")
results = runner.run(n_workers=8, verbose=True)
runner.save_results()
```

Run analysis/plots after the run:
```bash
python scripts/run_analysis.py --basic --regimes --plots --experiment killer_test --latest
```

## Logger (`logger.py`)

Tracked metrics per timestep:
- Trust components: `tau_accuracy`, `reliability`, `trust`
- ToM metrics: `weight_entropy`, `confidence`, `reliability`, `n_eff`
- Social update: `eta_t`, `social_update_norm`
- State: `entropy_hidden`, `param_self`, `param_other_mean`
- Progress: `progress`

Outcome metrics:
- `is_success`, `is_timeout`
- `time_to_success`, `final_progress`

## Metrics and Visualization

### Basic Metrics (`clean_up.experiment.metrics`)

Basic metrics for experiment results:

```python
from clean_up.experiment.metrics import (
    compute_final_error,
    compute_drift_metrics,
    compute_detection_time,
    compute_stability_metrics,
    compute_regime_label,
)
```

**Note:** Import metrics directly from the module:
```python
from clean_up.experiment.metrics import compute_final_error
```

**Note:** Plotting utilities have been moved to the `analysis` module. See the analysis module documentation.

### Deep Analysis (`analysis/` module)

For deeper analysis functions (discovery metrics, echo chamber detection, baseline comparisons), see the `analysis/` module:

```python
from analysis import (
    compute_discovery_success_rate,
    compute_tom_accuracy_vs_belief_similarity,
    join_baseline_results,
)
```

See [`analysis/README.md`](../../analysis/README.md) for full documentation.

## Usage

```bash
# Run experiment
python scripts/run_experiment.py --experiment killer_test --n_workers 8
```

## Creating New Configs

Run the interactive config helper:

```bash
python scripts/create_config.py
```

You can also list configs or use a template:

```bash
python scripts/create_config.py --list
python scripts/create_config.py --create --template killer_test
```

## References

**Key Experiments:**
- **Clean Up Task:** Primary task with three conditions (solo, co-present, ToM) - Section 4.10
- **Killer Test:** Mismatch protection validation - Section 6.2
- **Phase Diagram:** Trust × mismatch severity regimes - Section 6.2

See [`docs/experiments.md`](../../docs/experiments.md) for implementation details.
