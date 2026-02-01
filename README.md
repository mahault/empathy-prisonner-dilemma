# Empathy Research

Active Inference research on empathy, theory of mind, and social learning.

## Quick Start

```bash
# Install
conda env create -f environment.yml
conda activate empathy-prisoner-dilemma

# Run tests
python scripts/run_tests.py

# Run PD experiments
python scripts/run_pd_experiments.py --mode smoke
```

## Projects

This repository contains three active inference research projects:

| Project | Task | Description |
|---------|------|-------------|
| `prisoners_dilemma` | Prisoner's Dilemma | Empathy via EFE weighting + ToM |
| `graph_navigation` | Graph Navigation | Emotional inference and ToM |
| `clean_up` | Clean Up Task | Particle-based ToM with accuracy-gated trust |

## Running Tests

```bash
# All tests
python scripts/run_tests.py

# Verbose output
python scripts/run_tests.py -v

# Specific module
python scripts/run_tests.py --module prisoners_dilemma
python scripts/run_tests.py --module clean_up

# With coverage
python scripts/run_tests.py --coverage
```

## Running Experiments

### Prisoner's Dilemma

The main experiment runner is `scripts/run_pd_experiments.py`:

```bash
# Smoke test (quick verification)
python scripts/run_pd_experiments.py --mode smoke

# Single experiment with custom parameters
python scripts/run_pd_experiments.py --mode single --lambda_i 0.5 --lambda_j 0.5 --T 50

# Parameter sweep
python scripts/run_pd_experiments.py --mode sweep --output results/sweep.json --n_seeds 10

# Validation checks (from roadmap)
python scripts/run_pd_experiments.py --mode validate
```

### Clean Up Task

```bash
python scripts/run_experiment.py --experiment smoke --n_workers 1
```

## Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate empathy-prisoner-dilemma
```

### Using pip

```bash
# Core only
pip install -e .

# With pymdp (for prisoners_dilemma and graph_navigation)
pip install -e ".[pymdp]"

# With numba (for clean_up)
pip install -e ".[clean-up]"

# Everything
pip install -e ".[all]"
```

## Project Structure

```
src/empathy/
├── core/                  # Shared utilities
├── prisoners_dilemma/     # Prisoner's Dilemma project
│   ├── agent.py           # EmpatheticAgent, ToMEmpatheticAgent
│   ├── env.py             # Environment
│   ├── sim.py             # Simulation runner
│   ├── tom/               # Theory of Mind module
│   │   ├── tom_core.py    # Best-response, social EFE
│   │   └── inversion.py   # Particle-based opponent inference
│   └── metrics/           # Analysis tools
│       └── exploitability.py
├── graph_navigation/      # Graph Navigation project
└── clean_up/              # Clean Up project

scripts/
├── run_tests.py           # Test runner (all tests)
├── run_pd_experiments.py  # PD experiment runner (main entry point)
└── run_experiment.py      # Clean Up experiment runner

docs/
├── ROADMAP.md             # Development roadmap
└── IMPLEMENTATION_PLAN.md # Detailed implementation plan
```

## Development Status

See [docs/ROADMAP.md](docs/ROADMAP.md) for current progress:

- **Phase 1-4**: Complete (ToM, inversion, exploitability)
- **Phase 5-6**: In progress (parameter sweeps, validation)

## Usage Examples

### Prisoner's Dilemma with ToM

```python
from empathy.prisoners_dilemma import ToMEmpatheticAgent, Environment

# Create agents with different empathy levels
config = {...}  # See scripts/run_pd_experiments.py for full config
agent_i = ToMEmpatheticAgent(config, agent_num=0, empathy_factor=0.7)
agent_j = ToMEmpatheticAgent(config, agent_num=1, empathy_factor=0.3)

# Run interaction
env = Environment(K=2)
for t in range(T):
    obs = env.step(t=t, actions=actions)
    results_i = agent_i.step(t=t, observation=obs[0])
    results_j = agent_j.step(t=t, observation=obs[1])
    actions = [results_i["exp_action"], results_j["exp_action"]]
```

### Graph Navigation

```python
from empathy.graph_navigation.agents import ToMAgent, ForageAgent
from empathy.graph_navigation.envs import GraphEnv
```

### Clean Up

```python
from empathy.clean_up import CleanUpAgent, CleanUpEnvironment
from empathy.clean_up.experiment.runner import ExperimentRunner

runner = ExperimentRunner("smoke")
results = runner.run()
```

## Running Tests

```bash
pytest tests/
pytest src/empathy/clean_up/agent/tests/
```
