# Empathy Research

Active Inference research on empathy, theory of mind, and social learning.

## Projects

This repository contains three active inference research projects:

| Project | Task | Description |
|---------|------|-------------|
| `prisoners_dilemma` | Prisoner's Dilemma | Empathy via EFE weighting |
| `graph_navigation` | Graph Navigation | Emotional inference and ToM |
| `clean_up` | Clean Up Task | Particle-based ToM with accuracy-gated trust |

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

## Usage

### Prisoner's Dilemma

```python
from empathy.prisoners_dilemma import Sim

config = {...}  # See notebooks for examples
sim = Sim(config)
history = sim.run()
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

## Project Structure

```
src/empathy/
├── core/                  # Shared utilities
├── prisoners_dilemma/     # Prisoner's Dilemma project
├── graph_navigation/      # Graph Navigation project
└── clean_up/              # Clean Up project
    ├── agent/
    ├── environment/
    └── experiment/

analysis/                  # Analysis tools
notebooks/                 # Jupyter notebooks
scripts/                   # CLI entry points
tests/                     # Test suite
```

## Running Tests

```bash
pytest tests/
pytest src/empathy/clean_up/agent/tests/
```
