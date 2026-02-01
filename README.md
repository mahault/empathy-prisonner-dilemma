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

## Prisoner's Dilemma: Theory of Mind + Empathy

### Core Contribution

This implementation demonstrates "proper ToM + empathy" that **does not bake in pro-sociality** but allows it to emerge from principled EFE weighting:

```
G_social(a_i) = (1 - lambda) * G_self(a_i | predicted_response) + lambda * G_other(predicted_response | a_i)
```

Unlike previous approaches where pro-sociality is assumed, here empathy emerges from:
1. **Theory of Mind**: Agent predicts opponent's response to each candidate action
2. **Social EFE**: Agent weighs own expected free energy against opponent's
3. **Empathy factor (lambda)**: Controls the weight given to opponent's welfare

### Architecture

```
EmpatheticAgent
├── self_agent (PyMDP)     # Own generative model
├── other_model (PyMDP)    # Model of opponent
├── tom                    # Theory of Mind module
│   ├── inversion          # Particle-based opponent inference
│   └── best_response      # Predict opponent's likely action
└── empathy_factor         # lambda in [0, 1]
```

### Key Equations

**Social EFE (main contribution):**
```
G_social(a_i) = (1 - lambda) * G_i(a_i | q(a_j|a_i)) + lambda * E[G_j(a_j | a_i)]
```

**Opponent Response (ToM depth-1):**
```
q(a_j | a_i) ~ exp(-beta_j * G_j(a_j | a_i))
```

**Reliability gating (from Harshil's work):**
```
H(w) = -sum_k w_k * log(w_k)         # Weight entropy
u_t = 1 - H(w) / log(N_particles)    # Confidence
r_t = sigmoid((u_t - u_0) / kappa)   # Reliability
```

### Phase 5 Results: Parameter Sweep

We ran 22,500 experiments sweeping over:
- `lambda_i, lambda_j`: [0, 0.25, 0.5, 0.75, 1.0]
- `beta_i, beta_j`: [1, 4, 16]
- `use_inversion`: [False, True]
- 50 seeds per condition

**Key Findings:**

| Condition | Cooperation Rate | Interpretation |
|-----------|------------------|----------------|
| lambda = 0, 0 | 2% | Baseline: mutual defection |
| lambda = 0.25, 0.25 | 70% | Some cooperation emerges |
| lambda >= 0.5, >= 0.5 | **100%** | Full mutual cooperation |
| lambda = 0.9, 0.1 | Exploited | High-empathy agent disadvantaged |

**Main result:** Empathy (lambda >= 0.5) combined with ToM reliably produces mutual cooperation without baking in pro-sociality.

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

# Parameter sweep (full - takes ~12 min)
python scripts/run_pd_experiments.py --mode sweep --output results/sweep.json --n_seeds 50

# Quick sweep (for testing)
python scripts/run_pd_experiments.py --mode sweep --quick --n_seeds 5

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

results/                   # Experiment outputs (gitignored)
```

## Development Status

See [docs/ROADMAP.md](docs/ROADMAP.md) for current progress:

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Refactoring | Complete | Module structure, ToMEmpatheticAgent |
| Phase 2: ToM Implementation | Complete | Best-response prediction, social EFE |
| Phase 3: Harshil's Inversion | Complete | Particle filter, reliability gating |
| Phase 4: Exploitability | Complete | Game-theoretic analysis, outcome classification |
| Phase 5: Parameter Sweeps | Complete | Full sweep over lambda, beta, inversion (22,500 runs) |
| Phase 6: Validation | Complete | Visualizations, statistical analysis |

## Generated Figures

After running the parameter sweep, generate analysis figures:

```bash
python scripts/analyze_sweep.py --input results/phase5_full_sweep.json --output figures/
```

This creates:
- `cooperation_heatmap.png` - Cooperation rate by empathy levels
- `outcome_distribution.png` - Outcomes by empathy configuration
- `beta_effect.png` - Effect of action precision
- `exploitation_dynamics.png` - Payoff gap analysis
- `summary_figure.png` - Combined 4-panel paper figure

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

## Validation Matrix

| Condition | Expected | Observed | Status |
|-----------|----------|----------|--------|
| lambda=0, ToM=on | Baseline defection | 2% cooperation | Pass |
| lambda>0, ToM=on | Pro-sociality emerges | 100% CC at lambda>=0.5 | Pass |
| Asymmetric lambda | Exploitation dynamics | High-lambda exploited | Pass |
| Symmetric high lambda | Mutual cooperation | 100% CC | Pass |

## References

- Implementation based on discussions with Mahault Albarracin, Sanjeev Namjoshi, Hongju, and Harshil
- Particle filter and reliability gating adapted from SocialLearningAgents
- Exploitability analysis adapted from alignment experiments
