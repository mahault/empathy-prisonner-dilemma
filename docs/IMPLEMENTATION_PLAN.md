# Empathy PD Implementation Plan

## Executive Summary

This plan implements "proper ToM + empathy" for the Prisoner's Dilemma, combining:
- **Hongju's exploitability analysis** (from alignment experiments)
- **Harshil's principled inversion** (from SocialLearningAgents)
- **ToM best-response rollouts** (from alignment experiments)
- **Current empathy weighting** (EFE-weighted action selection)

### Core Principle (from meeting)
> "It factors in the actual EFE into action selection. Whereas in Ry's thing, it doesn't.
> It just assumes that they want to coordinate... the pro-sociality here is sort of baked
> in the theory of mind agent, whereas we're not specifically baking it in here."
> — Mahault Albarracin

---

## Phase 0: Baseline & Target Definition

### 0.1 Freeze Current Baseline
- [ ] Run existing PD experiment as-is
- [ ] Save outputs: joint-action trajectories, action marginals, empathy_factor sweep results
- [ ] Document current behavior for comparison

### 0.2 Define Target Agent Loop

Each timestep, the agent should:
```
1. Perceive o_t (joint outcome from previous round)
2. Infer own state/beliefs (standard active inference)
3. Infer opponent action/type (Harshil's inversion)
4. Predict opponent response to candidate actions (ToM best-response)
5. Compute social EFE:
   G_social(a_i) = (1-λ) * G_i(a_i | â_j) + λ * G_j(â_j | a_i)
6. Act: sample from softmax(-β * G_social)
```

---

## Phase 1: Code Refactoring

### 1.1 New Module Structure

```
src/empathy/prisoners_dilemma/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── pd_model.py          # A/B/C/D matrices for PD
├── tom/
│   ├── __init__.py
│   ├── tom_core.py           # ToM rollouts + best-response
│   └── inversion.py          # Harshil-style opponent inference
├── metrics/
│   ├── __init__.py
│   └── exploitability.py     # Hongju-style analysis
├── experiments/
│   ├── __init__.py
│   └── sweep.py              # Parameter sweeps + logging
├── agent.py                  # Refactored EmpatheticAgent
├── env.py                    # Environment (unchanged)
├── sim.py                    # Simulation runner
└── config.py                 # Pydantic models
```

### 1.2 Refactor EmpatheticAgent

**Current structure** (K-copy ensemble averaging):
```python
# Current: K copies of agent, average their EFE
self.agents = [Agent(...) for _ in range(K)]
G_weighted = sum(empathy_factor[k] * G_k for k, G_k in enumerate(agent_Gs))
```

**New structure** (explicit roles):
```python
class EmpatheticAgent:
    def __init__(self, ...):
        self.self_agent: Agent      # My own generative model
        self.other_model: Agent     # Model of opponent (for ToM)
        self.tom: TheoryOfMind      # ToM module (rollouts + inversion)
        self.empathy_factor: float  # λ ∈ [0, 1]
```

---

## Phase 2: Harshil's Principled Inversion

### 2.1 What We're Inferring

**Minimal**: Posterior over opponent's last action
```python
q(a_other | history)  # Did they cooperate or defect?
```

**Full (recommended)**: Posterior over opponent type/parameters
```python
q(θ_other | history)  # What kind of player are they?
# θ ∈ {TFT, WSLS, always_defect, always_cooperate, rational_β}
```

### 2.2 Implementation: `tom/inversion.py`

```python
class OpponentInversion:
    """Harshil-style particle-based opponent inference."""

    def __init__(self, n_particles: int, hypotheses: list[str]):
        self.n_particles = n_particles
        self.hypotheses = hypotheses  # e.g., ["TFT", "WSLS", "always_C", "always_D"]
        self.weights = np.ones(n_particles) / n_particles
        self.particle_types = np.random.choice(hypotheses, n_particles)

    def update(self, observed_action: int, context: dict) -> None:
        """Update particle weights given observed opponent action."""
        for k in range(self.n_particles):
            # Compute likelihood of observed action under hypothesis k
            likelihood = self._action_likelihood(
                observed_action,
                self.particle_types[k],
                context
            )
            self.weights[k] *= likelihood

        # Normalize
        self.weights /= self.weights.sum()

        # Resample if effective sample size too low
        if self._effective_sample_size() < self.n_particles / 2:
            self._resample()

    def reliability(self) -> float:
        """Compute reliability from weight concentration (entropy-based)."""
        entropy = -np.sum(self.weights * np.log(self.weights + 1e-10))
        max_entropy = np.log(self.n_particles)
        confidence = 1 - entropy / max_entropy
        return sigmoid((confidence - 0.5) / 0.1)  # Reliability ∈ [0, 1]

    def expected_params(self) -> dict:
        """Return expected opponent parameters."""
        type_probs = {}
        for h in self.hypotheses:
            type_probs[h] = self.weights[self.particle_types == h].sum()
        return type_probs

    def _action_likelihood(self, action, hypothesis, context) -> float:
        """P(a_other | θ, context) for each hypothesis."""
        # Implement strategy-specific action probabilities
        ...
```

### 2.3 Reliability Gating

From SocialLearningAgents:
```python
# Only trust ToM predictions when reliability is high
if self.tom.inversion.reliability() < threshold:
    # Fall back to uniform prior over opponent actions
    q_other = np.array([0.5, 0.5])
else:
    q_other = self.tom.predict_opponent_response(a_i)
```

---

## Phase 3: ToM Best-Response (Alignment-Style)

### 3.1 Implementation: `tom/tom_core.py`

```python
class TheoryOfMind:
    """ToM rollouts and best-response prediction."""

    def __init__(self, other_model: Agent, inversion: OpponentInversion):
        self.other_model = other_model
        self.inversion = inversion

    def predict_opponent_response(self, my_action: int, depth: int = 1) -> np.ndarray:
        """
        Predict distribution over opponent actions given my action.

        For depth-1 ToM:
        q(a_j | a_i) ∝ exp(-β_j * G_j(a_j | a_i))
        """
        # Get opponent's believed state
        qs_other = self.other_model.qs

        # For each possible opponent action, compute their EFE
        G_other = np.zeros(2)  # [G(C), G(D)]
        for a_j in [0, 1]:  # C=0, D=1
            G_other[a_j] = self._compute_efe_for_action(
                self.other_model, a_j,
                given_my_action=my_action
            )

        # Softmax to get response distribution
        beta_other = self.other_model.beta  # opponent's precision
        q_response = softmax(-beta_other * G_other)

        return q_response

    def compute_social_efe(self, my_action: int, empathy: float) -> float:
        """
        Compute social EFE for my candidate action.

        G_social(a_i) = (1-λ) * G_i(a_i | q(a_j|a_i)) + λ * E[G_j(a_j | a_i)]
        """
        # Predict opponent's response
        q_response = self.predict_opponent_response(my_action)

        # My EFE given opponent's expected response
        G_self = self._expected_efe_self(my_action, q_response)

        # Opponent's expected EFE under their response
        G_other = self._expected_efe_other(my_action, q_response)

        # Weighted combination
        G_social = (1 - empathy) * G_self + empathy * G_other

        return G_social
```

### 3.2 Depth-2 ToM (Optional Extension)

For i anticipating j anticipating i:
```python
def predict_opponent_response_depth2(self, my_action: int) -> np.ndarray:
    """Opponent predicts my response to their response."""
    # j thinks: "if I do a_j, then i will do..."
    # This adds one more nesting layer
    ...
```

---

## Phase 4: New Empathy Formulation

### 4.1 Replace K-Copy Averaging

**Old (current code)**:
```python
# Weighted average over K agent copies
G = sum(empathy_factor[k] * agent.G for k, agent in enumerate(self.agents))
```

**New (proper ToM)**:
```python
def select_action(self) -> int:
    """Select action using social EFE."""
    G_social = np.zeros(2)  # [G(C), G(D)]

    for a_i in [0, 1]:
        G_social[a_i] = self.tom.compute_social_efe(a_i, self.empathy_factor)

    # Sample action
    q_action = softmax(-self.beta * G_social)
    action = np.random.choice([0, 1], p=q_action)

    return action
```

### 4.2 Asymmetric Empathy

Each agent has its own empathy parameter:
```python
agent_i = EmpatheticAgent(empathy_factor=0.8)  # High empathy
agent_j = EmpatheticAgent(empathy_factor=0.2)  # Low empathy
```

This enables studying exploitation dynamics.

---

## Phase 5: Exploitability Analysis (Hongju)

### 5.1 Implementation: `metrics/exploitability.py`

```python
class ExploitabilityAnalysis:
    """Game-theoretic exploitability and empirical exploitation detection."""

    @staticmethod
    def compute_exploitability(policy_i: np.ndarray, payoff_matrix: np.ndarray) -> float:
        """
        How much can a best-responding opponent gain against this policy?

        Exploitability(π_i) = U_j(BR(π_i), π_i) - U_j(π_j, π_i)
        """
        # Compute best response to policy_i
        br_j = compute_best_response(policy_i, payoff_matrix)

        # Expected payoffs
        u_br = expected_payoff(br_j, policy_i, payoff_matrix)
        u_eq = expected_payoff(policy_j_equilibrium, policy_i, payoff_matrix)

        return u_br - u_eq

    @staticmethod
    def classify_outcome(trajectory: dict) -> str:
        """
        Classify run-level outcome.

        Returns one of:
        - "mutual_cooperation"
        - "mutual_defection"
        - "i_exploited": i cooperates, j defects, j gains
        - "j_exploited": j cooperates, i defects, i gains
        - "cycling": high switching rate
        """
        # Compute action frequencies
        p_ci = np.mean(trajectory["actions_i"] == 0)  # i's cooperation rate
        p_cj = np.mean(trajectory["actions_j"] == 0)  # j's cooperation rate

        # Compute outcome frequencies
        outcomes = compute_outcome_frequencies(trajectory)

        # Compute payoff gap
        payoff_gap = trajectory["rewards_i"].mean() - trajectory["rewards_j"].mean()

        # Classification logic
        if outcomes["CC"] > 0.7:
            return "mutual_cooperation"
        elif outcomes["DD"] > 0.7:
            return "mutual_defection"
        elif p_ci > 0.6 and p_cj < 0.4 and payoff_gap < -1:
            return "i_exploited"
        elif p_cj > 0.6 and p_ci < 0.4 and payoff_gap > 1:
            return "j_exploited"
        else:
            return "cycling"
```

---

## Phase 6: Parameter Sweeps

### 6.1 Sweep Parameters

From meeting: "Parameter sweep over the different knobs that we're pulling"

| Parameter | Values | Description |
|-----------|--------|-------------|
| `lambda_i` | [0, 0.25, 0.5, 0.75, 1.0] | Agent i's empathy |
| `lambda_j` | [0, 0.25, 0.5, 0.75, 1.0] | Agent j's empathy |
| `beta_i` | [1, 4, 16] | Agent i's action precision |
| `beta_j` | [1, 4, 16] | Agent j's action precision |
| `tom_depth` | [0, 1] | No ToM vs depth-1 ToM |
| `inversion_on` | [False, True] | Harshil inversion enabled |
| `n_particles` | [1, 10, 30] | Particles for inversion |

### 6.2 Outputs Per Run

```python
@dataclass
class RunResult:
    # Outcome frequencies
    outcome_freq: dict  # {"CC": 0.4, "CD": 0.1, "DC": 0.1, "DD": 0.4}

    # Payoffs
    payoff_i_mean: float
    payoff_i_std: float
    payoff_j_mean: float
    payoff_j_std: float
    payoff_gap: float

    # Exploitability
    exploitability_i: float
    exploitability_j: float

    # Classification
    outcome_label: str  # "mutual_cooperation", "i_exploited", etc.

    # Reliability stats (if inversion on)
    reliability_mean: float
    reliability_std: float

    # Policy stability
    switching_rate_i: float
    switching_rate_j: float
```

### 6.3 Sweep Harness: `experiments/sweep.py`

```python
def run_sweep(config: SweepConfig) -> pd.DataFrame:
    """Run full parameter sweep."""
    results = []

    param_grid = create_param_grid(config)

    for params in tqdm(param_grid):
        for seed in range(config.n_seeds):
            result = run_single_experiment(params, seed)
            results.append({**params, **asdict(result), "seed": seed})

    df = pd.DataFrame(results)
    df.to_csv(config.output_path)

    return df
```

---

## Phase 7: Validation Checks

From meeting: "We're not specifically baking [pro-sociality] in here"

### 7.1 Sanity Checks

| Check | What to verify |
|-------|----------------|
| ToM off, λ > 0 | Empathy without ToM shouldn't produce the same effect |
| ToM on, λ = 0 | No empathy effect should appear with λ=0 |
| Asymmetric λ | High-λ agent should become exploitable |
| Unreliable inversion | ToM predictions should degrade gracefully |

### 7.2 Baseline Comparisons

```python
# Must show these differ:
baseline_no_empathy = run_experiment(lambda_i=0, lambda_j=0, tom=False)
with_empathy_no_tom = run_experiment(lambda_i=0.5, lambda_j=0.5, tom=False)
with_tom_no_empathy = run_experiment(lambda_i=0, lambda_j=0, tom=True)
with_tom_and_empathy = run_experiment(lambda_i=0.5, lambda_j=0.5, tom=True)

assert outcomes_differ(baseline_no_empathy, with_tom_and_empathy)
```

---

## Phase 8: Implementation Roadmap

### Week 1: Foundation
- [ ] Create module structure (`models/`, `tom/`, `metrics/`, `experiments/`)
- [ ] Implement `pd_model.py` (clean A/B/C/D generation)
- [ ] Refactor `agent.py` to new structure (self_agent, other_model, tom)

### Week 2: ToM Core
- [ ] Implement `tom/tom_core.py` (best-response prediction)
- [ ] Implement social EFE computation
- [ ] Test on simple scenarios

### Week 3: Inversion
- [ ] Implement `tom/inversion.py` (particle-based opponent inference)
- [ ] Implement reliability gating
- [ ] Integrate with agent

### Week 4: Analysis
- [ ] Implement `metrics/exploitability.py`
- [ ] Implement outcome classification
- [ ] Create visualization utilities

### Week 5: Sweeps
- [ ] Implement `experiments/sweep.py`
- [ ] Run baseline sweeps
- [ ] Run full parameter sweeps

### Week 6: Validation & Paper
- [ ] Run validation checks
- [ ] Generate figures
- [ ] Write results section

---

## File Edit Summary

| File | Action | Changes |
|------|--------|---------|
| `agent.py` | Major refactor | Replace K-copy with self/other/tom structure |
| `env.py` | Minor | Add reward extraction, optional noise |
| `sim.py` | Moderate | Update to use new agent interface |
| `models/pd_model.py` | New | Clean A/B/C/D generation |
| `tom/tom_core.py` | New | Best-response, social EFE |
| `tom/inversion.py` | New | Particle-based opponent inference |
| `metrics/exploitability.py` | New | Exploitability, outcome classification |
| `experiments/sweep.py` | New | Parameter sweeps, logging |

---

## Success Criteria

From meeting discussion:

1. **Empathy produces different behavior than no-empathy** (primary hypothesis)
2. **Parameter sweeps characterize conditions for pro-sociality emergence**
3. **Asymmetric empathy shows exploitation dynamics**
4. **Results support alignment paper framing**

> "The finding would be characterizing the conditions that are needed to be met in order
> for [pro-sociality] to emerge. Sweeping over those parameter sweeps and then comparing
> and now saying when these conditions are met this is what you see emerge."
> — Sanjeev Namjoshi
