# Empathy PD Project Roadmap

## Vision

Implement "proper ToM + empathy" that **does not bake in pro-sociality** but allows it to emerge from principled EFE weighting.

```
G_social(a_i) = (1-λ) * G_self(a_i | predicted_response) + λ * G_other(predicted_response | a_i)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        EmpatheticAgent                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │  self_agent  │   │ other_model  │   │       tom        │   │
│  │  (PyMDP)     │   │  (PyMDP)     │   │                  │   │
│  │              │   │              │   │  ┌────────────┐  │   │
│  │  - A,B,C,D   │   │  - A,B,C,D   │   │  │ inversion  │  │   │
│  │  - qs        │   │  - qs        │   │  │ (particles)│  │   │
│  │  - G         │   │  - G         │   │  └────────────┘  │   │
│  └──────────────┘   └──────────────┘   │                  │   │
│                                         │  ┌────────────┐  │   │
│                                         │  │best_response│ │   │
│                                         │  │ (rollouts) │  │   │
│                                         │  └────────────┘  │   │
│                                         └──────────────────┘   │
│                                                                 │
│  empathy_factor (λ) ────────────────────────────────────────▶  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Development Phases

### Phase 1: Refactoring (Foundation)
```
Status: [ ] Not Started  [●] In Progress  [✓] Complete

[✓] Create module structure
    ├── models/pd_model.py
    ├── tom/tom_core.py
    ├── tom/inversion.py
    ├── metrics/exploitability.py
    └── experiments/sweep.py

[✓] Refactor EmpatheticAgent
    ├── Remove K-copy ensemble (new ToMEmpatheticAgent)
    ├── Add self_agent, other_model, tom
    └── Update action selection
```

### Phase 2: ToM Implementation
```
[✓] Best-response prediction
    ├── predict_opponent_response(my_action) → q(a_j | a_i)
    ├── G_j computation for opponent
    └── Softmax response distribution

[✓] Social EFE computation
    ├── G_social = (1-λ) * G_self + λ * E[G_other]
    └── Integrate with action selection
```

### Phase 3: Harshil's Inversion
```
[✓] Particle-based opponent inference
    ├── Hypothesis space: {TFT, WSLS, always_C, always_D, rational_β}
    ├── Likelihood: P(a_observed | θ_k, context)
    ├── Weight update: w_k ∝ w_k * likelihood
    └── Resampling when ESS low

[✓] Reliability gating
    ├── Entropy-based confidence
    ├── Sigmoid reliability score
    └── Graceful degradation when unreliable
```

### Phase 4: Exploitability Analysis
```
[✓] Game-theoretic exploitability
    ├── Best response computation
    ├── Exploitability score
    └── Per-policy analysis

[✓] Empirical outcome classification
    ├── mutual_cooperation
    ├── mutual_defection
    ├── i_exploited / j_exploited
    └── cycling
```

### Phase 5: Parameter Sweeps
```
[✓] Sweep configuration
    ├── λ_i, λ_j: [0, 0.25, 0.5, 0.75, 1.0]
    ├── β_i, β_j: [1, 4, 16]
    ├── use_inversion: [False, True]
    └── n_seeds: 50 per cell (22,500 total runs)

[✓] Output logging
    ├── Outcome frequencies (CC/CD/DC/DD)
    ├── Payoffs (mean, std, gap)
    ├── Exploitability scores
    └── Results saved to results/phase5_full_sweep.json
```

### Phase 6: Validation & Analysis
```
[✓] Sanity checks (all passed)
    ├── ToM on + λ = 0 → 2% cooperation (no empathy leakage)
    ├── ToM on + λ ≥ 0.5 → 100% cooperation
    ├── Asymmetric λ → exploitation dynamics confirmed
    └── High-empathy agents disadvantaged when facing low-empathy

[✓] Visualizations generated (figures/)
    ├── cooperation_heatmap.png - CC rate by lambda_i x lambda_j
    ├── outcome_distribution.png - outcomes by empathy category
    ├── beta_effect.png - action precision analysis
    ├── exploitation_dynamics.png - payoff gap analysis
    ├── inversion_effect.png - with/without opponent inference
    └── summary_figure.png - combined 4-panel figure
```

### Phase 7: Reviewer Critiques (Theoretical Foundations)

Addresses three critiques from game-theory review. Three sub-phases that
strengthen the Active Inference framing, fix the simultaneous-move formulation,
and align the hypothesis space with the policy class.

#### Phase 7A: Fix Simultaneous-Move Conditioning
```
[✓] Replace within-round q(a_j | a_i) with history-conditioned q(a_j | h_t)
    ├── Problem: conditioning opponent action on my simultaneous action is invalid
    ├── Fix: opponent prediction based on history, not current-round action
    ├── Static ToM uses opponent's belief about my policy π_i from past rounds
    ├── Particle filter prediction q_learned(a_j | h_t) already history-based
    └── Sophisticated planner: step>0 CAN condition on simulated prior actions

[✓] Update interfaces
    ├── TheoryOfMind.predict_opponent_response(my_action) → predict_opponent_action()
    ├── GatedToM.predict_opponent_response(my_action, ctx) → predict_opponent_action(ctx)
    ├── OpponentSimulator.predict_response(my_action, step) → predict_response(step)
    ├── SocialEFE: single q_response instead of per-action overrides
    └── Agent: track empirical cooperation rate, call update_my_policy_belief()
```

New formulation:
```
q_prior(a_j) ∝ exp(-β_j * G_j(a_j))
G_j(a_j) = -Σ_i π_i(a_i) * payoff_j(a_i, a_j)

π_i = empirical cooperation rate from history  (or [0.5, 0.5] at t=0)

G_social(a_i) = (1-λ) * G_self(a_i | q(a_j|h_t)) + λ * E_{q(a_j|h_t)}[G_other(a_j)]
```

Note: `q(a_j|h_t)` is the same for all candidate `a_i`, but `G_self(a_i|q)` still
differentiates actions through the payoff function `payoff_i(a_i, a_j)`.

#### Phase 7B: Parametric Behavioral Hypotheses
```
[✓] Replace discrete OpponentHypothesis enum with continuous BehavioralProfile
    ├── Problem: TFT/WSLS hypotheses don't match agent's own policy class
    ├── Fix: parametric model P(C | h_t) = σ(β * (α + ρ * f(h_t) + empathy_shift))
    │   ├── α = cooperation bias  (ALLC: α >> 0, ALLD: α << 0)
    │   ├── ρ = reciprocity       (TFT-like: ρ >> 0, unconditional: ρ ≈ 0)
    │   ├── β = precision         (deterministic: β >> 0, random: β ≈ 0)
    │   └── f(h_t) = history feature (my_last_action mapped to ±1)
    ├── Particles: sample α ~ N(0, 2), ρ ~ N(0, 1.5), β ~ Gamma(2, 2)
    └── Captures all old hypotheses as special cases

[✓] Update reporting and visualization
    ├── get_type_distribution() → get_profile_summary() (mean/std of params)
    ├── Learning figure: parameter convergence instead of type posterior bars
    └── Agent step results: profile summaries instead of type distributions
```

#### Phase 7C: Opponent Empathy Inference (Epistemic Value)
```
[✓] Add λ_j as latent variable to particle filter
    ├── Problem: identity A matrix → zero epistemic value → AIF adds nothing
    ├── Fix: each particle carries (α, ρ, β, λ_j), infer opponent empathy
    ├── Empathy feature: empathy_shift(λ_j, p) = 5*λ_j - p - 1
    │   derived from social EFE payoff difference (PD: R=3, S=0, T=5, P=1)
    ├── Initialize λ_j ~ Uniform(0, 1) per particle
    └── Provides genuine hidden-state inference that justifies AIF framing

[✓] Implement epistemic value in EFE
    ├── G_epistemic(a_i) = -IG(a_i) (one-step-ahead information gain about λ_j)
    ├── IG computed via hypothetical next-round posterior entropy reduction
    ├── Key insight: cooperating is more epistemically valuable than defecting
    │   because it better disambiguates opponent empathy
    └── Decays as λ_j posterior concentrates (uncertainty resolves)

[✓] Full EFE with epistemic term
    ├── G(a_i) = G_pragmatic(a_i) + G_epistemic(a_i)
    ├── G_pragmatic(a_i) = (1-λ_i) * G_self(a_i|q(a_j)) + λ_i * E[G_other]
    └── SocialEFE accepts optional inversion for epistemic computation

[✓] New figure: empathy inference visualization
    ├── Panel A: λ_j posterior convergence (facing λ_j=0.7 vs λ_j=0.1)
    ├── Panel B: epistemic value contribution over time
    └── Panel C: cooperation rate with/without epistemic term
```

#### Phase 7D: Rerun All Experiments and Regenerate Figures
```
[✓] Rerun experiments
    ├── Full parameter sweep → results/phase7_sweep.json (50 seeds)
    ├── Validation checks: 4/4 passed
    └── Smoke test: passed

[✓] Regenerate all figures
    ├── analyze_sweep.py (heatmap, outcome dist, beta effect, exploitation, summary)
    ├── generate_timeseries_figures.py (cooperation dynamics, belief synchrony)
    ├── generate_learning_figure.py (updated for parametric profiles)
    ├── generate_near_symmetric_figure.py (boundary layer analysis)
    ├── generate_sophisticated_figure.py (myopic vs multi-step)
    └── generate_empathy_inference_figure.py (NEW: λ_j inference)

[✓] Update tests (117 tests, all passing)
    ├── test_tom.py - new prediction interface, parametric profiles, epistemic value
    ├── test_sophisticated.py - unconditional prediction in rollouts
    └── test_integration.py - end-to-end with all three fixes
```

---

## Key Equations

### Social EFE (Main Contribution)
```
G_social(a_i) = (1-λ) G_i(a_i | q(a_j|h_t)) + λ E_{q(a_j|h_t)}[G_j(a_j)]
```

### Opponent Prediction (History-Conditioned)
```
q(a_j | h_t) = r * q_learned(a_j | h_t) + (1 - r) * q_prior(a_j)

q_prior(a_j) ∝ exp(-β_j * G_j(a_j))
G_j(a_j) = -Σ_i π_i(a_i) * payoff_j(a_i, a_j)
```

### Parametric Behavioral Model
```
P(a_j = C | h_t) = σ(α + ρ * f(h_t))
  α = cooperation bias, ρ = reciprocity, f(h_t) = history feature
```

### Epistemic Value (Empathy Inference)
```
G_epistemic(a_i) = -Σ_j q(a_j|h_t) * KL(q(λ_j | a_j observed) ‖ q(λ_j | h_t))
G(a_i) = G_pragmatic(a_i) + G_epistemic(a_i)
```

### Reliability (from Harshil)
```
H(w) = -Σ_k w_k log(w_k)           # Weight entropy
u_t = 1 - H(w) / log(N_particles)  # Confidence
r_t = σ((u_t - u_0) / κ)           # Reliability
```

### Exploitability (from Hongju)
```
Exploitability(π_i) = U_j(BR(π_i), π_i) - U_j(π_j*, π_i)
```

---

## Validation Matrix

| Condition | Expected Behavior | Why It Matters |
|-----------|-------------------|----------------|
| λ=0, ToM=off | Baseline defection | No empathy effect |
| λ=0, ToM=on | Unchanged from baseline | ToM alone ≠ empathy |
| λ>0, ToM=off | Minimal effect | Empathy needs ToM |
| λ>0, ToM=on | Pro-sociality emerges | Main hypothesis |
| λ_i >> λ_j | i becomes exploitable | Asymmetric dynamics |
| Low reliability | ToM prediction degrades | Graceful failure |
| λ_j inference on | Faster adaptation to opponent | AIF epistemic value |
| Early rounds | Higher cooperation (epistemic) | Information-seeking probe |
| λ_j posterior converged | Epistemic value → 0 | Reduces to pragmatic |

---

## Success Metrics

From the meeting:

> "The finding would be characterizing the conditions that are needed to be met
> in order for [pro-sociality] to emerge."

1. **Primary**: Empathy + ToM produces significantly different outcomes than baseline
2. **Secondary**: Parameter sweeps reveal emergence conditions
3. **Alignment angle**: High-empathy agents can be exploited by low-empathy agents

---

## Dependencies

### From Other Projects

| Source | What to Port | Target |
|--------|--------------|--------|
| SocialLearningAgents | Particle filter, reliability gating | `tom/inversion.py` |
| Alignment experiments | Best-response, exploitability | `tom/tom_core.py`, `metrics/` |
| Current PD repo | A/B/C/D matrices, env | `models/`, keep `env.py` |

### External
- `pymdp` (custom branch: si_branch_observations)
- `numpy`
- `pandas` (for sweep results)
- `matplotlib` (for visualizations)
