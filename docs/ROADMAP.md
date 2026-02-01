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

---

## Key Equations

### Social EFE (Main Contribution)
```
G_social(a_i) = (1-λ) G_i(a_i | q(a_j|a_i)) + λ E_{q(a_j|a_i)}[G_j(a_j | a_i)]
```

### Opponent Response (ToM Depth-1)
```
q(a_j | a_i) ∝ exp(-β_j * G_j(a_j | a_i))
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
