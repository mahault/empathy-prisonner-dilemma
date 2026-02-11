# Roadmap: Figure 3 (Near-Symmetric) + Learning & Convergence

## Context
- Alejandro suggested zooming into near-symmetric empathy (|λ_i - λ_j| ≈ 0.25)
- Mao suggested learning/convergence
- Both approved for implementation

## Status Tracker

- [x] Figure 3 script written: `scripts/generate_near_symmetric_figure.py`
- [x] Figure 3 generated and verified
- [x] Learning: add `predict_action()` to OpponentInversion
- [x] Learning: add `q_response_override` to SocialEFE
- [x] Learning: wire inversion into `ToMEmpatheticAgent.step()`
- [x] Tests pass after learning changes (95/95)
- [x] Validation pass (4/4 checks)
- [x] Learning figure script: `scripts/generate_learning_figure.py`
- [x] Learning figure generated and verified

---

## Part 1: Figure 3 — Near-Symmetric Empathy Dynamics (COMPLETE)

### Script: `scripts/generate_near_symmetric_figure.py`

- Fix λ_j = 0.5, sweep λ_i ∈ {0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75}
- T=100, n_seeds=30
- Panel A: rolling CC rate ± std, color-coded by λ_i (coolwarm colormap)
- Panel B: example traces for λ_i = 0.20, 0.35, 0.50 (near transition)
- Output: `figures/near_symmetric_dynamics.png/.pdf`

### Key results

| λ_i | Mean CC | Std CC |
|-----|---------|--------|
| 0.10 | 0.101 | 0.034 |
| 0.20 | 0.443 | 0.052 |
| 0.25 | 0.686 | 0.045 |
| 0.30 | 0.859 | 0.029 |
| 0.35 | 0.942 | 0.016 |
| 0.40 | 0.979 | 0.014 |
| 0.45 | 0.990 | 0.010 |
| 0.50 | 0.995 | 0.008 |
| ≥0.55 | ~0.997 | ~0.006 |

**Finding:** Sharp transition between λ_i=0.10 and λ_i=0.40. Above λ_i ≈ 0.45,
cooperation saturates. Variance peaks at λ_i ≈ 0.20 (max std = 0.052).
Asymmetric: cooperation is fragile to empathy *deficit* but robust to empathy *excess*.

---

## Part 2: Learning Infrastructure (COMPLETE)

### Changes made

1. **`src/empathy/prisoners_dilemma/tom/inversion.py`** — Added `predict_action()` method
   to OpponentInversion. Bayesian model-averaged prediction across all particles.

2. **`src/empathy/prisoners_dilemma/tom/tom_core.py`** — Added `q_response_override`
   parameter to `SocialEFE.compute()` and `compute_all_actions()`. When provided,
   bypasses static ToM and uses learned opponent model.

3. **`src/empathy/prisoners_dilemma/agent.py`** — Wired inversion into
   `ToMEmpatheticAgent.step()`. When `use_inversion=True` and the particle filter
   is reliable, the learned prediction overrides the static ToM.

### Architecture (updated)

```
ToMEmpatheticAgent.step()
  │
  ├─ 1. Extract opponent action from observation
  ├─ 2. OpponentInversion.update()  ← particle filter learns opponent type
  ├─ 3. self_agent.infer_states()   ← PyMDP belief update
  ├─ 4. inversion.predict_action() → q_response_override (when reliable)
  ├─ 5. SocialEFE.compute_all_actions(q_response_override=...)
  │      G_social = (1-λ)*G_self + λ*G_other
  ├─ 6. softmax(-G_social) → q_action → sample action
  └─ 7. Return step results + inversion state
```

---

## Part 3: Learning Convergence Figure (COMPLETE)

### Script: `scripts/generate_learning_figure.py`

**Panel A — Type posterior convergence (facing λ_j=0.7):**
- Particle filter converges to ALWAYS_COOPERATE (98%) by round ~60
- TFT initially competes (~30%) but is distinguished from pure cooperation

**Panel B — Type posterior convergence (facing λ_j=0.1):**
- Converges to ALWAYS_DEFECT (90%) + RANDOM (10%) by round ~20
- Fast convergence because defection is more discriminating

**Panel C — Cooperation: learned vs static (fixing λ_j=0.7):**
- λ_i=0.2: CC ≈ 45%, learning has no effect (filter unreliable at low CC)
- λ_i=0.3: CC ≈ 87%, tiny threshold effect (-0.8%)
- λ_i=0.5: CC ≈ 100%, identical
- λ_i=0.7: CC ≈ 100%, identical

### Revised key finding

The cooperation threshold shift (λ≈0.2 → λ≈0.4) predicted analytically is
**attenuated in practice** because:
1. At low λ, cooperation is too low for the filter to converge reliably
2. At high λ, cooperation is already maximal regardless of beliefs
3. The narrow window where learning matters (λ≈0.3-0.4) shows only a small effect

**Paper-ready interpretation:** Cooperation in this model is primarily driven by
the empathy parameter λ, not by learned beliefs about the opponent. The empathy
mechanism is **robust to accurate opponent modelling** — agents cooperate because
they weight the other's welfare, not because they believe the other will cooperate.
This distinguishes empathy-based cooperation from reciprocity-based cooperation.

---

## Files Summary

| File | Status |
|------|--------|
| `scripts/generate_near_symmetric_figure.py` | COMPLETE |
| `scripts/generate_learning_figure.py` | COMPLETE |
| `src/empathy/prisoners_dilemma/tom/inversion.py` | MODIFIED (predict_action) |
| `src/empathy/prisoners_dilemma/tom/tom_core.py` | MODIFIED (q_response_override) |
| `src/empathy/prisoners_dilemma/agent.py` | MODIFIED (wired inversion) |
| `figures/near_symmetric_dynamics.png/.pdf` | GENERATED |
| `figures/learning_convergence.png/.pdf` | GENERATED |

## Verification (all pass)

```bash
python -m pytest tests/                          # 95/95 passed
python scripts/run_pd_experiments.py --mode smoke     # PASSED
python scripts/run_pd_experiments.py --mode validate  # 4/4 passed
python scripts/generate_near_symmetric_figure.py      # Generated
python scripts/generate_learning_figure.py            # Generated
```
