# Referee analyses for JRSI rsif-2026-0208 revision

Run 2026-07-28 · `scripts/run_referee_analyses.py` · full grid: T=100, 25 seeds/cell.
Raw data: `comparison_results.json` (225 runs), `horizon_results.json` (900 runs).

Reproduce: `python scripts/run_referee_analyses.py --mode all --T 100 --n_seeds 25`

---

## Analysis 1 — Model comparison (Referee 1, point 8)

Three AIF agent types, differing only in λ and opponent inversion:

| type | λ | particle-filter inversion |
|---|---|---|
| self_interested | 0.0 | off |
| tom_only | 0.0 | **on** (accurate opponent prediction, no welfare weighting) |
| empathic | 0.6 | on |

**Symmetric pairings (mean over 25 seeds):**

| pairing | CC freq | DD freq |
|---|---|---|
| self_interested vs self_interested | 0.00 | 0.97 |
| tom_only vs tom_only | 0.00 | 0.95 |
| empathic vs empathic | **1.00** | 0.00 |

**tom_only − self_interested CC delta: +0.001.** Belief accuracy alone does not
generate cooperation; only the welfare weighting λ does. This is the direct
experimental dissociation the referee asked for, and it makes the paper's
central conceptual claim (prediction ≠ valuation) a result rather than an
interpretation.

**Mixed pairings (exploitability = |coop_i − coop_j|, Hongju's metric):**

| pairing | exploitability | payoff gap |
|---|---|---|
| self_interested vs tom_only | 0.02 | +0.04 |
| empathic vs self_interested | 0.98 | −4.91 |
| empathic vs tom_only | 0.98 | −4.89 |

An unconditional λ=0.6 empathic agent is fully exploited by either non-empathic
type (payoff 0.1 vs 5.0). Connects to the paper's existing asymmetric-empathy /
exploitability findings and motivates the adaptive-λ discussion in the revision.

## Analysis 2 — Planning-horizon robustness (Referee 1, point 6)

Symmetric dyads, sophisticated planner (as in the paper's Fig on sophisticated
inference), H ∈ {1,2,3,4} × λ ∈ {0.3, 0.5, 0.7} × three payoff structures
(all valid PDs: T > R > P > S, 2R > T + S).

Mean cooperation rate:

**standard (R,S,T,P) = (3,0,5,1)** — the paper's decision payoffs

| λ | H=1 | H=2 | H=3 | H=4 |
|---|---|---|---|---|
| 0.3 | 0.88 | 0.73 | 0.67 | 0.63 |
| 0.5 | 1.00 | 0.95 | 0.88 | 0.82 |
| 0.7 | 1.00 | 0.99 | 0.97 | 0.92 |

**weak_temptation (3,1,4,2)**

| λ | H=1 | H=2 | H=3 | H=4 |
|---|---|---|---|---|
| 0.3 | 0.40 | 0.45 | 0.46 | **0.47** ← rises |
| 0.5 | 0.88 | 0.73 | 0.67 | 0.63 |
| 0.7 | 0.99 | 0.90 | 0.82 | 0.76 |

**high_temptation (5,0,8,1)**

| λ | H=1 | H=2 | H=3 | H=4 |
|---|---|---|---|---|
| 0.3 | 1.00 | 0.94 | 0.87 | 0.81 |
| 0.5 | 1.00 | 1.00 | 0.98 | 0.95 |
| 0.7 | 1.00 | 1.00 | 1.00 | 0.99 |

**Reading.** The referee's hypothesis is partially confirmed:

1. In the regime where myopic cooperation is high, deeper planning erodes
   cooperation under **all three** payoff structures — the erosion itself is a
   general property, not an artifact of one matrix.
2. But the **direction can reverse**: under weak temptation at λ=0.3, myopic
   cooperation is already largely collapsed (0.40) and deeper planning slightly
   *recovers* it (0.47). So the correct statement for the revision is:
   *deeper planning amplifies whatever the payoff-empathy configuration favors —
   it erodes cooperation above the cooperation threshold and mildly supports it
   below — rather than uniformly reducing cooperation.*
3. Note the λ=0.5/weak_temptation row is numerically identical to the
   λ=0.3/standard row: λ and the payoff margins enter the social EFE through
   the same weighted sum, so empathy effectively rescales the temptation
   margin. Worth one sentence in the revision — it explains *why* the horizon
   effect is payoff-dependent in exactly the way it is.

## Implementation note discovered en route (affects Methods text)

The agents' decisions are driven by the module-level `PD_PAYOFFS` dict in
`tom/tom_core.py` = (R=3, S=0, T=5, P=1). The pymdp C matrices and the
realized-payoff logging in `run_pd_experiments.py` historically used
(3, 1, 4, 2). Behavior is entirely governed by the former. The revision's
Methods should state the decision payoffs explicitly — this also resolves the
reader question (Hyojun Choi, 2 Jul) about `empathy_shift`, whose constants
are derived from (3, 0, 5, 1). This script swaps `PD_PAYOFFS` in place for the
payoff-robustness sweep and restores it after each run.
