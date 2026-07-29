# Referee analyses for JRSI rsif-2026-0208 revision

Run 2026-07-28 · `scripts/run_referee_analyses.py`.
Raw data: `comparison_results.json` (225 runs), `horizon_results.json`
(720 runs, paper protocol), `horizon_results_symmetric.json` (900 runs,
earlier symmetric-planning variant, kept for reference).

Reproduce:
`python scripts/run_referee_analyses.py --mode comparison --T 100 --n_seeds 25`
`python scripts/run_referee_analyses.py --mode horizon --T 100 --n_seeds 20 --protocol paper`

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

Config note: the state-inference C matrices have NO behavioral effect (verified:
legacy (3,1,4,2) vs payoff-aligned C give identical results to 3 decimals), so
these numbers are directly comparable to the paper's other experiments.

## Analysis 2 — Planning-horizon robustness (Referee 1, point 6)

### Protocol (aligned with the paper's Table 2 / Fig 7, verified by exact reproduction)

The paper's published sophistication numbers come from
`generate_sophisticated_figure.py`, whose protocol is:

- **metric: mutual cooperation frequency (freq_CC)**, not mean cooperation rate;
- **one-sided planning**: the sophisticated agent (horizon H) is paired with a
  myopic partner (H=1) of equal λ — only agent i receives `use_sophisticated`;
- legacy state-inference C = (3,1,4,2) (no behavioral effect, see above);
- static ToM (no inversion), T=100, seeds 0–19.

Under this protocol the standard-payoff column below reproduces the paper's
Table 2 λ=0.3 entries **exactly**: 0.782 / 0.657 / 0.597 for H=1/2/3.

Mutual cooperation frequency:

**standard (R,S,T,P) = (3,0,5,1)** — the paper's decision payoffs

| λ | H=1 | H=2 | H=3 | H=4 |
|---|---|---|---|---|
| 0.3 | 0.78 | 0.66 | 0.60 | 0.56 |
| 0.5 | 1.00 | 0.95 | 0.89 | 0.83 |
| 0.7 | 1.00 | 0.99 | 0.97 | 0.92 |

**weak_temptation (3,1,4,2)**

| λ | H=1 | H=2 | H=3 | H=4 |
|---|---|---|---|---|
| 0.3 | 0.17 | 0.19 | 0.19 | **0.20** ← no erosion, slight rise |
| 0.5 | 0.78 | 0.66 | 0.60 | 0.56 |
| 0.7 | 0.98 | 0.89 | 0.81 | 0.76 |

**high_temptation (5,0,8,1)**

| λ | H=1 | H=2 | H=3 | H=4 |
|---|---|---|---|---|
| 0.3 | 0.99 | 0.94 | 0.87 | 0.81 |
| 0.5 | 1.00 | 1.00 | 0.98 | 0.95 |
| 0.7 | 1.00 | 1.00 | 1.00 | 0.99 |

**Reading.** The referee's hypothesis is partially confirmed:

1. Where myopic cooperation is high, deeper planning erodes it under **all
   three** payoff structures — the erosion is a general property, not an
   artifact of one matrix.
2. The direction can reverse: under weak temptation at λ=0.3, myopic
   cooperation has already collapsed (0.17) and deeper planning no longer
   erodes it, rising slightly (0.20 at H=4). Correct statement for the
   revision: *deeper planning amplifies whatever the payoff-empathy
   configuration favors — it erodes cooperation above the cooperation
   threshold and leaves it flat or mildly supported below.*
3. The λ=0.5/weak_temptation row coincides with the λ=0.3/standard row
   (max per-seed CC difference 0.01, i.e. one round): λ and the payoff
   margins enter the social EFE through the same weighted sum, so empathy
   effectively rescales the temptation margin. This explains *why* the
   horizon effect is payoff-dependent in exactly the way it is.

The earlier symmetric-planning variant (both agents at horizon H, payoff-
aligned C, mean-cooperation metric, 25 seeds) is preserved in
`horizon_results_symmetric.json`; its qualitative conclusions are the same.

## Analysis 3 — Self/other weight space (Referee 2, point 2)

`scripts/run_referee2_weights.py`. Referee 2 asked why empathic concern is a
single bipolar scalar rather than separate self- and other-concern weights.

Analytically, with G = w_self·G_self + w_other·G_other and softmax precision β,

    β(w_self·G_self + w_other·G_other)
      = β' [(1−λ)G_self + λ·G_other],
    λ = w_other/(w_self+w_other),  β' = β(w_self+w_other)

so on the positive quadrant the *direction* of the weight vector sets λ and
its *magnitude* only rescales action precision.

**Verified empirically:**

- `--mode precision`: (c·w_self, c·w_other) at β reproduces (w_self, w_other)
  at c·β **identically per seed** (all 6 tested configurations, c ∈ {0.5, 2}).
- `--mode ratio`: over a 5×5 weight grid (20 seeds each), cooperation is a
  function of the ratio; residual spread within a ratio is the precision
  effect (e.g. λ=0.5 across 5 different magnitudes: mean CC 0.965, spread
  0.164, monotone in magnitude).
- `--mode quadrants`: the genuine extension is the sign-extended space, which
  λ ∈ [0,1] cannot reach:

| regime | (w_self, w_other) | CC | DD |
|---|---|---|---|
| cooperative | (0.4, 0.6) | 1.000 | 0.000 |
| self-interested | (1.0, 0.0) | 0.000 | 0.965 |
| spite | (0.6, −0.4) | 0.000 | **0.999** |
| pure spite | (0.0, −1.0) | 0.000 | **1.000** |
| self-abnegation | (−0.4, 0.6) | 1.000 | 0.000 |

Spite is behaviourally distinct from indifference: valuing the partner's loss
produces *more* stable mutual defection than merely disregarding the partner's
welfare. Reported in the revision and linked to the psychopathy literature
Referee 2 cites.

Raw data: `referee2_weight_grid.json` (500 runs), `referee2_quadrants.json`
(100 runs).

## Implementation notes discovered en route (affect Methods text)

1. The agents' decisions are driven by the module-level `PD_PAYOFFS` dict in
   `tom/tom_core.py` = (R=3, S=0, T=5, P=1). The pymdp C matrices and the
   realized-payoff logging in `run_pd_experiments.py` historically used
   (3, 1, 4, 2). Behavior is entirely governed by the former (C matrices have
   no behavioral effect). The revision's Methods now state the decision
   payoffs explicitly; this also resolves the reader question (Hyojun Choi,
   2 Jul) about `empathy_shift`, whose constants derive from (3, 0, 5, 1).
2. The paper's Table 2 / Fig 7 protocol pairs a sophisticated agent with a
   MYOPIC partner and reports mutual cooperation frequency. This was
   confirmed by exact reproduction (0.782/0.657/0.597) and is now stated in
   the revised manuscript (sophistication subsection + captions).
