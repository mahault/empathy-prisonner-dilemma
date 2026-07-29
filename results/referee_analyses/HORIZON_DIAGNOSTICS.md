# Planning-horizon diagnostics

Run 2026-07-29 · `scripts/run_horizon_diagnostics.py` · standard protocol
(sophisticated agent vs myopic partner, static ToM, T=100, seeds 0-19).

Prompted by Hongju's review of the implementation (email 2026-07-29). His two
observations were correct, and the consequence is stronger than suspected.

## Result

**The planning-horizon effect reported in the paper is an artifact of the
softmax normalisation. It is not strategic lookahead.**

### Why, analytically

`OpponentSimulator.predict_response(step=t)` returns the same distribution for
every rollout step: it is computed from the opponent's belief about our
empirical cooperation rate, which is fixed for the duration of a rollout and
never depends on the candidate policy. So the per-step social EFE is a function
of that step's action alone, and

    G(pi) = (1/H) * sum_t g(a_t)

with the same `g` at every step. The policy softmax therefore factorises,

    exp(-beta * G(pi)) = prod_t exp(-(beta/H) * g(a_t))

and marginalising to the first action leaves every t >= 1 factor contributing
an identical constant to both a_0 = C and a_0 = D. What remains is

    P(a_0 = C) = softmax over g with precision beta/H

The sophisticated planner at horizon H is *exactly* a myopic planner whose
action precision has been divided by H. Averaging the accumulated EFE over the
horizon, while holding beta fixed, is the entire mechanism.

### Confirmed numerically

| check | result |
|---|---|
| sophisticated H=1 vs myopic, matched precision | identical (sanity) |
| sophisticated at horizon H with beta **vs** myopic with beta/H | **identical to 1e-12 in all 36 cells** (3 lambdas x 4 horizons x 3 payoff structures) |
| cumulative (summed) EFE instead of 1/H | cooperation **exactly constant** in H (spread 0.0) |
| rollout prediction updated from simulated prefix, summed EFE | lambda=0.3: 0.7820 -> 0.7790 across H=1..4 (**-0.003**); lambda=0.5 and 0.7 exactly flat |

For comparison, the effect reported in the manuscript is -0.185 at lambda=0.3
(H=1 to H=3). All of it is reproduced by changing beta to beta/3 with no
lookahead at all. Genuine policy-dependent lookahead, once the rollout
prediction is allowed to respond to the simulated prefix, contributes about
-0.003.

## What this invalidates

1. **"Increasing planning depth reduces cooperation at moderate empathy"**
   (Table 2, Fig 7, Discussion, Conclusion point four). As implemented, this
   says only that dividing the EFE by H lowers effective precision.
2. **The threshold shift** lambda* = 0.35 -> 0.45 with horizon: same cause.
3. **The revision's payoff-structure sweep** (new Table, Fig 9) and the claim
   that "deeper planning amplifies whatever the payoff-empathy configuration
   favours". This is not just unsupported, it is the wrong direction: lowering
   precision moves behaviour *toward* indifference, not away from it. It is
   why the weak-temptation cell at lambda=0.3 *rose* (0.169 -> 0.196) while
   the high-cooperation cells fell. Everything regresses toward 0.5.

## Options

- Remove the planning-depth claim. The prediction-vs-valuation dissociation
  and the exploitation asymmetry stand on their own and are unaffected.
- Keep a corrected version: report that under this planner the horizon enters
  only through effective precision, which is itself a defensible negative
  result about naive EFE averaging in sophisticated inference.
- Fix the planner (sum rather than average; make rollout predictions depend on
  the simulated prefix) and re-run. On the evidence above the effect would be
  roughly -0.003 rather than -0.185, so the claim would not survive in its
  current form either way.

Nothing here touches Analysis 1 (model comparison) or Analysis 3 (weight
space), which do not use the sophisticated planner.

Reproduce: `python scripts/run_horizon_diagnostics.py --mode all`
