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

## Does implementing the rollout properly rescue the claim? No.

The Methods say future-step predictions are "conditioned on the simulated
history induced by the partial rollout". The code does not do this. We
implemented what the Methods describe (`--mode proper`): at each rollout step
t > 0 the opponent prediction is recomputed from an `ObservationContext` whose
`my_last_action` is the previous *simulated* action, with opponent inversion
ON so the reciprocity term rho * f(h_t) actually responds to it, and with
cumulative rather than averaged EFE so precision stays fixed.

Mutual cooperation frequency, 20 seeds, standard payoffs, inversion on:

| lambda | variant | H=1 | H=2 | H=3 | H=4 | H4-H1 |
|---|---|---|---|---|---|---|
| 0.3 | as published (static rollout, 1/H) | 0.7530 | 0.6300 | 0.5705 | 0.5300 | **-0.2230** |
| 0.3 | precision fixed (static rollout, sum) | 0.7530 | 0.7540 | 0.7540 | 0.7540 | +0.0010 |
| 0.3 | **proper (prefix-conditioned, sum)** | 0.7530 | 0.7575 | 0.7575 | 0.7575 | **+0.0045** |
| 0.5 | as published | 0.9870 | 0.9355 | 0.8715 | 0.8080 | **-0.1790** |
| 0.5 | precision fixed | 0.9870 | 0.9910 | 0.9910 | 0.9910 | +0.0040 |
| 0.5 | **proper** | 0.9870 | 0.9910 | 0.9910 | 0.9910 | **+0.0040** |
| 0.7 | as published | 1.0000 | 0.9920 | 0.9595 | 0.9145 | **-0.0855** |
| 0.7 | precision fixed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 0.7 | **proper** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.0000** |

Reading:

- The published erosion survives none of it. With precision held fixed the
  horizon dependence disappears whether or not the rollout is corrected.
- Conditioning the rollout on the simulated prefix adds essentially nothing:
  +0.0045 at lambda=0.3, and nothing at all at lambda=0.5 and 0.7 where
  behaviour is at ceiling. The prefix conditioning is active (it moves
  lambda=0.3 from 0.7540 to 0.7575), it simply has almost no leverage.
- A prior expectation that a corrected rollout would *reverse* the sign, with
  anticipated retaliation supporting cooperation, is **not** supported. The
  genuine effect is not negative, but it is not meaningfully positive either.
  It is approximately zero.

So the claim cannot be rescued by fixing the implementation. There is no
planning-horizon effect in this model in either direction, at H <= 4.

Scope of this conclusion: we tested one corrected rollout, the one the
Methods describe. A fuller sophisticated-inference treatment (branching over
opponent responses with belief updating in the rollout, depth-2 ToM, longer
horizons) could behave differently, and the codebase does contain an unused
`DepthTwoToM`. What we can say is that implementing what the paper claims to
implement yields no effect.

## A full sophisticated-inference treatment does not rescue it either

`scripts/run_sophisticated_rollout.py` implements what the previous section
listed as untested: expectimax branching over the partner's responses (not a
single simulated path), particle-posterior updating *inside* the rollout so the
agent plans over what it will come to believe, properly implemented depth-2 ToM
(the shipped `DepthTwoToM` is a stub that calls `super()` and is depth-1), and
accumulated rather than averaged EFE.

The decision layer is re-implemented standalone for speed. It is validated
against the shipped myopic agent first: 0.7705 / 0.9945 / 1.0000 versus the
shipped 0.7820 / 0.9975 / 1.0000 at lambda = 0.3 / 0.5 / 0.7, a maximum gap of
0.0115 against a per-seed s.e.m. of about 0.009.

Mutual cooperation frequency, 20 seeds, T=100, partner myopic:

| config | lambda | H=1 | H=2 | H=3 | H=4 | H4-H1 |
|---|---|---|---|---|---|---|
| depth-1 ToM, static | 0.3 | 0.7705 | 0.7700 | 0.7700 | 0.7700 | -0.0005 |
| depth-1 ToM, static | 0.5 | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.0000 |
| depth-1 ToM, static | 0.7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| **depth-2 ToM**, static | 0.3 | 0.7565 | 0.7565 | 0.7565 | 0.7565 | **0.0000** |
| **depth-2 ToM**, static | 0.5 | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.0000 |
| **depth-2 ToM**, static | 0.7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| learning in rollout | 0.3 | 0.2190 | 0.2100 | 0.1960 | 0.1975 | **-0.0215** |
| learning in rollout | 0.5 | 0.9480 | 0.9495 | 0.9535 | 0.9525 | +0.0045 |
| learning in rollout | 0.7 | 0.9995 | 0.9995 | 1.0000 | 1.0000 | +0.0005 |

Reading:

- **Depth-2 ToM changes the level, not the horizon dependence.** It moves
  lambda=0.3 cooperation from 0.7705 to 0.7565 and then stays flat to four
  decimals at every horizon. Recursive mentalising does not create a
  planning-depth effect.
- **Branching and belief updating are what can create one at all.** Only the
  learning config moves with H, and only at lambda=0.3.
- **Every fidelity improvement shrank it.** Learning in rollout gave -0.081
  before the empathy shift and reliability gating were added to match the
  shipped inversion model, and -0.0215 after. The manuscript reports -0.185.

Fidelity caveat, stated plainly: the learning config reproduces the shipped
agent at lambda = 0.5 (0.948 vs 0.987) and 0.7 (0.9995 vs 1.000) but *not* at
lambda = 0.3 (0.219 vs 0.753), which is the sensitive cell. The most likely
missing ingredient is the epistemic (expected information gain) term in the
EFE, which the paper credits with producing "early cooperation as
information-seeking" and which this standalone planner omits. So the -0.0215
should be read as an indication that in-rollout learning can produce a small
negative effect, not as a calibrated estimate. The direction of travel across
all three fidelity fixes was toward zero.

## Root cause: the static ToM predicts unconditional defection

The obvious question about the tables above is why the horizon effect is
*exactly* zero rather than merely small. Two candidate explanations were
tested.

**Rejected: sluggish partner memory.** The partner's belief about my strategy
is a running average over the whole history, so one simulated action moves it
by 1/n. Plausible bottleneck, but replacing it with exponential recency
weighting does not unlock the horizon (`--mode memory`). Even at alpha = 0.9,
essentially last-action memory, lambda=0.3 gives -0.0075 and lambda=0.5 gives
exactly 0.0000.

**Confirmed: the predicted partner never responds, because it is modelled as
selfish.** The static ToM computes the partner's expected free energy as

    G_j(a_j) = - sum_{a_i} pi_i(a_i) * payoff_j(a_i, a_j)

with no empathy term and no reciprocity term. Defection therefore strictly
dominates for the partner at every value of my cooperation rate p:

| p (partner's belief about my cooperation) | G_j(C) | G_j(D) | D better by |
|---|---|---|---|
| 0.0 | 0.00 | -1.00 | +1.00 |
| 0.5 | -1.50 | -3.00 | +1.50 |
| 1.0 | -3.00 | -5.00 | +2.00 |

Passed through the softmax this yields a predicted partner cooperation
probability of **0.03% to 1.8%**, and the residual dependence runs the *wrong*
way: the more I am believed to cooperate, the less likely the partner is
predicted to cooperate, because exploiting a cooperator pays better.

So under static ToM the agent believes, with near certainty, that its partner
will defect no matter what it does. There is no shadow of the future to plan
over. Every candidate policy faces the identical predicted partner, which is
exactly the condition under which G(pi) decomposes additively, the policy
softmax factorises, and the planner collapses to myopic-at-beta/H. The 1/H
averaging then converts that degeneracy into a smooth monotone "erosion" trend
that looks precisely like the hypothesis being tested.

### A consequence worth taking seriously

Under static ToM the agents cooperate roughly 77% of rounds at lambda=0.3
while predicting their partner cooperates about 1% of the time. The theory of
mind is not mildly miscalibrated, it is almost maximally wrong, and cooperation
happens anyway.

That is a much stronger version of the paper's own central claim than the paper
currently makes. Cooperation is not merely "not driven by" belief accuracy; it
survives beliefs that are close to inverted. The agent cooperates *while
expecting to be exploited*, purely because lambda weights the partner's
outcome. It also explains the model-comparison null (ToM-only minus
self-interested = +0.001) mechanically: the ToM prediction is a near-constant
"they will defect", so adding it changes nothing.

The static ToM's blind spot is specifically that it models the partner as
purely self-interested. The particle-filter inversion carries lambda_j as a
latent and can in principle infer an empathic partner. The horizon experiments
all used the static path.

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
