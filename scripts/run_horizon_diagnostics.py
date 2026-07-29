#!/usr/bin/env python
"""
Diagnostics for the planning-horizon result (Hongju's checks, 2026-07-29).

Concern raised: during rollouts the opponent prediction is reused unchanged for
future steps, and policy EFE is divided by the horizon before the softmax. If
so, part or all of the "deeper planning erodes cooperation" effect could be a
reduction in effective decision precision rather than strategic lookahead.

ANALYTIC CLAIM this script tests numerically
-------------------------------------------
In the current implementation `OpponentSimulator.predict_response(step=t)`
returns the same distribution q for every t (it depends only on the opponent's
belief about our empirical cooperation rate, which is fixed during a rollout,
and never on the candidate policy). So the per-step social EFE depends only on
the action taken at that step:

    G(pi) = (1/H) * sum_t g(a_t)

with g the same function at every step. The policy softmax then factorises:

    exp(-beta*G(pi)) = prod_t exp(-(beta/H) * g(a_t))

and when we marginalise to the first action every factor for t >= 1 contributes
the same constant to both a_0 = C and a_0 = D. Hence

    P(a_0 = C) = softmax over g with precision beta/H

i.e. the sophisticated planner at horizon H is EXACTLY a myopic planner with
its precision divided by H. Under this account the horizon effect is a
precision artifact, and it predicts cooperation should regress towards
indifference as H grows, in whichever direction that happens to be.

Checks
------
  equiv    : sophisticated H=1 vs myopic, matched precision (sanity)
  precision: sophisticated at horizon H, beta  ==  myopic at beta/H
  sum      : replace the 1/H average with a plain sum. Under the analysis above
             this removes the H dependence entirely.
  history  : make the rollout prediction depend on the simulated prefix, so
             lookahead can actually do something, and see what survives.

Usage:
    python scripts/run_horizon_diagnostics.py --mode all
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from empathy.prisoners_dilemma.tom import sophisticated_planner as sp_mod
from empathy.prisoners_dilemma.tom.sophisticated_planner import SophisticatedPlanner
from empathy.prisoners_dilemma.tom.tom_core import COOPERATE, DEFECT, PD_PAYOFFS

from run_referee_analyses import run_pair  # noqa: E402

LAMBDAS = (0.3, 0.5, 0.7)
HORIZONS = (1, 2, 3, 4)
BETA = 4.0
T = 100

# --- switches consulted by the patched planner -------------------------
CONFIG = {"normalize": "mean", "rollout": "static", "hist_alpha": 0.5}

_orig_evaluate = SophisticatedPlanner.evaluate_policy


def _patched_evaluate(self, policy):
    """evaluate_policy with selectable normalisation and rollout prediction."""
    lam = self.empathy_factor
    total_G = 0.0
    steps = []

    tom = getattr(self.opponent_sim, "tom", None)
    saved = None
    if CONFIG["rollout"] == "history" and tom is not None:
        saved = np.array(tom._believed_my_policy, dtype=float).copy()
        p_coop = float(saved[0])

    for t, my_action in enumerate(policy):
        if CONFIG["rollout"] == "history" and tom is not None:
            # Let the opponent's belief about our policy absorb the simulated
            # prefix, so the prediction depends on the candidate policy.
            tom.update_my_policy_belief(float(np.clip(p_coop, 0.0, 1.0)))

        q_response = self.opponent_sim.predict_response(step=t)

        G_self = 0.0
        for a_j in (COOPERATE, DEFECT):
            my_payoff, _ = PD_PAYOFFS[(my_action, a_j)]
            G_self += q_response[a_j] * (-my_payoff)

        G_other_per_action = np.zeros(2)
        for a_j in (COOPERATE, DEFECT):
            _, other_payoff = PD_PAYOFFS[(my_action, a_j)]
            G_other_per_action[a_j] = -other_payoff
        G_other_expected = float(np.sum(q_response * G_other_per_action))

        G_t = (1 - lam) * G_self + lam * G_other_expected
        total_G += G_t
        steps.append({"action": my_action, "G_step": G_t})

        if CONFIG["rollout"] == "history" and tom is not None:
            a = CONFIG["hist_alpha"]
            p_coop = (1 - a) * p_coop + a * (1.0 if my_action == COOPERATE else 0.0)

    if saved is not None and tom is not None:
        tom.update_my_policy_belief(float(saved[0]))

    G_policy = total_G / self.horizon if CONFIG["normalize"] == "mean" else total_G
    return G_policy, {"steps": steps, "total_G": total_G, "horizon": self.horizon}


SophisticatedPlanner.evaluate_policy = _patched_evaluate
sp_mod.SophisticatedPlanner.evaluate_policy = _patched_evaluate


# ----------------------------------------------------------------------

def cc(lam, H, beta=BETA, sophisticated=None, seeds=range(20), beta_j=BETA):
    """Mean mutual cooperation frequency, paper protocol (soph vs myopic partner).

    beta applies to the planning agent i only; the partner j keeps beta_j, since
    only agent i's decision rule is under test.
    """
    if sophisticated is None:
        sophisticated = H > 1
    ki = dict(empathy_factor=lam, use_inversion=False,
              use_sophisticated=sophisticated, planning_horizon=H,
              beta_self=beta)
    kj = dict(empathy_factor=lam, use_inversion=False, beta_self=beta_j)
    rs = [run_pair("diag", f"H{H}", "H1", dict(ki), dict(kj), T=T, seed=s,
                   payoff_structure="standard", planning_horizon=H,
                   legacy_C=True)
          for s in seeds]
    return float(np.mean([r.freq_CC for r in rs]))


def check_equiv():
    print("=" * 72)
    print("CHECK 1: sophisticated H=1 vs myopic, same precision")
    print("=" * 72)
    CONFIG.update(normalize="mean", rollout="static")
    ok = True
    for lam in LAMBDAS:
        a = cc(lam, 1, sophisticated=True)
        b = cc(lam, 1, sophisticated=False)
        same = abs(a - b) < 1e-12
        ok &= same
        print(f"  lambda={lam}: soph H=1 CC={a:.4f}  myopic CC={b:.4f}  match={same}")
    print(f"\n  Sanity check passed: {ok}")
    return ok


def check_precision():
    print("=" * 72)
    print("CHECK 2: sophisticated at horizon H (beta) == myopic at beta/H ?")
    print("=" * 72)
    CONFIG.update(normalize="mean", rollout="static")
    ok = True
    print(f"  {'lambda':>7} {'H':>3} {'soph(beta)':>11} {'myopic(beta/H)':>15} {'match':>7}")
    for lam in LAMBDAS:
        for H in HORIZONS:
            a = cc(lam, H, beta=BETA, sophisticated=True)
            b = cc(lam, 1, beta=BETA / H, sophisticated=False)
            same = abs(a - b) < 1e-12
            ok &= same
            print(f"  {lam:>7} {H:>3} {a:>11.4f} {b:>15.4f} {str(same):>7}")
    print(f"\n  All horizons reduce exactly to a precision rescaling: {ok}")
    if ok:
        print("  => the horizon effect in this implementation IS a precision artifact.")
    return ok


def check_sum():
    print("=" * 72)
    print("CHECK 3: horizon effect with cumulative (summed) EFE instead of 1/H")
    print("=" * 72)
    print(f"  {'lambda':>7}" + "".join(f"{'H='+str(h):>9}" for h in HORIZONS)
          + f"{'spread':>9}")
    flat = True
    for lam in LAMBDAS:
        CONFIG.update(normalize="sum", rollout="static")
        vals = [cc(lam, H, sophisticated=True) for H in HORIZONS]
        spread = max(vals) - min(vals)
        flat &= spread < 1e-12
        print(f"  {lam:>7}" + "".join(f"{v:>9.4f}" for v in vals) + f"{spread:>9.2e}")
    CONFIG.update(normalize="mean")
    print(f"\n  Cooperation independent of H once EFE is summed: {flat}")
    if flat:
        print("  => no genuine lookahead effect remains. The 1/H average was the effect.")
    return flat


def check_history():
    print("=" * 72)
    print("CHECK 4: rollout prediction updated from the simulated prefix")
    print("=" * 72)
    print("  (diagnostic only: opponent's belief about our policy absorbs the")
    print("   simulated actions with weight alpha=0.5 per step)")
    for norm in ("mean", "sum"):
        print(f"\n  normalize={norm}")
        print(f"  {'lambda':>7}" + "".join(f"{'H='+str(h):>9}" for h in HORIZONS))
        for lam in LAMBDAS:
            CONFIG.update(normalize=norm, rollout="history")
            vals = [cc(lam, H, sophisticated=True) for H in HORIZONS]
            print(f"  {lam:>7}" + "".join(f"{v:>9.4f}" for v in vals))
    CONFIG.update(normalize="mean", rollout="static")
    print("\n  Compare the sum rows against Check 3: any H dependence here is")
    print("  attributable to genuine policy-dependent lookahead, not precision.")


def main():
    p = argparse.ArgumentParser(description="Planning-horizon diagnostics")
    p.add_argument("--mode", choices=["equiv", "precision", "sum", "history", "all"],
                   default="all")
    a = p.parse_args()
    if a.mode in ("equiv", "all"):
        check_equiv()
    if a.mode in ("precision", "all"):
        check_precision()
    if a.mode in ("sum", "all"):
        check_sum()
    if a.mode in ("history", "all"):
        check_history()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
