#!/usr/bin/env python
"""Verification pass for run_referee_analyses.py results.

Re-runs the load-bearing cells with a DIFFERENT seed range (100-124) to check
the headline numbers are not seed-dependent, and asserts the PD_PAYOFFS
swap/restore mechanism leaves no state behind.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_referee_analyses import (  # noqa: E402
    AGENT_TYPES, PAYOFF_STRUCTURES, run_pair,
    _set_decision_payoffs, _restore_decision_payoffs,
)
from empathy.prisoners_dilemma.tom.tom_core import PD_PAYOFFS  # noqa: E402

SEEDS = range(100, 125)
T = 100
failures = []


def check(name, condition, detail):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not condition:
        failures.append(name)


# --- Check 0: payoff swap + restore leaves original state ---
original = dict(PD_PAYOFFS)
_set_decision_payoffs(PAYOFF_STRUCTURES["high_temptation"])
swapped = dict(PD_PAYOFFS)
_restore_decision_payoffs()
check("payoff_restore", dict(PD_PAYOFFS) == original,
      f"restored={dict(PD_PAYOFFS) == original}, swap_changed={swapped != original}")

# --- Check 1: tom_only vs self_interested CC delta ~ 0 (new seeds) ---
def sym_cc(agent_type):
    rs = [run_pair("verify", agent_type, agent_type,
                   dict(AGENT_TYPES[agent_type]), dict(AGENT_TYPES[agent_type]),
                   T=T, seed=s) for s in SEEDS]
    return np.mean([r.freq_CC for r in rs]), np.mean([r.freq_DD for r in rs])

cc_si, dd_si = sym_cc("self_interested")
cc_to, dd_to = sym_cc("tom_only")
cc_em, dd_em = sym_cc("empathic")
delta = cc_to - cc_si
check("tom_only_delta", abs(delta) < 0.05,
      f"CC self_interested={cc_si:.3f}, tom_only={cc_to:.3f}, delta={delta:+.3f}")
check("empathic_cooperates", cc_em > 0.9, f"empathic CC={cc_em:.3f}")
check("defectors_defect", dd_si > 0.85 and dd_to > 0.85,
      f"DD self_interested={dd_si:.3f}, tom_only={dd_to:.3f}")

# --- Check 2: empathic exploited in mixed pairing (new seeds) ---
rs = [run_pair("verify", "empathic", "self_interested",
               dict(AGENT_TYPES["empathic"]), dict(AGENT_TYPES["self_interested"]),
               T=T, seed=s) for s in SEEDS]
expl = np.mean([r.exploitability for r in rs])
gap = np.mean([r.payoff_gap for r in rs])
check("empathic_exploited", expl > 0.8 and gap < -3,
      f"exploitability={expl:.2f}, payoff_gap={gap:+.2f}")

# --- Check 3: horizon reversal under weak_temptation lambda=0.3 (new seeds) ---
def horizon_coop(pname, lam, H):
    kwargs = dict(empathy_factor=lam, use_inversion=False,
                  use_sophisticated=(H > 1), planning_horizon=H)
    rs = [run_pair("verify", f"H{H}", f"H{H}", dict(kwargs), dict(kwargs),
                   T=T, seed=s, payoff_structure=pname, planning_horizon=H)
          for s in SEEDS]
    return np.mean([(r.coop_rate_i + r.coop_rate_j) / 2 for r in rs])

std_h1 = horizon_coop("standard", 0.3, 1)
std_h4 = horizon_coop("standard", 0.3, 4)
check("standard_erosion", std_h1 - std_h4 > 0.1,
      f"standard lam=0.3: H1={std_h1:.2f} -> H4={std_h4:.2f}")

wk_h1 = horizon_coop("weak_temptation", 0.3, 1)
wk_h4 = horizon_coop("weak_temptation", 0.3, 4)
check("weak_temptation_reversal", wk_h4 >= wk_h1 - 0.02,
      f"weak_temptation lam=0.3: H1={wk_h1:.2f} -> H4={wk_h4:.2f} "
      f"(reversal holds if H4 not lower)")

# --- Check 4: payoff structures actually produce different behavior ---
check("payoffs_differentiate", abs(std_h1 - wk_h1) > 0.1,
      f"standard H1={std_h1:.2f} vs weak_temptation H1={wk_h1:.2f}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED (seeds 100-124, independent of the reported run)")
