"""Why does planning depth do nothing? Measure the lever arm lookahead needs."""
import sys
from pathlib import Path
ROOT = Path("C:/Users/User/Desktop/projects/github/empathy-prisonner-dilemma")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from empathy.prisoners_dilemma.tom.tom_core import COOPERATE, DEFECT
from empathy.prisoners_dilemma.tom.inversion import ObservationContext
from run_partner_types import play, STRATS, PAY
from run_referee_analyses import run_pair

C, D = COOPERATE, DEFECT
SEEDS = range(10)

# ---------------------------------------------------------------- lever arm
# Lookahead can only matter if what I do now changes what the partner does
# next. Under the learned model that is exactly:
#     dq = q(partner C | I cooperated) - q(partner C | I defected)
print("=" * 78)
print("1. LEVER ARM: does my action change the partner's predicted next move?")
print("=" * 78)
print("   dq = q(partner C | I cooperated last) - q(partner C | I defected last)")
print(f"\n{'partner':<12}{'inferred rho':>14}{'q(C|I coop)':>13}{'q(C|I def)':>12}{'dq':>9}")


def lever(strat, lam=0.3, T=100, seed=0):
    r = play(dict(empathy_factor=lam, use_inversion=True), strat, T=T, seed=seed,
             return_agent=True)
    ag = r["agent"]
    inv = ag.inversion
    base = ag._build_observation_context(T)
    out = []
    for mine in (C, D):
        ctx = ObservationContext(
            my_last_action=mine, their_last_action=base.their_last_action,
            joint_outcome=base.joint_outcome, round_number=T)
        out.append(float(inv.predict_action(ctx)[C]))
    rho = inv.get_profile_summary()["mean_reciprocity"]
    return rho, out[0], out[1]


for sname, strat in STRATS.items():
    rows = np.array([lever(strat, seed=s) for s in SEEDS])
    rho, qc, qd = rows.mean(axis=0)
    print(f"{sname:<12}{rho:>14.3f}{qc:>13.3f}{qd:>12.3f}{qc-qd:>9.3f}")

# self-play: the only condition the paper ever tested
print("\n   self-play (the paper's setting) measured inside the pair:")
for lam in (0.3, 0.5):
    rhos = []
    for s in SEEDS:
        np.random.seed(s)
        res = run_pair("d", "i", "j",
                       dict(empathy_factor=lam, use_inversion=True),
                       dict(empathy_factor=lam, use_inversion=True),
                       T=100, seed=s, legacy_C=True)
        rhos.append(res)
    # rerun one pair to grab the agent's own posterior
    print(f"     lambda={lam}: mean CC={np.mean([r.freq_CC for r in rhos]):.3f}, "
          f"partner coop rate={np.mean([r.coop_rate_j for r in rhos]):.3f} "
          f"(near-constant => nothing to reciprocate)")

# ------------------------------------------------------- horizon vs partner
print("\n" + "=" * 78)
print("2. PLANNING HORIZON AGAINST PARTNERS THAT ACTUALLY RECIPROCATE")
print("=" * 78)
print("   agent cooperation rate by horizon, lambda = 0.3")
print(f"\n{'partner':<12}{'H=1':>9}{'H=2':>9}{'H=3':>9}{'H=4':>9}{'H4-H1':>9}")
for sname, strat in STRATS.items():
    vals = []
    for H in (1, 2, 3, 4):
        kw = dict(empathy_factor=0.3, use_inversion=True,
                  use_sophisticated=(H > 1), planning_horizon=H)
        vals.append(np.mean([play(kw, strat, T=60, seed=s)["coop"] for s in SEEDS]))
    print(f"{sname:<12}" + "".join(f"{v:>9.3f}" for v in vals)
          + f"{vals[-1]-vals[0]:>+9.3f}")

print("\n   agent MEAN PAYOFF by horizon, lambda = 0.3 (does lookahead pay?)")
print(f"\n{'partner':<12}{'H=1':>9}{'H=2':>9}{'H=3':>9}{'H=4':>9}{'H4-H1':>9}")
for sname, strat in STRATS.items():
    vals = []
    for H in (1, 2, 3, 4):
        kw = dict(empathy_factor=0.3, use_inversion=True,
                  use_sophisticated=(H > 1), planning_horizon=H)
        vals.append(np.mean([play(kw, strat, T=60, seed=s)["my_pay"] for s in SEEDS]))
    print(f"{sname:<12}" + "".join(f"{v:>9.3f}" for v in vals)
          + f"{vals[-1]-vals[0]:>+9.3f}")
