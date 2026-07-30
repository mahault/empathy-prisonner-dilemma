"""Do empathic agents cope with non-learning partners, and do they learn?"""
import sys
from pathlib import Path
ROOT = Path("C:/Users/User/Desktop/projects/github/empathy-prisonner-dilemma")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from empathy.prisoners_dilemma.agent import ToMEmpatheticAgent
from empathy.prisoners_dilemma.tom.tom_core import COOPERATE, DEFECT
from run_referee_analyses import create_pd_config, EMPATHIC_LAMBDA

C, D = COOPERATE, DEFECT
R, S, T_, P = 3.0, 0.0, 5.0, 1.0
PAY = {(0, 0): R, (0, 1): S, (1, 0): T_, (1, 1): P}


# ---- fixed partner strategies: (my_hist, their_hist) -> action -------------
def allc(mine, theirs):   return C
def alld(mine, theirs):   return D
def tft(mine, theirs):    return C if not mine else mine[-1]
def grim(mine, theirs):   return D if any(a == D for a in mine) else C
def rand(mine, theirs):   return int(np.random.rand() < 0.5)

STRATS = {"ALLC": allc, "ALLD": alld, "TFT": tft, "GRIM": grim, "RANDOM": rand}
AGENTS = {
    "self_interested": dict(empathy_factor=0.0, use_inversion=False),
    "tom_only":        dict(empathy_factor=0.0, use_inversion=True),
    "empathic":        dict(empathy_factor=EMPATHIC_LAMBDA, use_inversion=True),
}


def play(agent_kwargs, strat, T=100, seed=0):
    """Agent is player 0; the fixed strategy is player 1."""
    np.random.seed(seed)
    ag = ToMEmpatheticAgent(config=create_pd_config(T=T, payoffs=(R, S, T_, P),
                                                    legacy_C=True),
                            agent_num=0, **agent_kwargs)
    # Record the prediction the agent actually used each round.
    preds = []
    if ag.gated_tom is not None:
        orig = ag.gated_tom.predict_opponent_action
        def spy(context=None, _o=orig):
            q = _o(context)
            preds.append(float(q[C]))
            return q
        ag.gated_tom.predict_opponent_action = spy

    mine, theirs, lam_tr, rel_tr, alpha_tr = [], [], [], [], []
    obs = None
    for t in range(T):
        r = ag.step(t=t, observation=obs)
        a = r["exp_action"]
        b = strat(mine, theirs)
        mine.append(a); theirs.append(b)
        obs = 2 * a + b                      # encoding from player 0's view
        if "inversion" in r:
            lam_tr.append(r["inversion"]["lambda_j_belief"]["mean"])
            rel_tr.append(r["inversion"]["reliability"])
            alpha_tr.append(r["inversion"]["profile_summary"]["mean_alpha"])

    mine_a, theirs_a = np.array(mine), np.array(theirs)
    # Brier score of the one-step-ahead partner prediction (lower is better).
    brier = None
    if preds:
        k = min(len(preds), len(theirs))
        p = np.array(preds[:k])
        truth = (theirs_a[-k:] == C).astype(float)
        brier = float(np.mean((p - truth) ** 2))
    return dict(
        coop=float(1 - mine_a.mean()),
        my_pay=float(np.mean([PAY[(x, y)] for x, y in zip(mine, theirs)])),
        their_pay=float(np.mean([PAY[(y, x)] for x, y in zip(mine, theirs)])),
        brier=brier, lam=lam_tr, rel=rel_tr, alpha=alpha_tr,
    )


SEEDS = range(10)
print("=" * 74)
print("A. AGENTS vs FIXED-STRATEGY PARTNERS  (20 rounds x 10 seeds, standard PD)")
print("=" * 74)
print(f"{'partner':<9}{'agent':<17}{'agent coop':>11}{'agent pay':>11}"
      f"{'partner pay':>13}{'exploited?':>12}")
store = {}
for sname, strat in STRATS.items():
    for aname, kw in AGENTS.items():
        rs = [play(kw, strat, T=100, seed=s) for s in SEEDS]
        store[(sname, aname)] = rs
        co = np.mean([r["coop"] for r in rs])
        mp = np.mean([r["my_pay"] for r in rs])
        tp = np.mean([r["their_pay"] for r in rs])
        flag = "YES" if tp - mp > 0.5 else ""
        print(f"{sname:<9}{aname:<17}{co:>11.3f}{mp:>11.2f}{tp:>13.2f}{flag:>12}")
    print()

print("=" * 74)
print("B. IS THE PARTNER MODEL ACTUALLY LEARNING?")
print("=" * 74)
print("Brier score of one-step-ahead partner prediction (0 = perfect, "
      "0.25 = coin flip)")
print(f"{'partner':<9}{'tom_only':>11}{'empathic':>11}")
for sname in STRATS:
    row = [np.mean([r["brier"] for r in store[(sname, a)]])
           for a in ("tom_only", "empathic")]
    print(f"{sname:<9}{row[0]:>11.4f}{row[1]:>11.4f}")

print("\nPosterior over partner empathy lambda_j (prior mean 0.5), empathic agent")
print(f"{'partner':<9}{'round 1':>10}{'round 25':>10}{'round 100':>11}"
      f"{'mean alpha':>12}{'reliability':>13}")
for sname in STRATS:
    rs = store[(sname, "empathic")]
    lam = np.array([r["lam"] for r in rs])
    al = np.array([r["alpha"] for r in rs])
    rel = np.array([r["rel"] for r in rs])
    print(f"{sname:<9}{lam[:, 0].mean():>10.3f}{lam[:, 24].mean():>10.3f}"
          f"{lam[:, -1].mean():>11.3f}{al[:, -1].mean():>12.3f}"
          f"{rel[:, -1].mean():>13.3f}")

print("\nIdentifiability: can the filter tell ALLC from ALLD?")
lc = np.array([r["lam"][-1] for r in store[("ALLC", "empathic")]])
ld = np.array([r["lam"][-1] for r in store[("ALLD", "empathic")]])
ac = np.array([r["alpha"][-1] for r in store[("ALLC", "empathic")]])
ad = np.array([r["alpha"][-1] for r in store[("ALLD", "empathic")]])
print(f"  lambda_j : ALLC {lc.mean():.3f} +/- {lc.std():.3f}   "
      f"ALLD {ld.mean():.3f} +/- {ld.std():.3f}   separation {lc.mean()-ld.mean():+.3f}")
print(f"  alpha    : ALLC {ac.mean():.3f} +/- {ac.std():.3f}   "
      f"ALLD {ad.mean():.3f} +/- {ad.std():.3f}   separation {ac.mean()-ad.mean():+.3f}")
