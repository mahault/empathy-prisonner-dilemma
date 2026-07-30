#!/usr/bin/env python
"""
Figure 5: transition to cooperation, empirical against analytic.

Left  -- P(CC) as a function of empathy. The analytic curve is
             P(CC)(lambda) = sigma(beta (A + lambda B))^2,
         with A = q0 R + (1-q0) S - q1 T - (1-q1) P and B = (T-S)(1-q0+q1),
         q0 = q(C|C), q1 = q(C|D) read off the opponent model. The derivation
         assumes identical agents without learning or inversion, so the
         no-inversion simulation is the like-for-like comparison; the
         inversion-on curve is shown alongside it.
Right -- Memory effects: probability that an agent cooperates given its own
         previous action, which separates stubborn defection at low empathy
         from fragile cooperation near the transition.

No script for this figure existed in the repository; the published version
appears to have been produced by hand.

Run with the project virtualenv:
    .venv/Scripts/python.exe scripts/generate_pcc_figure.py
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from empathy.prisoners_dilemma import ToMEmpatheticAgent, Environment  # noqa: E402
from empathy.prisoners_dilemma.tom import TheoryOfMind  # noqa: E402
from empathy.prisoners_dilemma.tom.tom_core import COOPERATE  # noqa: E402
from run_referee_analyses import create_pd_config  # noqa: E402

R, S, T_, P = 3.0, 0.0, 5.0, 1.0
BETA = 4.0


class _Mock:
    qs = [np.array([0.5, 0.5])]
    beta = BETA


def opponent_qs(lambda_j=0.5):
    """q0 = q(C | I cooperate), q1 = q(C | I defect) under the opponent model."""
    prev = TheoryOfMind.DEFAULT_LAMBDA_J
    try:
        TheoryOfMind.DEFAULT_LAMBDA_J = lambda_j
        t = TheoryOfMind(other_model=_Mock(), beta_other=BETA)
        t.update_my_policy_belief(1.0)
        q0 = float(t.predict_opponent_action().q_response[COOPERATE])
        t.update_my_policy_belief(0.0)
        q1 = float(t.predict_opponent_action().q_response[COOPERATE])
    finally:
        TheoryOfMind.DEFAULT_LAMBDA_J = prev
    return q0, q1


def analytic(lams, q0, q1):
    A = q0 * R + (1 - q0) * S - q1 * T_ - (1 - q1) * P
    B = (T_ - S) * (1 - q0 + q1)
    p = 1.0 / (1.0 + np.exp(-BETA * (A + np.asarray(lams) * B)))
    lam_star = (np.log(2) / BETA - A) / B
    return p ** 2, A, B, lam_star


def simulate(lam, seed, T, inversion):
    np.random.seed(seed)
    cfg = create_pd_config(T=T, payoffs=(R, S, T_, P), legacy_C=True)
    a = [ToMEmpatheticAgent(config=cfg, agent_num=k, empathy_factor=lam,
                            use_inversion=inversion) for k in (0, 1)]
    env = Environment(K=2)
    acts, hi, hj = [0, 0], [], []
    for t in range(T):
        o = env.step(t=t, actions=acts)
        oi, oj = (a[0].o_init, a[1].o_init) if t == 0 else (o[0], o[1])
        x = a[0].step(t=t, observation=oi)["exp_action"]
        y = a[1].step(t=t, observation=oj)["exp_action"]
        acts = [x, y]
        hi.append(x)
        hj.append(y)
    return np.array(hi), np.array(hj)


def collect(lams, seeds, T):
    out = {"pcc_inv": [], "pcc_noinv": [], "p_c_given_c": [], "p_c_given_d": []}
    for lam in lams:
        cc_i, cc_n, pcc, pcd = [], [], [], []
        for s in range(seeds):
            hi, hj = simulate(lam, s, T, True)
            cc_i.append(np.mean((hi == 0) & (hj == 0)))
            prev, cur = hi[:-1], hi[1:]
            if (prev == 0).any():
                pcc.append(np.mean(cur[prev == 0] == 0))
            if (prev == 1).any():
                pcd.append(np.mean(cur[prev == 1] == 0))
            hi2, hj2 = simulate(lam, s, T, False)
            cc_n.append(np.mean((hi2 == 0) & (hj2 == 0)))
        out["pcc_inv"].append(float(np.mean(cc_i)))
        out["pcc_noinv"].append(float(np.mean(cc_n)))
        out["p_c_given_c"].append(float(np.mean(pcc)) if pcc else np.nan)
        out["p_c_given_d"].append(float(np.mean(pcd)) if pcd else np.nan)
        print(f"  lambda {lam:.2f}  P(CC) inv {out['pcc_inv'][-1]:.3f}  "
              f"no-inv {out['pcc_noinv'][-1]:.3f}", flush=True)
    return out


def draw(lams, data, q0, q1, out_path):
    an, A, B, lam_star = analytic(lams, q0, q1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(lams, an, "k-", lw=2, label="analytic $\\sigma(\\beta\\Delta U)^2$")
    ax1.plot(lams, data["pcc_noinv"], "o--", color="#C44E52", ms=5,
             label="simulated, no inversion")
    ax1.plot(lams, data["pcc_inv"], "s-", color="#4C72B0", ms=5,
             label="simulated, inversion on")
    ax1.axvline(lam_star, color="grey", ls=":", lw=1.5)
    ax1.annotate(f"$\\lambda^*={lam_star:.2f}$", xy=(lam_star, 0.44),
                 xytext=(lam_star + 0.06, 0.30), fontsize=10,
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="grey"))
    ax1.axhline(4 / 9, color="grey", ls=":", lw=0.8)
    ax1.set_xlabel(r"empathy $\lambda$", fontsize=12)
    ax1.set_ylabel("P(mutual cooperation)", fontsize=12)
    ax1.set_title("A.  Transition to cooperation", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=9, frameon=False)
    ax1.grid(alpha=0.3)

    ax2.plot(lams, data["p_c_given_c"], "o-", color="#55A868", ms=5,
             label=r"$P(C_t \mid C_{t-1})$")
    ax2.plot(lams, data["p_c_given_d"], "s-", color="#C44E52", ms=5,
             label=r"$P(C_t \mid D_{t-1})$")
    ax2.axvline(lam_star, color="grey", ls=":", lw=1.5)
    ax2.set_xlabel(r"empathy $\lambda$", fontsize=12)
    ax2.set_ylabel("P(cooperate now)", fontsize=12)
    ax2.set_title("B.  Memory effects", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=10, frameon=False)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")
    return dict(A=A, B=B, lam_star=float(lam_star), q0=q0, q1=q1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "images" / "fig5_pcc_new.png"))
    a = ap.parse_args()

    q0, q1 = opponent_qs()
    lams = [round(x, 2) for x in np.arange(0.05, 0.91, 0.05)]
    print(f"q0 = {q0:.4f}, q1 = {q1:.4f};  T={a.T}, {a.seeds} seeds")
    data = collect(lams, a.seeds, a.T)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    meta = draw(lams, data, q0, q1, a.out)
    Path(a.out).with_suffix(".json").write_text(
        json.dumps({"lambdas": lams, **data, **meta}, indent=1), encoding="utf-8")
    print(f"A = {meta['A']:.3f}, B = {meta['B']:.3f}, "
          f"lambda* = {meta['lam_star']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
