#!/usr/bin/env python
"""
Figure 7 (replacement): planning depth pays only against a reciprocal partner.

Panel A: agent cooperation rate by planning horizon, one line per partner.
Panel B: agent mean payoff by planning horizon, same partners.
Panel C: the lever arm that lookahead acts through,
             dq = q(partner C | I cooperated) - q(partner C | I defected),
         which is near zero for every partner except tit-for-tat.

Panel C is the explanation for panels A and B: multi-step planning can only
change behaviour when my action changes what the partner does next. Among
these agents in self-play that quantity is ~0, which is why the horizon has
no effect there.

Run with the project virtualenv:
    .venv/Scripts/python.exe scripts/generate_partner_planning_figure.py
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

from empathy.prisoners_dilemma.tom.inversion import ObservationContext  # noqa: E402
from run_partner_types import STRATS, play  # noqa: E402

HORIZONS = (1, 2, 3, 4)
LAM = 0.3
COLOURS = {"ALLC": "#4C72B0", "ALLD": "#C44E52", "TFT": "#55A868",
           "GRIM": "#8172B2", "RANDOM": "#937860"}


def collect(T, seeds):
    data = {}
    for name, strat in STRATS.items():
        co, pay = [], []
        for H in HORIZONS:
            kw = dict(empathy_factor=LAM, use_inversion=True,
                      use_sophisticated=(H > 1), planning_horizon=H)
            rs = [play(kw, strat, T=T, seed=s) for s in range(seeds)]
            co.append(float(np.mean([r["coop"] for r in rs])))
            pay.append(float(np.mean([r["my_pay"] for r in rs])))
        # lever arm, measured on a myopic run so it reflects what was learned
        kw = dict(empathy_factor=LAM, use_inversion=True)
        dqs = []
        for s in range(seeds):
            r = play(kw, strat, T=T, seed=s, return_agent=True)
            inv = r["agent"].inversion
            q = [float(inv.predict_action(ObservationContext(
                    my_last_action=m, their_last_action=None,
                    joint_outcome=None, round_number=T))[0]) for m in (0, 1)]
            dqs.append(q[0] - q[1])
        data[name] = dict(coop=co, payoff=pay, dq=float(np.mean(dqs)),
                          dq_sd=float(np.std(dqs)))
        print(f"  {name:<8} coop {co[0]:.3f}->{co[-1]:.3f}   "
              f"payoff {pay[0]:.2f}->{pay[-1]:.2f}   dq {data[name]['dq']:+.3f}",
              flush=True)
    return data


def draw(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for name, d in data.items():
        axes[0].plot(HORIZONS, d["coop"], marker="o", label=name,
                     color=COLOURS[name])
        axes[1].plot(HORIZONS, d["payoff"], marker="o", label=name,
                     color=COLOURS[name])

    axes[0].set_xlabel("planning horizon $H$")
    axes[0].set_ylabel("agent cooperation rate")
    axes[0].set_title(f"A. Cooperation by horizon ($\\lambda={LAM}$)")
    axes[0].set_xticks(HORIZONS)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("planning horizon $H$")
    axes[1].set_ylabel("agent mean payoff")
    axes[1].set_title("B. Payoff by horizon")
    axes[1].set_xticks(HORIZONS)
    axes[1].grid(alpha=0.3)

    names = list(data.keys())
    vals = [data[n]["dq"] for n in names]
    errs = [data[n]["dq_sd"] for n in names]
    axes[2].bar(names, vals, yerr=errs, capsize=4,
                color=[COLOURS[n] for n in names])
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_ylabel(r"$\Delta q$  (lever arm)")
    axes[2].set_title("C. Does my action move the partner?")
    axes[2].set_ylim(-0.15, 1.05)
    axes[2].grid(alpha=0.3, axis="y")
    axes[2].annotate("only TFT reciprocates,\nand only there does\nhorizon matter",
                     xy=(2, data["TFT"]["dq"]), xytext=(2.6, 0.62),
                     fontsize=9, ha="left",
                     arrowprops=dict(arrowstyle="->", lw=0.9))

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=60)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default=str(PROJECT_ROOT / "images" /
                                        "fig7_partner_planning.png"))
    a = p.parse_args()
    print(f"collecting: lambda={LAM}, T={a.T}, {a.seeds} seeds, "
          f"horizons {HORIZONS}")
    data = collect(a.T, a.seeds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    draw(data, a.out)
    js = Path(a.out).with_suffix(".json")
    js.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {js}  (cross-check against results/rebaseline/PAPER_NUMBERS.md)")


if __name__ == "__main__":
    raise SystemExit(main())
