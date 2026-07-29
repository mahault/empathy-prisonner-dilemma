#!/usr/bin/env python
"""Publication figures for the JRSI rsif-2026-0208 revision.

Reads the JSON produced by run_referee_analyses.py and run_referee2_weights.py
and writes three figures matching the style of the existing paper figures:

    fig8_model_comparison.png   Referee 1 point 8  (prediction vs valuation)
    fig9_payoff_horizon.png     Referee 1 point 6  (horizon x payoff structure)
    fig10_weight_space.png      Referee 2 point 2  (self/other weight space)

Usage:
    python scripts/generate_revision_figures.py [--out images]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results" / "referee_analyses"

BLUE, RED, GREEN = "#3498db", "#e74c3c", "#2ecc71"
GREY = "#7f8c8d"
LAMBDA_COLORS = {0.3: BLUE, 0.5: RED, 0.7: GREEN}


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def sem(rows, key):
    if not rows:
        return float("nan")
    v = np.array([r[key] for r in rows], dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


# -----------------------------------------------------------------------
# Figure 8: model comparison
# -----------------------------------------------------------------------

def fig_model_comparison(out_dir):
    rows = load("comparison_results.json")
    types = ["self_interested", "tom_only", "empathic"]
    labels = {"self_interested": "self-interested\n$\\lambda=0$, no inversion",
              "tom_only": "ToM-only\n$\\lambda=0$, inversion on",
              "empathic": "empathic\n$\\lambda=0.6$, inversion on"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: symmetric pairings, CC and DD
    ax = axes[0]
    cc = [mean([r for r in rows if r["type_i"] == t and r["type_j"] == t], "freq_CC")
          for t in types]
    dd = [mean([r for r in rows if r["type_i"] == t and r["type_j"] == t], "freq_DD")
          for t in types]
    cc_e = [sem([r for r in rows if r["type_i"] == t and r["type_j"] == t], "freq_CC")
            for t in types]
    dd_e = [sem([r for r in rows if r["type_i"] == t and r["type_j"] == t], "freq_DD")
            for t in types]

    x = np.arange(len(types))
    w = 0.36
    ax.bar(x - w / 2, cc, w, yerr=cc_e, capsize=4, color=GREEN,
           edgecolor="black", linewidth=0.6, label="mutual cooperation (CC)")
    ax.bar(x + w / 2, dd, w, yerr=dd_e, capsize=4, color=RED,
           edgecolor="black", linewidth=0.6, label="mutual defection (DD)")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in types], fontsize=9)
    ax.set_ylabel("Frequency of rounds", fontsize=12)
    ax.set_ylim(0, 1.14)
    ax.set_title("A.  Symmetric pairings", fontsize=13, loc="left", fontweight="bold")
    ax.legend(fontsize=9, loc="upper center", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # annotate the null result
    delta = cc[types.index("tom_only")] - cc[types.index("self_interested")]
    ax.annotate(f"$\\Delta$CC = {delta:+.3f}",
                xy=(0.5, 0.08), xycoords="data", ha="center", fontsize=10)
    ax.annotate("", xy=(0 + w / 2, 0.05), xytext=(1 - w / 2, 0.05),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))

    # Panel B: mixed pairings. Plot each agent's own cooperation rate, which
    # shows the exploitation directly; annotate the payoff gap as text rather
    # than as a line, since the pairings are categories and not a trend.
    ax = axes[1]
    mixed = [("empathic", "self_interested", "empathic vs\nself-interested"),
             ("empathic", "tom_only", "empathic vs\nToM-only"),
             ("self_interested", "tom_only", "self-interested vs\nToM-only")]
    ci, cj, gap, names = [], [], [], []
    for ti, tj, lab in mixed:
        cell = [r for r in rows if r["type_i"] == ti and r["type_j"] == tj]
        ci.append(mean(cell, "coop_rate_i"))
        cj.append(mean(cell, "coop_rate_j"))
        gap.append(mean(cell, "payoff_gap"))
        names.append(lab)

    x = np.arange(len(mixed))
    w = 0.36
    ax.bar(x - w / 2, ci, w, color=BLUE, edgecolor="black", linewidth=0.6,
           label="first agent")
    ax.bar(x + w / 2, cj, w, color=GREY, edgecolor="black", linewidth=0.6,
           label="second agent")
    for xi, (a, b, g) in enumerate(zip(ci, cj, gap)):
        top = max(a, b)
        ax.annotate(f"payoff gap {g:+.2f}", xy=(xi, top + 0.06),
                    ha="center", fontsize=9, color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Cooperation rate", fontsize=12)
    ax.set_ylim(0, 1.24)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left", ncol=2, frameon=False)
    ax.set_title("B.  Mixed pairings", fontsize=13, loc="left", fontweight="bold")

    fig.tight_layout()
    path = out_dir / "fig8_model_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# -----------------------------------------------------------------------
# Figure 9: horizon x payoff structure
# -----------------------------------------------------------------------

def fig_payoff_horizon(out_dir):
    rows = load("horizon_results.json")
    structures = [("standard", "standard  $(R,S,T,P)=(3,0,5,1)$"),
                  ("weak_temptation", "weak temptation  $(3,1,4,2)$"),
                  ("high_temptation", "high temptation  $(5,0,8,1)$")]
    horizons = sorted({r["planning_horizon"] for r in rows})
    lambdas = sorted({r["lambda_i"] for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (key, title), letter in zip(axes, structures, "ABC"):
        for lam in lambdas:
            ys, es = [], []
            for H in horizons:
                cell = [r for r in rows if r["payoff_structure"] == key
                        and r["planning_horizon"] == H and r["lambda_i"] == lam]
                ys.append(mean(cell, "freq_CC"))
                es.append(sem(cell, "freq_CC"))
            ys, es = np.array(ys), np.array(es)
            ax.plot(horizons, ys, "o-", color=LAMBDA_COLORS[lam], lw=2.2,
                    markersize=6, label=f"$\\lambda={lam}$")
            ax.fill_between(horizons, ys - es, ys + es,
                            color=LAMBDA_COLORS[lam], alpha=0.18)
        ax.set_xticks(horizons)
        ax.set_xlabel("Planning horizon $H$", fontsize=12)
        ax.set_title(f"{letter}.  {title}", fontsize=11.5, loc="left",
                     fontweight="bold")
        ax.set_ylim(-0.03, 1.08)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mutual cooperation frequency", fontsize=12)
    axes[0].legend(fontsize=10, loc="lower left", frameon=True)

    # flag the two rows that coincide (empathy rescales the temptation margin)
    axes[0].annotate("$\\lambda=0.3$ here matches\n$\\lambda=0.5$ under weak temptation",
                     xy=(2, 0.66), xytext=(2.15, 0.30), fontsize=8.5, color=GREY,
                     arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    axes[1].annotate("no erosion:\ncooperation already collapsed",
                     xy=(3, 0.19), xytext=(1.6, 0.42), fontsize=8.5, color=GREY,
                     arrowprops=dict(arrowstyle="->", color=GREY, lw=1))

    fig.tight_layout()
    path = out_dir / "fig9_payoff_horizon.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# -----------------------------------------------------------------------
# Figure 10: self/other weight space
# -----------------------------------------------------------------------

def fig_weight_space(out_dir):
    grid = load("referee2_weight_grid.json")
    quad = load("referee2_quadrants.json")

    ws = sorted({r["w_self"] for r in grid})
    wo = sorted({r["w_other"] for r in grid})
    M = np.full((len(wo), len(ws)), np.nan)
    agg = defaultdict(list)
    for r in grid:
        agg[(r["w_self"], r["w_other"])].append(r["freq_CC"])
    for i, o in enumerate(wo):
        for j, s in enumerate(ws):
            vals = agg.get((s, o))
            if vals:
                M[i, j] = float(np.mean(vals))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel A: heatmap over the positive quadrant, with iso-ratio lines
    ax = axes[0]
    im = ax.imshow(M, origin="lower", cmap="RdYlGn", vmin=0, vmax=1,
                   extent=[min(ws) - 0.1, max(ws) + 0.1,
                           min(wo) - 0.1, max(wo) + 0.1], aspect="auto")
    ax.set_xticks(ws); ax.set_yticks(wo)
    ax.set_xlabel("$w_{\\mathrm{self}}$", fontsize=13)
    ax.set_ylabel("$w_{\\mathrm{other}}$", fontsize=13)
    ax.set_title("A.  Cooperation over the weight plane", fontsize=12.5,
                 loc="left", fontweight="bold")
    for i, o in enumerate(wo):
        for j, s in enumerate(ws):
            if not np.isnan(M[i, j]):
                ax.text(s, o, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="black" if 0.25 < M[i, j] < 0.85 else "white")
    # iso-ratio rays: w_other/w_self constant => straight lines through origin
    xlo, xhi = min(ws) - 0.1, max(ws) + 0.1
    ylo, yhi = min(wo) - 0.1, max(wo) + 0.1
    for lam, style in [(0.33, ":"), (0.5, "-"), (0.67, ":")]:
        k = lam / (1 - lam)               # w_other = k * w_self
        xs = np.array([0, xhi])
        ax.plot(xs, k * xs, style, color="black", lw=1.3, alpha=0.75)
        # place the label where the ray leaves the axes, whichever edge that is
        x_end = min(xhi, yhi / k)
        y_end = k * x_end
        ax.annotate(f"$\\lambda={lam:.2f}$", xy=(x_end, y_end),
                    xytext=(-4, -10 if k < 1.5 else 4), textcoords="offset points",
                    fontsize=8.5, color="black", ha="right",
                    va="bottom" if k < 1.5 else "top")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Mutual cooperation frequency", fontsize=10)

    # Panel B: sign-extended regimes
    ax = axes[1]
    order = [(0.4, 0.6, "cooperative\n$(0.4,\\,0.6)$"),
             (1.0, 0.0, "self-interested\n$(1,\\,0)$"),
             (0.6, -0.4, "spite\n$(0.6,\\,-0.4)$"),
             (0.0, -1.0, "pure spite\n$(0,\\,-1)$"),
             (-0.4, 0.6, "self-abnegation\n$(-0.4,\\,0.6)$")]
    cc, dd, names = [], [], []
    for s, o, lab in order:
        cell = [r for r in quad
                if abs(r["w_self"] - s) < 1e-9 and abs(r["w_other"] - o) < 1e-9]
        cc.append(mean(cell, "freq_CC"))
        dd.append(mean(cell, "freq_DD"))
        names.append(lab)

    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, cc, w, color=GREEN, edgecolor="black", linewidth=0.6,
           label="mutual cooperation (CC)")
    ax.bar(x + w / 2, dd, w, color=RED, edgecolor="black", linewidth=0.6,
           label="mutual defection (DD)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Frequency of rounds", fontsize=12)
    ax.set_ylim(0, 1.3)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title("B.  Regimes outside $\\lambda \\in [0,1]$", fontsize=12.5,
                 loc="left", fontweight="bold")
    # all three negative-weight regimes are unreachable, not just the spiteful ones
    ax.axvspan(1.5, len(order) - 0.4, color=GREY, alpha=0.10)
    ax.annotate("negative weights: unreachable with a bipolar $\\lambda$",
                xy=((1.5 + len(order) - 0.4) / 2, 1.21), ha="center",
                fontsize=9, color=GREY)
    ax.legend(fontsize=9, loc="upper left", frameon=False)

    fig.tight_layout()
    path = out_dir / "fig10_weight_space.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Revision figures for rsif-2026-0208")
    p.add_argument("--out", type=str, default=str(PROJECT_ROOT / "images"))
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {out_dir}")
    fig_model_comparison(out_dir)
    fig_payoff_horizon(out_dir)
    fig_weight_space(out_dir)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
