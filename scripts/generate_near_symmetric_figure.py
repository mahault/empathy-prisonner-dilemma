#!/usr/bin/env python
"""
Figure 3: Near-symmetric empathy dynamics.

Zooms into the boundary layer around λ_i ≈ λ_j to show that empathy
modulates variance and temporal stability, not just mean cooperation.

Fix λ_j = 0.5, sweep λ_i ∈ {0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75}.

Usage:
    python scripts/generate_near_symmetric_figure.py [--T 100] [--n_seeds 30]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pymdp.utils import obj_array, obj_array_uniform
from empathy.prisoners_dilemma import ToMEmpatheticAgent, Environment


# ───────────────────────────────────────────────────────────────────────────
# PD config (same as other scripts)
# ───────────────────────────────────────────────────────────────────────────

def create_pd_config(T: int = 50) -> dict:
    n_mod, n_fac, n_obs, n_st = 1, 1, 4, 4

    A0 = obj_array(n_mod); A0[0] = np.eye(n_obs)
    B0 = obj_array(n_fac); B0[0] = np.zeros((4, 4, 2))
    B0[0][0, :, 0] = 0.5; B0[0][1, :, 0] = 0.5
    B0[0][2, :, 1] = 0.5; B0[0][3, :, 1] = 0.5
    C0 = obj_array(n_mod); C0[0] = np.array([3, 1, 4, 2])
    D0 = obj_array_uniform([n_st])

    A1 = obj_array(n_mod); A1[0] = np.eye(n_obs)
    B1 = obj_array(n_fac); B1[0] = np.zeros((4, 4, 2))
    B1[0][0, :, 0] = 0.5; B1[0][2, :, 0] = 0.5
    B1[0][1, :, 1] = 0.5; B1[0][3, :, 1] = 0.5
    C1 = obj_array(n_mod); C1[0] = np.array([3, 4, 1, 2])
    D1 = obj_array_uniform([n_st])

    return {
        "T": T, "K": 2,
        "A": [A0, A1], "B": [B0, B1],
        "C": [C0, C1], "D": [D0, D1],
        "empathy_factor": [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        "actions": ["C", "D"], "learn": False,
        "policy_len": 2, "same_pref": False,
    }


# ───────────────────────────────────────────────────────────────────────────
# Traced experiment
# ───────────────────────────────────────────────────────────────────────────

def run_traced(lambda_i, lambda_j, T=100, seed=42):
    np.random.seed(seed)
    config = create_pd_config(T=T)
    env = Environment(K=2)

    ag_i = ToMEmpatheticAgent(config=config, agent_num=0,
                              empathy_factor=lambda_i, use_inversion=False)
    ag_j = ToMEmpatheticAgent(config=config, agent_num=1,
                              empathy_factor=lambda_j, use_inversion=False)

    act_i = np.zeros(T, dtype=int)
    act_j = np.zeros(T, dtype=int)
    actions = [0, 0]

    for t in range(T):
        obs = env.step(t=t, actions=actions)
        obs_i = ag_i.o_init if t == 0 else obs[0]
        obs_j = ag_j.o_init if t == 0 else obs[1]

        res_i = ag_i.step(t=t, observation=obs_i)
        res_j = ag_j.step(t=t, observation=obs_j)

        a_i, a_j = res_i["exp_action"], res_j["exp_action"]
        act_i[t], act_j[t] = a_i, a_j
        actions = [a_i, a_j]

    return dict(actions_i=act_i, actions_j=act_j)


def rolling(x, w=5):
    v = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), v])


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Figure 3: near-symmetric dynamics")
    parser.add_argument("--output", default="figures/", help="Output directory")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    T, ns, w = args.T, args.n_seeds, args.window

    lambda_j = 0.5
    lambda_i_values = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75]

    print("=" * 60)
    print(f"Figure 3: Near-symmetric dynamics  |  T={T}  seeds={ns}")
    print(f"  Fixed lambda_j = {lambda_j}")
    print(f"  Sweep lambda_i = {lambda_i_values}")
    print("=" * 60)

    # ── Run all conditions ────────────────────────────────────────────
    all_data = {}
    for li in lambda_i_values:
        print(f"\n  lambda_i = {li:.2f} ...")
        traces = [run_traced(li, lambda_j, T=T, seed=s) for s in range(ns)]
        cc_rates = [np.mean((tr["actions_i"] == 0) & (tr["actions_j"] == 0))
                    for tr in traces]
        print(f"    CC = {np.mean(cc_rates):.3f} +/- {np.std(cc_rates):.3f}")
        all_data[li] = traces

    # ── Color map ─────────────────────────────────────────────────────
    cmap = plt.get_cmap("coolwarm", len(lambda_i_values))

    # ── Figure ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    t_ax = np.arange(T)

    # Panel A: rolling cooperation ± variance
    for idx, li in enumerate(lambda_i_values):
        col = cmap(idx)
        traces = all_data[li]
        seeds = []
        for tr in traces:
            cc = ((tr["actions_i"] == 0) & (tr["actions_j"] == 0)).astype(float)
            seeds.append(rolling(cc, w))
        arr = np.array(seeds)
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)

        diff = abs(li - lambda_j)
        lbl = (rf"$\lambda_i$={li:.2f}  ($\Delta$={diff:.2f})"
               if li != lambda_j
               else rf"$\lambda_i$={li:.2f}  (symmetric)")
        ax1.plot(t_ax, mu, c=col, lw=2, label=lbl)
        ax1.fill_between(t_ax, mu - sd, mu + sd, color=col, alpha=0.12)

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel(f"Cooperation Rate (rolling w={w})", fontsize=12)
    ax1.set_title(rf"A.  Cooperation near symmetry ($\lambda_j$={lambda_j})",
                  fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(True, alpha=0.3)

    # Panel B: example traces for 3 near-transition values
    trace_lambdas = [0.20, 0.35, 0.50]
    trace_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    offsets = [-0.03, 0.0, 0.03]

    for li, col, off in zip(trace_lambdas, trace_colors, offsets):
        traces = all_data[li]
        # pick seed with most variance (most interesting dynamics)
        variances = []
        for tr in traces:
            cc = ((tr["actions_i"] == 0) & (tr["actions_j"] == 0)).astype(float)
            variances.append(np.var(cc))
        best_seed = int(np.argmax(variances))
        tr = traces[best_seed]
        cc = ((tr["actions_i"] == 0) & (tr["actions_j"] == 0)).astype(float)

        ax2.step(t_ax, cc + off, where="mid", color=col, lw=1.3, alpha=0.8,
                 label=rf"$\lambda_i$={li:.2f} (seed {best_seed})")

    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Mutual Cooperation (0 / 1)", fontsize=12)
    ax2.set_title("B.  Example traces near transition", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.15, 1.2)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"near_symmetric_dynamics.{ext}", dpi=150)
    plt.close(fig)
    print("\nSaved: near_symmetric_dynamics.png / .pdf")

    # ── Print variance summary ────────────────────────────────────────
    print("\n--- Variance Summary ---")
    print(f"  {'lambda_i':>10}  {'mean_CC':>8}  {'std_CC':>8}  {'variance':>10}")
    for li in lambda_i_values:
        traces = all_data[li]
        cc_rates = [np.mean((tr["actions_i"] == 0) & (tr["actions_j"] == 0))
                    for tr in traces]
        print(f"  {li:10.2f}  {np.mean(cc_rates):8.3f}  {np.std(cc_rates):8.3f}"
              f"  {np.var(cc_rates):10.6f}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
