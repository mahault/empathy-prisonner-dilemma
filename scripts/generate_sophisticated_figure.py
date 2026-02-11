#!/usr/bin/env python
"""
Figure 5: Sophisticated inference — myopic vs multi-step planning.

Panel A: Cooperation rate sweep for H=1 (myopic), H=2, H=3 across empathy values.
Panel B: P(C) traces over rounds at lambda=0.3 (near threshold) for different H.
Panel C: Final cooperation rate bar chart at selected empathy values.

Usage:
    python scripts/generate_sophisticated_figure.py [--T 100] [--n_seeds 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pymdp.utils import obj_array, obj_array_uniform
from empathy.prisoners_dilemma import ToMEmpatheticAgent, Environment


# -----------------------------------------------------------------------
# PD config
# -----------------------------------------------------------------------

def create_pd_config(T: int = 100) -> dict:
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


# -----------------------------------------------------------------------
# Run experiment
# -----------------------------------------------------------------------

def run_experiment(lambda_i, lambda_j, T=100, seed=42,
                   use_sophisticated=False, planning_horizon=1):
    """Run a PD experiment with given parameters."""
    np.random.seed(seed)
    config = create_pd_config(T=T)
    env = Environment(K=2)

    ag_i = ToMEmpatheticAgent(
        config=config, agent_num=0,
        empathy_factor=lambda_i,
        use_inversion=False,
        use_sophisticated=use_sophisticated,
        planning_horizon=planning_horizon,
    )
    ag_j = ToMEmpatheticAgent(
        config=config, agent_num=1,
        empathy_factor=lambda_j,
        use_inversion=False,
    )

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

    return act_i, act_j


def rolling(x, w=5):
    v = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), v])


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Figure 5: Sophisticated inference comparison"
    )
    parser.add_argument("--output", default="figures/", help="Output directory")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    T, ns, w = args.T, args.n_seeds, args.window

    print("=" * 60)
    print(f"Figure 5: Sophisticated inference  |  T={T}  seeds={ns}")
    print("=" * 60)

    # Horizon configs: H=1 is effectively myopic
    horizons = [1, 2, 3]
    horizon_labels = {1: "H=1 (myopic)", 2: "H=2", 3: "H=3"}
    horizon_colors = {1: "#3498db", 2: "#e74c3c", 3: "#2ecc71"}

    # ---- Panel A: Cooperation rate sweep --------------------------------
    print("\n--- Panel A: Cooperation rate sweep ---")
    lambda_values = np.arange(0.0, 1.05, 0.05)
    panel_a_results = {}  # (H, lambda) -> mean_cc

    for H in horizons:
        print(f"  Horizon H={H}:")
        use_soph = H > 1
        for lam in lambda_values:
            cc_rates = []
            for s in range(ns):
                act_i, act_j = run_experiment(
                    lam, lam, T=T, seed=s,
                    use_sophisticated=use_soph, planning_horizon=H,
                )
                cc = np.mean((act_i == 0) & (act_j == 0))
                cc_rates.append(cc)
            mean_cc = np.mean(cc_rates)
            std_cc = np.std(cc_rates)
            panel_a_results[(H, lam)] = (mean_cc, std_cc)
            if lam in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]:
                print(f"    lambda={lam:.2f}: CC={mean_cc:.3f} +/- {std_cc:.3f}")

    # ---- Panel B: P(C) traces at lambda=0.3 ----------------------------
    print("\n--- Panel B: P(C) traces at lambda=0.3 ---")
    lambda_fixed = 0.3
    panel_b_results = {}

    for H in horizons:
        use_soph = H > 1
        all_coop = []
        for s in range(ns):
            act_i, act_j = run_experiment(
                lambda_fixed, lambda_fixed, T=T, seed=s,
                use_sophisticated=use_soph, planning_horizon=H,
            )
            cc = ((act_i == 0) & (act_j == 0)).astype(float)
            all_coop.append(rolling(cc, w))
        arr = np.array(all_coop)
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        panel_b_results[H] = (mu, sd)
        print(f"  H={H}: mean CC = {np.nanmean(mu):.3f}")

    # ---- Panel C: Bar chart at selected empathy values ------------------
    print("\n--- Panel C: Bar chart at selected lambdas ---")
    bar_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    panel_c_results = {}

    for lam in bar_lambdas:
        for H in horizons:
            key = (H, round(lam, 2))
            if key in panel_a_results:
                panel_c_results[(H, lam)] = panel_a_results[key][0]
            else:
                # Compute if not already in sweep (shouldn't happen with 0.05 steps)
                use_soph = H > 1
                cc_rates = []
                for s in range(ns):
                    act_i, act_j = run_experiment(
                        lam, lam, T=T, seed=s,
                        use_sophisticated=use_soph, planning_horizon=H,
                    )
                    cc_rates.append(np.mean((act_i == 0) & (act_j == 0)))
                panel_c_results[(H, lam)] = np.mean(cc_rates)

    # ---- Plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    t_ax = np.arange(T)

    # Panel A: Cooperation sweep
    ax = axes[0]
    for H in horizons:
        lams = sorted(set(l for (h, l) in panel_a_results if h == H))
        means = [panel_a_results[(H, l)][0] for l in lams]
        stds = [panel_a_results[(H, l)][1] for l in lams]
        ax.plot(lams, means, color=horizon_colors[H], lw=2.5,
                label=horizon_labels[H], marker="o", markersize=3)
        ax.fill_between(lams,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=horizon_colors[H], alpha=0.15)
    ax.set_xlabel(r"Empathy $\lambda$ (symmetric)", fontsize=12)
    ax.set_ylabel("Cooperation Rate (CC)", fontsize=12)
    ax.set_title("A.  Cooperation vs Empathy by Horizon",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: P(C) traces at lambda=0.3
    ax = axes[1]
    for H in horizons:
        mu, sd = panel_b_results[H]
        ax.plot(t_ax, mu, color=horizon_colors[H], lw=2,
                label=horizon_labels[H])
        ax.fill_between(t_ax, mu - sd, mu + sd,
                        color=horizon_colors[H], alpha=0.15)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(f"CC Rate (rolling w={w})", fontsize=12)
    ax.set_title(r"B.  Cooperation Traces ($\lambda$=0.3)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel C: Bar chart
    ax = axes[2]
    x = np.arange(len(bar_lambdas))
    bar_width = 0.25
    for i, H in enumerate(horizons):
        vals = [panel_c_results.get((H, l), 0) for l in bar_lambdas]
        offset = (i - 1) * bar_width
        ax.bar(x + offset, vals, bar_width,
               color=horizon_colors[H], label=horizon_labels[H], alpha=0.85)
    ax.set_xlabel(r"Empathy $\lambda$", fontsize=12)
    ax.set_ylabel("Mean CC Rate", fontsize=12)
    ax.set_title("C.  Cooperation by Horizon",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:.1f}" for l in bar_lambdas])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sophisticated_comparison.{ext}", dpi=150)
    plt.close(fig)
    print(f"\nSaved: sophisticated_comparison.png / .pdf")

    # ---- Summary --------------------------------------------------------
    print(f"\n--- Summary: CC rate by horizon ---")
    print(f"  {'lambda':>8}  {'H=1':>8}  {'H=2':>8}  {'H=3':>8}  {'H2-H1':>8}  {'H3-H1':>8}")
    for lam in bar_lambdas:
        h1 = panel_c_results.get((1, lam), 0)
        h2 = panel_c_results.get((2, lam), 0)
        h3 = panel_c_results.get((3, lam), 0)
        print(f"  {lam:8.1f}  {h1:8.3f}  {h2:8.3f}  {h3:8.3f}  {h2-h1:+8.3f}  {h3-h1:+8.3f}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
