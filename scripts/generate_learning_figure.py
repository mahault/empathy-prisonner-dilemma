#!/usr/bin/env python
"""
Figure 4: Learning convergence and cooperation with/without opponent modelling.

Panel A: Parametric profile convergence — shows the particle filter's inferred
         alpha (cooperation bias) and reciprocity converging over rounds when
         facing a cooperative opponent.
Panel B: Parametric profile convergence — same, facing a selfish opponent.
Panel C: Cooperation rate with learning vs static ToM — shows how wiring
         the learned opponent model into action selection affects cooperation.

Key finding: Learning raises the cooperation threshold from λ≈0.2 to λ≈0.4.
Accurate opponent modelling is necessary but not sufficient — sufficient empathy
is also required to resist temptation to exploit a known cooperator.

Usage:
    python scripts/generate_learning_figure.py [--T 100] [--n_seeds 20]
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


# ───────────────────────────────────────────────────────────────────────────
# PD config
# ───────────────────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────────────────
# Traced experiment with optional inversion tracking
# ───────────────────────────────────────────────────────────────────────────

def run_traced(lambda_i, lambda_j, T=100, seed=42, use_inversion=False):
    """Run a traced experiment, optionally capturing inversion state per step."""
    np.random.seed(seed)
    config = create_pd_config(T=T)
    env = Environment(K=2)

    ag_i = ToMEmpatheticAgent(config=config, agent_num=0,
                              empathy_factor=lambda_i, use_inversion=use_inversion)
    ag_j = ToMEmpatheticAgent(config=config, agent_num=1,
                              empathy_factor=lambda_j, use_inversion=False)

    act_i = np.zeros(T, dtype=int)
    act_j = np.zeros(T, dtype=int)
    actions = [0, 0]

    # Track inversion state per timestep
    profile_history = []  # list of dicts with mean/std of alpha, reciprocity, beta
    reliability_history = []

    for t in range(T):
        obs = env.step(t=t, actions=actions)
        obs_i = ag_i.o_init if t == 0 else obs[0]
        obs_j = ag_j.o_init if t == 0 else obs[1]

        res_i = ag_i.step(t=t, observation=obs_i)
        res_j = ag_j.step(t=t, observation=obs_j)

        a_i, a_j = res_i["exp_action"], res_j["exp_action"]
        act_i[t], act_j[t] = a_i, a_j
        actions = [a_i, a_j]

        # Capture inversion state
        if use_inversion and ag_i.inversion is not None:
            profile_history.append(ag_i.inversion.get_profile_summary())
            reliability_history.append(ag_i.inversion.reliability())
        else:
            profile_history.append(None)
            reliability_history.append(None)

    return dict(
        actions_i=act_i,
        actions_j=act_j,
        profile_history=profile_history,
        reliability_history=reliability_history,
    )


def rolling(x, w=5):
    v = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), v])


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Figure 4: learning convergence")
    parser.add_argument("--output", default="figures/", help="Output directory")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    T, ns, w = args.T, args.n_seeds, args.window

    print("=" * 60)
    print(f"Figure 4: Learning convergence  |  T={T}  seeds={ns}")
    print("=" * 60)

    # ── Panel A & B: Parametric profile convergence ──────────────────────
    # Two conditions: facing cooperative (λ=0.7) vs selfish (λ=0.1) opponent
    print("\n--- Panels A & B: Profile parameter convergence ---")

    panel_ab_conditions = [
        {"label": r"Facing $\lambda_j$=0.7 (cooperative)", "lambda_j": 0.7},
        {"label": r"Facing $\lambda_j$=0.1 (selfish)", "lambda_j": 0.1},
    ]

    panel_ab_results = {}
    param_keys = ["mean_alpha", "mean_reciprocity", "mean_beta",
                  "std_alpha", "std_reciprocity", "std_beta"]

    for cond in panel_ab_conditions:
        lj = cond["lambda_j"]
        print(f"  Running lambda_i=0.5 vs lambda_j={lj} ...")
        all_profile_histories = []
        for s in range(ns):
            result = run_traced(0.5, lj, T=T, seed=s, use_inversion=True)
            all_profile_histories.append(result["profile_history"])

        # Aggregate: for each parameter, compute mean and std across seeds
        avg_params = {}
        std_params = {}
        for key in param_keys:
            vals = np.zeros((ns, T))
            for s in range(ns):
                for t in range(T):
                    ph = all_profile_histories[s][t]
                    if ph is not None:
                        vals[s, t] = ph[key]
                    else:
                        vals[s, t] = 0.0  # prior mean
            avg_params[key] = np.mean(vals, axis=0)
            std_params[key] = np.std(vals, axis=0)

        panel_ab_results[lj] = {"mean": avg_params, "std": std_params}
        print(f"    Final alpha: {avg_params['mean_alpha'][-1]:.2f} "
              f"± {avg_params['std_alpha'][-1]:.2f}")
        print(f"    Final reciprocity: {avg_params['mean_reciprocity'][-1]:.2f} "
              f"± {avg_params['std_reciprocity'][-1]:.2f}")

    # ── Panel C: Cooperation with learning vs static ───────────────────
    print("\n--- Panel C: Learning vs static ToM (asymmetric) ---")

    lambda_j_fixed = 0.7
    lambda_i_values_b = [0.2, 0.3, 0.5, 0.7]
    panel_c_results = {}

    for li in lambda_i_values_b:
        print(f"  lambda_i={li} vs lambda_j={lambda_j_fixed}")
        for use_inv, label in [(False, "static"), (True, "learned")]:
            traces = [run_traced(li, lambda_j_fixed, T=T, seed=s, use_inversion=use_inv)
                      for s in range(ns)]
            cc_per_seed = []
            for tr in traces:
                cc = ((tr["actions_i"] == 0) & (tr["actions_j"] == 0)).astype(float)
                cc_per_seed.append(rolling(cc, w))
            arr = np.array(cc_per_seed)
            mu = np.nanmean(arr, axis=0)
            sd = np.nanstd(arr, axis=0)
            panel_c_results[(li, label)] = (mu, sd)
            cc_mean = np.nanmean([np.mean((tr["actions_i"] == 0) & (tr["actions_j"] == 0))
                                  for tr in traces])
            print(f"    {label:>8}: CC = {cc_mean:.3f}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    t_ax = np.arange(T)

    # --- Panel A: Profile convergence — facing cooperative opponent ---
    ax = axes[0]
    lj = 0.7
    data = panel_ab_results[lj]
    param_colors = {"mean_alpha": "#2ecc71", "mean_reciprocity": "#3498db", "mean_beta": "#e74c3c"}
    param_labels = {"mean_alpha": r"$\alpha$ (coop. bias)",
                    "mean_reciprocity": r"$\rho$ (reciprocity)",
                    "mean_beta": r"$\beta$ (precision)"}

    for key, color in param_colors.items():
        mu = data["mean"][key]
        sd = data["std"][key]
        ax.plot(t_ax, mu, color=color, lw=2, label=param_labels[key])
        ax.fill_between(t_ax, mu - sd, mu + sd, color=color, alpha=0.15)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_title(r"A.  Profile inference — facing $\lambda_j$=0.7",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # --- Panel B: Profile convergence — facing selfish opponent ---
    ax = axes[1]
    lj = 0.1
    data = panel_ab_results[lj]
    for key, color in param_colors.items():
        mu = data["mean"][key]
        sd = data["std"][key]
        ax.plot(t_ax, mu, color=color, lw=2, label=param_labels[key])
        ax.fill_between(t_ax, mu - sd, mu + sd, color=color, alpha=0.15)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_title(r"B.  Profile inference — facing $\lambda_j$=0.1",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # --- Panel C: cooperation with/without learning (asymmetric) ---
    ax = axes[2]
    colors_c = {0.2: "#e67e22", 0.3: "#e74c3c", 0.5: "#3498db", 0.7: "#2ecc71"}

    for li in lambda_i_values_b:
        col = colors_c[li]
        mu_s, sd_s = panel_c_results[(li, "static")]
        mu_l, sd_l = panel_c_results[(li, "learned")]

        ax.plot(t_ax, mu_s, color=col, lw=2, linestyle="--", alpha=0.7,
                label=rf"$\lambda_i$={li} static")
        ax.fill_between(t_ax, mu_s - sd_s, mu_s + sd_s, color=col, alpha=0.06)
        ax.plot(t_ax, mu_l, color=col, lw=2, linestyle="-",
                label=rf"$\lambda_i$={li} learned")
        ax.fill_between(t_ax, mu_l - sd_l, mu_l + sd_l, color=col, alpha=0.10)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(f"Cooperation Rate (rolling w={w})", fontsize=12)
    ax.set_title(rf"C.  Learned vs static ($\lambda_j$={lambda_j_fixed})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"learning_convergence.{ext}", dpi=150)
    plt.close(fig)
    print(f"\nSaved: learning_convergence.png / .pdf")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n--- Cooperation Summary (learning vs static, lambda_j={lambda_j_fixed}) ---")
    print(f"  {'lambda_i':>10}  {'static':>8}  {'learned':>8}  {'diff':>8}")
    for li in lambda_i_values_b:
        mu_s = np.nanmean(panel_c_results[(li, "static")][0])
        mu_l = np.nanmean(panel_c_results[(li, "learned")][0])
        print(f"  {li:10.1f}  {mu_s:8.3f}  {mu_l:8.3f}  {mu_l - mu_s:+8.3f}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
