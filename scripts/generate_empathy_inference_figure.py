#!/usr/bin/env python
"""
Figure 8: Empathy parameter inference (λ_j) and epistemic value.

Panel A: λ_j posterior convergence over rounds — facing cooperative (λ_j=0.7)
         vs selfish (λ_j=0.1) opponent.
Panel B: Epistemic value contribution over time — should decrease as λ_j
         becomes better known.
Panel C: Cooperation rate comparison with/without epistemic term — epistemic
         exploration should boost early cooperation.

Key finding: The epistemic value of cooperation is higher than defection
because cooperating better disambiguates opponent empathy. A selfish opponent
always defects regardless; an empathetic opponent cooperates back. Against
defection, both types tend to defect, so it's less informative.

Usage:
    python scripts/generate_empathy_inference_figure.py [--T 100] [--n_seeds 20]
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
# Traced experiment capturing lambda_j inference and epistemic value
# ───────────────────────────────────────────────────────────────────────────

def run_traced(lambda_i, lambda_j, T=100, seed=42, use_inversion=True):
    """Run a traced experiment capturing lambda_j posterior and epistemic value."""
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

    lambda_j_mean_history = []
    lambda_j_std_history = []
    lambda_j_entropy_history = []
    epistemic_C_history = []
    epistemic_D_history = []

    for t in range(T):
        obs = env.step(t=t, actions=actions)
        obs_i = ag_i.o_init if t == 0 else obs[0]
        obs_j = ag_j.o_init if t == 0 else obs[1]

        res_i = ag_i.step(t=t, observation=obs_i)
        res_j = ag_j.step(t=t, observation=obs_j)

        a_i, a_j = res_i["exp_action"], res_j["exp_action"]
        act_i[t], act_j[t] = a_i, a_j
        actions = [a_i, a_j]

        # Capture lambda_j inference state
        if use_inversion and ag_i.inversion is not None:
            posterior = ag_i.inversion.get_lambda_j_posterior()
            lambda_j_mean_history.append(posterior["mean"])
            lambda_j_std_history.append(posterior["std"])
            lambda_j_entropy_history.append(posterior["entropy"])

            # Capture epistemic values from info
            if "info" in res_i and isinstance(res_i["info"], dict):
                # For myopic path, info is dict with action indices
                if 0 in res_i["info"]:
                    epistemic_C_history.append(res_i["info"][0].get("G_epistemic", 0.0))
                    epistemic_D_history.append(res_i["info"][1].get("G_epistemic", 0.0))
                else:
                    epistemic_C_history.append(0.0)
                    epistemic_D_history.append(0.0)
            else:
                epistemic_C_history.append(0.0)
                epistemic_D_history.append(0.0)
        else:
            lambda_j_mean_history.append(0.5)
            lambda_j_std_history.append(0.29)
            lambda_j_entropy_history.append(np.log(10))
            epistemic_C_history.append(0.0)
            epistemic_D_history.append(0.0)

    return dict(
        actions_i=act_i,
        actions_j=act_j,
        lambda_j_mean=np.array(lambda_j_mean_history),
        lambda_j_std=np.array(lambda_j_std_history),
        lambda_j_entropy=np.array(lambda_j_entropy_history),
        epistemic_C=np.array(epistemic_C_history),
        epistemic_D=np.array(epistemic_D_history),
    )


def rolling(x, w=5):
    v = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), v])


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Figure 8: empathy inference")
    parser.add_argument("--output", default="figures/", help="Output directory")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    T, ns, w = args.T, args.n_seeds, args.window

    print("=" * 60)
    print(f"Figure 8: Empathy inference  |  T={T}  seeds={ns}")
    print("=" * 60)

    # ── Panel A: λ_j posterior convergence ────────────────────────────
    print("\n--- Panel A: Lambda_j posterior convergence ---")

    conditions = [
        {"label": r"$\lambda_j$=0.7 (cooperative)", "lambda_j": 0.7, "color": "#2ecc71"},
        {"label": r"$\lambda_j$=0.1 (selfish)", "lambda_j": 0.1, "color": "#e74c3c"},
    ]

    panel_a_results = {}
    for cond in conditions:
        lj = cond["lambda_j"]
        print(f"  lambda_i=0.5 vs lambda_j={lj} ...")
        all_means = []
        all_stds = []
        all_entropies = []
        all_ep_C = []
        all_ep_D = []
        for s in range(ns):
            res = run_traced(0.5, lj, T=T, seed=s)
            all_means.append(res["lambda_j_mean"])
            all_stds.append(res["lambda_j_std"])
            all_entropies.append(res["lambda_j_entropy"])
            all_ep_C.append(res["epistemic_C"])
            all_ep_D.append(res["epistemic_D"])

        panel_a_results[lj] = {
            "mean": np.mean(all_means, axis=0),
            "mean_std": np.std(all_means, axis=0),
            "uncertainty": np.mean(all_stds, axis=0),
            "entropy": np.mean(all_entropies, axis=0),
            "epistemic_C": np.mean(all_ep_C, axis=0),
            "epistemic_D": np.mean(all_ep_D, axis=0),
        }
        print(f"    Final mean lambda_j: {panel_a_results[lj]['mean'][-1]:.3f}")
        print(f"    Final std: {panel_a_results[lj]['uncertainty'][-1]:.3f}")

    # ── Panel C: Cooperation comparison ──────────────────────────────
    print("\n--- Panel C: Cooperation with/without inversion ---")
    panel_c_results = {}
    for use_inv, label in [(False, "no_inversion"), (True, "with_inversion")]:
        traces = [run_traced(0.5, 0.7, T=T, seed=s, use_inversion=use_inv)
                  for s in range(ns)]
        coop_per_seed = []
        for tr in traces:
            coop = (tr["actions_i"] == 0).astype(float)
            coop_per_seed.append(rolling(coop, w))
        arr = np.array(coop_per_seed)
        panel_c_results[label] = (np.nanmean(arr, axis=0), np.nanstd(arr, axis=0))
        print(f"  {label}: mean coop = {np.nanmean(arr):.3f}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    t_ax = np.arange(T)

    # --- Panel A: Lambda_j convergence ---
    ax = axes[0]
    for cond in conditions:
        lj = cond["lambda_j"]
        data = panel_a_results[lj]
        mu = data["mean"]
        sd = data["mean_std"]
        ax.plot(t_ax, mu, color=cond["color"], lw=2, label=cond["label"])
        ax.fill_between(t_ax, mu - sd, mu + sd, color=cond["color"], alpha=0.15)
        ax.axhline(lj, color=cond["color"], lw=1, ls=":", alpha=0.5)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(r"Inferred $\hat{\lambda}_j$", fontsize=12)
    ax.set_title(r"A.  $\lambda_j$ posterior convergence", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, alpha=0.3)

    # --- Panel B: Epistemic value over time ---
    ax = axes[1]
    for cond in conditions:
        lj = cond["lambda_j"]
        data = panel_a_results[lj]
        # Plot negative epistemic value (information gain, positive)
        ig_C = -data["epistemic_C"]
        ig_D = -data["epistemic_D"]
        ax.plot(t_ax, rolling(ig_C, w), color=cond["color"], lw=2, ls="-",
                label=f"IG(C) vs {cond['label']}")
        ax.plot(t_ax, rolling(ig_D, w), color=cond["color"], lw=2, ls="--",
                label=f"IG(D) vs {cond['label']}")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Information Gain", fontsize=12)
    ax.set_title("B.  Epistemic value over time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Panel C: Cooperation with/without epistemic exploration ---
    ax = axes[2]
    for label, (col, ls) in [("with_inversion", ("#2ecc71", "-")),
                               ("no_inversion", ("#95a5a6", "--"))]:
        mu, sd = panel_c_results[label]
        nice_label = "With empathy inference" if "with" in label else "Static ToM"
        ax.plot(t_ax, mu, color=col, lw=2, ls=ls, label=nice_label)
        ax.fill_between(t_ax, mu - sd, mu + sd, color=col, alpha=0.1)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(f"Agent i Cooperation Rate (rolling w={w})", fontsize=12)
    ax.set_title(r"C.  Cooperation ($\lambda_i$=0.5 vs $\lambda_j$=0.7)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"empathy_inference.{ext}", dpi=150)
    plt.close(fig)
    print(f"\nSaved: empathy_inference.png / .pdf")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
