#!/usr/bin/env python
"""
Generate time-series figures for the paper:

  Figure A - Cooperation dynamics (apology-forgiveness recovery)
  Figure B - Prediction error / belief divergence

Usage:
    python scripts/generate_timeseries_figures.py [--output figures/] [--T 100] [--n_seeds 20]
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
# PD config (self-contained copy from run_pd_experiments.py)
# ───────────────────────────────────────────────────────────────────────────

def create_pd_config(T: int = 50) -> dict:
    num_mod = 1
    num_fac = 1
    num_obs = 4
    num_st = 4

    A0 = obj_array(num_mod); A0[0] = np.eye(num_obs)
    B0 = obj_array(num_fac); B0[0] = np.zeros((4, 4, 2))
    B0[0][0, :, 0] = 0.5; B0[0][1, :, 0] = 0.5
    B0[0][2, :, 1] = 0.5; B0[0][3, :, 1] = 0.5
    C0 = obj_array(num_mod); C0[0] = np.array([3, 1, 4, 2])
    D0 = obj_array_uniform([num_st])

    A1 = obj_array(num_mod); A1[0] = np.eye(num_obs)
    B1 = obj_array(num_fac); B1[0] = np.zeros((4, 4, 2))
    B1[0][0, :, 0] = 0.5; B1[0][2, :, 0] = 0.5
    B1[0][1, :, 1] = 0.5; B1[0][3, :, 1] = 0.5
    C1 = obj_array(num_mod); C1[0] = np.array([3, 4, 1, 2])
    D1 = obj_array_uniform([num_st])

    return {
        "T": T, "K": 2,
        "A": [A0, A1], "B": [B0, B1],
        "C": [C0, C1], "D": [D0, D1],
        "empathy_factor": [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        "actions": ["C", "D"], "learn": False,
        "policy_len": 2, "same_pref": False,
    }


# ───────────────────────────────────────────────────────────────────────────
# Traced experiment runner
# ───────────────────────────────────────────────────────────────────────────

def run_traced(lambda_i, lambda_j, beta_i=4.0, beta_j=4.0,
               T=100, seed=42, use_inversion=False):
    """Run one experiment returning per-timestep traces."""
    np.random.seed(seed)
    config = create_pd_config(T=T)
    env = Environment(K=2)

    ag_i = ToMEmpatheticAgent(
        config=config, agent_num=0,
        empathy_factor=lambda_i, beta_self=beta_i,
        use_inversion=use_inversion,
    )
    ag_j = ToMEmpatheticAgent(
        config=config, agent_num=1,
        empathy_factor=lambda_j, beta_self=beta_j,
        use_inversion=use_inversion,
    )

    act_i = np.zeros(T, dtype=int)
    act_j = np.zeros(T, dtype=int)
    pe_i  = np.full(T, np.nan)
    pe_j  = np.full(T, np.nan)
    qc_i  = np.zeros(T)
    qc_j  = np.zeros(T)

    actions = [0, 0]

    for t in range(T):
        obs = env.step(t=t, actions=actions)
        obs_i = ag_i.o_init if t == 0 else obs[0]
        obs_j = ag_j.o_init if t == 0 else obs[1]

        res_i = ag_i.step(t=t, observation=obs_i)
        res_j = ag_j.step(t=t, observation=obs_j)

        a_i = res_i["exp_action"]
        a_j = res_j["exp_action"]

        act_i[t] = a_i
        act_j[t] = a_j
        qc_i[t] = res_i["q_action"][0]   # P(Cooperate) for agent i
        qc_j[t] = res_j["q_action"][0]

        # Prediction error (surprisal about opponent's actual action)
        if t > 0:
            pe_i[t] = -np.log(res_i["info"][a_i]["q_response"][a_j] + 1e-10)
            pe_j[t] = -np.log(res_j["info"][a_j]["q_response"][a_i] + 1e-10)

        actions = [a_i, a_j]

    return dict(actions_i=act_i, actions_j=act_j,
                pe_i=pe_i, pe_j=pe_j, qc_i=qc_i, qc_j=qc_j)


def run_condition(lambda_i, lambda_j, n_seeds=20, T=100,
                  beta_i=4.0, beta_j=4.0, **kw):
    """Run multiple seeds, return list of trace dicts."""
    return [run_traced(lambda_i, lambda_j, beta_i=beta_i, beta_j=beta_j,
                       T=T, seed=s, **kw)
            for s in range(n_seeds)]


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def rolling(x, w=5):
    """Rolling mean, NaN-padded at start."""
    v = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), v])


def cumulative_mean(x):
    """Cumulative mean: y[t] = mean(x[0..t])."""
    cs = np.nancumsum(x)
    counts = np.arange(1, len(x) + 1)
    return cs / counts


# ───────────────────────────────────────────────────────────────────────────
# Figure A : Cooperation dynamics (apology-forgiveness)
# ───────────────────────────────────────────────────────────────────────────

def figure_cooperation(conds, out_dir, T, w, noisy_cond=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    t_ax = np.arange(T)

    # ── Left panel: rolling cooperation rate (mean +/- std) ──────────
    for label, traces, col in conds:
        seeds = []
        for tr in traces:
            cc = ((tr["actions_i"] == 0) & (tr["actions_j"] == 0)).astype(float)
            seeds.append(rolling(cc, w))
        arr = np.array(seeds)
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax1.plot(t_ax, mu, c=col, lw=2, label=label)
        ax1.fill_between(t_ax, mu - sd, mu + sd, color=col, alpha=0.15)

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel(f"Cooperation Rate (rolling w={w})", fontsize=12)
    ax1.set_title("A.  Cooperation Dynamics", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9, loc="center right")
    ax1.grid(True, alpha=0.3)

    # ── Right panel: representative single-seed traces ───────────────
    # Use noisy high-empathy condition to show defection-then-recovery.
    # Fall back to moderate empathy (index 1) if noisy not available.
    if noisy_cond is not None:
        rec_label, rec_traces, rec_col = noisy_cond
    else:
        rec_label, rec_traces, rec_col = conds[1]
    lo_label, lo_traces, lo_col = conds[2]   # low empathy

    # pick a seed with the most defections (best demo of recovery)
    best_idx, best_n = 0, 0
    for i, tr in enumerate(rec_traces):
        n = int(np.sum(tr["actions_i"] == 1) + np.sum(tr["actions_j"] == 1))
        if n > best_n:
            best_idx, best_n = i, n

    # also find a low-empathy seed that has at least one cooperation
    lo_idx = 0
    for i, tr in enumerate(lo_traces):
        if np.any(tr["actions_i"] == 0) or np.any(tr["actions_j"] == 0):
            lo_idx = i
            break

    tr_rec = rec_traces[best_idx]
    tr_lo = lo_traces[lo_idx]

    cc_rec = ((tr_rec["actions_i"] == 0) & (tr_rec["actions_j"] == 0)).astype(float)
    cc_lo = ((tr_lo["actions_i"] == 0) & (tr_lo["actions_j"] == 0)).astype(float)

    ax2.step(t_ax, cc_rec + 0.02, where="mid", color=rec_col, lw=1.5, alpha=0.85,
             label=f"High empathy ({rec_label})")
    ax2.step(t_ax, cc_lo - 0.02, where="mid", color=lo_col, lw=1.5, alpha=0.85,
             label=f"{lo_label}")

    # highlight defection rounds in recovery trace
    defs = np.where(cc_rec == 0)[0]
    for d in defs:
        ax2.axvspan(d - 0.5, d + 0.5, color="red", alpha=0.15)
    if len(defs):
        ax2.annotate(
            "defection\nrecovery",
            xy=(defs[0], 0.5), xytext=(min(defs[0] + 12, T - 25), 0.55),
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        )

    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Mutual Cooperation (0 / 1)", fontsize=12)
    ax2.set_title("B.  Example Traces (apology-forgiveness)", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.15, 1.25)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"cooperation_timeseries.{ext}", dpi=150)
    plt.close(fig)
    print("Saved: cooperation_timeseries.png / .pdf")


# ───────────────────────────────────────────────────────────────────────────
# Figure B : Prediction error & belief synchronization
# ───────────────────────────────────────────────────────────────────────────

def figure_belief(conds, out_dir, T, w):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    t_ax = np.arange(T)

    # ── Left panel: rolling action agreement rate ────────────────────
    # Agreement = both agents chose the same action (CC or DD).
    # High symmetric empathy → both cooperate → high agreement.
    # Asymmetric empathy → one cooperates, one defects → low agreement.
    for label, traces, col in conds:
        seeds = []
        for tr in traces:
            agree = (tr["actions_i"] == tr["actions_j"]).astype(float)
            seeds.append(rolling(agree, w))
        arr = np.array(seeds)
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax1.plot(t_ax, mu, c=col, lw=2, label=label)
        ax1.fill_between(t_ax, mu - sd, mu + sd, color=col, alpha=0.15)

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel(f"Action Agreement Rate (rolling w={w})", fontsize=12)
    ax1.set_title("A.  Behavioural Synchrony", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Right panel: cumulative cooperation convergence ─────────────
    # Shows how rapidly each regime reaches its equilibrium cooperation
    # rate.  High empathy converges to ~1 within a few rounds.
    for label, traces, col in conds:
        seeds = []
        for tr in traces:
            cc = ((tr["actions_i"] == 0) & (tr["actions_j"] == 0)).astype(float)
            seeds.append(cumulative_mean(cc))
        arr = np.array(seeds)
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax2.plot(t_ax, mu, c=col, lw=2, label=label)
        ax2.fill_between(t_ax, mu - sd, mu + sd, color=col, alpha=0.15)

    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Cumulative Cooperation Rate", fontsize=12)
    ax2.set_title("B.  Cooperation Convergence", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"belief_synchronization.{ext}", dpi=150)
    plt.close(fig)
    print("Saved: belief_synchronization.png / .pdf")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate time-series figures")
    parser.add_argument("--output", default="figures/", help="Output directory")
    parser.add_argument("--T", type=int, default=100, help="Rounds per episode")
    parser.add_argument("--n_seeds", type=int, default=20, help="Seeds per condition")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    T, ns, w = args.T, args.n_seeds, args.window

    print("=" * 60)
    print(f"Time-series figures  |  T={T}  seeds={ns}  window={w}")
    print("=" * 60)

    # Main conditions (shown in all averaged panels)
    main_specs = [
        # (label, lambda_i, lambda_j, color, beta_i, beta_j)
        (r"High empathy ($\lambda$=0.7)",   0.7, 0.7, "#2ecc71", 4.0, 4.0),
        (r"Moderate ($\lambda$=0.4)",       0.4, 0.4, "#3498db", 4.0, 4.0),
        (r"Low empathy ($\lambda$=0.1)",    0.1, 0.1, "#e74c3c", 4.0, 4.0),
        (r"Asymmetric ($\lambda$=0.9/0.1)", 0.9, 0.1, "#9b59b6", 4.0, 4.0),
    ]

    # Extra noisy condition (only for example-trace panel in Figure A)
    noisy_spec = (r"$\lambda$=0.7, $\beta$=2", 0.7, 0.7, "#27ae60", 2.0, 2.0)

    conds = []
    for label, li, lj, col, bi, bj in main_specs:
        print(f"\n  Running {label} ...")
        traces = run_condition(li, lj, n_seeds=ns, T=T, beta_i=bi, beta_j=bj)
        avg_cc = np.mean([
            np.mean((tr["actions_i"] == 0) & (tr["actions_j"] == 0))
            for tr in traces
        ])
        print(f"    mean CC = {avg_cc:.3f}")
        conds.append((label, traces, col))

    # Run noisy condition
    nl, nli, nlj, ncol, nbi, nbj = noisy_spec
    print(f"\n  Running noisy: {nl} ...")
    noisy_traces = run_condition(nli, nlj, n_seeds=ns, T=T, beta_i=nbi, beta_j=nbj)
    avg_cc = np.mean([
        np.mean((tr["actions_i"] == 0) & (tr["actions_j"] == 0))
        for tr in noisy_traces
    ])
    print(f"    mean CC = {avg_cc:.3f}")
    noisy_cond = (nl, noisy_traces, ncol)

    print("\nGenerating Figure A ...")
    figure_cooperation(conds, out_dir, T, w, noisy_cond=noisy_cond)

    print("Generating Figure B ...")
    figure_belief(conds, out_dir, T, w)

    print(f"\nDone. Figures in {out_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
