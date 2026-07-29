#!/usr/bin/env python
"""
Revision analyses for JRSI rsif-2026-0208 (Referee 1, points 6 and 8).

Analysis 1 (--mode comparison): model comparison of three AIF agent types
    - self_interested : lambda = 0, no opponent inversion
    - tom_only        : lambda = 0, particle-filter opponent inversion ON
                        (accurate opponent prediction, no welfare weighting)
    - empathic        : lambda > 0, opponent inversion ON
    Run as the full 3x3 pairing matrix so the off-diagonal cells expose
    exploitability (who exploits whom when types are mixed).
    Directly operationalises the claim that belief accuracy alone does not
    generate cooperation: tom_only should defect like self_interested.

Analysis 2 (--mode horizon): robustness of the planning-horizon result.
    Sweep planning horizon H x empathy lambda x payoff structure to test
    whether "deeper planning reduces cooperation at moderate empathy" is a
    general property or an artifact of the standard payoff matrix.
    Payoff structures (R, S, T, P), all satisfying T > R > P > S and 2R > T + S:
    - standard         (3, 0, 5, 1)  the decision payoffs used throughout the
                                     paper's ToM/planning modules (PD_PAYOFFS)
    - weak_temptation  (3, 1, 4, 2)  smaller T-R and P-S margins
    - high_temptation  (5, 0, 8, 1)  larger absolute temptation, still 2R > T+S

    NOTE ON MECHANISM: the agents' decision loop reads payoffs from the shared
    PD_PAYOFFS dict in tom.tom_core (NOT from the config C matrices, which only
    parameterise the pymdp state-inference model). This script therefore swaps
    payoff structures by mutating PD_PAYOFFS in place (all modules import the
    same dict object) and restores it afterwards. This also means the paper's
    Methods should state that decision payoffs are (3, 0, 5, 1) while the
    legacy realized-payoff logging in run_pd_experiments.py used (3, 1, 4, 2).

Usage:
    python scripts/run_referee_analyses.py --mode comparison --quick
    python scripts/run_referee_analyses.py --mode horizon --quick
    python scripts/run_referee_analyses.py --mode all            # full run
"""

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pymdp.utils import obj_array, obj_array_uniform

from empathy.prisoners_dilemma import ToMEmpatheticAgent, Environment
from empathy.prisoners_dilemma.tom.tom_core import PD_PAYOFFS

COOPERATE = 0

# Default empathic lambda for the comparison analysis (paper's cooperative regime)
EMPATHIC_LAMBDA = 0.6

AGENT_TYPES = {
    "self_interested": dict(empathy_factor=0.0, use_inversion=False),
    "tom_only":        dict(empathy_factor=0.0, use_inversion=True),
    "empathic":        dict(empathy_factor=EMPATHIC_LAMBDA, use_inversion=True),
}

PAYOFF_STRUCTURES = {
    # name: (R, S, T, P); all satisfy T > R > P > S and 2R > T + S
    "standard":        (3.0, 0.0, 5.0, 1.0),
    "weak_temptation": (3.0, 1.0, 4.0, 2.0),
    "high_temptation": (5.0, 0.0, 8.0, 1.0),
}

_PD_PAYOFFS_ORIGINAL = dict(PD_PAYOFFS)


def _set_decision_payoffs(payoffs: tuple):
    """Mutate the shared PD_PAYOFFS dict in place so every ToM/planning module
    (tom_core, sophisticated_planner) sees the chosen payoff structure."""
    R, S, Tt, P = payoffs
    PD_PAYOFFS.clear()
    PD_PAYOFFS.update({
        (0, 0): (R, R),    # CC
        (0, 1): (S, Tt),   # I cooperate, they defect
        (1, 0): (Tt, S),   # I defect, they cooperate
        (1, 1): (P, P),    # DD
    })


def _restore_decision_payoffs():
    PD_PAYOFFS.clear()
    PD_PAYOFFS.update(_PD_PAYOFFS_ORIGINAL)


def create_pd_config(T: int, payoffs: tuple, legacy_C: bool = False) -> dict:
    """PD config matching run_pd_experiments.py, parameterised by payoffs.

    legacy_C=True reproduces the paper's figure scripts exactly: the pymdp
    state-inference preference vectors stay fixed at (3,1,4,2) regardless of
    the decision payoff structure (behaviour is driven by PD_PAYOFFS either
    way; this only pins the state-inference model to the published config).
    """
    R, S, Tt, P = payoffs
    if legacy_C:
        R, S, Tt, P = 3.0, 1.0, 4.0, 2.0
    n_mod, n_fac, n_obs, n_st = 1, 1, 4, 4

    A0 = obj_array(n_mod); A0[0] = np.eye(n_obs)
    B0 = obj_array(n_fac); B0[0] = np.zeros((4, 4, 2))
    B0[0][0, :, 0] = 0.5; B0[0][1, :, 0] = 0.5
    B0[0][2, :, 1] = 0.5; B0[0][3, :, 1] = 0.5
    # Observation order: CC, CD, DC, DD (from agent 0's perspective)
    C0 = obj_array(n_mod); C0[0] = np.array([R, S, Tt, P])
    D0 = obj_array_uniform([n_st])

    A1 = obj_array(n_mod); A1[0] = np.eye(n_obs)
    B1 = obj_array(n_fac); B1[0] = np.zeros((4, 4, 2))
    B1[0][0, :, 0] = 0.5; B1[0][2, :, 0] = 0.5
    B1[0][1, :, 1] = 0.5; B1[0][3, :, 1] = 0.5
    C1 = obj_array(n_mod); C1[0] = np.array([R, Tt, S, P])
    D1 = obj_array_uniform([n_st])

    return {
        "T": T, "K": 2,
        "A": [A0, A1], "B": [B0, B1],
        "C": [C0, C1], "D": [D0, D1],
        "empathy_factor": [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        "actions": ["C", "D"], "learn": False,
        "policy_len": 2, "same_pref": False,
    }


@dataclass
class PairResult:
    analysis: str
    type_i: str
    type_j: str
    lambda_i: float
    lambda_j: float
    use_inversion_i: bool
    use_inversion_j: bool
    planning_horizon: int
    payoff_structure: str
    T: int
    seed: int
    coop_rate_i: float
    coop_rate_j: float
    freq_CC: float
    freq_CD: float
    freq_DC: float
    freq_DD: float
    payoff_i_mean: float
    payoff_j_mean: float
    payoff_gap: float
    exploitability: float  # |coop_i - coop_j|, Hongju's metric


def run_pair(
    analysis: str,
    type_i: str, type_j: str,
    kwargs_i: dict, kwargs_j: dict,
    T: int, seed: int,
    payoff_structure: str = "standard",
    planning_horizon: int = 1,
    legacy_C: bool = False,
) -> PairResult:
    np.random.seed(seed)
    payoffs = PAYOFF_STRUCTURES[payoff_structure]
    R, S, Tt, P = payoffs

    # Swap the DECISION payoffs (shared PD_PAYOFFS dict) - this is what the
    # ToM/planning modules actually read. Config C matrices are set
    # consistently for the pymdp state-inference model (or pinned to the
    # published legacy config with legacy_C=True).
    _set_decision_payoffs(payoffs)

    config = create_pd_config(T=T, payoffs=payoffs, legacy_C=legacy_C)
    env = Environment(K=2)

    ag_i = ToMEmpatheticAgent(config=config, agent_num=0, **kwargs_i)
    ag_j = ToMEmpatheticAgent(config=config, agent_num=1, **kwargs_j)

    # Realized payoffs use the SAME structure as the decision payoffs
    payoff_matrix = {
        (0, 0): R,   # CC
        (0, 1): S,   # I cooperate, they defect
        (1, 0): Tt,  # I defect, they cooperate
        (1, 1): P,   # DD
    }

    actions_i, actions_j, payoffs_i, payoffs_j = [], [], [], []
    actions = [0, 0]

    for t in range(T):
        obs = env.step(t=t, actions=actions)
        if t == 0:
            obs_i, obs_j = ag_i.o_init, ag_j.o_init
        else:
            obs_i, obs_j = obs[0], obs[1]

        a_i = ag_i.step(t=t, observation=obs_i)["exp_action"]
        a_j = ag_j.step(t=t, observation=obs_j)["exp_action"]

        actions_i.append(a_i); actions_j.append(a_j)
        actions = [a_i, a_j]
        payoffs_i.append(payoff_matrix[(a_i, a_j)])
        payoffs_j.append(payoff_matrix[(a_j, a_i)])

    _restore_decision_payoffs()

    actions_i = np.array(actions_i); actions_j = np.array(actions_j)
    payoffs_i = np.array(payoffs_i); payoffs_j = np.array(payoffs_j)

    both = list(zip(actions_i, actions_j))
    n = len(both)
    coop_i = 1.0 - actions_i.mean()
    coop_j = 1.0 - actions_j.mean()

    return PairResult(
        analysis=analysis,
        type_i=type_i, type_j=type_j,
        lambda_i=kwargs_i.get("empathy_factor", 0.0),
        lambda_j=kwargs_j.get("empathy_factor", 0.0),
        use_inversion_i=kwargs_i.get("use_inversion", False),
        use_inversion_j=kwargs_j.get("use_inversion", False),
        planning_horizon=planning_horizon,
        payoff_structure=payoff_structure,
        T=T, seed=seed,
        coop_rate_i=coop_i, coop_rate_j=coop_j,
        freq_CC=sum(1 for a, b in both if a == 0 and b == 0) / n,
        freq_CD=sum(1 for a, b in both if a == 0 and b == 1) / n,
        freq_DC=sum(1 for a, b in both if a == 1 and b == 0) / n,
        freq_DD=sum(1 for a, b in both if a == 1 and b == 1) / n,
        payoff_i_mean=float(payoffs_i.mean()),
        payoff_j_mean=float(payoffs_j.mean()),
        payoff_gap=float(payoffs_i.mean() - payoffs_j.mean()),
        exploitability=abs(coop_i - coop_j),
    )


# -----------------------------------------------------------------------
# Analysis 1: model comparison (3x3 pairing matrix)
# -----------------------------------------------------------------------

def run_comparison(T: int, n_seeds: int, out_dir: Path) -> list:
    print("=" * 70)
    print("ANALYSIS 1: model comparison "
          f"(self_interested / tom_only / empathic, lambda_emp={EMPATHIC_LAMBDA})")
    print(f"3x3 pairings x {n_seeds} seeds x T={T}")
    print("=" * 70)

    results = []
    pairings = list(itertools.product(AGENT_TYPES.keys(), repeat=2))
    start = datetime.now()

    for idx, (ti, tj) in enumerate(pairings):
        for seed in range(n_seeds):
            results.append(run_pair(
                analysis="comparison",
                type_i=ti, type_j=tj,
                kwargs_i=dict(AGENT_TYPES[ti]),
                kwargs_j=dict(AGENT_TYPES[tj]),
                T=T, seed=seed,
            ))
        done = (idx + 1) * n_seeds
        total = len(pairings) * n_seeds
        elapsed = (datetime.now() - start).total_seconds()
        eta = elapsed / done * (total - done)
        print(f"  [{ti} vs {tj}] done ({done}/{total}, ETA {eta/60:.1f} min)")

    _save(results, out_dir / "comparison_results.json")
    _print_comparison_summary(results)
    return results


def _print_comparison_summary(results: list):
    print("\n--- Pairing matrix: mean coop rate (agent i, agent j) | mean payoffs ---")
    header = f"{'i \\ j':<17}" + "".join(f"{t:<26}" for t in AGENT_TYPES)
    print(header)
    for ti in AGENT_TYPES:
        row = f"{ti:<17}"
        for tj in AGENT_TYPES:
            cell = [r for r in results if r.type_i == ti and r.type_j == tj]
            ci = np.mean([r.coop_rate_i for r in cell])
            cj = np.mean([r.coop_rate_j for r in cell])
            pi = np.mean([r.payoff_i_mean for r in cell])
            pj = np.mean([r.payoff_j_mean for r in cell])
            row += f"C:{ci:.2f}/{cj:.2f} U:{pi:.1f}/{pj:.1f}   "
        print(row)

    print("\n--- Key contrasts (symmetric pairings) ---")
    for t in AGENT_TYPES:
        cell = [r for r in results if r.type_i == t and r.type_j == t]
        cc = np.mean([r.freq_CC for r in cell])
        dd = np.mean([r.freq_DD for r in cell])
        print(f"  {t:<17} CC={cc:.2f}  DD={dd:.2f}")
    si = [r for r in results if r.type_i == "self_interested" and r.type_j == "self_interested"]
    to = [r for r in results if r.type_i == "tom_only" and r.type_j == "tom_only"]
    print(f"\n  tom_only vs self_interested CC delta: "
          f"{np.mean([r.freq_CC for r in to]) - np.mean([r.freq_CC for r in si]):+.3f}")
    print("  (near zero => belief accuracy alone does not generate cooperation)")

    print("\n--- Exploitability in mixed pairings (Hongju's |coop_i - coop_j|) ---")
    for ti, tj in itertools.product(AGENT_TYPES, repeat=2):
        if ti == tj:
            continue
        cell = [r for r in results if r.type_i == ti and r.type_j == tj]
        print(f"  {ti} vs {tj}: exploitability={np.mean([r.exploitability for r in cell]):.2f}, "
              f"payoff_gap={np.mean([r.payoff_gap for r in cell]):+.2f}")


# -----------------------------------------------------------------------
# Analysis 2: planning-horizon robustness across payoff structures
# -----------------------------------------------------------------------

def run_horizon(T: int, n_seeds: int, out_dir: Path,
                horizons=(1, 2, 3, 4), lambdas=(0.3, 0.5, 0.7),
                protocol: str = "paper") -> list:
    """protocol="paper" reproduces the exact Table 2 / Fig 7 setup of the
    manuscript, verified to give CC = 0.782/0.657/0.597 at lambda=0.3,
    H=1/2/3 under the standard payoffs (seeds 0-19, T=100):
      - the sophisticated agent is paired with a MYOPIC partner of equal
        lambda (generate_sophisticated_figure.py only passes the horizon to
        agent i),
      - legacy state-inference C matrices (3,1,4,2),
      - metric: mutual cooperation frequency (freq_CC).
    protocol="symmetric" gives both agents the same horizon and aligns the
    C matrices with the decision payoffs (the earlier referee-sweep setup).
    """
    print("=" * 70)
    print(f"ANALYSIS 2: planning-horizon robustness (protocol={protocol})")
    print(f"H={list(horizons)} x lambda={list(lambdas)} x "
          f"payoffs={list(PAYOFF_STRUCTURES)} x {n_seeds} seeds x T={T}")
    print("=" * 70)

    paper = protocol == "paper"
    results = []
    grid = list(itertools.product(PAYOFF_STRUCTURES, horizons, lambdas))
    start = datetime.now()

    for idx, (pname, H, lam) in enumerate(grid):
        kwargs_i = dict(
            empathy_factor=lam,
            use_inversion=False,          # matches generate_sophisticated_figure.py
            use_sophisticated=(H > 1),
            planning_horizon=H,
        )
        if paper:
            # partner is always myopic, exactly as in the published figure
            kwargs_j = dict(empathy_factor=lam, use_inversion=False)
        else:
            kwargs_j = dict(kwargs_i)
        for seed in range(n_seeds):
            results.append(run_pair(
                analysis="horizon",
                type_i=f"H{H}", type_j="H1" if paper else f"H{H}",
                kwargs_i=dict(kwargs_i), kwargs_j=dict(kwargs_j),
                T=T, seed=seed,
                payoff_structure=pname,
                planning_horizon=H,
                legacy_C=paper,
            ))
        done = (idx + 1) * n_seeds
        total = len(grid) * n_seeds
        elapsed = (datetime.now() - start).total_seconds()
        eta = elapsed / done * (total - done)
        print(f"  [{pname} H={H} lambda={lam}] done ({done}/{total}, ETA {eta/60:.1f} min)")

    suffix = "" if paper else "_symmetric"
    _save(results, out_dir / f"horizon_results{suffix}.json")
    _print_horizon_summary(results, horizons, lambdas)
    return results


def _print_horizon_summary(results: list, horizons, lambdas):
    print("\n--- Mutual cooperation frequency (CC) by payoff structure x H x lambda ---")
    for pname in PAYOFF_STRUCTURES:
        print(f"\n  {pname} (R,S,T,P) = {PAYOFF_STRUCTURES[pname]}")
        print("    " + f"{'lambda':<8}" + "".join(f"H={h:<6}" for h in horizons))
        for lam in lambdas:
            row = f"    {lam:<8}"
            for H in horizons:
                cell = [r for r in results
                        if r.payoff_structure == pname
                        and r.planning_horizon == H
                        and r.lambda_i == lam]
                cc = np.mean([r.freq_CC for r in cell])
                row += f"{cc:<8.2f}"
            print(row)
    print("\n  Metric matches paper Table 2 (freq_CC). Read across each row: if")
    print("  cooperation falls with H only under some payoff structures, the")
    print("  horizon effect is payoff-dependent; under all three it is general.")


# -----------------------------------------------------------------------

def _save(results: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved {len(results)} runs -> {path}")


def main():
    parser = argparse.ArgumentParser(description="JRSI revision analyses (Referee 1, points 6 & 8)")
    parser.add_argument("--mode", choices=["comparison", "horizon", "all"], default="all")
    parser.add_argument("--T", type=int, default=100, help="Rounds per game")
    parser.add_argument("--n_seeds", type=int, default=25, help="Seeds per cell")
    parser.add_argument("--quick", action="store_true", help="Tiny run to validate the pipeline")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--protocol", choices=["paper", "symmetric"], default="paper",
                        help="Horizon analysis protocol: 'paper' matches Table 2 / Fig 7 "
                             "(sophisticated vs myopic partner, legacy C, CC metric); "
                             "'symmetric' gives both agents the same horizon.")
    args = parser.parse_args()

    T = 20 if args.quick else args.T
    n_seeds = 2 if args.quick else args.n_seeds
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "results" / "referee_analyses"

    start = datetime.now()
    if args.mode in ("comparison", "all"):
        run_comparison(T=T, n_seeds=n_seeds, out_dir=out_dir)
    if args.mode in ("horizon", "all"):
        if args.quick:
            run_horizon(T=T, n_seeds=n_seeds, out_dir=out_dir, horizons=(1, 2),
                        lambdas=(0.3,), protocol=args.protocol)
        else:
            run_horizon(T=T, n_seeds=n_seeds, out_dir=out_dir, protocol=args.protocol)
    print(f"\nTotal time: {(datetime.now() - start).total_seconds()/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
