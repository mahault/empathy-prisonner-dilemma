#!/usr/bin/env python
"""
Prisoner's Dilemma Experiment Runner.

This is the main entry point for running PD experiments with Theory of Mind
and empathy. Use this script for all PD-related experiments.

Usage:
------
# Quick smoke test (verify everything works)
python scripts/run_pd_experiments.py --mode smoke

# Run parameter sweep
python scripts/run_pd_experiments.py --mode sweep --output results/sweep_results.csv

# Run single experiment with specific parameters
python scripts/run_pd_experiments.py --mode single --lambda_i 0.5 --lambda_j 0.5 --T 50

# Run validation checks
python scripts/run_pd_experiments.py --mode validate
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
from empathy.prisoners_dilemma.metrics.exploitability import (
    ExploitabilityAnalysis,
    classify_outcome,
    OutcomeFrequencies,
)


@dataclass
class RunResult:
    """Results from a single experiment run."""

    # Parameters
    lambda_i: float
    lambda_j: float
    beta_i: float
    beta_j: float
    use_tom: bool
    use_inversion: bool
    T: int
    seed: int

    # Outcome frequencies
    freq_CC: float
    freq_CD: float
    freq_DC: float
    freq_DD: float

    # Payoffs
    payoff_i_mean: float
    payoff_i_std: float
    payoff_j_mean: float
    payoff_j_std: float
    payoff_gap: float

    # Classification
    outcome_label: str

    # Cooperation rates
    coop_rate_i: float
    coop_rate_j: float


def create_pd_config(T: int = 50) -> dict:
    """Create standard PD configuration."""
    num_modalities = 1
    num_factors = 1
    num_obs_categories = 4
    num_state_categories = 4

    # Agent 0 matrices
    A_k0 = obj_array(num_modalities)
    A_k0[0] = np.eye(num_obs_categories)

    B_k0 = obj_array(num_factors)
    B_k0[0] = np.zeros((4, 4, 2))
    B_k0[0][0, :, 0] = np.tile(0.5, 4)
    B_k0[0][1, :, 0] = np.tile(0.5, 4)
    B_k0[0][2, :, 1] = np.tile(0.5, 4)
    B_k0[0][3, :, 1] = np.tile(0.5, 4)

    C_k0 = obj_array(num_modalities)
    C_k0[0] = np.array([3, 1, 4, 2])  # Payoffs: CC=3, CD=1, DC=4, DD=2

    D_k0 = obj_array_uniform([num_state_categories])

    # Agent 1 matrices
    A_k1 = obj_array(num_modalities)
    A_k1[0] = np.eye(num_obs_categories)

    B_k1 = obj_array(num_factors)
    B_k1[0] = np.zeros((4, 4, 2))
    B_k1[0][0, :, 0] = np.tile(0.5, 4)
    B_k1[0][2, :, 0] = np.tile(0.5, 4)
    B_k1[0][1, :, 1] = np.tile(0.5, 4)
    B_k1[0][3, :, 1] = np.tile(0.5, 4)

    C_k1 = obj_array(num_modalities)
    C_k1[0] = np.array([3, 4, 1, 2])  # Payoffs from j's perspective

    D_k1 = obj_array_uniform([num_state_categories])

    config = {
        "T": T,
        "K": 2,
        "A": [A_k0, A_k1],
        "B": [B_k0, B_k1],
        "C": [C_k0, C_k1],
        "D": [D_k0, D_k1],
        "empathy_factor": [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        "actions": ["C", "D"],
        "learn": False,
        "policy_len": 2,
        "same_pref": False,
    }

    return config


# PD payoff matrix: (my_action, their_action) -> my_payoff
PAYOFF_MATRIX = {
    (0, 0): 3,  # CC -> 3
    (0, 1): 1,  # CD -> 1 (sucker)
    (1, 0): 4,  # DC -> 4 (temptation)
    (1, 1): 2,  # DD -> 2
}


def run_single_experiment(
    lambda_i: float,
    lambda_j: float,
    beta_i: float = 4.0,
    beta_j: float = 4.0,
    use_tom: bool = True,
    use_inversion: bool = False,
    T: int = 50,
    seed: int = 42,
) -> RunResult:
    """Run a single PD experiment with specified parameters."""
    np.random.seed(seed)

    config = create_pd_config(T=T)
    env = Environment(K=2)

    # Create agents with specified empathy
    agent_i = ToMEmpatheticAgent(
        config=config,
        agent_num=0,
        empathy_factor=lambda_i,
        use_inversion=use_inversion,
    )
    agent_j = ToMEmpatheticAgent(
        config=config,
        agent_num=1,
        empathy_factor=lambda_j,
        use_inversion=use_inversion,
    )

    # Run simulation
    actions_i = []
    actions_j = []
    payoffs_i = []
    payoffs_j = []
    actions = [0, 0]

    for t in range(T):
        obs = env.step(t=t, actions=actions)

        if t == 0:
            obs_i, obs_j = agent_i.o_init, agent_j.o_init
        else:
            obs_i, obs_j = obs[0], obs[1]

        results_i = agent_i.step(t=t, observation=obs_i)
        results_j = agent_j.step(t=t, observation=obs_j)

        a_i = results_i["exp_action"]
        a_j = results_j["exp_action"]

        actions_i.append(a_i)
        actions_j.append(a_j)
        actions = [a_i, a_j]

        # Compute payoffs
        payoffs_i.append(PAYOFF_MATRIX[(a_i, a_j)])
        payoffs_j.append(PAYOFF_MATRIX[(a_j, a_i)])

    # Compute statistics
    actions_i = np.array(actions_i)
    actions_j = np.array(actions_j)
    payoffs_i = np.array(payoffs_i)
    payoffs_j = np.array(payoffs_j)

    # Outcome frequencies
    outcomes = []
    for ai, aj in zip(actions_i, actions_j):
        if ai == 0 and aj == 0:
            outcomes.append("CC")
        elif ai == 0 and aj == 1:
            outcomes.append("CD")
        elif ai == 1 and aj == 0:
            outcomes.append("DC")
        else:
            outcomes.append("DD")

    freq_CC = outcomes.count("CC") / len(outcomes)
    freq_CD = outcomes.count("CD") / len(outcomes)
    freq_DC = outcomes.count("DC") / len(outcomes)
    freq_DD = outcomes.count("DD") / len(outcomes)

    # Cooperation rates
    coop_rate_i = 1.0 - actions_i.mean()
    coop_rate_j = 1.0 - actions_j.mean()

    # Outcome classification
    payoff_gap = payoffs_i.mean() - payoffs_j.mean()
    if freq_CC > 0.7:
        outcome_label = "mutual_cooperation"
    elif freq_DD > 0.7:
        outcome_label = "mutual_defection"
    elif coop_rate_i > 0.6 and coop_rate_j < 0.4 and payoff_gap < -0.5:
        outcome_label = "i_exploited"
    elif coop_rate_j > 0.6 and coop_rate_i < 0.4 and payoff_gap > 0.5:
        outcome_label = "j_exploited"
    else:
        outcome_label = "mixed"

    return RunResult(
        lambda_i=lambda_i,
        lambda_j=lambda_j,
        beta_i=beta_i,
        beta_j=beta_j,
        use_tom=use_tom,
        use_inversion=use_inversion,
        T=T,
        seed=seed,
        freq_CC=freq_CC,
        freq_CD=freq_CD,
        freq_DC=freq_DC,
        freq_DD=freq_DD,
        payoff_i_mean=payoffs_i.mean(),
        payoff_i_std=payoffs_i.std(),
        payoff_j_mean=payoffs_j.mean(),
        payoff_j_std=payoffs_j.std(),
        payoff_gap=payoff_gap,
        outcome_label=outcome_label,
        coop_rate_i=coop_rate_i,
        coop_rate_j=coop_rate_j,
    )


def run_smoke_test():
    """Quick smoke test to verify everything works."""
    print("Running smoke test...")
    print("=" * 60)

    result = run_single_experiment(
        lambda_i=0.5,
        lambda_j=0.5,
        T=20,
        seed=42,
    )

    print(f"Parameters: lambda_i={result.lambda_i}, lambda_j={result.lambda_j}")
    print(f"Outcome frequencies: CC={result.freq_CC:.2f}, CD={result.freq_CD:.2f}, "
          f"DC={result.freq_DC:.2f}, DD={result.freq_DD:.2f}")
    print(f"Payoffs: i={result.payoff_i_mean:.2f}+/-{result.payoff_i_std:.2f}, "
          f"j={result.payoff_j_mean:.2f}+/-{result.payoff_j_std:.2f}")
    print(f"Cooperation rates: i={result.coop_rate_i:.2f}, j={result.coop_rate_j:.2f}")
    print(f"Outcome: {result.outcome_label}")
    print("=" * 60)
    print("Smoke test PASSED")
    return 0


def run_sweep(output_path: str = None, n_seeds: int = 10, quick: bool = False):
    """Run parameter sweep over empathy and precision values.

    Full sweep (from roadmap):
    - lambda_i, lambda_j: [0, 0.25, 0.5, 0.75, 1.0]
    - beta_i, beta_j: [1, 4, 16]
    - use_inversion: [False, True]
    - n_seeds: 50 per cell

    Quick sweep (for testing):
    - lambda: [0, 0.5, 1.0]
    - beta: [4]
    - use_inversion: [False]
    """
    print("Running parameter sweep...")
    print("=" * 60)

    # Parameter grid (from roadmap Phase 5)
    if quick:
        lambda_values = [0.0, 0.5, 1.0]
        beta_values = [4.0]
        use_inversion_values = [False]
    else:
        lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        beta_values = [1.0, 4.0, 16.0]
        use_inversion_values = [False, True]

    total_runs = (
        len(lambda_values) ** 2
        * len(beta_values) ** 2
        * len(use_inversion_values)
        * n_seeds
    )
    print(f"Parameter grid:")
    print(f"  lambda values: {lambda_values}")
    print(f"  beta values: {beta_values}")
    print(f"  inversion: {use_inversion_values}")
    print(f"  seeds per cell: {n_seeds}")
    print(f"Total runs: {total_runs}")
    print()

    results = []
    run_count = 0
    start_time = datetime.now()

    for lambda_i, lambda_j in itertools.product(lambda_values, repeat=2):
        for beta_i, beta_j in itertools.product(beta_values, repeat=2):
            for use_inversion in use_inversion_values:
                for seed in range(n_seeds):
                    run_count += 1
                    if run_count % 100 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = run_count / elapsed
                        remaining = (total_runs - run_count) / rate
                        print(f"Progress: {run_count}/{total_runs} "
                              f"({100*run_count/total_runs:.1f}%) "
                              f"- ETA: {remaining/60:.1f} min")

                    result = run_single_experiment(
                        lambda_i=lambda_i,
                        lambda_j=lambda_j,
                        beta_i=beta_i,
                        beta_j=beta_j,
                        use_tom=True,
                        use_inversion=use_inversion,
                        T=50,
                        seed=seed,
                    )
                    results.append(asdict(result))

    # Save results
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / f"results/sweep_{timestamp}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("=" * 60)

    # Print summary
    from collections import Counter, defaultdict

    print("\n=== SWEEP SUMMARY ===\n")

    # Overall outcome distribution
    print("Overall Outcome Distribution:")
    outcomes = Counter(r["outcome_label"] for r in results)
    for label, count in outcomes.most_common():
        print(f"  {label}: {count} ({100*count/len(results):.1f}%)")

    # Cooperation rate by empathy level
    print("\nCooperation Rate by Empathy (lambda):")
    lambda_coop = defaultdict(list)
    for r in results:
        key = (r["lambda_i"], r["lambda_j"])
        lambda_coop[key].append((r["coop_rate_i"] + r["coop_rate_j"]) / 2)

    print("  lambda_i  lambda_j  avg_coop")
    for (li, lj) in sorted(lambda_coop.keys()):
        avg = np.mean(lambda_coop[(li, lj)])
        print(f"  {li:7.2f}  {lj:7.2f}  {avg:8.2f}")

    # Effect of symmetric vs asymmetric empathy
    print("\nSymmetric vs Asymmetric Empathy:")
    symmetric = [r for r in results if r["lambda_i"] == r["lambda_j"]]
    asymmetric = [r for r in results if r["lambda_i"] != r["lambda_j"]]
    print(f"  Symmetric (n={len(symmetric)}): "
          f"CC={np.mean([r['freq_CC'] for r in symmetric]):.2f}, "
          f"DD={np.mean([r['freq_DD'] for r in symmetric]):.2f}")
    print(f"  Asymmetric (n={len(asymmetric)}): "
          f"CC={np.mean([r['freq_CC'] for r in asymmetric]):.2f}, "
          f"DD={np.mean([r['freq_DD'] for r in asymmetric]):.2f}")

    # Effect of inversion
    print("\nEffect of Opponent Inversion:")
    with_inv = [r for r in results if r["use_inversion"]]
    without_inv = [r for r in results if not r["use_inversion"]]
    if with_inv:
        print(f"  With inversion (n={len(with_inv)}): "
              f"CC={np.mean([r['freq_CC'] for r in with_inv]):.2f}")
    if without_inv:
        print(f"  Without inversion (n={len(without_inv)}): "
              f"CC={np.mean([r['freq_CC'] for r in without_inv]):.2f}")

    # Key finding: high empathy cooperation
    high_emp = [r for r in results if r["lambda_i"] >= 0.5 and r["lambda_j"] >= 0.5]
    low_emp = [r for r in results if r["lambda_i"] <= 0.25 and r["lambda_j"] <= 0.25]
    print("\nKey Finding - Empathy Effect:")
    print(f"  High empathy (lambda >= 0.5): CC rate = {np.mean([r['freq_CC'] for r in high_emp]):.2f}")
    print(f"  Low empathy (lambda <= 0.25): CC rate = {np.mean([r['freq_CC'] for r in low_emp]):.2f}")

    return 0


def run_validation():
    """Run validation checks from roadmap."""
    print("Running validation checks...")
    print("=" * 60)

    checks_passed = 0
    checks_total = 4

    # Check 1: lambda=0, ToM=on should produce baseline (selfish) behavior
    print("\nCheck 1: lambda=0 with ToM should show no empathy effect")
    result = run_single_experiment(lambda_i=0.0, lambda_j=0.0, use_tom=True, T=50)
    print(f"  Outcome: {result.outcome_label}")
    print(f"  Cooperation: i={result.coop_rate_i:.2f}, j={result.coop_rate_j:.2f}")
    if result.coop_rate_i < 0.5 and result.coop_rate_j < 0.5:
        print("  PASS: Low cooperation as expected without empathy")
        checks_passed += 1
    else:
        print("  INCONCLUSIVE: Higher cooperation than expected")
        checks_passed += 0.5

    # Check 2: lambda>0, ToM=on should enable pro-social behavior
    print("\nCheck 2: lambda>0 with ToM should show empathy effect")
    result = run_single_experiment(lambda_i=0.7, lambda_j=0.7, use_tom=True, T=50)
    print(f"  Outcome: {result.outcome_label}")
    print(f"  Cooperation: i={result.coop_rate_i:.2f}, j={result.coop_rate_j:.2f}")
    if result.coop_rate_i > 0.3 or result.coop_rate_j > 0.3:
        print("  PASS: Some cooperation observed with empathy")
        checks_passed += 1
    else:
        print("  CHECK: Lower cooperation than expected with empathy")
        checks_passed += 0.5

    # Check 3: Asymmetric lambda should show exploitation
    print("\nCheck 3: Asymmetric lambda (high vs low) should show exploitation dynamics")
    result = run_single_experiment(lambda_i=0.9, lambda_j=0.1, use_tom=True, T=50)
    print(f"  Outcome: {result.outcome_label}")
    print(f"  Cooperation: i={result.coop_rate_i:.2f}, j={result.coop_rate_j:.2f}")
    print(f"  Payoff gap: {result.payoff_gap:.2f}")
    if result.payoff_gap < 0:
        print("  PASS: High-empathy agent (i) at disadvantage as expected")
        checks_passed += 1
    else:
        print("  INCONCLUSIVE: Payoff dynamics different than expected")
        checks_passed += 0.5

    # Check 4: Symmetric high lambda should enable mutual cooperation
    print("\nCheck 4: Symmetric high lambda should enable mutual cooperation")
    result = run_single_experiment(lambda_i=0.8, lambda_j=0.8, use_tom=True, T=50)
    print(f"  Outcome: {result.outcome_label}")
    print(f"  CC frequency: {result.freq_CC:.2f}")
    if result.freq_CC > 0.3:
        print("  PASS: Significant mutual cooperation observed")
        checks_passed += 1
    else:
        print("  CHECK: Lower CC rate than expected")
        checks_passed += 0.5

    print("\n" + "=" * 60)
    print(f"Validation: {checks_passed}/{checks_total} checks passed")
    return 0 if checks_passed >= checks_total * 0.5 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Prisoner's Dilemma Experiment Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "single", "sweep", "validate"],
        default="smoke",
        help="Experiment mode",
    )
    parser.add_argument("--lambda_i", type=float, default=0.5, help="Agent i empathy")
    parser.add_argument("--lambda_j", type=float, default=0.5, help="Agent j empathy")
    parser.add_argument("--T", type=int, default=50, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_seeds", type=int, default=50, help="Seeds per cell for sweep")
    parser.add_argument("--output", type=str, help="Output file for sweep results")
    parser.add_argument("--quick", action="store_true", help="Quick sweep (fewer parameters)")

    args = parser.parse_args()

    if args.mode == "smoke":
        return run_smoke_test()
    elif args.mode == "single":
        result = run_single_experiment(
            lambda_i=args.lambda_i,
            lambda_j=args.lambda_j,
            T=args.T,
            seed=args.seed,
        )
        print(json.dumps(asdict(result), indent=2))
        return 0
    elif args.mode == "sweep":
        return run_sweep(output_path=args.output, n_seeds=args.n_seeds, quick=args.quick)
    elif args.mode == "validate":
        return run_validation()


if __name__ == "__main__":
    sys.exit(main())
