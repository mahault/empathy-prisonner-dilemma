#!/usr/bin/env python
"""
Phase 6: Visualization and Analysis of Parameter Sweep Results.

Generates figures for the paper:
1. Cooperation heatmap (lambda_i x lambda_j)
2. Outcome distribution by empathy level
3. Effect of beta (action precision)
4. Exploitation dynamics (asymmetric empathy)
5. Payoff analysis

Usage:
------
python scripts/analyze_sweep.py --input results/phase5_full_sweep.json --output figures/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_results(input_path: str) -> list:
    """Load sweep results from JSON."""
    with open(input_path, "r") as f:
        return json.load(f)


def create_cooperation_heatmap(results: list, output_dir: Path, beta_filter: float = None):
    """
    Create heatmap of cooperation rate by lambda_i x lambda_j.

    This is the main figure showing how empathy affects cooperation.
    """
    # Get unique lambda values
    lambda_vals = sorted(set(r["lambda_i"] for r in results))
    n = len(lambda_vals)

    # Filter by beta if specified
    if beta_filter:
        results = [r for r in results if r["beta_i"] == beta_filter and r["beta_j"] == beta_filter]

    # Aggregate cooperation rates
    coop_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))

    for r in results:
        i = lambda_vals.index(r["lambda_i"])
        j = lambda_vals.index(r["lambda_j"])
        coop_matrix[i, j] += r["freq_CC"]
        count_matrix[i, j] += 1

    # Average
    coop_matrix = coop_matrix / np.maximum(count_matrix, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(coop_matrix, cmap="RdYlGn", vmin=0, vmax=1, origin="lower")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mutual Cooperation Rate", fontsize=12)

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{v:.2f}" for v in lambda_vals])
    ax.set_yticklabels([f"{v:.2f}" for v in lambda_vals])
    ax.set_xlabel("Agent j Empathy (lambda_j)", fontsize=12)
    ax.set_ylabel("Agent i Empathy (lambda_i)", fontsize=12)

    title = "Mutual Cooperation Rate by Empathy Level"
    if beta_filter:
        title += f" (beta={beta_filter})"
    ax.set_title(title, fontsize=14)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = coop_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()

    suffix = f"_beta{int(beta_filter)}" if beta_filter else ""
    fig.savefig(output_dir / f"cooperation_heatmap{suffix}.png", dpi=150)
    fig.savefig(output_dir / f"cooperation_heatmap{suffix}.pdf")
    plt.close(fig)

    print(f"Saved: cooperation_heatmap{suffix}.png")


def create_outcome_distribution(results: list, output_dir: Path):
    """
    Create bar chart showing outcome distribution by empathy category.
    """
    # Categorize by empathy level
    categories = {
        "Both Low\n(lambda < 0.25)": lambda r: r["lambda_i"] <= 0.25 and r["lambda_j"] <= 0.25,
        "Both Medium\n(0.25-0.5)": lambda r: 0.25 < r["lambda_i"] <= 0.5 and 0.25 < r["lambda_j"] <= 0.5,
        "Both High\n(lambda > 0.5)": lambda r: r["lambda_i"] > 0.5 and r["lambda_j"] > 0.5,
        "Asymmetric": lambda r: abs(r["lambda_i"] - r["lambda_j"]) >= 0.5,
    }

    outcomes = ["mutual_cooperation", "mutual_defection", "i_exploited", "j_exploited", "mixed"]
    outcome_colors = {
        "mutual_cooperation": "#2ecc71",  # Green
        "mutual_defection": "#e74c3c",    # Red
        "i_exploited": "#f39c12",         # Orange
        "j_exploited": "#9b59b6",         # Purple
        "mixed": "#95a5a6",               # Gray
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.15

    for idx, outcome in enumerate(outcomes):
        rates = []
        for cat_name, cat_filter in categories.items():
            cat_results = [r for r in results if cat_filter(r)]
            if cat_results:
                rate = sum(1 for r in cat_results if r["outcome_label"] == outcome) / len(cat_results)
            else:
                rate = 0
            rates.append(rate)

        offset = (idx - len(outcomes)/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=outcome.replace("_", " ").title(),
                     color=outcome_colors[outcome])

    ax.set_xlabel("Empathy Configuration", fontsize=12)
    ax.set_ylabel("Proportion of Outcomes", fontsize=12)
    ax.set_title("Outcome Distribution by Empathy Configuration", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories.keys())
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(output_dir / "outcome_distribution.png", dpi=150)
    fig.savefig(output_dir / "outcome_distribution.pdf")
    plt.close(fig)

    print("Saved: outcome_distribution.png")


def create_beta_effect_plot(results: list, output_dir: Path):
    """
    Show how action precision (beta) affects cooperation.
    """
    beta_vals = sorted(set(r["beta_i"] for r in results))
    lambda_vals = sorted(set(r["lambda_i"] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cooperation rate vs beta for different lambda levels
    ax1 = axes[0]

    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        coop_by_beta = []
        for beta in beta_vals:
            subset = [r for r in results
                     if r["lambda_i"] == lam and r["lambda_j"] == lam
                     and r["beta_i"] == beta and r["beta_j"] == beta]
            if subset:
                coop_by_beta.append(np.mean([r["freq_CC"] for r in subset]))
            else:
                coop_by_beta.append(0)

        ax1.plot(beta_vals, coop_by_beta, 'o-', label=f"lambda={lam}", linewidth=2, markersize=8)

    ax1.set_xlabel("Action Precision (beta)", fontsize=12)
    ax1.set_ylabel("Mutual Cooperation Rate", fontsize=12)
    ax1.set_title("Effect of Beta on Cooperation\n(Symmetric Empathy)", fontsize=14)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Heatmap of beta effect at fixed lambda=0.5
    ax2 = axes[1]

    n = len(beta_vals)
    coop_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))

    subset = [r for r in results if r["lambda_i"] == 0.5 and r["lambda_j"] == 0.5]
    for r in subset:
        i = beta_vals.index(r["beta_i"])
        j = beta_vals.index(r["beta_j"])
        coop_matrix[i, j] += r["freq_CC"]
        count_matrix[i, j] += 1

    coop_matrix = coop_matrix / np.maximum(count_matrix, 1)

    im = ax2.imshow(coop_matrix, cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("CC Rate", fontsize=10)

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels([f"{int(v)}" for v in beta_vals])
    ax2.set_yticklabels([f"{int(v)}" for v in beta_vals])
    ax2.set_xlabel("Agent j Beta", fontsize=12)
    ax2.set_ylabel("Agent i Beta", fontsize=12)
    ax2.set_title("Beta Effect at lambda=0.5", fontsize=14)

    for i in range(n):
        for j in range(n):
            val = coop_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "beta_effect.png", dpi=150)
    fig.savefig(output_dir / "beta_effect.pdf")
    plt.close(fig)

    print("Saved: beta_effect.png")


def create_exploitation_plot(results: list, output_dir: Path):
    """
    Show exploitation dynamics with asymmetric empathy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Payoff gap as function of empathy difference
    ax1 = axes[0]

    # Group by empathy difference
    emp_diffs = defaultdict(list)
    for r in results:
        diff = r["lambda_i"] - r["lambda_j"]
        emp_diffs[round(diff, 2)].append(r["payoff_gap"])

    diffs = sorted(emp_diffs.keys())
    means = [np.mean(emp_diffs[d]) for d in diffs]
    stds = [np.std(emp_diffs[d]) for d in diffs]

    ax1.errorbar(diffs, means, yerr=stds, fmt='o-', capsize=3, linewidth=2, markersize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(diffs, means, 0, alpha=0.3,
                     where=[m > 0 for m in means], color='green', label='Agent i advantaged')
    ax1.fill_between(diffs, means, 0, alpha=0.3,
                     where=[m < 0 for m in means], color='red', label='Agent i disadvantaged')

    ax1.set_xlabel("Empathy Difference (lambda_i - lambda_j)", fontsize=12)
    ax1.set_ylabel("Payoff Gap (i - j)", fontsize=12)
    ax1.set_title("Exploitation: Payoff Gap vs Empathy Asymmetry", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Exploitation outcomes heatmap
    ax2 = axes[1]

    lambda_vals = sorted(set(r["lambda_i"] for r in results))
    n = len(lambda_vals)

    # -1 = i exploited, 0 = neither, +1 = j exploited
    exploit_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))

    for r in results:
        i = lambda_vals.index(r["lambda_i"])
        j = lambda_vals.index(r["lambda_j"])
        if r["outcome_label"] == "i_exploited":
            exploit_matrix[i, j] -= 1
        elif r["outcome_label"] == "j_exploited":
            exploit_matrix[i, j] += 1
        count_matrix[i, j] += 1

    exploit_matrix = exploit_matrix / np.maximum(count_matrix, 1)

    # Custom colormap: red (i exploited) -> white -> blue (j exploited)
    cmap = plt.cm.RdBu
    im = ax2.imshow(exploit_matrix, cmap=cmap, vmin=-1, vmax=1, origin="lower")
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Exploitation Direction\n(<0: i exploited, >0: j exploited)", fontsize=10)

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels([f"{v:.2f}" for v in lambda_vals])
    ax2.set_yticklabels([f"{v:.2f}" for v in lambda_vals])
    ax2.set_xlabel("Agent j Empathy (lambda_j)", fontsize=12)
    ax2.set_ylabel("Agent i Empathy (lambda_i)", fontsize=12)
    ax2.set_title("Exploitation Direction by Empathy", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / "exploitation_dynamics.png", dpi=150)
    fig.savefig(output_dir / "exploitation_dynamics.pdf")
    plt.close(fig)

    print("Saved: exploitation_dynamics.png")


def create_inversion_effect_plot(results: list, output_dir: Path):
    """
    Compare results with and without opponent inversion.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lambda_vals = sorted(set(r["lambda_i"] for r in results))

    # Get symmetric empathy results
    with_inv = []
    without_inv = []

    for lam in lambda_vals:
        subset_with = [r for r in results
                      if r["lambda_i"] == lam and r["lambda_j"] == lam and r["use_inversion"]]
        subset_without = [r for r in results
                         if r["lambda_i"] == lam and r["lambda_j"] == lam and not r["use_inversion"]]

        with_inv.append(np.mean([r["freq_CC"] for r in subset_with]) if subset_with else 0)
        without_inv.append(np.mean([r["freq_CC"] for r in subset_without]) if subset_without else 0)

    x = np.arange(len(lambda_vals))
    width = 0.35

    bars1 = ax.bar(x - width/2, without_inv, width, label='Without Inversion', color='#3498db')
    bars2 = ax.bar(x + width/2, with_inv, width, label='With Inversion', color='#e74c3c')

    ax.set_xlabel("Symmetric Empathy Level (lambda)", fontsize=12)
    ax.set_ylabel("Mutual Cooperation Rate", fontsize=12)
    ax.set_title("Effect of Opponent Inversion on Cooperation", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in lambda_vals])
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "inversion_effect.png", dpi=150)
    fig.savefig(output_dir / "inversion_effect.pdf")
    plt.close(fig)

    print("Saved: inversion_effect.png")


def create_summary_figure(results: list, output_dir: Path):
    """
    Create a single summary figure with key findings.
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    lambda_vals = sorted(set(r["lambda_i"] for r in results))
    n = len(lambda_vals)

    # Panel A: Cooperation heatmap
    ax1 = fig.add_subplot(gs[0, 0])

    coop_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))
    for r in results:
        i = lambda_vals.index(r["lambda_i"])
        j = lambda_vals.index(r["lambda_j"])
        coop_matrix[i, j] += r["freq_CC"]
        count_matrix[i, j] += 1
    coop_matrix = coop_matrix / np.maximum(count_matrix, 1)

    im1 = ax1.imshow(coop_matrix, cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
    plt.colorbar(im1, ax=ax1, label="CC Rate")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels([f"{v:.1f}" for v in lambda_vals])
    ax1.set_yticklabels([f"{v:.1f}" for v in lambda_vals])
    ax1.set_xlabel("lambda_j")
    ax1.set_ylabel("lambda_i")
    ax1.set_title("A. Cooperation Rate by Empathy", fontsize=12, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = coop_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)

    # Panel B: Cooperation curve
    ax2 = fig.add_subplot(gs[0, 1])

    symmetric_coop = []
    for lam in lambda_vals:
        subset = [r for r in results if r["lambda_i"] == lam and r["lambda_j"] == lam]
        symmetric_coop.append(np.mean([r["freq_CC"] for r in subset]))

    ax2.plot(lambda_vals, symmetric_coop, 'o-', linewidth=3, markersize=12, color='#2ecc71')
    ax2.fill_between(lambda_vals, symmetric_coop, alpha=0.3, color='#2ecc71')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Symmetric Empathy (lambda)")
    ax2.set_ylabel("Mutual Cooperation Rate")
    ax2.set_title("B. Cooperation Threshold Effect", fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Annotate threshold
    ax2.annotate('Threshold\nlambda=0.5', xy=(0.5, 0.85), xytext=(0.6, 0.6),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    # Panel C: Payoff gap
    ax3 = fig.add_subplot(gs[1, 0])

    emp_diffs = defaultdict(list)
    for r in results:
        diff = r["lambda_i"] - r["lambda_j"]
        emp_diffs[round(diff, 2)].append(r["payoff_gap"])

    diffs = sorted(emp_diffs.keys())
    means = [np.mean(emp_diffs[d]) for d in diffs]

    ax3.bar(diffs, means, width=0.2, color=['#e74c3c' if m < 0 else '#2ecc71' for m in means])
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_xlabel("Empathy Difference (lambda_i - lambda_j)")
    ax3.set_ylabel("Payoff Gap (i - j)")
    ax3.set_title("C. Exploitation: High Empathy Disadvantaged", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel D: Key statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Calculate key statistics
    high_emp = [r for r in results if r["lambda_i"] >= 0.5 and r["lambda_j"] >= 0.5]
    low_emp = [r for r in results if r["lambda_i"] <= 0.25 and r["lambda_j"] <= 0.25]
    asymm = [r for r in results if abs(r["lambda_i"] - r["lambda_j"]) >= 0.5]

    stats_text = """
    D. Key Findings

    Total experiments: {:,}

    High empathy (both >= 0.5):
      - Cooperation rate: {:.1%}
      - N = {:,}

    Low empathy (both <= 0.25):
      - Cooperation rate: {:.1%}
      - N = {:,}

    Asymmetric (|diff| >= 0.5):
      - i exploited: {:.1%}
      - j exploited: {:.1%}
      - N = {:,}

    Main Result:
    Empathy >= 0.5 reliably produces
    mutual cooperation without
    baking in pro-sociality.
    """.format(
        len(results),
        np.mean([r["freq_CC"] for r in high_emp]),
        len(high_emp),
        np.mean([r["freq_CC"] for r in low_emp]),
        len(low_emp),
        sum(1 for r in asymm if r["outcome_label"] == "i_exploited") / len(asymm),
        sum(1 for r in asymm if r["outcome_label"] == "j_exploited") / len(asymm),
        len(asymm),
    )

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title("D. Summary Statistics", fontsize=12, fontweight='bold')

    plt.suptitle("Empathy + ToM in Prisoner's Dilemma: Phase 5 Results",
                fontsize=16, fontweight='bold', y=0.98)

    fig.savefig(output_dir / "summary_figure.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "summary_figure.pdf", bbox_inches='tight')
    plt.close(fig)

    print("Saved: summary_figure.png")


def print_statistical_summary(results: list):
    """Print statistical summary of results."""
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)

    print(f"\nTotal experiments: {len(results):,}")

    # Outcome distribution
    from collections import Counter
    outcomes = Counter(r["outcome_label"] for r in results)
    print("\nOutcome Distribution:")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome}: {count:,} ({100*count/len(results):.1f}%)")

    # Effect of empathy
    print("\nEffect of Symmetric Empathy:")
    lambda_vals = sorted(set(r["lambda_i"] for r in results))
    for lam in lambda_vals:
        subset = [r for r in results if r["lambda_i"] == lam and r["lambda_j"] == lam]
        cc_rate = np.mean([r["freq_CC"] for r in subset])
        print(f"  lambda={lam:.2f}: CC={cc_rate:.2f} (n={len(subset)})")

    # Effect of beta
    print("\nEffect of Beta (at lambda=0.5):")
    beta_vals = sorted(set(r["beta_i"] for r in results))
    for beta in beta_vals:
        subset = [r for r in results
                 if r["lambda_i"] == 0.5 and r["lambda_j"] == 0.5
                 and r["beta_i"] == beta and r["beta_j"] == beta]
        if subset:
            cc_rate = np.mean([r["freq_CC"] for r in subset])
            print(f"  beta={beta:.0f}: CC={cc_rate:.2f} (n={len(subset)})")

    # Effect of inversion
    print("\nEffect of Inversion:")
    with_inv = [r for r in results if r["use_inversion"]]
    without_inv = [r for r in results if not r["use_inversion"]]
    print(f"  With inversion: CC={np.mean([r['freq_CC'] for r in with_inv]):.2f} (n={len(with_inv)})")
    print(f"  Without inversion: CC={np.mean([r['freq_CC'] for r in without_inv]):.2f} (n={len(without_inv)})")

    # Exploitation analysis
    print("\nExploitation Analysis (|lambda_diff| >= 0.5):")
    asymm = [r for r in results if abs(r["lambda_i"] - r["lambda_j"]) >= 0.5]
    i_exp = sum(1 for r in asymm if r["outcome_label"] == "i_exploited")
    j_exp = sum(1 for r in asymm if r["outcome_label"] == "j_exploited")
    print(f"  Total asymmetric cases: {len(asymm)}")
    print(f"  i exploited (higher empathy): {i_exp} ({100*i_exp/len(asymm):.1f}%)")
    print(f"  j exploited (higher empathy): {j_exp} ({100*j_exp/len(asymm):.1f}%)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze parameter sweep results")
    parser.add_argument("--input", type=str, default="results/phase5_full_sweep.json",
                       help="Input JSON file with sweep results")
    parser.add_argument("--output", type=str, default="figures/",
                       help="Output directory for figures")

    args = parser.parse_args()

    # Setup
    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {input_path}")
    results = load_results(input_path)
    print(f"Loaded {len(results):,} experiment results")

    # Generate all figures
    print("\nGenerating figures...")

    create_cooperation_heatmap(results, output_dir)
    create_cooperation_heatmap(results, output_dir, beta_filter=4.0)
    create_outcome_distribution(results, output_dir)
    create_beta_effect_plot(results, output_dir)
    create_exploitation_plot(results, output_dir)
    create_inversion_effect_plot(results, output_dir)
    create_summary_figure(results, output_dir)

    # Print statistical summary
    print_statistical_summary(results)

    print(f"\nAll figures saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
