#!/usr/bin/env python3
"""
Detailed parameter combination analysis - shows how different priors shape outcomes.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from empathy.clean_up.experiment.config import get_experiment_config, create_experiment_grid

def analyze_parameter_combinations():
    """Analyze all parameter combinations and their outcomes."""
    
    # Get the config and grid
    config = get_experiment_config('mega_sweep')
    grid = create_experiment_grid(config)
    
    # Load results from new structure
    results_dir = Path('results/mega_sweep')
    data_dir = results_dir / 'data'
    
    # Check for new structure (data/) or fallback to old structure
    if data_dir.exists():
        search_dir = data_dir
    else:
        search_dir = results_dir
    
    # Load all timeseries.json files
    run_dirs = [d for d in search_dir.iterdir() if d.is_dir() and (d / 'timeseries.json').exists()]
    if not run_dirs:
        return None
    
    errors = []
    regimes_list = []
    regime_lookup = {}
    
    # Load from timeseries.json files
    for run_dir in sorted(run_dirs):
        try:
            with open(run_dir / 'timeseries.json') as f:
                data = json.load(f)
            metadata = data.get('metadata', {})
            metrics = metadata.get('metrics', {})
            outcomes = metadata.get('outcomes', {})
            
            # Extract error
            error = metrics.get('final_error')
            if error is not None and not np.isnan(error):
                errors.append(error)
            
            # Extract regime (if available in outcomes or from analysis)
            regime = outcomes.get('regime_label', 'Unknown')
            regimes_list.append(regime)
            
            # Create lookup by run index
            run_idx = metadata.get('run_idx', len(regime_lookup))
            regime_lookup[run_idx - 1] = regime  # Convert to 0-indexed
        except Exception as e:
            print(f"Warning: Failed to load {run_dir}: {e}")
            continue
    
    # Group by parameter combination
    combinations = defaultdict(lambda: {
        'errors': [],
        'regimes': defaultdict(int),
        'count': 0
    })
    
    print(f"Processing {len(grid)} grid entries with {len(errors)} errors...")
    
    missing_tom_mode_count = 0
    for i, ep_config in enumerate(grid):
        if i < len(errors):
            # Infer use_tom from tom_mode
            # tom_mode should always be present (validated during grid creation),
            # but handle missing values defensively for backward compatibility
            if 'tom_mode' not in ep_config:
                missing_tom_mode_count += 1
                if missing_tom_mode_count == 1:
                    print(f"Warning: Episode config at index {i} is missing 'tom_mode'. "
                          f"This should not happen if grid was created via create_experiment_grid(). "
                          f"Defaulting to 'softmax' for analysis. This may cause incorrect classification.")
                # Skip this entry to avoid incorrect analysis
                continue
            
            tom_mode = ep_config.get('tom_mode')
            use_tom = (tom_mode != 'off')
            eta_0 = ep_config.get('eta_0')
            lambda_mix = ep_config.get('lambda_mix')
            u_0 = ep_config.get('u_0')
            kappa = ep_config.get('kappa')
            
            if all(v is not None for v in [use_tom, eta_0, lambda_mix, u_0, kappa]):
                combo_key = (
                    f"ToM={'ON' if use_tom else 'OFF'}",
                    f"η₀={eta_0}",
                    f"λ={lambda_mix}",
                    f"u₀={u_0}",
                    f"κ={kappa}"
                )
                combo_str = " | ".join(combo_key)
                
                combinations[combo_str]['errors'].append(errors[i])
                combinations[combo_str]['count'] += 1
                
                # Get regime if available
                if i in regime_lookup:
                    regime = regime_lookup[i]
                    combinations[combo_str]['regimes'][regime] += 1
                elif i < len(regimes_list) and isinstance(regimes_list[i], str):
                    regime = regimes_list[i]
                    combinations[combo_str]['regimes'][regime] += 1
    
    if missing_tom_mode_count > 0:
        print(f"Warning: Skipped {missing_tom_mode_count} entries due to missing 'tom_mode' field.")
    
    print(f"Found {len(combinations)} unique parameter combinations")
    return combinations

def generate_tables(combinations):
    """Generate markdown tables for parameter combinations."""
    
    # Sort combinations by mean error
    combo_list = []
    for combo_str, data in combinations.items():
        if data['errors']:
            stats = {
                'combo': combo_str,
                'mean': np.mean(data['errors']),
                'std': np.std(data['errors']),
                'min': np.min(data['errors']),
                'max': np.max(data['errors']),
                'median': np.median(data['errors']),
                'n': len(data['errors']),
                'regimes': dict(data['regimes']),
            }
            combo_list.append(stats)
    
    combo_list.sort(key=lambda x: x['mean'])
    
    # Generate markdown
    lines = []
    lines.append("# Parameter Combination Analysis: How Different Priors Shape Outcomes")
    lines.append("")
    lines.append("This analysis shows how different parameter combinations (priors) affect system performance.")
    lines.append("Each combination represents a different set of hyperparameters that shape how the agent learns from social information.")
    lines.append("")
    lines.append("## Complete Parameter Combination Table")
    lines.append("")
    lines.append("**Sorted by mean error (best to worst)**")
    lines.append("")
    lines.append("| Combination | Mean Error | Std Error | Min | Max | Median | N | Top Regime |")
    lines.append("|-------------|------------|-----------|-----|-----|--------|---|------------|")
    
    for combo in combo_list:
        # Get top regime
        if combo['regimes']:
            top_regime = max(combo['regimes'].items(), key=lambda x: x[1])[0]
            top_regime_count = combo['regimes'][top_regime]
            top_regime_pct = 100 * top_regime_count / combo['n']
            regime_str = f"{top_regime} ({top_regime_pct:.0f}%)"
        else:
            regime_str = "N/A"
        
        lines.append(
            f"| {combo['combo']} | "
            f"{combo['mean']:.4f} | "
            f"{combo['std']:.4f} | "
            f"{combo['min']:.4f} | "
            f"{combo['max']:.4f} | "
            f"{combo['median']:.4f} | "
            f"{combo['n']} | "
            f"{regime_str} |"
        )
    
    lines.append("")
    lines.append("## Detailed Breakdown by Individual Parameter")
    lines.append("")
    lines.append("These tables show how each individual parameter affects outcomes when averaged across all other parameter settings.")
    lines.append("")
    
    # Aggregate by individual parameters
    eta_0_data = defaultdict(lambda: {'errors': [], 'count': 0})
    lambda_mix_data = defaultdict(lambda: {'errors': [], 'count': 0})
    u_0_data = defaultdict(lambda: {'errors': [], 'count': 0})
    kappa_data = defaultdict(lambda: {'errors': [], 'count': 0})
    tom_data = defaultdict(lambda: {'errors': [], 'count': 0})
    
    for combo in combo_list:
        # Extract individual parameters and aggregate
        for part in combo['combo'].split(' | '):
            if part.startswith('η₀='):
                eta_0_val = part.split('=')[1]
                eta_0_data[eta_0_val]['errors'].extend([combo['mean']] * combo['n'])
                eta_0_data[eta_0_val]['count'] += combo['n']
            elif part.startswith('λ='):
                lm_val = part.split('=')[1]
                lambda_mix_data[lm_val]['errors'].extend([combo['mean']] * combo['n'])
                lambda_mix_data[lm_val]['count'] += combo['n']
            elif part.startswith('u₀='):
                u0_val = part.split('=')[1]
                u_0_data[u0_val]['errors'].extend([combo['mean']] * combo['n'])
                u_0_data[u0_val]['count'] += combo['n']
            elif part.startswith('κ='):
                k_val = part.split('=')[1]
                kappa_data[k_val]['errors'].extend([combo['mean']] * combo['n'])
                kappa_data[k_val]['count'] += combo['n']
            elif part.startswith('ToM='):
                tom_val = part.split('=')[1]
                tom_data[tom_val]['errors'].extend([combo['mean']] * combo['n'])
                tom_data[tom_val]['count'] += combo['n']
    
    # Generate individual parameter tables
    if eta_0_data:
        lines.append("### By eta_0 (Base Social Learning Rate)")
        lines.append("")
        lines.append("Higher values = more aggressive social learning")
        lines.append("")
        lines.append("| eta_0 | Mean Error | Std Error | Min | Max | N |")
        lines.append("|-------|-----------|-----------|-----|-----|---|")
        for eta_0 in sorted(eta_0_data.keys(), key=float):
            data = eta_0_data[eta_0]
            if data['errors']:
                lines.append(
                    f"| {eta_0} | "
                    f"{np.mean(data['errors']):.4f} | "
                    f"{np.std(data['errors']):.4f} | "
                    f"{np.min(data['errors']):.4f} | "
                    f"{np.max(data['errors']):.4f} | "
                    f"{data['count']} |"
                )
        lines.append("")
    
    if lambda_mix_data:
        lines.append("### By lambda_mix (Deprecated - replaced by accuracy gate)")
        lines.append("")
        lines.append("Higher values = more emphasis on past performance vs predicted performance")
        lines.append("")
        lines.append("| lambda_mix | Mean Error | Std Error | Min | Max | N |")
        lines.append("|------------|-----------|-----------|-----|-----|---|")
        for lm in sorted(lambda_mix_data.keys(), key=float):
            data = lambda_mix_data[lm]
            if data['errors']:
                lines.append(
                    f"| {lm} | "
                    f"{np.mean(data['errors']):.4f} | "
                    f"{np.std(data['errors']):.4f} | "
                    f"{np.min(data['errors']):.4f} | "
                    f"{np.max(data['errors']):.4f} | "
                    f"{data['count']} |"
                )
        lines.append("")
    
    if u_0_data:
        lines.append("### By u_0 (Reliability Threshold)")
        lines.append("")
        lines.append("Higher values = more conservative filtering (rejects more social information)")
        lines.append("")
        lines.append("| u_0 | Mean Error | Std Error | Min | Max | N |")
        lines.append("|-----|-----------|-----------|-----|-----|---|")
        for u0 in sorted(u_0_data.keys(), key=float):
            data = u_0_data[u0]
            if data['errors']:
                lines.append(
                    f"| {u0} | "
                    f"{np.mean(data['errors']):.4f} | "
                    f"{np.std(data['errors']):.4f} | "
                    f"{np.min(data['errors']):.4f} | "
                    f"{np.max(data['errors']):.4f} | "
                    f"{data['count']} |"
                )
        lines.append("")
    
    if kappa_data:
        lines.append("### By kappa (Reliability Sharpness)")
        lines.append("")
        lines.append("Lower values = sharper threshold transition")
        lines.append("")
        lines.append("| kappa | Mean Error | Std Error | Min | Max | N |")
        lines.append("|-------|-----------|-----------|-----|-----|---|")
        for k in sorted(kappa_data.keys(), key=float):
            data = kappa_data[k]
            if data['errors']:
                lines.append(
                    f"| {k} | "
                    f"{np.mean(data['errors']):.4f} | "
                    f"{np.std(data['errors']):.4f} | "
                    f"{np.min(data['errors']):.4f} | "
                    f"{np.max(data['errors']):.4f} | "
                    f"{data['count']} |"
                )
        lines.append("")
    
    if tom_data:
        lines.append("### By ToM Gate")
        lines.append("")
        lines.append("ON = Theory of Mind enabled, OFF = social learning disabled")
        lines.append("")
        lines.append("| ToM Gate | Mean Error | Std Error | Min | Max | N |")
        lines.append("|----------|-----------|-----------|-----|-----|---|")
        for tom in sorted(tom_data.keys()):
            data = tom_data[tom]
            if data['errors']:
                lines.append(
                    f"| {tom} | "
                    f"{np.mean(data['errors']):.4f} | "
                    f"{np.std(data['errors']):.4f} | "
                    f"{np.min(data['errors']):.4f} | "
                    f"{np.max(data['errors']):.4f} | "
                    f"{data['count']} |"
                )
        lines.append("")
    
    # Top and bottom combinations
    lines.append("## Best and Worst Combinations")
    lines.append("")
    lines.append("### Top 30 Best (Lowest Mean Error)")
    lines.append("")
    lines.append("| Rank | Combination | Mean Error | Std Error | N |")
    lines.append("|------|-------------|------------|-----------|---|")
    for i, combo in enumerate(combo_list[:30], 1):
        lines.append(
            f"| {i} | {combo['combo']} | "
            f"{combo['mean']:.4f} | "
            f"{combo['std']:.4f} | "
            f"{combo['n']} |"
        )
    
    lines.append("")
    lines.append("### Bottom 30 Worst (Highest Mean Error)")
    lines.append("")
    lines.append("| Rank | Combination | Mean Error | Std Error | N |")
    lines.append("|------|-------------|------------|-----------|---|")
    for i, combo in enumerate(combo_list[-30:], len(combo_list)-29):
        lines.append(
            f"| {i} | {combo['combo']} | "
            f"{combo['mean']:.4f} | "
            f"{combo['std']:.4f} | "
            f"{combo['n']} |"
        )
    
    # Summary statistics
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")
    all_errors = [e for combo in combo_list for e in [combo['mean']] * combo['n']]
    lines.append(f"- **Total Unique Combinations:** {len(combo_list)}")
    lines.append(f"- **Total Episodes:** {sum(c['n'] for c in combo_list)}")
    lines.append(f"- **Best Mean Error:** {combo_list[0]['mean']:.4f}")
    lines.append(f"- **Worst Mean Error:** {combo_list[-1]['mean']:.4f}")
    lines.append(f"- **Overall Mean Error:** {np.mean(all_errors):.4f}")
    lines.append(f"- **Overall Std Error:** {np.std(all_errors):.4f}")
    lines.append(f"- **Error Range:** {combo_list[-1]['mean'] - combo_list[0]['mean']:.4f}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    print("Analyzing parameter combinations...")
    combinations = analyze_parameter_combinations()
    
    if combinations:
        print(f"Found {len(combinations)} unique combinations")
        report = generate_tables(combinations)
        
        output_file = Path('PARAMETER_COMBINATION_ANALYSIS.md')
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"\nAnalysis saved to: {output_file}")
        print(f"\nTotal combinations: {len(combinations)}")
    else:
        print("No combinations found!")
