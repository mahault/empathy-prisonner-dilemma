"""
Report generation functions for experiment analysis.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any
from datetime import datetime


def format_analysis_summary(analysis: Dict[str, Any]) -> str:
    """Format a single experiment analysis summary as markdown."""
    lines = []
    
    if "error_summary" in analysis and analysis["error_summary"]:
        es = analysis["error_summary"]
        lines.append("## Error Metrics")
        lines.append("")
        lines.append(f"- **Mean Error:** {es['mean']:.4f}")
        lines.append(f"- **Std Error:** {es['std']:.4f}")
        lines.append(f"- **Min Error:** {es['min']:.4f}")
        lines.append(f"- **Max Error:** {es['max']:.4f}")
        lines.append(f"- **Median Error:** {es['median']:.4f}")
        lines.append("")
    
    if "regimes" in analysis and analysis["regimes"]:
        lines.append("## Regime Distribution")
        lines.append("")
        total = sum(analysis["regimes"].values())
        for regime, count in sorted(analysis["regimes"].items()):
            pct = 100 * count / total if total > 0 else 0
            lines.append(f"- **{regime}:** {count} ({pct:.1f}%)")
        lines.append("")
    
    return "\n".join(lines)


def generate_cross_experiment_comparison(all_analyses: Dict[str, Dict[str, Any]]) -> str:
    """Generate cross-experiment comparison report."""
    report = []
    report.append("# Cross-Experiment Comparison")
    report.append("")
    report.append(f"**Analysis Date:** {datetime.now().isoformat()}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Compare baseline vs social learning
    if "phase_diagram_baseline" in all_analyses and "phase_diagram" in all_analyses:
        baseline = all_analyses["phase_diagram_baseline"]
        social = all_analyses["phase_diagram"]
        
        baseline_error = baseline.get("error_summary")
        social_error = social.get("error_summary")
        
        if baseline_error and social_error:
            report.append("## Baseline vs Social Learning (Phase Diagram)")
            report.append("")
            report.append(f"- **Baseline Mean Error:** {baseline_error['mean']:.4f} ± {baseline_error['std']:.4f}")
            report.append(f"- **Social Learning Mean Error:** {social_error['mean']:.4f} ± {social_error['std']:.4f}")
            
            improvement = baseline_error['mean'] - social_error['mean']
            pct_change = 100 * improvement / baseline_error['mean'] if baseline_error['mean'] != 0 else 0
            
            if improvement > 0:
                report.append(f"- **Improvement:** {improvement:.4f} ({pct_change:.1f}% better)")
            elif improvement < 0:
                report.append(f"- **Degradation:** {abs(improvement):.4f} ({abs(pct_change):.1f}% worse)")
            else:
                report.append("- **No Change**")
            
            report.append("")
    
    # Overall comparison table
    report.append("## Experiment Comparison")
    report.append("")
    report.append("| Experiment | Mean Error | Std Error | Episodes |")
    report.append("|------------|------------|-----------|----------|")
    
    for exp_name in sorted(all_analyses.keys()):
        analysis = all_analyses[exp_name]
        error_summary = analysis.get("error_summary")
        n_episodes = analysis.get("n_episodes", 0)
        
        if error_summary:
            report.append(
                f"| {exp_name} | {error_summary['mean']:.4f} | "
                f"{error_summary['std']:.4f} | {n_episodes} |"
            )
        else:
            report.append(f"| {exp_name} | N/A | N/A | {n_episodes} |")
    
    report.append("")
    
    return "\n".join(report)


def generate_findings_report(all_analyses: Dict[str, Dict[str, Any]]) -> str:
    """Generate a comprehensive markdown findings report."""
    report = []
    
    report.append("# Experiment Results: Comprehensive Findings")
    report.append("")
    report.append(f"**Analysis Date:** {datetime.now().isoformat()}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Overview
    report.append("## Overview")
    report.append("")
    total_episodes = sum(a.get("n_episodes", 0) for a in all_analyses.values())
    report.append(f"- **Total Experiments:** {len(all_analyses)}")
    report.append(f"- **Total Episodes:** {total_episodes}")
    report.append("")
    
    for exp_name in sorted(all_analyses.keys()):
        analysis = all_analyses[exp_name]
        report.append(f"- **{exp_name}:** {analysis.get('n_episodes', 0)} episodes")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Detailed analysis per experiment
    for exp_name in sorted(all_analyses.keys()):
        analysis = all_analyses[exp_name]
        report.append(f"## {exp_name.replace('_', ' ').title()}")
        report.append("")
        
        # Error summary
        if analysis.get("error_summary"):
            es = analysis["error_summary"]
            report.append("### Error Metrics")
            report.append("")
            report.append(f"- **Mean Error:** {es['mean']:.4f}")
            report.append(f"- **Std Error:** {es['std']:.4f}")
            report.append(f"- **Min Error:** {es['min']:.4f}")
            report.append(f"- **Max Error:** {es['max']:.4f}")
            report.append(f"- **Median Error:** {es['median']:.4f}")
            report.append("")
        
        # Regime distribution
        if analysis.get("regimes"):
            report.append("### Regime Distribution")
            report.append("")
            total = sum(analysis["regimes"].values())
            for regime, count in sorted(analysis["regimes"].items()):
                pct = 100 * count / total if total > 0 else 0
                report.append(f"- **{regime}:** {count} ({pct:.1f}%)")
            report.append("")
        
        # Task breakdown
        if analysis.get("task_breakdown"):
            report.append("### Task Breakdown")
            report.append("")
            for task, stats in sorted(analysis["task_breakdown"].items()):
                if isinstance(stats, dict):
                    report.append(f"- **{task}:** mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")
            report.append("")
        
        # use_tom breakdown
        if analysis.get("use_tom_breakdown"):
            report.append("### ToM Gate Breakdown")
            report.append("")
            for use_tom, stats in sorted(analysis["use_tom_breakdown"].items()):
                if isinstance(stats, dict):
                    gate_status = "ON" if use_tom else "OFF"
                    report.append(f"- **ToM {gate_status}:** mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")
            report.append("")
        
        # eta_0 breakdown
        if analysis.get("eta_0_breakdown"):
            report.append("### Trust Level (eta_0) Breakdown")
            report.append("")
            for eta_0, stats in sorted(analysis["eta_0_breakdown"].items()):
                if isinstance(stats, dict):
                    report.append(f"- **eta_0={eta_0}:** mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")
            report.append("")
        
        # Social learning stats
        if analysis.get("social_learning_stats"):
            report.append("### Social Learning Metrics")
            report.append("")
            sls = analysis["social_learning_stats"]
            
            if sls.get("reliability_mean") and sls["reliability_mean"].get("mean") is not None:
                rel_mean = sls["reliability_mean"]["mean"]
                rel_std = sls["reliability_mean"].get("std", 0.0)
                report.append(f"- **Reliability:** mean={rel_mean:.4f}, std={rel_std:.4f}")
            
            if sls.get("trust_mean") and sls["trust_mean"].get("mean") is not None:
                trust_mean = sls["trust_mean"]["mean"]
                trust_std = sls["trust_mean"].get("std", 0.0)
                report.append(f"- **Trust:** mean={trust_mean:.4f}, std={trust_std:.4f}")
            
            if sls.get("effective_influence_mean") and sls["effective_influence_mean"].get("mean") is not None:
                eff_inf_mean = sls["effective_influence_mean"]["mean"]
                report.append(f"- **Effective Influence:** mean={eff_inf_mean:.4f}")
            
            report.append("")
        
        report.append("---")
        report.append("")
    
    # Cross-experiment comparisons
    report.append("## Cross-Experiment Comparisons")
    report.append("")
    
    # Compare baseline vs social learning
    if "phase_diagram_baseline" in all_analyses and "phase_diagram" in all_analyses:
        baseline = all_analyses["phase_diagram_baseline"].get("error_summary")
        social = all_analyses["phase_diagram"].get("error_summary")
        
        if baseline and social:
            report.append("### Baseline vs Social Learning (Phase Diagram)")
            report.append("")
            report.append(f"- **Baseline Mean Error:** {baseline['mean']:.4f} ± {baseline['std']:.4f}")
            report.append(f"- **Social Learning Mean Error:** {social['mean']:.4f} ± {social['std']:.4f}")
            
            improvement = baseline['mean'] - social['mean']
            pct_change = 100 * improvement / baseline['mean'] if baseline['mean'] != 0 else 0
            
            if improvement > 0:
                report.append(f"- **Improvement:** {improvement:.4f} ({pct_change:.1f}% better)")
            elif improvement < 0:
                report.append(f"- **Degradation:** {abs(improvement):.4f} ({abs(pct_change):.1f}% worse)")
            else:
                report.append("- **No Change**")
            
            report.append("")
    
    report.append("---")
    report.append("")
    
    # Key findings
    report.append("## Key Findings")
    report.append("")
    
    # Smoke test
    if "smoke" in all_analyses:
        smoke = all_analyses["smoke"]
        smoke_error = smoke.get("error_summary")
        if smoke_error and smoke_error["mean"] < 0.01:
            report.append("### ✓ Smoke Test: PASSED")
            report.append("")
            report.append("- All tasks achieve near-perfect parameter recovery")
            report.append("- System functioning correctly across all environments")
            report.append("")
        else:
            report.append("### ✗ Smoke Test: ISSUES DETECTED")
            report.append("")
            if smoke_error:
                report.append(f"- Mean error: {smoke_error['mean']:.4f} (expected < 0.01)")
            report.append("")
    
    # Killer test
    if "killer_test" in all_analyses:
        killer = all_analyses["killer_test"]
        report.append("### Killer Test: Mismatch Protection")
        report.append("")
        
        if killer.get("regimes"):
            protected = killer["regimes"].get("Independent", 0) + killer["regimes"].get("Beneficial", 0)
            overcopying = killer["regimes"].get("Overcopying", 0)
            total = sum(killer["regimes"].values())
            
            if protected > overcopying:
                report.append(f"- **Protection Working:** {protected}/{total} episodes protected or beneficial")
                report.append(f"- **Overcopying:** {overcopying}/{total} episodes")
            else:
                report.append(f"- **Protection Issues:** {overcopying}/{total} episodes overcopying")
                report.append(f"- **Protected:** {protected}/{total} episodes")
        
        report.append("")
    
    # Phase diagram
    if "phase_diagram" in all_analyses:
        phase = all_analyses["phase_diagram"]
        report.append("### Phase Diagram: Trust × Mismatch Regimes")
        report.append("")
        
        if phase.get("regimes"):
            for regime, count in sorted(phase["regimes"].items(), key=lambda x: -x[1]):
                pct = 100 * count / phase.get("n_episodes", 1)
                report.append(f"- **{regime}:** {count} episodes ({pct:.1f}%)")
        
        report.append("")
    
    report.append("---")
    report.append("")
    
    # Issues and concerns
    report.append("## Issues and Concerns")
    report.append("")
    
    issues_found = False
    
    for exp_name, analysis in all_analyses.items():
        # Check for high error rates
        error_summary = analysis.get("error_summary")
        if error_summary and error_summary["mean"] > 1.5:
            report.append(f"- **{exp_name}:** High mean error ({error_summary['mean']:.4f})")
            issues_found = True
        
        # Check for particle collapse
        stability_metrics = analysis.get("stability_metrics", [])
        if stability_metrics:
            n_eff_mins = [m.get("n_eff_min", 30) for m in stability_metrics if isinstance(m, dict)]
            n_eff_mins = [x for x in n_eff_mins if not np.isnan(x)]
            if n_eff_mins and np.mean(n_eff_mins) < 20:
                report.append(f"- **{exp_name}:** Particle collapse detected (mean n_eff_min = {np.mean(n_eff_mins):.1f})")
                issues_found = True
    
    if not issues_found:
        report.append("No major issues detected.")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # What's working well
    report.append("## What's Working Well")
    report.append("")
    
    working_well = []
    
    # Check smoke test
    if "smoke" in all_analyses:
        smoke = all_analyses["smoke"]
        smoke_error = smoke.get("error_summary")
        if smoke_error and smoke_error["mean"] < 0.01:
            working_well.append("- **Parameter Recovery:** Agents achieve perfect recovery in ideal conditions")
    
    # Check social learning engagement
    for exp_name, analysis in all_analyses.items():
        sls = analysis.get("social_learning_stats", {})
        if sls and isinstance(sls, dict):
            eff_inf = sls.get("effective_influence_mean")
            if eff_inf and isinstance(eff_inf, dict) and eff_inf.get("mean", 0) > 0.01:
                mean_eff_inf = eff_inf["mean"]
                working_well.append(f"- **Social Learning ({exp_name}):** Active engagement (mean influence = {mean_eff_inf:.4f})")
                break
    
    # Check regime diversity
    if "killer_test" in all_analyses:
        killer = all_analyses["killer_test"]
        if killer.get("regimes") and len(killer["regimes"]) >= 3:
            working_well.append("- **Regime Diversity:** System exhibits multiple behavioral regimes as expected")
    
    if working_well:
        report.extend(working_well)
    else:
        report.append("Analysis in progress...")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("1. **Continue monitoring:** Track error metrics across experiments")
    report.append("2. **Investigate high-error cases:** Examine episodes with error > 1.5")
    report.append("3. **Analyze regime transitions:** Study when agents switch between regimes")
    report.append("4. **Parameter sensitivity:** Test robustness to hyperparameter variations")
    report.append("")
    
    return "\n".join(report)
