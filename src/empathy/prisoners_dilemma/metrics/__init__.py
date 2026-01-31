"""Metrics module for Prisoner's Dilemma analysis.

Provides:
- Exploitability analysis (Hongju-style)
- Outcome classification
- Payoff analysis
"""

from empathy.prisoners_dilemma.metrics.exploitability import (
    ExploitabilityAnalysis,
    OutcomeClassifier,
    PayoffAnalysis,
    compute_best_response,
    compute_exploitability,
    classify_outcome,
)

__all__ = [
    "ExploitabilityAnalysis",
    "OutcomeClassifier",
    "PayoffAnalysis",
    "compute_best_response",
    "compute_exploitability",
    "classify_outcome",
]
