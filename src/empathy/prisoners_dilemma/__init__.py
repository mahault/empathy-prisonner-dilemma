"""Prisoner's Dilemma with empathy via EFE weighting.

This module implements empathetic active inference agents for the
Iterated Prisoner's Dilemma task.

Two agent implementations are available:
- EmpatheticAgent: Original K-copy ensemble averaging
- ToMEmpatheticAgent: Proper ToM with best-response prediction
"""
from empathy.prisoners_dilemma.agent import EmpatheticAgent, ToMEmpatheticAgent
from empathy.prisoners_dilemma.env import Environment
from empathy.prisoners_dilemma.sim import Sim
from empathy.prisoners_dilemma.sweep import Sweep
from empathy.prisoners_dilemma.tom import TheoryOfMind, SocialEFE, OpponentInversion
from empathy.prisoners_dilemma.metrics import (
    ExploitabilityAnalysis,
    OutcomeClassifier,
    PayoffAnalysis,
    compute_exploitability,
    classify_outcome,
)

__all__ = [
    # Agents
    "EmpatheticAgent",
    "ToMEmpatheticAgent",
    # Environment and simulation
    "Environment",
    "Sim",
    "Sweep",
    # Theory of Mind
    "TheoryOfMind",
    "SocialEFE",
    "OpponentInversion",
    # Metrics
    "ExploitabilityAnalysis",
    "OutcomeClassifier",
    "PayoffAnalysis",
    "compute_exploitability",
    "classify_outcome",
]
