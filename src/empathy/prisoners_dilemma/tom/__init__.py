"""Theory of Mind module for Prisoner's Dilemma.

This module provides:
- ToM best-response prediction (opponent responds to my action)
- Social EFE computation (weighted self + other EFE)
- Particle-based opponent inference (Harshil's inversion)
"""

from empathy.prisoners_dilemma.tom.tom_core import TheoryOfMind, SocialEFE
from empathy.prisoners_dilemma.tom.inversion import OpponentInversion, OpponentHypothesis

__all__ = [
    "TheoryOfMind",
    "SocialEFE",
    "OpponentInversion",
    "OpponentHypothesis",
]
