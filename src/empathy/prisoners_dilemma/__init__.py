"""Prisoner's Dilemma with empathy via EFE weighting.

This module implements empathetic active inference agents for the
Iterated Prisoner's Dilemma task.
"""
from empathy.prisoners_dilemma.agent import EmpatheticAgent
from empathy.prisoners_dilemma.env import Environment
from empathy.prisoners_dilemma.sim import Sim
from empathy.prisoners_dilemma.sweep import Sweep

__all__ = ["EmpatheticAgent", "Environment", "Sim", "Sweep"]
