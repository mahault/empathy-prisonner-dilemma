"""Opponent Simulator for multi-step rollouts.

Bridges the particle filter (OpponentInversion) and GatedToM with future-step
prediction needed during sophisticated planning rollouts.

At each rollout step, predicts what the opponent will do:
- Step 0: Uses GatedToM (blends learned + static ToM based on reliability)
- Step > 0: Falls back to static ToM (no new observations during rollout)

Predictions are history-conditioned q(a_j | h_t), not conditioned on the
agent's current-round action (simultaneous-move game).
"""

import dataclasses

import numpy as np
from typing import Optional, Any


class OpponentSimulator:
    """Predict opponent actions during multi-step planning rollouts.

    Wraps GatedToM (for step 0) and static TheoryOfMind (for future steps)
    into a single interface that the SophisticatedPlanner can call at each
    rollout step.
    """

    def __init__(
        self,
        tom: Any,
        gated_tom: Optional[Any] = None,
        context: Optional[Any] = None,
        my_coop_rate: Optional[float] = None,
        n_rounds: int = 0,
    ):
        """
        Args:
            tom: TheoryOfMind module (static prior, always available)
            gated_tom: GatedToM module (learned + static blend, optional)
            context: ObservationContext for the current timestep (for gated predictions)
        """
        self.tom = tom
        self.gated_tom = gated_tom
        self.context = context
        self.my_coop_rate = my_coop_rate
        self.n_rounds = n_rounds

    def simulated_policy_belief(self, prefix_coops: int, step: int) -> Optional[float]:
        """The opponent's belief about my cooperation rate after `step`
        simulated rounds of which `prefix_coops` were cooperations."""
        if self.my_coop_rate is None or step <= 0:
            return None
        n = self.n_rounds
        return (n * self.my_coop_rate + prefix_coops) / (n + step)

    def apply_policy_belief(self, value: Optional[float]) -> Optional[float]:
        """Temporarily set the ToM's belief about my policy; returns the old
        value so the caller can restore it."""
        if value is None:
            return None
        prev = float(self.tom._believed_my_policy[0])
        self.tom.update_my_policy_belief(float(value))
        return prev

    def predict_response(self, step: int = 0,
                         simulated_last_action: Optional[int] = None) -> np.ndarray:
        """Predict q(a_j | h_t) at a given rollout step.

        For simultaneous-move games the prediction is unconditional on the
        agent's current-round action, but at future rollout steps it IS
        conditioned on the simulated history induced by the partial rollout:
        `simulated_last_action` is the action the candidate policy took at the
        previous step, which the opponent model reads through its reciprocity
        term. Without this the prediction is identical for every candidate
        policy and multi-step planning degenerates to a myopic rule.

        Args:
            step: Rollout step index (0 = current, >0 = future)
            simulated_last_action: my action at step-1 within this rollout

        Returns:
            np.ndarray: [P(C), P(D)] -- opponent's action distribution
        """
        if self.gated_tom is not None and self.context is not None:
            ctx = self.context
            if step > 0 and simulated_last_action is not None:
                ctx = dataclasses.replace(
                    ctx, my_last_action=int(simulated_last_action))
            return self.gated_tom.predict_opponent_action(ctx)
        prediction = self.tom.predict_opponent_action()
        return prediction.q_response
