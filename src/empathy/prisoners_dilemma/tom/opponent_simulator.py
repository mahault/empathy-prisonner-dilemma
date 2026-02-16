"""Opponent Simulator for multi-step rollouts.

Bridges the particle filter (OpponentInversion) and GatedToM with future-step
prediction needed during sophisticated planning rollouts.

At each rollout step, predicts what the opponent will do:
- Step 0: Uses GatedToM (blends learned + static ToM based on reliability)
- Step > 0: Falls back to static ToM (no new observations during rollout)

Predictions are history-conditioned q(a_j | h_t), not conditioned on the
agent's current-round action (simultaneous-move game).
"""

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

    def predict_response(self, step: int = 0) -> np.ndarray:
        """Predict q(a_j | h_t) at a given rollout step.

        For simultaneous-move games, prediction is unconditional on the
        agent's current-round action.

        Args:
            step: Rollout step index (0 = current, >0 = future)

        Returns:
            np.ndarray: [P(C), P(D)] -- opponent's action distribution
        """
        if step == 0 and self.gated_tom is not None and self.context is not None:
            return self.gated_tom.predict_opponent_action(self.context)
        else:
            prediction = self.tom.predict_opponent_action()
            return prediction.q_response
