"""Opponent Simulator for multi-step rollouts.

Bridges the particle filter (OpponentInversion) and GatedToM with future-step
prediction needed during sophisticated planning rollouts.

At each rollout step, predicts what the opponent will do given the agent's action:
- Step 0: Uses GatedToM (blends learned + static ToM based on reliability)
- Step > 0: Falls back to static ToM (no new observations during rollout)
"""

import numpy as np
from typing import Optional, Any


class OpponentSimulator:
    """Predict opponent responses during multi-step planning rollouts.

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

    def predict_response(self, my_action: int, step: int = 0) -> np.ndarray:
        """Predict q(a_j | a_i) at a given rollout step.

        Args:
            my_action: My candidate action (0=C, 1=D)
            step: Rollout step index (0 = current, >0 = future)

        Returns:
            np.ndarray: [P(C), P(D)] — opponent's response distribution
        """
        if step == 0 and self.gated_tom is not None and self.context is not None:
            return self.gated_tom.predict_opponent_response(
                my_action, self.context
            )
        else:
            prediction = self.tom.predict_opponent_response(my_action)
            return prediction.q_response
