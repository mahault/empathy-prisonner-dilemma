"""Theory of Mind core module.

Implements:
- History-conditioned opponent prediction: q(a_j | h_t)
- Social EFE computation: G_social = (1-λ) * G_self + λ * E[G_other]

For simultaneous-move games, the opponent's action distribution is conditioned
on history (not on the agent's current-round action), since neither player
observes the other's within-round choice.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Prisoner's Dilemma constants
COOPERATE = 0
DEFECT = 1
ACTION_NAMES = {0: "C", 1: "D"}

# Standard PD payoff matrix (row = my action, col = opponent action)
# Format: (my_payoff, other_payoff)
PD_PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3),  # Mutual cooperation (R, R)
    (COOPERATE, DEFECT): (0, 5),     # Sucker's payoff (S, T)
    (DEFECT, COOPERATE): (5, 0),     # Temptation (T, S)
    (DEFECT, DEFECT): (1, 1),        # Mutual defection (P, P)
}


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature parameter."""
    x_scaled = x / temperature
    x_scaled = x_scaled - np.max(x_scaled)  # Numerical stability
    exp_x = np.exp(x_scaled)
    return exp_x / np.sum(exp_x)


@dataclass
class ToMPrediction:
    """Result of a ToM prediction."""
    q_response: np.ndarray  # Distribution over opponent actions [P(C), P(D)]
    G_other: np.ndarray     # Opponent's EFE for each action
    expected_action: int    # Most likely opponent action
    confidence: float       # Confidence in prediction (1 - entropy)


class TheoryOfMind:
    """
    Theory of Mind module for predicting opponent actions.

    For simultaneous-move games, prediction is history-conditioned:

        q(a_j | h_t) ∝ exp(-β_j * G_j(a_j))

    where the opponent's EFE is computed under their belief about my policy:

        G_j(a_j) = -Σ_i π_i(a_i) * payoff_j(a_i, a_j)

    π_i is the opponent's belief about my mixed strategy, derived from
    my empirical cooperation rate over past rounds.
    """

    def __init__(
        self,
        other_model: Any,  # PyMDP Agent representing opponent
        beta_other: float = 4.0,  # Opponent's action precision
        use_pragmatic_value: bool = True,
        use_epistemic_value: bool = True,
    ):
        self.other_model = other_model
        self.beta_other = beta_other
        self.use_pragmatic_value = use_pragmatic_value
        self.use_epistemic_value = use_epistemic_value

        # Cache for opponent's believed state
        self._qs_other: Optional[np.ndarray] = None

        # Opponent's belief about my policy (updated each round from history)
        # Default: uniform (no prior information)
        self._believed_my_policy = np.array([0.5, 0.5])

    def update_opponent_beliefs(self, qs_other: np.ndarray) -> None:
        """Update our estimate of opponent's beliefs."""
        self._qs_other = qs_other

    def update_my_policy_belief(self, cooperation_rate: float) -> None:
        """Update the opponent's belief about my policy from empirical history.

        Args:
            cooperation_rate: My empirical cooperation rate from past rounds.
        """
        self._believed_my_policy = np.array([cooperation_rate, 1.0 - cooperation_rate])

    def predict_opponent_action(
        self,
        my_beliefs: Optional[np.ndarray] = None,
    ) -> ToMPrediction:
        """
        Predict distribution over opponent actions given history.

        This is a simultaneous-move prediction: q(a_j | h_t).
        The opponent's EFE is computed under their belief about my policy
        (derived from past cooperation rate), NOT conditioned on my
        current-round action.

        Args:
            my_beliefs: My current state beliefs (optional)

        Returns:
            ToMPrediction with response distribution and metadata
        """
        G_other = np.zeros(2)

        for a_j in [COOPERATE, DEFECT]:
            G_other[a_j] = self._compute_opponent_efe(opponent_action=a_j)

        # Opponent's action distribution (softmax over negative EFE)
        q_response = softmax(-G_other, temperature=1.0/self.beta_other)

        # Confidence from entropy
        entropy = -np.sum(q_response * np.log(q_response + 1e-10))
        max_entropy = np.log(2)
        confidence = 1 - entropy / max_entropy

        return ToMPrediction(
            q_response=q_response,
            G_other=G_other,
            expected_action=int(np.argmax(q_response)),
            confidence=confidence
        )

    def _compute_opponent_efe(
        self,
        opponent_action: int,
    ) -> float:
        """
        Compute opponent's expected free energy for their action.

        G_j(a_j) = -Σ_i π_i(a_i) * payoff_j(a_i, a_j)

        The opponent computes their expected payoff under their belief
        about my policy π_i (from observed history).
        """
        G = 0.0

        if self.use_pragmatic_value:
            expected_payoff = 0.0
            for a_i in [COOPERATE, DEFECT]:
                _, other_payoff = PD_PAYOFFS[(a_i, opponent_action)]
                expected_payoff += self._believed_my_policy[a_i] * other_payoff
            G -= expected_payoff

        if self.use_epistemic_value:
            # Epistemic value placeholder — implemented in Phase 7C
            pass

        return G

    def predict_response_distribution(self) -> np.ndarray:
        """Simplified interface: just return q(a_j | h_t)."""
        prediction = self.predict_opponent_action()
        return prediction.q_response


class SocialEFE:
    """
    Compute social expected free energy.

    G_social(a_i) = G_pragmatic(a_i) + G_epistemic(a_i)

    G_pragmatic(a_i) = (1-λ) * G_self(a_i | q(a_j|h_t)) + λ * E_{q(a_j|h_t)}[G_other(a_j)]

    G_epistemic(a_i) = -E[IG(a_i)] where IG is the expected information gain
    about opponent empathy λ_j from observing their next response.

    The opponent prediction q(a_j|h_t) is the same for all candidate actions a_i
    (simultaneous-move game). My G_self still depends on a_i through the payoff
    function payoff_i(a_i, a_j). The epistemic term depends on a_i through the
    next-round context (my_last_action affects what we learn from their response).
    """

    def __init__(
        self,
        tom: TheoryOfMind,
        empathy_factor: float = 0.5,
        beta_self: float = 4.0,
        inversion: 'Optional[OpponentInversion]' = None,
    ):
        self.tom = tom
        self.empathy_factor = empathy_factor
        self.beta_self = beta_self
        self.inversion = inversion  # For epistemic value computation

    def compute(
        self,
        my_action: int,
        my_beliefs: Optional[np.ndarray] = None,
        q_response_override: Optional[np.ndarray] = None,
        context: 'Optional[ObservationContext]' = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute social EFE for a candidate action.

        Args:
            my_action: Candidate action (0=C, 1=D)
            my_beliefs: Current state beliefs (optional)
            q_response_override: If provided, use this as q(a_j|h_t) instead
                of the static ToM prediction. From learned opponent model.
            context: Observation context for epistemic value computation.

        Returns:
            G_social: Social expected free energy
            info: Dictionary with breakdown
        """
        if q_response_override is not None:
            q_response = q_response_override
            confidence = 1.0
        else:
            prediction = self.tom.predict_opponent_action(my_beliefs)
            q_response = prediction.q_response
            confidence = prediction.confidence

        # G_other for empathy weighting: opponent's payoff given MY specific
        # action. This is NOT the same as the opponent's EFE (which uses their
        # belief about my policy). Here we evaluate: "if I commit to a_i and
        # opponent plays according to q(a_j|h_t), what is opponent's payoff?"
        G_other = np.zeros(2)
        for a_j in [COOPERATE, DEFECT]:
            _, other_payoff = PD_PAYOFFS[(my_action, a_j)]
            G_other[a_j] = -other_payoff

        # My EFE given opponent's expected response
        G_self = self._compute_my_efe(my_action, q_response)

        # Opponent's expected EFE under their response
        G_other_expected = np.sum(q_response * G_other)

        # Pragmatic EFE: weighted combination
        lam = self.empathy_factor
        G_pragmatic = (1 - lam) * G_self + lam * G_other_expected

        # Epistemic value: expected information gain about opponent empathy
        G_epistemic = 0.0
        if self.inversion is not None and context is not None:
            G_epistemic = self.inversion.compute_epistemic_value(my_action, context)

        # Full EFE = pragmatic + epistemic
        G_social = G_pragmatic + G_epistemic

        info = {
            "G_self": G_self,
            "G_other_expected": G_other_expected,
            "G_pragmatic": G_pragmatic,
            "G_epistemic": G_epistemic,
            "q_response": q_response,
            "G_other": G_other,
            "empathy_factor": lam,
            "prediction_confidence": confidence,
        }

        return G_social, info

    def _compute_my_efe(
        self,
        my_action: int,
        q_opponent: np.ndarray,
    ) -> float:
        """
        Compute my EFE given opponent's action distribution.

        G_i(a_i | q(a_j|h_t)) = E_{q(a_j|h_t)}[-payoff_i(a_i, a_j)]
        """
        expected_payoff = 0.0
        for a_j in [COOPERATE, DEFECT]:
            my_payoff, _ = PD_PAYOFFS[(my_action, a_j)]
            expected_payoff += q_opponent[a_j] * my_payoff

        return -expected_payoff

    def compute_all_actions(
        self,
        my_beliefs: Optional[np.ndarray] = None,
        q_response_override: Optional[np.ndarray] = None,
        context: 'Optional[ObservationContext]' = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute social EFE for all candidate actions.

        Args:
            my_beliefs: Current state beliefs (optional)
            q_response_override: Single q(a_j|h_t) used for all actions
                (since opponent prediction is action-independent in
                simultaneous moves).
            context: Observation context for epistemic value computation.

        Returns:
            G_social: Array of social EFE for each action [G(C), G(D)]
            info: Dictionary with breakdown for each action
        """
        G_social = np.zeros(2)
        info_all = {}

        for a_i in [COOPERATE, DEFECT]:
            G_social[a_i], info_all[a_i] = self.compute(
                a_i, my_beliefs, q_response_override=q_response_override,
                context=context,
            )

        return G_social, info_all

    def select_action(
        self,
        my_beliefs: Optional[np.ndarray] = None,
        return_info: bool = False,
    ) -> int | Tuple[int, Dict[str, Any]]:
        """Select action by sampling from softmax(-β * G_social)."""
        G_social, info = self.compute_all_actions(my_beliefs)

        q_action = softmax(-G_social, temperature=1.0/self.beta_self)
        action = np.random.choice([COOPERATE, DEFECT], p=q_action)

        if return_info:
            info["G_social"] = G_social
            info["q_action"] = q_action
            info["selected_action"] = action
            return action, info

        return action


class DepthTwoToM(TheoryOfMind):
    """
    Depth-2 Theory of Mind: I anticipate that j anticipates my response.

    For PD, depth-1 is usually sufficient, but this is available for extensions.
    """

    def __init__(
        self,
        other_model: Any,
        beta_other: float = 4.0,
        beta_self_in_other_view: float = 4.0,
        **kwargs
    ):
        super().__init__(other_model, beta_other, **kwargs)
        self.beta_self_in_other_view = beta_self_in_other_view

    def predict_opponent_action(
        self,
        my_beliefs: Optional[np.ndarray] = None,
    ) -> ToMPrediction:
        """Depth-2 prediction. Falls back to depth-1 for now."""
        return super().predict_opponent_action(my_beliefs)
