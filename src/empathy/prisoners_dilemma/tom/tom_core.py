"""Theory of Mind core module.

Implements:
- Best-response prediction: q(a_j | a_i) - opponent's response to my action
- Social EFE computation: G_social = (1-λ) * G_self + λ * E[G_other]

This replaces the K-copy ensemble averaging with proper ToM rollouts.
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
    Theory of Mind module for predicting opponent responses.

    Implements depth-1 ToM: predict how opponent will respond to my action.

    q(a_j | a_i) ∝ exp(-β_j * G_j(a_j | a_i))

    Where G_j is the opponent's expected free energy.
    """

    def __init__(
        self,
        other_model: Any,  # PyMDP Agent representing opponent
        beta_other: float = 4.0,  # Opponent's action precision
        use_pragmatic_value: bool = True,
        use_epistemic_value: bool = True,
    ):
        """
        Initialize Theory of Mind module.

        Args:
            other_model: PyMDP Agent representing our model of the opponent
            beta_other: Opponent's inverse temperature (action precision)
            use_pragmatic_value: Include reward-seeking in opponent's EFE
            use_epistemic_value: Include information gain in opponent's EFE
        """
        self.other_model = other_model
        self.beta_other = beta_other
        self.use_pragmatic_value = use_pragmatic_value
        self.use_epistemic_value = use_epistemic_value

        # Cache for opponent's believed state
        self._qs_other: Optional[np.ndarray] = None

    def update_opponent_beliefs(self, qs_other: np.ndarray) -> None:
        """Update our estimate of opponent's beliefs."""
        self._qs_other = qs_other

    def predict_opponent_response(
        self,
        my_action: int,
        my_beliefs: Optional[np.ndarray] = None,
    ) -> ToMPrediction:
        """
        Predict distribution over opponent actions given my action.

        This is depth-1 ToM: "If I do a_i, what will j do?"

        Args:
            my_action: My candidate action (0=C, 1=D)
            my_beliefs: My current state beliefs (optional)

        Returns:
            ToMPrediction with response distribution and metadata
        """
        # Compute opponent's EFE for each action given my action
        G_other = np.zeros(2)

        for a_j in [COOPERATE, DEFECT]:
            G_other[a_j] = self._compute_opponent_efe(
                opponent_action=a_j,
                my_action=my_action
            )

        # Opponent's response distribution (softmax over negative EFE)
        # Note: pymdp uses negative EFE convention (lower = better)
        q_response = softmax(-G_other, temperature=1.0/self.beta_other)

        # Confidence from entropy
        entropy = -np.sum(q_response * np.log(q_response + 1e-10))
        max_entropy = np.log(2)  # Maximum entropy for 2 actions
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
        my_action: int,
    ) -> float:
        """
        Compute opponent's expected free energy for their action.

        G_j(a_j | a_i) = pragmatic_value + epistemic_value

        In PD, this is primarily about expected payoff (pragmatic value).
        """
        G = 0.0

        if self.use_pragmatic_value:
            # Pragmatic value: expected (negative) utility
            # Higher payoff = lower EFE (more preferred)
            _, other_payoff = PD_PAYOFFS[(my_action, opponent_action)]
            # Convert to EFE: negative of expected log preference
            # Using simple linear mapping: G = -payoff
            G -= other_payoff

        if self.use_epistemic_value:
            # Epistemic value: expected information gain
            # In simple PD, this is minimal, but can add if needed
            # For now, set to 0 (pure pragmatic)
            pass

        return G

    def predict_response_distribution(
        self,
        my_action: int,
    ) -> np.ndarray:
        """
        Simplified interface: just return q(a_j | a_i).
        """
        prediction = self.predict_opponent_response(my_action)
        return prediction.q_response


class SocialEFE:
    """
    Compute social expected free energy.

    G_social(a_i) = (1-λ) * G_self(a_i | q(a_j|a_i)) + λ * E[G_other(a_j | a_i)]

    This is the key innovation: empathy weights self vs other EFE
    without baking in coordination.
    """

    def __init__(
        self,
        tom: TheoryOfMind,
        empathy_factor: float = 0.5,
        beta_self: float = 4.0,
    ):
        """
        Initialize Social EFE computer.

        Args:
            tom: Theory of Mind module for predicting opponent responses
            empathy_factor: λ ∈ [0,1], weight on opponent's EFE
            beta_self: My action precision
        """
        self.tom = tom
        self.empathy_factor = empathy_factor
        self.beta_self = beta_self

    def compute(
        self,
        my_action: int,
        my_beliefs: Optional[np.ndarray] = None,
        q_response_override: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute social EFE for a candidate action.

        Args:
            my_action: Candidate action (0=C, 1=D)
            my_beliefs: Current state beliefs (optional)
            q_response_override: If provided, use this as q(a_j|a_i) instead
                of the static ToM prediction. From learned opponent model.

        Returns:
            G_social: Social expected free energy
            info: Dictionary with breakdown (G_self, G_other_expected, etc.)
        """
        if q_response_override is not None:
            q_response = q_response_override
            # Compute G_other for each opponent action under known payoffs
            G_other = np.zeros(2)
            for a_j in [COOPERATE, DEFECT]:
                _, other_payoff = PD_PAYOFFS[(my_action, a_j)]
                G_other[a_j] = -other_payoff
            confidence = 1.0
        else:
            # Get opponent's predicted response from static ToM
            prediction = self.tom.predict_opponent_response(my_action, my_beliefs)
            q_response = prediction.q_response
            G_other = prediction.G_other
            confidence = prediction.confidence

        # My EFE given opponent's expected response
        G_self = self._compute_my_efe(my_action, q_response)

        # Opponent's expected EFE under their response
        G_other_expected = np.sum(q_response * G_other)

        # Social EFE: weighted combination
        λ = self.empathy_factor
        G_social = (1 - λ) * G_self + λ * G_other_expected

        info = {
            "G_self": G_self,
            "G_other_expected": G_other_expected,
            "q_response": q_response,
            "G_other": G_other,
            "empathy_factor": λ,
            "prediction_confidence": confidence,
        }

        return G_social, info

    def _compute_my_efe(
        self,
        my_action: int,
        q_opponent: np.ndarray,
    ) -> float:
        """
        Compute my EFE given opponent's response distribution.

        G_i(a_i | q(a_j|a_i)) = E_{q(a_j|a_i)}[-payoff_i(a_i, a_j)]
        """
        expected_payoff = 0.0
        for a_j in [COOPERATE, DEFECT]:
            my_payoff, _ = PD_PAYOFFS[(my_action, a_j)]
            expected_payoff += q_opponent[a_j] * my_payoff

        # EFE is negative of expected payoff (lower EFE = better)
        return -expected_payoff

    def compute_all_actions(
        self,
        my_beliefs: Optional[np.ndarray] = None,
        q_response_overrides: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute social EFE for all candidate actions.

        Args:
            my_beliefs: Current state beliefs (optional)
            q_response_overrides: If provided, per-action mapping
                {action: q(a_j|a_i)} from gated opponent model.

        Returns:
            G_social: Array of social EFE for each action [G(C), G(D)]
            info: Dictionary with breakdown for each action
        """
        G_social = np.zeros(2)
        info_all = {}

        for a_i in [COOPERATE, DEFECT]:
            override = q_response_overrides[a_i] if q_response_overrides else None
            G_social[a_i], info_all[a_i] = self.compute(
                a_i, my_beliefs, q_response_override=override
            )

        return G_social, info_all

    def select_action(
        self,
        my_beliefs: Optional[np.ndarray] = None,
        return_info: bool = False,
    ) -> int | Tuple[int, Dict[str, Any]]:
        """
        Select action by sampling from softmax(-β * G_social).

        Returns:
            action: Selected action (0=C, 1=D)
            info: (optional) Dictionary with EFE breakdown
        """
        G_social, info = self.compute_all_actions(my_beliefs)

        # Action distribution
        q_action = softmax(-G_social, temperature=1.0/self.beta_self)

        # Sample action
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

    This adds one more nesting layer:
    - Depth 1: "If I do a_i, j will do a_j"
    - Depth 2: "If I do a_i, j thinks I would respond to their a_j, so j does..."

    For PD, depth-1 is usually sufficient, but this is available for extensions.
    """

    def __init__(
        self,
        other_model: Any,
        beta_other: float = 4.0,
        beta_self_in_other_view: float = 4.0,  # What j thinks my β is
        **kwargs
    ):
        super().__init__(other_model, beta_other, **kwargs)
        self.beta_self_in_other_view = beta_self_in_other_view

    def predict_opponent_response(
        self,
        my_action: int,
        my_beliefs: Optional[np.ndarray] = None,
    ) -> ToMPrediction:
        """
        Depth-2 prediction: j considers my response to their response.

        For now, falls back to depth-1. Override for full implementation.
        """
        # For simplicity, use depth-1 for now
        # Full depth-2 would involve j predicting my response to each of their actions
        return super().predict_opponent_response(my_action, my_beliefs)
