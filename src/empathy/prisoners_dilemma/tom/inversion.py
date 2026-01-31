"""Opponent Inversion module (Harshil's principled inversion).

Implements particle-based inference over opponent types/parameters.

Key components:
- Hypothesis space: Different opponent strategies (TFT, WSLS, etc.)
- Likelihood: P(observed_action | hypothesis, context)
- Particle filter: Update weights based on observations
- Reliability gating: Only trust ToM when inference is reliable
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class OpponentHypothesis(Enum):
    """Hypotheses about opponent's strategy type."""
    ALWAYS_COOPERATE = "always_C"
    ALWAYS_DEFECT = "always_D"
    TIT_FOR_TAT = "TFT"  # Copy opponent's last action
    WIN_STAY_LOSE_SHIFT = "WSLS"  # Repeat if won, switch if lost
    RANDOM = "random"  # 50/50
    RATIONAL = "rational"  # Rational with some β (minimize EFE)


@dataclass
class InversionState:
    """State of the opponent inversion."""
    weights: np.ndarray  # Particle weights
    particle_types: np.ndarray  # Strategy type for each particle
    particle_params: np.ndarray  # Optional parameters (e.g., β for rational)
    reliability: float  # Current reliability score
    entropy: float  # Weight entropy
    effective_sample_size: float


@dataclass
class ObservationContext:
    """Context for computing action likelihood."""
    my_last_action: Optional[int]  # My action in previous round
    their_last_action: Optional[int]  # Their action in previous round
    joint_outcome: Optional[int]  # Observation index (CC=0, CD=1, DC=2, DD=3)
    round_number: int
    my_cumulative_payoff: float = 0.0
    their_cumulative_payoff: float = 0.0


def sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    """Sigmoid function for reliability gating."""
    return 1.0 / (1.0 + np.exp(-(x - center) / scale))


class OpponentInversion:
    """
    Particle-based opponent inference (Harshil's inversion pattern).

    Maintains a distribution over opponent "types" (strategies) and updates
    based on observed actions. Provides reliability gating to know when
    ToM predictions can be trusted.
    """

    def __init__(
        self,
        n_particles: int = 30,
        hypotheses: Optional[List[OpponentHypothesis]] = None,
        reliability_threshold: float = 0.5,
        resample_threshold: float = 0.5,  # ESS fraction
        initial_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize opponent inversion.

        Args:
            n_particles: Number of particles
            hypotheses: List of opponent strategy hypotheses
            reliability_threshold: Threshold for trusting ToM
            resample_threshold: ESS threshold for resampling (as fraction of n_particles)
            initial_weights: Initial particle weights (uniform if None)
        """
        self.n_particles = n_particles
        self.reliability_threshold = reliability_threshold
        self.resample_threshold = resample_threshold

        # Default hypotheses if not provided
        if hypotheses is None:
            hypotheses = [
                OpponentHypothesis.ALWAYS_COOPERATE,
                OpponentHypothesis.ALWAYS_DEFECT,
                OpponentHypothesis.TIT_FOR_TAT,
                OpponentHypothesis.WIN_STAY_LOSE_SHIFT,
                OpponentHypothesis.RANDOM,
            ]
        self.hypotheses = hypotheses
        self.n_hypotheses = len(hypotheses)

        # Initialize particles
        self._initialize_particles(initial_weights)

        # Track observation history
        self.observation_history: List[Tuple[int, ObservationContext]] = []

    def _initialize_particles(self, initial_weights: Optional[np.ndarray] = None):
        """Initialize particle weights and types."""
        # Uniform weights
        if initial_weights is not None:
            self.weights = initial_weights.copy()
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Assign hypothesis types to particles (uniform distribution over types)
        self.particle_types = np.random.choice(
            len(self.hypotheses),
            size=self.n_particles,
            replace=True
        )

        # Parameters for each particle (e.g., β for rational type)
        # For rational particles, sample β from a prior
        self.particle_params = np.ones(self.n_particles) * 4.0  # Default β=4
        rational_idx = np.where(
            np.array([self.hypotheses[t] for t in self.particle_types]) == OpponentHypothesis.RATIONAL
        )[0]
        if len(rational_idx) > 0:
            self.particle_params[rational_idx] = np.random.gamma(4, 1, size=len(rational_idx))

    def update(
        self,
        observed_action: int,
        context: ObservationContext,
    ) -> InversionState:
        """
        Update particle weights given observed opponent action.

        Args:
            observed_action: Observed opponent action (0=C, 1=D)
            context: Observation context (previous actions, etc.)

        Returns:
            InversionState with updated weights and reliability
        """
        # Store observation
        self.observation_history.append((observed_action, context))

        # Compute likelihood for each particle
        likelihoods = np.zeros(self.n_particles)
        for k in range(self.n_particles):
            hypothesis = self.hypotheses[self.particle_types[k]]
            params = self.particle_params[k]
            likelihoods[k] = self._action_likelihood(
                observed_action, hypothesis, context, params
            )

        # Update weights
        self.weights *= likelihoods

        # Normalize (handle case where all weights are 0)
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:
            self.weights /= weight_sum
        else:
            # Reset to uniform if all weights collapsed
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Compute reliability
        reliability = self.reliability()
        entropy = self._weight_entropy()
        ess = self._effective_sample_size()

        # Resample if ESS too low
        if ess < self.resample_threshold * self.n_particles:
            self._resample()

        return InversionState(
            weights=self.weights.copy(),
            particle_types=self.particle_types.copy(),
            particle_params=self.particle_params.copy(),
            reliability=reliability,
            entropy=entropy,
            effective_sample_size=ess,
        )

    def _action_likelihood(
        self,
        action: int,
        hypothesis: OpponentHypothesis,
        context: ObservationContext,
        params: float = 4.0,
    ) -> float:
        """
        Compute P(action | hypothesis, context).

        Args:
            action: Observed action (0=C, 1=D)
            hypothesis: Opponent strategy hypothesis
            context: Observation context
            params: Parameters (e.g., β for rational)

        Returns:
            Likelihood of observed action
        """
        # Get action probabilities under this hypothesis
        p_action = self._hypothesis_action_probs(hypothesis, context, params)

        # Return likelihood (with small floor for numerical stability)
        return max(p_action[action], 1e-10)

    def _hypothesis_action_probs(
        self,
        hypothesis: OpponentHypothesis,
        context: ObservationContext,
        params: float,
    ) -> np.ndarray:
        """
        Get P(a) for each action under a hypothesis.

        Returns:
            Array [P(C), P(D)] under the hypothesis
        """
        if hypothesis == OpponentHypothesis.ALWAYS_COOPERATE:
            return np.array([0.99, 0.01])

        elif hypothesis == OpponentHypothesis.ALWAYS_DEFECT:
            return np.array([0.01, 0.99])

        elif hypothesis == OpponentHypothesis.RANDOM:
            return np.array([0.5, 0.5])

        elif hypothesis == OpponentHypothesis.TIT_FOR_TAT:
            # Copy my last action (from opponent's perspective, my action)
            if context.my_last_action is None:
                # First round: TFT typically starts with cooperation
                return np.array([0.9, 0.1])
            else:
                # Copy what I did
                if context.my_last_action == 0:  # I cooperated
                    return np.array([0.95, 0.05])  # They cooperate
                else:
                    return np.array([0.05, 0.95])  # They defect

        elif hypothesis == OpponentHypothesis.WIN_STAY_LOSE_SHIFT:
            # Repeat if outcome was good, switch if bad
            if context.joint_outcome is None:
                # First round: often starts with cooperation
                return np.array([0.7, 0.3])
            else:
                # Check if opponent "won" last round
                # CC=0, CD=1, DC=2, DD=3
                # From opponent's view: CD (I got suckered) is bad, DC (I exploited) is good
                if context.their_last_action == 0:  # They cooperated
                    # CC is good (mutual), CD is bad (suckered)
                    if context.joint_outcome in [0]:  # CC - good, stay
                        return np.array([0.9, 0.1])
                    else:  # CD - bad, shift
                        return np.array([0.1, 0.9])
                else:  # They defected
                    # DC is good (exploited), DD is bad (mutual defection)
                    if context.joint_outcome in [2]:  # DC - good, stay
                        return np.array([0.1, 0.9])
                    else:  # DD - bad, shift
                        return np.array([0.9, 0.1])

        elif hypothesis == OpponentHypothesis.RATIONAL:
            # Rational agent with precision β
            # Simplified: compute expected payoffs and softmax
            # If I defected, rational opponent should defect (Nash equilibrium)
            # If I cooperated, rational opponent might cooperate if empathetic
            β = params
            if context.my_last_action is None:
                # First round: slight preference for defection (Nash)
                G = np.array([-3.0, -3.5])  # Slight preference for C in uncertainty
            elif context.my_last_action == 0:  # I cooperated
                # Opponent EFE: C gives them 3, D gives them 5
                G = np.array([-3.0, -5.0])  # D is better for them
            else:  # I defected
                # Opponent EFE: C gives them 0, D gives them 1
                G = np.array([0.0, -1.0])  # D is better

            # Softmax
            exp_g = np.exp(β * G)  # Note: G is already negative EFE
            p = exp_g / np.sum(exp_g)
            return p

        else:
            # Unknown hypothesis - uniform
            return np.array([0.5, 0.5])

    def reliability(self) -> float:
        """
        Compute reliability from weight concentration.

        Reliability is high when particles are concentrated on a few hypotheses.
        Uses entropy-based confidence.

        Returns:
            Reliability score ∈ [0, 1]
        """
        entropy = self._weight_entropy()
        max_entropy = np.log(self.n_particles)

        # Confidence: 1 when entropy is 0 (all weight on one particle)
        if max_entropy > 0:
            confidence = 1 - entropy / max_entropy
        else:
            confidence = 1.0

        # Transform to reliability using sigmoid
        # Center at 0.5 confidence, scale controls sharpness
        reliability = sigmoid(confidence, center=0.5, scale=0.1)

        return reliability

    def _weight_entropy(self) -> float:
        """Compute entropy of particle weights."""
        # Filter out zero weights
        nonzero = self.weights[self.weights > 1e-10]
        if len(nonzero) == 0:
            return np.log(self.n_particles)  # Maximum entropy
        return -np.sum(nonzero * np.log(nonzero))

    def _effective_sample_size(self) -> float:
        """Compute effective sample size."""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self):
        """Resample particles using systematic resampling."""
        # Systematic resampling
        positions = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
        cumsum = np.cumsum(self.weights)

        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, self.n_particles - 1)

        # Resample
        self.particle_types = self.particle_types[indices]
        self.particle_params = self.particle_params[indices]

        # Add small jitter to params
        self.particle_params += np.random.normal(0, 0.1, self.n_particles)
        self.particle_params = np.clip(self.particle_params, 0.1, 20.0)

        # Reset weights
        self.weights = np.ones(self.n_particles) / self.n_particles

    def get_type_distribution(self) -> Dict[OpponentHypothesis, float]:
        """
        Get posterior distribution over opponent types.

        Returns:
            Dictionary mapping hypothesis to probability
        """
        type_probs = {}
        for i, h in enumerate(self.hypotheses):
            mask = self.particle_types == i
            type_probs[h] = np.sum(self.weights[mask])
        return type_probs

    def get_most_likely_type(self) -> Tuple[OpponentHypothesis, float]:
        """
        Get most likely opponent type and its probability.

        Returns:
            (hypothesis, probability)
        """
        type_probs = self.get_type_distribution()
        best_type = max(type_probs, key=type_probs.get)
        return best_type, type_probs[best_type]

    def get_expected_beta(self) -> float:
        """Get expected β parameter (for rational hypothesis)."""
        return np.sum(self.weights * self.particle_params)

    def is_reliable(self) -> bool:
        """Check if current inference is reliable enough to trust."""
        return self.reliability() >= self.reliability_threshold

    def reset(self):
        """Reset inversion state."""
        self._initialize_particles()
        self.observation_history = []


class GatedToM:
    """
    Theory of Mind with reliability gating.

    Uses inversion reliability to gate how much ToM predictions influence
    decisions. When reliability is low, falls back to uniform prior.
    """

    def __init__(
        self,
        tom: 'TheoryOfMind',  # Forward reference
        inversion: OpponentInversion,
        fallback_distribution: np.ndarray = np.array([0.5, 0.5]),
    ):
        """
        Initialize gated ToM.

        Args:
            tom: Theory of Mind module
            inversion: Opponent inversion module
            fallback_distribution: Distribution to use when unreliable
        """
        self.tom = tom
        self.inversion = inversion
        self.fallback_distribution = fallback_distribution

    def predict_opponent_response(
        self,
        my_action: int,
        context: Optional[ObservationContext] = None,
    ) -> np.ndarray:
        """
        Predict opponent response with reliability gating.

        Returns:
            q(a_j | a_i) - possibly interpolated with fallback if unreliable
        """
        reliability = self.inversion.reliability()

        # Get ToM prediction
        tom_prediction = self.tom.predict_opponent_response(my_action)
        q_tom = tom_prediction.q_response

        # Interpolate based on reliability
        # High reliability: use ToM prediction
        # Low reliability: use fallback (uniform)
        q_gated = reliability * q_tom + (1 - reliability) * self.fallback_distribution

        return q_gated

    def update(self, observed_action: int, context: ObservationContext) -> InversionState:
        """Update inversion with new observation."""
        return self.inversion.update(observed_action, context)
