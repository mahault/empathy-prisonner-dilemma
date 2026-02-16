"""Opponent Inversion module (Harshil's principled inversion).

Implements particle-based inference over opponent behavioral parameters,
including opponent empathy λ_j as a latent variable.

Each particle represents a parametric behavioral profile:
    P(a_j = C | h_t) = sigma(beta * (alpha + rho * f(h_t) + empathy_shift(lambda_j, p)))

where:
    alpha = cooperation bias (ALLC-like when high, ALLD-like when low)
    rho   = reciprocity (TFT-like when positive, contrarian when negative)
    beta  = action precision (deterministic when high, random when low)
    lambda_j = opponent empathy [0, 1] (how much opponent values our payoff)
    f(h_t) = history feature (+1 if I cooperated last, -1 if I defected, 0 if no history)
    empathy_shift = social EFE-derived cooperation advantage given lambda_j

The empathy_shift is derived from PD payoffs (R=3, S=0, T=5, P=1):
    empathy_shift(lambda_j, p) = 5 * lambda_j - p - 1

where p is the opponent's belief about my cooperation rate. This means:
    lambda_j = 0  → shift = -p-1 (selfish, D preferred)
    lambda_j > 0  → shift increases (empathy pushes toward C)

Classic strategies as special cases:
    ALLC:   alpha >> 0, rho ~ 0
    ALLD:   alpha << 0, rho ~ 0
    TFT:    alpha ~ 0,  rho >> 0
    Random: alpha ~ 0,  rho ~ 0, beta ~ 0
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class BehavioralProfile:
    """Parametric behavioral profile for a particle."""
    alpha: float       # cooperation bias (logit scale)
    reciprocity: float  # sensitivity to my last action (TFT-like)
    beta: float        # action precision (inverse temperature)
    lambda_j: float = 0.0  # opponent empathy level [0, 1]


@dataclass
class InversionState:
    """State of the opponent inversion."""
    weights: np.ndarray
    profiles: List[BehavioralProfile]
    reliability: float
    entropy: float
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


def _logistic(x: float) -> float:
    """Standard logistic sigmoid, numerically stable."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ez = np.exp(x)
        return ez / (1.0 + ez)


class OpponentInversion:
    """
    Particle-based opponent inference using parametric behavioral profiles.

    Each particle carries (alpha, reciprocity, beta, lambda_j) and produces:
        P(C | h_t) = logistic(beta * (alpha + reciprocity * f(h_t) + empathy_shift))

    where f(h_t) encodes the history feature and empathy_shift captures the
    social EFE-derived cooperation advantage from opponent empathy.
    """

    def __init__(
        self,
        n_particles: int = 30,
        reliability_threshold: float = 0.5,
        resample_threshold: float = 0.5,
        initial_weights: Optional[np.ndarray] = None,
        # Keep hypotheses parameter for backward compat (ignored)
        hypotheses=None,
    ):
        self.n_particles = n_particles
        self.reliability_threshold = reliability_threshold
        self.resample_threshold = resample_threshold
        self.my_cooperation_rate: float = 0.5  # Updated by agent each round

        self._initialize_particles(initial_weights)
        self.observation_history: List[Tuple[int, ObservationContext]] = []

    def _initialize_particles(self, initial_weights: Optional[np.ndarray] = None):
        """Initialize particles with priors over behavioral parameters."""
        if initial_weights is not None:
            self.weights = initial_weights.copy()
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        self.profiles: List[BehavioralProfile] = []
        for _ in range(self.n_particles):
            alpha = np.random.normal(0.0, 2.0)
            reciprocity = np.random.normal(0.0, 1.5)
            beta = np.random.gamma(2.0, 2.0)
            lambda_j = np.random.uniform(0.0, 1.0)
            self.profiles.append(BehavioralProfile(alpha, reciprocity, beta, lambda_j))

    def _history_feature(self, context: ObservationContext) -> float:
        """Extract history feature f(h_t) for the parametric model.

        Returns +1 if I cooperated last, -1 if I defected, 0 if no history.
        """
        if context.my_last_action is None:
            return 0.0
        return 1.0 - 2.0 * context.my_last_action  # C=0 → +1, D=1 → -1

    def _empathy_feature(self, lambda_j: float, my_coop_rate: float) -> float:
        """Compute social-EFE utility advantage of cooperation for opponent with empathy lambda_j.

        Derived from PD payoffs (R=3, S=0, T=5, P=1):
            E[U_j(C)] - E[U_j(D)] = 5*lambda_j - p - 1

        where p = my_coop_rate (opponent's belief about my cooperation probability).

        Returns positive when cooperation is preferred (high empathy),
        negative when defection is preferred (low empathy).
        """
        return 5.0 * lambda_j - my_coop_rate - 1.0

    def _particle_action_probs(
        self,
        particle_idx: int,
        context: ObservationContext,
    ) -> np.ndarray:
        """Compute P(a_j) = [P(C), P(D)] for a single particle.

        P(C | h_t) = logistic(beta * (alpha + reciprocity * f(h_t) + empathy_shift))
        """
        profile = self.profiles[particle_idx]
        f = self._history_feature(context)
        empathy_shift = self._empathy_feature(profile.lambda_j, self.my_cooperation_rate)
        logit = profile.beta * (profile.alpha + profile.reciprocity * f + empathy_shift)
        p_c = _logistic(logit)
        return np.array([p_c, 1.0 - p_c])

    def update(
        self,
        observed_action: int,
        context: ObservationContext,
    ) -> InversionState:
        """Update particle weights given observed opponent action."""
        self.observation_history.append((observed_action, context))

        # Compute likelihood for each particle
        likelihoods = np.zeros(self.n_particles)
        for k in range(self.n_particles):
            probs = self._particle_action_probs(k, context)
            likelihoods[k] = max(probs[observed_action], 1e-10)

        # Update weights
        self.weights *= likelihoods

        # Normalize
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Compute metrics
        rel = self.reliability()
        ent = self._weight_entropy()
        ess = self._effective_sample_size()

        # Resample if ESS too low
        if ess < self.resample_threshold * self.n_particles:
            self._resample()

        return InversionState(
            weights=self.weights.copy(),
            profiles=[BehavioralProfile(p.alpha, p.reciprocity, p.beta, p.lambda_j)
                      for p in self.profiles],
            reliability=rel,
            entropy=ent,
            effective_sample_size=ess,
        )

    def reliability(self) -> float:
        """Compute reliability from weight concentration (entropy-based)."""
        entropy = self._weight_entropy()
        max_entropy = np.log(self.n_particles)

        if max_entropy > 0:
            confidence = 1 - entropy / max_entropy
        else:
            confidence = 1.0

        return sigmoid(confidence, center=0.5, scale=0.1)

    def _weight_entropy(self) -> float:
        """Compute entropy of particle weights."""
        nonzero = self.weights[self.weights > 1e-10]
        if len(nonzero) == 0:
            return np.log(self.n_particles)
        return -np.sum(nonzero * np.log(nonzero))

    def _effective_sample_size(self) -> float:
        """Compute effective sample size."""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self):
        """Resample particles using systematic resampling with jitter."""
        positions = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
        cumsum = np.cumsum(self.weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, self.n_particles - 1)

        # Resample profiles with jitter
        new_profiles = []
        for idx in indices:
            old = self.profiles[idx]
            new_profiles.append(BehavioralProfile(
                alpha=old.alpha + np.random.normal(0, 0.1),
                reciprocity=old.reciprocity + np.random.normal(0, 0.1),
                beta=max(0.01, old.beta + np.random.normal(0, 0.1)),
                lambda_j=np.clip(old.lambda_j + np.random.normal(0, 0.05), 0.0, 1.0),
            ))
        self.profiles = new_profiles

        # Reset weights
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict_action(self, context: ObservationContext) -> np.ndarray:
        """Bayesian model-averaged prediction of opponent's next action.

        Returns:
            np.ndarray: [P(Cooperate), P(Defect)]
        """
        action_probs = np.zeros(2)
        for k in range(self.n_particles):
            probs = self._particle_action_probs(k, context)
            action_probs += self.weights[k] * probs
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        else:
            action_probs = np.array([0.5, 0.5])
        return action_probs

    def get_profile_summary(self) -> Dict[str, float]:
        """Get weighted summary statistics of behavioral profile parameters.

        Returns:
            Dictionary with mean and std of alpha, reciprocity, beta, lambda_j.
        """
        alphas = np.array([p.alpha for p in self.profiles])
        reciprocities = np.array([p.reciprocity for p in self.profiles])
        betas = np.array([p.beta for p in self.profiles])
        lambdas = np.array([p.lambda_j for p in self.profiles])

        mean_alpha = float(np.sum(self.weights * alphas))
        mean_recip = float(np.sum(self.weights * reciprocities))
        mean_beta = float(np.sum(self.weights * betas))
        mean_lambda = float(np.sum(self.weights * lambdas))

        return {
            "mean_alpha": mean_alpha,
            "mean_reciprocity": mean_recip,
            "mean_beta": mean_beta,
            "mean_lambda_j": mean_lambda,
            "std_alpha": float(np.sqrt(np.sum(self.weights * (alphas - mean_alpha)**2))),
            "std_reciprocity": float(np.sqrt(np.sum(self.weights * (reciprocities - mean_recip)**2))),
            "std_beta": float(np.sqrt(np.sum(self.weights * (betas - mean_beta)**2))),
            "std_lambda_j": float(np.sqrt(np.sum(self.weights * (lambdas - mean_lambda)**2))),
        }

    def get_mean_profile(self) -> BehavioralProfile:
        """Get weighted mean behavioral profile."""
        summary = self.get_profile_summary()
        return BehavioralProfile(
            alpha=summary["mean_alpha"],
            reciprocity=summary["mean_reciprocity"],
            beta=summary["mean_beta"],
            lambda_j=summary["mean_lambda_j"],
        )

    def get_lambda_j_posterior(self) -> Dict[str, float]:
        """Get posterior statistics for opponent empathy lambda_j.

        Returns:
            Dictionary with mean, std, and entropy of the lambda_j posterior.
        """
        lambdas = np.array([p.lambda_j for p in self.profiles])
        mean = float(np.sum(self.weights * lambdas))
        std = float(np.sqrt(np.sum(self.weights * (lambdas - mean)**2)))
        entropy = self._lambda_j_entropy()

        return {
            "mean": mean,
            "std": std,
            "entropy": entropy,
        }

    def _lambda_j_entropy(self, weights: Optional[np.ndarray] = None) -> float:
        """Compute entropy of lambda_j distribution using histogram approximation."""
        if weights is None:
            weights = self.weights
        lambdas = np.array([p.lambda_j for p in self.profiles])

        # Discretize into bins
        n_bins = 10
        bin_probs = np.zeros(n_bins)
        for k in range(self.n_particles):
            bin_idx = int(np.clip(lambdas[k] * n_bins, 0, n_bins - 1))
            bin_probs[bin_idx] += weights[k]

        # Normalize
        total = bin_probs.sum()
        if total < 1e-10:
            return np.log(n_bins)
        bin_probs /= total

        # Shannon entropy
        nonzero = bin_probs[bin_probs > 1e-10]
        return -float(np.sum(nonzero * np.log(nonzero)))

    def compute_epistemic_value(
        self,
        my_action: int,
        context: Optional[ObservationContext],
    ) -> float:
        """Compute epistemic value (expected information gain about lambda_j).

        G_epistemic(a_i) = -IG(a_i)

        where IG(a_i) is the expected reduction in lambda_j entropy from
        observing the opponent's *next-round* response, given that I play
        a_i this round.

        Key insight: cooperating is typically more epistemically valuable
        because it better disambiguates opponent empathy. A selfish opponent
        defects regardless; an empathetic opponent cooperates back. Against
        defection, both types tend to defect.

        Args:
            my_action: My candidate action (0=C, 1=D)
            context: Current observation context

        Returns:
            float: Negative expected information gain (lower = more informative)
        """
        # Current lambda_j entropy
        H_prior = self._lambda_j_entropy()

        # Build hypothetical next-round context (my_last_action = my_action)
        next_context = ObservationContext(
            my_last_action=my_action,
            their_last_action=context.their_last_action if context else None,
            joint_outcome=None,
            round_number=(context.round_number + 1) if context else 1,
        )

        # For each possible next-round opponent response
        expected_H_posterior = 0.0
        for a_j_next in [0, 1]:  # C, D
            # Compute marginal probability and hypothetical posterior weights
            marginal_p = 0.0
            hyp_weights = np.zeros(self.n_particles)
            for k in range(self.n_particles):
                probs = self._particle_action_probs(k, next_context)
                likelihood_k = max(probs[a_j_next], 1e-10)
                hyp_weights[k] = self.weights[k] * likelihood_k
                marginal_p += self.weights[k] * probs[a_j_next]

            if marginal_p < 1e-10:
                continue

            # Normalize hypothetical posterior
            hyp_weights /= hyp_weights.sum()

            # Compute lambda_j entropy under hypothetical posterior
            H_post = self._lambda_j_entropy(hyp_weights)
            expected_H_posterior += marginal_p * H_post

        IG = H_prior - expected_H_posterior
        return -IG  # Negative: lower EFE = better = more informative

    def is_reliable(self) -> bool:
        """Check if current inference is reliable enough to trust."""
        return self.reliability() >= self.reliability_threshold

    def reset(self):
        """Reset inversion state."""
        self._initialize_particles()
        self.observation_history = []
        self.my_cooperation_rate = 0.5


class GatedToM:
    """
    Theory of Mind with reliability-gated opponent inversion.

    Smoothly interpolates between the static ToM prediction (prior) and the
    learned prediction from the particle filter (posterior), weighted by
    inversion reliability.
    """

    def __init__(
        self,
        tom: 'TheoryOfMind',
        inversion: OpponentInversion,
    ):
        self.tom = tom
        self.inversion = inversion

    def predict_opponent_action(
        self,
        context: Optional[ObservationContext] = None,
    ) -> np.ndarray:
        """
        Predict opponent action with reliability gating.

        q_gated = r * q_learned + (1 - r) * q_static_tom

        Returns:
            q(a_j | h_t) - gated distribution over opponent actions
        """
        reliability = self.inversion.reliability()

        tom_prediction = self.tom.predict_opponent_action()
        q_tom = tom_prediction.q_response

        if context is not None:
            q_learned = self.inversion.predict_action(context)
        else:
            q_learned = q_tom

        q_gated = reliability * q_learned + (1 - reliability) * q_tom

        return q_gated

    def update(self, observed_action: int, context: ObservationContext) -> InversionState:
        """Update inversion with new observation."""
        return self.inversion.update(observed_action, context)
