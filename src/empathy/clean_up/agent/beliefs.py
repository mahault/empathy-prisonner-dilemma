"""
Dirichlet-Categorical belief representation for discrete contexts.

Implements Dirichlet beliefs over binary outcomes (spawn/no-spawn) for each
pollution context level in the Clean Up environment.
"""

from typing import Optional
import numpy as np


class DirichletBeliefs:
    """
    Dirichlet beliefs over binary outcomes per context.

    For each context (pollution level), maintains:
        alpha[context] = [alpha_0, alpha_1]
        - alpha_0: pseudo-count for "no spawn"
        - alpha_1: pseudo-count for "spawn"

    Expected spawn probability: p_spawn = alpha_1 / (alpha_0 + alpha_1)
    """

    def __init__(
        self,
        n_contexts: int = 5,
        initial_alpha: float = 1.0,
        alpha_dict: Optional[dict[int, np.ndarray]] = None
    ) -> None:
        """
        Initialize Dirichlet beliefs with uniform prior.

        Args:
            n_contexts: Number of discrete contexts (default 5 for pollution 0-4)
            initial_alpha: Initial concentration for uniform prior
            alpha_dict: Optional pre-specified alpha values
        """
        self.n_contexts = n_contexts

        if alpha_dict is not None:
            # validate all context keys exist
            missing = [c for c in range(n_contexts) if c not in alpha_dict]
            if missing:
                raise ValueError(f"alpha_dict missing keys: {missing}")
            # use provided alpha values
            self.alpha = {
                context: np.array(alpha_dict[context], dtype=np.float64)
                for context in range(n_contexts)
            }
        else:
            # Initialize with uniform prior
            self.alpha: dict[int, np.ndarray] = {
                ctx: np.array([initial_alpha, initial_alpha], dtype=np.float64)
                for ctx in range(n_contexts)
            }

    def get_probability(self, context: int) -> float:
        """
        Get expected spawn probability for context.

        Args:
            context: Pollution level (0-4)

        Returns:
            p_spawn = alpha_1 / (alpha[0] + alpha_1)
        """
        if context not in self.alpha:
            raise ValueError(f"Context {context} not in belief model (valid: 0-{self.n_contexts-1})")

        alpha = self.alpha[context]
        return float(alpha[1] / (alpha[0] + alpha[1]))

    def get_uncertainty(self, context: int) -> float:
        """
        Get uncertainty (inverse of total concentration) for context.

        Args:
            context: Pollution level (0-4)

        Returns:
            1.0 / (alpha_0 + alpha_1)
        """
        if context not in self.alpha:
            raise ValueError(f"Context {context} not in belief model (valid: 0-{self.n_contexts-1})")

        alpha = self.alpha[context]
        total = alpha[0] + alpha[1]
        return float(1.0 / total)

    def get_alpha(self, context: int) -> np.ndarray:
        """
        Get Dirichlet concentration parameters for context.

        Args:
            context: Pollution level (0-4)

        Returns:
            Copy of alpha array [alpha_0, alpha_1]
        """
        if context not in self.alpha:
            raise ValueError(f"Context {context} not in belief model (valid: 0-{self.n_contexts-1})")

        return self.alpha[context].copy()

    def get_all_probabilities(self) -> dict[int, float]:
        """Get spawn probabilities for all contexts."""
        return {ctx: self.get_probability(ctx) for ctx in range(self.n_contexts)}

    def update(self, context: int, *, outcome: bool, learning_rate: float = 1.0) -> None:
        """
        Bayesian update from observation.

        Args:
            context: Pollution level where outcome was observed
            outcome: True if apple spawned, False otherwise (keyword-only)
            learning_rate: Pseudo-counts to add per observation
        """
        if context not in self.alpha:
            raise ValueError(f"Context {context} not in belief model (valid: 0-{self.n_contexts-1})")

        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")

        self.alpha[context][1 if outcome else 0] += learning_rate

    def copy(self) -> 'DirichletBeliefs':
        """Create deep copy of beliefs."""
        new = DirichletBeliefs.__new__(DirichletBeliefs)
        new.n_contexts = self.n_contexts
        new.alpha = {k: v.copy() for k, v in self.alpha.items()}
        return new

    def to_dict(self) -> dict:
        """Serialize to dictionary for logging."""
        return {
            'n_contexts': self.n_contexts,
            'alpha': {str(ctx): self.alpha[ctx].tolist() for ctx in range(self.n_contexts)}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DirichletBeliefs':
        """
        Deserialize beliefs from dictionary.

        Args:
            data: Dictionary with 'n_contexts' and 'alpha' keys

        Returns:
            DirichletBeliefs instance
        """
        n_contexts = data['n_contexts']
        alpha_dict = {
            int(context): np.array(alpha, dtype=np.float64)
            for context, alpha in data['alpha'].items()
        }
        return cls(n_contexts=n_contexts, alpha_dict=alpha_dict)

    @classmethod
    def from_array(cls, alpha_array: np.ndarray) -> 'DirichletBeliefs':
        """
        Create beliefs from numpy array of alpha parameters.

        Args:
            alpha_array: Array of shape (n_contexts, 2) with alpha values

        Returns:
            DirichletBeliefs instance
        """
        n_contexts = alpha_array.shape[0]
        alpha_dict = {
            ctx: np.array(alpha_array[ctx], dtype=np.float64)
            for ctx in range(n_contexts)
        }
        return cls(n_contexts=n_contexts, alpha_dict=alpha_dict)

    def __repr__(self) -> str:
        probs = [self.get_probability(ctx) for ctx in range(self.n_contexts)]
        prob_str = ", ".join([f"{ctx}:{p:.3f}" for ctx, p in enumerate(probs)])
        return f"DirichletBeliefs(p_spawn=[{prob_str}])"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = ["DirichletBeliefs:"]
        for context in range(self.n_contexts):
            alpha = self.alpha[context]
            prob = self.get_probability(context)
            uncertainty = self.get_uncertainty(context)
            lines.append(
                f"  Context {context}: p={prob:.3f}, "
                f"alpha=[{alpha[0]:.2f}, {alpha[1]:.2f}], "
                f"uncertainty={uncertainty:.3f}"
            )
        return "\n".join(lines)


def create_beliefs_from_spawn_probs(
    spawn_probs: np.ndarray,
    concentration: float = 100.0
) -> DirichletBeliefs:
    """
    Create Dirichlet beliefs from spawn probability array.

    Args:
        spawn_probs: Array of spawn probabilities, one per context (e.g., [0.25, 0.15, 0.05, 0.0, 0.0])
        concentration: Total concentration (pseudo-observations)
            - Low values (1-2): weak/uncertain prior
            - High values (100+): confident/expert beliefs

    Returns:
        DirichletBeliefs with alpha values matching the given probabilities
    """
    spawn_probs = np.asarray(spawn_probs, dtype=np.float64)
    n_contexts = len(spawn_probs)
    
    alpha_dict = {}
    epsilon = 1e-10  # ensure valid dirichlet (alpha > 0)
    for context in range(n_contexts):
        p_spawn = np.clip(spawn_probs[context], 0.0, 1.0)
        alpha_1 = max(epsilon, concentration * p_spawn)
        alpha_0 = max(epsilon, concentration * (1 - p_spawn))
        alpha_dict[context] = np.array([alpha_0, alpha_1], dtype=np.float64)

    return DirichletBeliefs(n_contexts=n_contexts, alpha_dict=alpha_dict)
