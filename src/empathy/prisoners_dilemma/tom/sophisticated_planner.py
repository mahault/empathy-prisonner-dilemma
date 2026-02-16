"""Sophisticated multi-step planner for the Prisoner's Dilemma.

Implements planning via policy enumeration and forward rollout:
1. Enumerate all 2^H candidate policies (sequences of C/D)
2. For each policy, roll forward H steps accumulating social EFE
3. Select action by marginalizing the softmax policy distribution to the first action

Opponent predictions are history-conditioned q(a_j | h_t) for simultaneous
moves: the opponent's action distribution does not depend on the agent's
within-round action choice.
"""

import numpy as np
from itertools import product
from typing import Optional, Dict, Any, Tuple, List

from empathy.prisoners_dilemma.tom.tom_core import (
    COOPERATE, DEFECT, PD_PAYOFFS, softmax,
)
from empathy.prisoners_dilemma.tom.opponent_simulator import OpponentSimulator


class SophisticatedPlanner:
    """Multi-step rollout planner using social EFE.

    For each candidate H-step policy pi = (a_0, a_1, ..., a_{H-1}):
      G(pi) = (1/H) * sum_t [ (1-lambda)*G_self_t + lambda*G_other_t ]

    where at each step t:
      - Opponent action q(a_j | h_t) is predicted via OpponentSimulator
      - G_self_t = -E[payoff_i(a_i^t, a_j)]
      - G_other_t = E[G_j(a_j)] (opponent's expected EFE)

    Action selection: marginalize softmax(-beta * G(pi)) to the first action.
    """

    def __init__(
        self,
        opponent_sim: OpponentSimulator,
        empathy_factor: float = 0.5,
        horizon: int = 3,
        beta_self: float = 4.0,
    ):
        """
        Args:
            opponent_sim: OpponentSimulator for predicting opponent responses
            empathy_factor: lambda in [0, 1], weight on opponent's EFE
            horizon: Planning horizon H (number of steps to look ahead)
            beta_self: Action precision (inverse temperature)
        """
        self.opponent_sim = opponent_sim
        self.empathy_factor = empathy_factor
        self.horizon = horizon
        self.beta_self = beta_self

        # Pre-generate all 2^H policies
        self.policies = list(product([COOPERATE, DEFECT], repeat=horizon))

    def evaluate_policy(self, policy: Tuple[int, ...]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a single H-step policy by forward rollout.

        For each step t in [0, H):
          1. Predict opponent action: q(a_j | h_t) (history-conditioned)
          2. Compute G_self_t = -E[payoff_i(a_i^t, a_j)]
          3. Compute G_other_t = E[-payoff_j(a_i^t, a_j)]
          4. Combine: G_t = (1-lambda)*G_self_t + lambda*G_other_t
          5. Accumulate

        Returns:
            G_policy: Averaged social EFE for this policy
            info: Per-step breakdown
        """
        total_G = 0.0
        step_info = []
        lam = self.empathy_factor

        for t, my_action in enumerate(policy):
            # Predict opponent action at this rollout step (history-conditioned)
            q_response = self.opponent_sim.predict_response(step=t)

            # My EFE: expected negative payoff
            G_self = 0.0
            for a_j in [COOPERATE, DEFECT]:
                my_payoff, _ = PD_PAYOFFS[(my_action, a_j)]
                G_self += q_response[a_j] * (-my_payoff)

            # Opponent's EFE: expected negative payoff for opponent
            G_other_per_action = np.zeros(2)
            for a_j in [COOPERATE, DEFECT]:
                _, other_payoff = PD_PAYOFFS[(my_action, a_j)]
                G_other_per_action[a_j] = -other_payoff

            G_other_expected = np.sum(q_response * G_other_per_action)

            # Social EFE for this step
            G_t = (1 - lam) * G_self + lam * G_other_expected
            total_G += G_t

            step_info.append({
                "action": my_action,
                "q_response": q_response.copy(),
                "G_self": G_self,
                "G_other_expected": G_other_expected,
                "G_step": G_t,
            })

        # Average over horizon
        G_policy = total_G / self.horizon

        info = {
            "steps": step_info,
            "total_G": total_G,
            "horizon": self.horizon,
        }

        return G_policy, info

    def plan(
        self,
        initial_beliefs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Tuple[int, ...], Dict[str, Any]]:
        """Enumerate all 2^H policies, evaluate, and select first action.

        Returns:
            q_action: [P(C), P(D)] — marginal distribution over first action
            best_policy: The policy with lowest social EFE
            info: Full breakdown including per-policy EFE
        """
        n_policies = len(self.policies)
        G_policies = np.zeros(n_policies)
        policy_info = {}

        for i, policy in enumerate(self.policies):
            G_policies[i], policy_info[policy] = self.evaluate_policy(policy)

        # Policy distribution: softmax over negative EFE
        q_pi = softmax(-G_policies, temperature=1.0 / self.beta_self)

        # Marginalize to first action
        q_action = np.zeros(2)
        for i, policy in enumerate(self.policies):
            q_action[policy[0]] += q_pi[i]

        # Normalize (should already sum to 1, but be safe)
        total = q_action.sum()
        if total > 0:
            q_action /= total

        best_idx = int(np.argmin(G_policies))
        best_policy = self.policies[best_idx]

        info = {
            "G_policies": G_policies,
            "q_pi": q_pi,
            "policies": self.policies,
            "policy_info": policy_info,
            "best_policy": best_policy,
            "best_G": G_policies[best_idx],
        }

        return q_action, best_policy, info
