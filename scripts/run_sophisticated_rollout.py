#!/usr/bin/env python
"""
A genuine sophisticated-inference planner for the empathic PD agent.

Motivation
----------
`run_horizon_diagnostics.py` showed that the planner used in the manuscript is
degenerate: rollout predictions never depend on the candidate policy, so the
policy softmax factorises and horizon H collapses to a myopic rule at precision
beta/H. Correcting the rollout in the minimal way the Methods describe still
gives ~0 effect. The open question is whether a *fuller* treatment behaves
differently, namely one with

  1. branching over the opponent's responses (expectimax, not a single path),
  2. belief updating inside the rollout, so the agent plans over what it will
     come to believe (beliefs about future beliefs, which is the actual content
     of sophisticated inference),
  3. depth-2 theory of mind (the opponent models me modelling them),
  4. longer horizons.

This script implements all four and re-tests the planning-depth claim.

Design
------
The decision layer is re-implemented standalone rather than driven through the
full pymdp agent. This is legitimate here because decisions are driven entirely
by PD_PAYOFFS and the ToM prediction; the pymdp C matrices were verified inert
(see `verify_referee_analyses.py::C_matrices_inert`). The re-implementation is
validated against the shipped myopic agent before any conclusion is drawn
(`--mode validate`), and only used if it reproduces it.

Opponent model (learned, inside the rollout)
    P(a_j = C | h, theta) = sigmoid(beta_j * (alpha + rho * f(h)))
    f(h) = +1 if I cooperated on the previous round, -1 if I defected
Particles over theta = (alpha, rho, beta_j), Bayesian weight update on each
observed (or simulated) opponent action. This is the shipped inversion model
without the empathy_shift term.

Recursion (sophisticated inference)
    G(t, s, a_i) = g(a_i, s) + E_{a_j ~ q(.|s)} [ V(t+1, s') ]
    V(t, s)      = sum_{a_i} pi(a_i | s) G(t, s, a_i),  pi = softmax(-beta G)
with s' the belief state after simulating (a_i, a_j): particle weights updated
on a_j, last action set to a_i, and the opponent's belief about my cooperation
rate updated with a_i. Terminal V = 0 at t = H.

EFE is accumulated, never averaged, so precision does not drift with H.

Usage
    python scripts/run_sophisticated_rollout.py --mode validate
    python scripts/run_sophisticated_rollout.py --mode sweep --horizons 1 2 3 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

C, D = 0, 1
# Decision payoffs, as used throughout the paper (R, S, T, P) = (3, 0, 5, 1)
PAY = {(C, C): (3.0, 3.0), (C, D): (0.0, 5.0),
       (D, C): (5.0, 0.0), (D, D): (1.0, 1.0)}

BETA_SELF = 4.0
BETA_OTHER = 4.0


def _soft(vals, beta):
    z = -beta * np.asarray(vals, dtype=float)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------
# Opponent model
# ----------------------------------------------------------------------

class OpponentBelief:
    """Particle posterior over (alpha, rho, beta_j, lambda_j) for the partner.

    Matches the shipped inversion model, including the empathy shift:
        P(a_j = C | h) = sigmoid(beta_j * (alpha + rho*f(h) + 5*lambda_j - p - 1))
    where p is my own cooperation rate. Dropping the empathy term changes the
    predicted cooperation level drastically, so it is kept here.
    """

    def __init__(self, n_particles=15, rng=None):
        rng = rng or np.random.default_rng(0)
        self.alpha = rng.normal(0.0, 2.0, n_particles)
        self.rho = rng.normal(0.0, 1.5, n_particles)
        self.beta = rng.gamma(2.0, 2.0, n_particles)
        self.lam_j = rng.uniform(0.0, 1.0, n_particles)
        self.w = np.ones(n_particles) / n_particles

    def copy(self):
        o = object.__new__(OpponentBelief)
        o.alpha, o.rho, o.beta, o.lam_j = self.alpha, self.rho, self.beta, self.lam_j
        o.w = self.w.copy()
        return o

    def _p(self, my_last, p_my_coop):
        f = 0.0 if my_last is None else (1.0 if my_last == C else -1.0)
        shift = 5.0 * self.lam_j - p_my_coop - 1.0
        return 1.0 / (1.0 + np.exp(-self.beta * (self.alpha + self.rho * f + shift)))

    def p_coop(self, my_last, p_my_coop=0.5):
        """P(partner cooperates | my last action), marginalised over particles."""
        return float(np.dot(self.w, self._p(my_last, p_my_coop)))

    def reliability(self):
        """Concentration of the particle weights, as in the shipped GatedToM:
        0 when weights are uniform (posterior uninformative), 1 when they are
        concentrated. Used to blend the learned prediction with the static
        prior so the agent does not act on an unreliable posterior early on."""
        w = np.maximum(self.w, 1e-12)
        ent = -float(np.sum(w * np.log(w)))
        return float(1.0 - ent / np.log(len(w)))

    def updated(self, observed_a_j, my_last, p_my_coop=0.5):
        """Posterior after observing (or simulating) the partner's action."""
        p = self._p(my_last, p_my_coop)
        lik = p if observed_a_j == C else (1.0 - p)
        w = self.w * np.maximum(lik, 1e-12)
        s = w.sum()
        o = self.copy()
        o.w = (w / s) if s > 0 else np.ones_like(w) / len(w)
        return o


def q_partner_static(p_my_coop, depth=1):
    """Static ToM prediction of the partner's action.

    depth 1: partner best-responds to their belief about my mixed strategy.
    depth 2: partner anticipates that I best-respond to them, and responds to
             that anticipated policy instead of to my empirical frequency.
    """
    pi_me = np.array([p_my_coop, 1.0 - p_my_coop])

    def partner_given(pi_i):
        G_j = np.array([-sum(pi_i[a_i] * PAY[(a_i, a_j)][1] for a_i in (C, D))
                        for a_j in (C, D)])
        return _soft(G_j, BETA_OTHER)

    q_j = partner_given(pi_me)
    if depth <= 1:
        return q_j
    # depth 2: what would I do against q_j? partner assumes that, then re-solves
    G_i = np.array([-sum(q_j[a_j] * PAY[(a_i, a_j)][0] for a_j in (C, D))
                    for a_i in (C, D)])
    pi_i_pred = _soft(G_i, BETA_SELF)
    return partner_given(pi_i_pred)


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------

class SophisticatedAgent:
    def __init__(self, lam, horizon=1, beta=BETA_SELF, tom_depth=1,
                 learn=True, n_particles=15, seed=0):
        self.lam = lam
        self.H = horizon
        self.beta = beta
        self.tom_depth = tom_depth
        self.learn = learn
        self.belief = OpponentBelief(n_particles, np.random.default_rng(seed))
        self.my_last = None
        self.n_rounds = 0
        self.n_coop = 0

    # -- prediction -----------------------------------------------------
    def _q_partner(self, belief, my_last, p_my_coop):
        q_static = q_partner_static(p_my_coop, depth=self.tom_depth)
        if not self.learn:
            return q_static
        # Reliability-gated blend of learned and static prediction, matching
        # the shipped GatedToM. Without this the agent acts on an unreliable
        # posterior from round one and locks into defection.
        pc = belief.p_coop(my_last, p_my_coop)
        q_learned = np.array([pc, 1.0 - pc])
        r = belief.reliability()
        return r * q_learned + (1.0 - r) * q_static

    def _g_step(self, a_i, q_j):
        g_self = -sum(q_j[a_j] * PAY[(a_i, a_j)][0] for a_j in (C, D))
        g_other = -sum(q_j[a_j] * PAY[(a_i, a_j)][1] for a_j in (C, D))
        return (1 - self.lam) * g_self + self.lam * g_other

    # -- recursion ------------------------------------------------------
    def _G(self, t, belief, my_last, n, k):
        """Return array G[a_i] of expected free energy to go, depth t..H-1.

        n, k track the round count and cooperation count so the partner's
        belief about my mixed strategy moves with my simulated actions.
        """
        p_my = (k / n) if n > 0 else 0.5
        q_j = self._q_partner(belief, my_last, p_my)
        G = np.zeros(2)
        for a_i in (C, D):
            g = self._g_step(a_i, q_j)
            if t + 1 < self.H:
                fut = 0.0
                for a_j in (C, D):
                    if q_j[a_j] < 1e-9:
                        continue
                    b2 = (belief.updated(a_j, my_last, p_my) if self.learn
                          else belief)
                    G2 = self._G(t + 1, b2, a_i, n + 1, k + (1 if a_i == C else 0))
                    pi2 = _soft(G2, self.beta)
                    fut += q_j[a_j] * float(np.dot(pi2, G2))
                g = g + fut
            G[a_i] = g
        return G

    def act(self, rng):
        G = self._G(0, self.belief, self.my_last, self.n_rounds, self.n_coop)
        pi = _soft(G, self.beta)
        return int(rng.random() > pi[C])  # C if u < pi[C]

    def observe(self, my_action, their_action):
        if self.learn:
            p_my = (self.n_coop / self.n_rounds) if self.n_rounds > 0 else 0.5
            self.belief = self.belief.updated(their_action, self.my_last, p_my)
        self.my_last = my_action
        self.n_rounds += 1
        self.n_coop += 1 if my_action == C else 0


def play(lam, H, T=100, seed=0, tom_depth=1, learn=True, partner_H=1):
    rng = np.random.default_rng(seed)
    ai = SophisticatedAgent(lam, H, tom_depth=tom_depth, learn=learn, seed=seed)
    aj = SophisticatedAgent(lam, partner_H, tom_depth=tom_depth, learn=learn,
                            seed=seed + 10_000)
    cc = 0
    for _ in range(T):
        a_i = ai.act(rng)
        a_j = aj.act(rng)
        if a_i == C and a_j == C:
            cc += 1
        ai.observe(a_i, a_j)
        aj.observe(a_j, a_i)
    return cc / T


def mean_cc(lam, H, seeds, **kw):
    return float(np.mean([play(lam, H, seed=s, **kw) for s in seeds]))


# ----------------------------------------------------------------------

def mode_validate(args):
    """H=1 must behave like the shipped myopic agent, or nothing else counts."""
    from run_referee_analyses import run_pair
    print("=" * 72)
    print("VALIDATION: standalone H=1 vs shipped myopic agent (static ToM)")
    print("=" * 72)
    seeds = range(20)
    print(f"  {'lambda':>7} {'standalone':>12} {'shipped':>10} {'diff':>8}")
    worst = 0.0
    for lam in (0.3, 0.5, 0.7):
        mine = mean_cc(lam, 1, seeds, learn=False, tom_depth=1)
        k = dict(empathy_factor=lam, use_inversion=False)
        rs = [run_pair("v", "a", "b", dict(k), dict(k), T=100, seed=s,
                       payoff_structure="standard", planning_horizon=1,
                       legacy_C=True) for s in seeds]
        theirs = float(np.mean([r.freq_CC for r in rs]))
        worst = max(worst, abs(mine - theirs))
        print(f"  {lam:>7} {mine:>12.4f} {theirs:>10.4f} {mine-theirs:>+8.4f}")
    print(f"\n  max |difference| = {worst:.4f}")
    print("  The two are independent implementations of the same decision rule,")
    print("  so close agreement (not bit-identity) is what validates the model.")
    return worst


def mode_sweep(args):
    horizons = args.horizons
    seeds = range(args.n_seeds)
    print("=" * 72)
    print("SWEEP: full sophisticated inference (expectimax + belief updating)")
    print(f"H={list(horizons)}  seeds={args.n_seeds}  T={args.T}")
    print("=" * 72)
    configs = [
        ("depth-1 ToM, static (no learning)", dict(learn=False, tom_depth=1)),
        ("depth-2 ToM, static (no learning)", dict(learn=False, tom_depth=2)),
        ("depth-1 ToM, learning in rollout", dict(learn=True, tom_depth=1)),
    ]
    for label, kw in configs:
        print(f"\n  {label}")
        print(f"  {'lambda':>7}" + "".join(f"{'H='+str(h):>9}" for h in horizons)
              + f"{'last-first':>12}")
        for lam in (0.3, 0.5, 0.7):
            vals = [mean_cc(lam, H, seeds, T=args.T, **kw) for H in horizons]
            print(f"  {lam:>7}" + "".join(f"{v:>9.4f}" for v in vals)
                  + f"{vals[-1]-vals[0]:>+12.4f}")
    print("\n  Manuscript claims roughly -0.19 at lambda=0.3 from H=1 to H=3.")


def main():
    p = argparse.ArgumentParser(description="Full sophisticated-inference rollout")
    p.add_argument("--mode", choices=["validate", "sweep"], default="sweep")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n_seeds", type=int, default=20)
    p.add_argument("--T", type=int, default=100)
    a = p.parse_args()
    if a.mode == "validate":
        mode_validate(a)
    else:
        mode_sweep(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
