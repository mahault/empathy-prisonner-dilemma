#!/usr/bin/env python
"""
Referee 2: why a single bipolar scalar lambda rather than separate self- and
other-concern weights (a 2D parameter space)?

The paper's pragmatic social EFE is
    G = (1 - lambda) G_self + lambda G_other,                       (bipolar)
and the referee asks about the more general
    G = w_self G_self + w_other G_other.                            (2D)

Analytically the two coincide up to precision. Action selection is
softmax(-beta G), so

    beta (w_self G_self + w_other G_other)
        = [beta (w_self + w_other)] [(1-lambda) G_self + lambda G_other],
      with lambda = w_other / (w_self + w_other).

So on the positive quadrant the DIRECTION of (w_self, w_other) sets lambda and
its MAGNITUDE rescales the action precision beta. This script tests that claim
in simulation and quantifies what the 2D space adds beyond it.

Three experiments:
  ratio      : (w_self, w_other) grid; cooperation should be a function of the
               ratio lambda, with magnitude acting only through precision.
  precision  : exact equivalence check. (c*w_self, c*w_other) at precision beta
               must reproduce (w_self, w_other) at precision c*beta.
  quadrants  : negative weights, which lambda in [0,1] cannot express:
               spite (w_other < 0) and self-abnegation (w_self < 0).

Usage:
    python scripts/run_referee2_weights.py --mode all
"""

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from empathy.prisoners_dilemma import ToMEmpatheticAgent, Environment
from empathy.prisoners_dilemma.tom import tom_core
from empathy.prisoners_dilemma.tom.tom_core import COOPERATE, DEFECT, PD_PAYOFFS

from run_referee_analyses import create_pd_config  # noqa: E402

# -----------------------------------------------------------------------
# Two-weight social EFE: patch SocialEFE.compute to use (w_self, w_other)
# instead of (1-lambda, lambda). Falls back to the paper's behaviour when
# the weights are not set, so the patch is inert unless used.
# -----------------------------------------------------------------------

_original_compute = tom_core.SocialEFE.compute


def _two_weight_compute(self, my_action, my_beliefs=None,
                        q_response_override=None, context=None):
    w_self = getattr(self, "w_self", None)
    w_other = getattr(self, "w_other", None)
    if w_self is None or w_other is None:
        return _original_compute(self, my_action, my_beliefs,
                                 q_response_override, context)

    if q_response_override is not None:
        q_response, confidence = q_response_override, 1.0
    else:
        prediction = self.tom.predict_opponent_action(my_beliefs)
        q_response, confidence = prediction.q_response, prediction.confidence

    G_other = np.zeros(2)
    for a_j in (COOPERATE, DEFECT):
        _, other_payoff = PD_PAYOFFS[(my_action, a_j)]
        G_other[a_j] = -other_payoff

    G_self = self._compute_my_efe(my_action, q_response)
    G_other_expected = float(np.sum(q_response * G_other))

    G_pragmatic = w_self * G_self + w_other * G_other_expected

    G_epistemic = 0.0
    if self.inversion is not None and context is not None:
        G_epistemic = self.inversion.compute_epistemic_value(my_action, context)

    G_social = G_pragmatic + G_epistemic
    return G_social, {
        "G_self": G_self, "G_other_expected": G_other_expected,
        "G_pragmatic": G_pragmatic, "G_epistemic": G_epistemic,
        "q_response": q_response, "G_other": G_other,
        "w_self": w_self, "w_other": w_other,
        "prediction_confidence": confidence,
    }


tom_core.SocialEFE.compute = _two_weight_compute


@dataclass
class WeightResult:
    w_self: float
    w_other: float
    scale: float           # w_self + w_other
    ratio: float           # w_other / (w_self + w_other) == lambda, if scale != 0
    beta: float
    T: int
    seed: int
    coop_rate_i: float
    coop_rate_j: float
    freq_CC: float
    freq_DD: float


def run_weight_pair(w_self, w_other, T=100, seed=0, beta=4.0) -> WeightResult:
    """Symmetric dyad, both agents using (w_self, w_other), static ToM, H=1."""
    np.random.seed(seed)
    config = create_pd_config(T=T, payoffs=(3.0, 0.0, 5.0, 1.0), legacy_C=True)
    env = Environment(K=2)

    agents = []
    for num in (0, 1):
        ag = ToMEmpatheticAgent(config=config, agent_num=num,
                                empathy_factor=0.0, use_inversion=False,
                                beta_self=beta)
        ag.social_efe.w_self = w_self
        ag.social_efe.w_other = w_other
        agents.append(ag)
    ag_i, ag_j = agents

    act_i, act_j = [], []
    actions = [0, 0]
    for t in range(T):
        obs = env.step(t=t, actions=actions)
        obs_i = ag_i.o_init if t == 0 else obs[0]
        obs_j = ag_j.o_init if t == 0 else obs[1]
        a_i = ag_i.step(t=t, observation=obs_i)["exp_action"]
        a_j = ag_j.step(t=t, observation=obs_j)["exp_action"]
        act_i.append(a_i); act_j.append(a_j)
        actions = [a_i, a_j]

    act_i = np.array(act_i); act_j = np.array(act_j)
    scale = w_self + w_other
    return WeightResult(
        w_self=w_self, w_other=w_other, scale=scale,
        ratio=(w_other / scale) if scale != 0 else float("nan"),
        beta=beta, T=T, seed=seed,
        coop_rate_i=float(1.0 - act_i.mean()),
        coop_rate_j=float(1.0 - act_j.mean()),
        freq_CC=float(np.mean((act_i == 0) & (act_j == 0))),
        freq_DD=float(np.mean((act_i == 1) & (act_j == 1))),
    )


def _cc(results):
    return float(np.mean([r.freq_CC for r in results]))


# -----------------------------------------------------------------------

def run_ratio(T, n_seeds, out_dir):
    print("=" * 70)
    print("EXPERIMENT 1: (w_self, w_other) grid vs the bipolar lambda")
    print("=" * 70)
    weights = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    for w_s, w_o in itertools.product(weights, weights):
        cell = [run_weight_pair(w_s, w_o, T=T, seed=s) for s in range(n_seeds)]
        results.extend(cell)
        print(f"  w_self={w_s:.1f} w_other={w_o:.1f} "
              f"(lambda={w_o/(w_s+w_o):.2f}, scale={w_s+w_o:.1f}): CC={_cc(cell):.3f}")

    print("\n--- Grouped by ratio lambda = w_other/(w_self+w_other) ---")
    by_ratio = {}
    for r in results:
        by_ratio.setdefault(round(r.ratio, 3), []).append(r)
    print(f"  {'lambda':>8}  {'n cells':>8}  {'mean CC':>8}  {'spread':>8}")
    for lam in sorted(by_ratio):
        cells = {}
        for r in by_ratio[lam]:
            cells.setdefault(round(r.scale, 2), []).append(r)
        means = [_cc(v) for v in cells.values()]
        print(f"  {lam:>8.3f}  {len(cells):>8}  {np.mean(means):>8.3f}  "
              f"{(max(means)-min(means)):>8.3f}")
    print("\n  Spread within a ratio is the pure magnitude (precision) effect.")

    _save(results, out_dir / "referee2_weight_grid.json")
    return results


def run_precision(T, n_seeds, out_dir):
    """(c*w_s, c*w_o) at beta must equal (w_s, w_o) at c*beta, exactly."""
    print("=" * 70)
    print("EXPERIMENT 2: magnitude is exactly a precision rescaling")
    print("=" * 70)
    base = [(0.7, 0.3), (0.5, 0.5), (0.4, 0.6)]
    beta0 = 4.0
    ok = True
    for (w_s, w_o) in base:
        for c in (0.5, 2.0):
            a = [run_weight_pair(w_s * c, w_o * c, T=T, seed=s, beta=beta0)
                 for s in range(n_seeds)]
            b = [run_weight_pair(w_s, w_o, T=T, seed=s, beta=beta0 * c)
                 for s in range(n_seeds)]
            same = all(abs(x.freq_CC - y.freq_CC) < 1e-12 for x, y in zip(a, b))
            ok &= same
            print(f"  ({w_s},{w_o}) x{c}: CC={_cc(a):.3f} vs beta x{c}: "
                  f"CC={_cc(b):.3f}  identical_per_seed={same}")
    print(f"\n  All identical: {ok}")
    print("  => the 2D positive quadrant adds no behaviour beyond (lambda, beta).")
    return ok


def run_quadrants(T, n_seeds, out_dir):
    """What the bipolar scalar cannot express: negative weights."""
    print("=" * 70)
    print("EXPERIMENT 3: regions outside lambda in [0,1]")
    print("=" * 70)
    cases = [
        ("cooperative      (w_s=0.4, w_o=0.6)",  0.4,  0.6),
        ("self-interested  (w_s=1.0, w_o=0.0)",  1.0,  0.0),
        ("spite            (w_s=0.6, w_o=-0.4)", 0.6, -0.4),
        ("pure spite       (w_s=0.0, w_o=-1.0)", 0.0, -1.0),
        ("self-abnegation  (w_s=-0.4, w_o=0.6)", -0.4, 0.6),
    ]
    results = []
    for label, w_s, w_o in cases:
        cell = [run_weight_pair(w_s, w_o, T=T, seed=s) for s in range(n_seeds)]
        results.extend(cell)
        print(f"  {label}: CC={_cc(cell):.3f}  DD={np.mean([r.freq_DD for r in cell]):.3f}")
    print("\n  Negative w_other (spite) and negative w_self are unreachable with")
    print("  lambda in [0,1]; they are the genuine extension the 2D space offers.")
    _save(results, out_dir / "referee2_quadrants.json")
    return results


def _save(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved {len(results)} runs -> {path}")


def main():
    p = argparse.ArgumentParser(description="Referee 2: 2D self/other weights")
    p.add_argument("--mode", choices=["ratio", "precision", "quadrants", "all"],
                   default="all")
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--n_seeds", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "results" / "referee_analyses"
    start = datetime.now()
    if args.mode in ("ratio", "all"):
        run_ratio(args.T, args.n_seeds, out_dir)
    if args.mode in ("precision", "all"):
        run_precision(args.T, args.n_seeds, out_dir)
    if args.mode in ("quadrants", "all"):
        run_quadrants(args.T, args.n_seeds, out_dir)
    print(f"\nTotal time: {(datetime.now()-start).total_seconds()/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
