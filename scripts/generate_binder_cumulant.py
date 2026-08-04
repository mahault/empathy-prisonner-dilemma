#!/usr/bin/env python
"""Binder-cumulant analysis of the cooperation transition.

Adapted from A. Jimenez Rodriguez's cooperation_transition notebook
(notebooks/prisoners_dilemma/analysis_oucomes.ipynb on the
cooperation_transition branch) and rerun under the corrected model on
fix/inferred-lambda-j. Three changes from the notebook version:

  * beta = 4 and standard payoffs, matching the manuscript's analytic
    treatment (the notebook used beta = 5);
  * deterministic seeds, so every number regenerates exactly (the notebook
    drew seeds from np.random.randint);
  * the lambda grid spans the corrected transition at lambda* = 0.45 (the
    notebook's grid stopped at 0.4, short of the rebaselined crossover).

For each interaction length T the order parameter is the per-run mean of the
CC-indicator spin s_t = 2*1[outcome_t = CC] - 1 after a 20% burn-in, and

    U4(lambda, T) = 1 - <m^4> / (3 <m^2>^2)

over R independent runs. U4 -> 2/3 in an ordered phase and dips at the
crossover; the dip sharpening with T is the finite-size signature reported
in the Discussion.

Run with the project virtualenv:
    .venv/Scripts/python.exe scripts/generate_binder_cumulant.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from generate_pcc_figure import simulate  # noqa: E402  (beta = 4, standard payoffs)

BURN_FRAC = 0.2


def binder(lams, sizes, runs, inversion):
    U = np.zeros((len(sizes), len(lams)))
    t0 = time.time()
    for k, T in enumerate(sizes):
        burn = int(BURN_FRAC * T)
        for j, lam in enumerate(lams):
            m = np.zeros(runs)
            for r in range(runs):
                seed = 100000 * k + 1000 * j + r
                hi, hj = simulate(lam, seed, T, inversion)
                s = ((hi == 0) & (hj == 0)).astype(float)[burn:]
                spin = 2.0 * s - 1.0
                m[r] = spin.mean()
            m2 = np.mean(m ** 2)
            m4 = np.mean(m ** 4)
            U[k, j] = 1.0 - m4 / (3.0 * (m2 ** 2 + 1e-15))
            print(f"T={T} lam={lam:.3f} U4={U[k, j]:.3f} [{time.time()-t0:.0f}s]",
                  flush=True)
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--inversion", action="store_true", default=False)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "binder_cumulant.json"))
    args = ap.parse_args()

    lams = [round(l, 3) for l in np.arange(0.30, 0.625, 0.025)]
    sizes = [200, 350, 500]
    U = binder(lams, sizes, args.runs, args.inversion)

    dips = {sizes[k]: float(lams[int(np.argmin(U[k]))]) for k in range(len(sizes))}
    out = {
        "lambdas": lams, "sizes": sizes, "runs": args.runs,
        "inversion": args.inversion, "U4": U.tolist(), "dip_locations": dips,
        "U4_min": {sizes[k]: float(U[k].min()) for k in range(len(sizes))},
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("dip locations by T:", dips)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
