#!/usr/bin/env python
"""
Re-baseline the paper's headline numbers under the corrected model.

Model changes on this branch (fix/inferred-lambda-j):
  1. The Theory of Mind evaluates the opponent under a social EFE using the
     INFERRED opponent empathy lambda_j, instead of assuming the opponent is
     purely self-interested. This is what the Methods describe.
  2. Sophisticated planning accumulates EFE over the horizon instead of
     dividing by H, which had rescaled action precision to beta/H.
  3. Rollout predictions at future steps are conditioned on the simulated
     prefix, so multi-step planning is genuinely policy-dependent.

ENVIRONMENT
-----------
Run this with the project virtualenv, which has pymdp and its dependencies:

    .venv/Scripts/python.exe scripts/rerun_all.py --mode all      (Windows)
    .venv/bin/python scripts/rerun_all.py --mode all              (POSIX)

A bare `python` on this machine resolves to a global interpreter that has no
pymdp installed and will fail at import. The check below turns that into a
readable message instead of a bare traceback.

Usage:
    python scripts/rerun_all.py --mode validate
    python scripts/rerun_all.py --mode all
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def check_environment():
    """Fail loudly and usefully if run outside the project virtualenv."""
    try:
        import numpy  # noqa: F401
        import pymdp  # noqa: F401
    except Exception as exc:
        venv = PROJECT_ROOT / ".venv" / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        sys.exit(
            f"\nCannot import pymdp: {type(exc).__name__}: {exc}\n\n"
            f"You are running {sys.executable}\n"
            f"Use the project virtualenv instead:\n\n"
            f"    {venv} scripts/rerun_all.py --mode all\n\n"
            f"If .venv does not exist, create it with:\n"
            f"    python -m venv .venv && {venv} -m pip install -e \".[pymdp,dev]\"\n")


check_environment()

import numpy as np  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from empathy.prisoners_dilemma.tom.tom_core import TheoryOfMind  # noqa: E402
from run_referee_analyses import run_pair, AGENT_TYPES  # noqa: E402

SEEDS = range(20)
T = 100

# Known values produced by the pre-change model, measured repeatedly earlier.
LEGACY_MYOPIC_CC = {0.3: 0.7820, 0.5: 0.9975, 0.7: 1.0000}


def sym_cc(lam, seeds=SEEDS, inversion=False, H=1, T=T):
    k = dict(empathy_factor=lam, use_inversion=inversion)
    ki = dict(k, use_sophisticated=(H > 1), planning_horizon=H)
    rs = [run_pair("rerun", "i", "j", dict(ki), dict(k), T=T, seed=s,
                   payoff_structure="standard", planning_horizon=H,
                   legacy_C=True) for s in seeds]
    return float(np.mean([r.freq_CC for r in rs]))


def mode_validate(_):
    print("=" * 72)
    print("VALIDATION: force lambda_j = 0 and reproduce the legacy baseline")
    print("=" * 72)
    TheoryOfMind.DEFAULT_LAMBDA_J = 0.0
    ok = True
    print(f"  {'lambda':>7} {'now':>10} {'legacy':>10} {'diff':>9}")
    for lam, want in LEGACY_MYOPIC_CC.items():
        got = sym_cc(lam)
        d = got - want
        ok &= abs(d) < 1e-9
        print(f"  {lam:>7} {got:>10.4f} {want:>10.4f} {d:>+9.6f}")
    TheoryOfMind.DEFAULT_LAMBDA_J = 0.5
    print(f"\n  exact reproduction: {ok}")
    if ok:
        print("  => the port is behaviour-preserving at lambda_j = 0.")
    else:
        print("  => DO NOT TRUST any numbers from this run.")
    return ok


def mode_all(args):
    if not mode_validate(args):
        print("\nAborting: validation failed.")
        return False

    print()
    print("=" * 72)
    print("RE-BASELINE under inferred lambda_j (inversion ON, as the paper)")
    print("=" * 72)

    print("\n1. Cooperation vs empathy, and the transition threshold")
    lams = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.9]
    print(f"  {'lambda':>7} {'legacy(static)':>16} {'new(inferred)':>15}")
    legacy, new = [], []
    for lam in lams:
        TheoryOfMind.DEFAULT_LAMBDA_J = 0.0
        a = sym_cc(lam, inversion=False)
        TheoryOfMind.DEFAULT_LAMBDA_J = 0.5
        b = sym_cc(lam, inversion=True)
        legacy.append(a); new.append(b)
        print(f"  {lam:>7} {a:>16.4f} {b:>15.4f}")

    def thr(vals):
        for l, v in zip(lams, vals):
            if v >= 0.8:
                return l
        return None
    print(f"\n  lambda* (first lambda with CC >= 0.8): "
          f"legacy {thr(legacy)}, new {thr(new)}")

    print("\n2. Planning horizon (inversion ON, corrected planner)")
    print(f"  {'lambda':>7}" + "".join(f"{'H='+str(h):>9}" for h in (1, 2, 3, 4))
          + f"{'H4-H1':>9}")
    for lam in (0.3, 0.5, 0.7):
        vals = [sym_cc(lam, inversion=True, H=H) for H in (1, 2, 3, 4)]
        print(f"  {lam:>7}" + "".join(f"{v:>9.4f}" for v in vals)
              + f"{vals[-1]-vals[0]:>+9.4f}")
    print("  manuscript reports -0.185 at lambda=0.3 from H=1 to H=3")

    print("\n3. Model comparison (prediction vs valuation)")
    for ti in ("self_interested", "tom_only", "empathic"):
        rs = [run_pair("rerun", ti, ti, dict(AGENT_TYPES[ti]),
                       dict(AGENT_TYPES[ti]), T=T, seed=s) for s in SEEDS]
        print(f"  {ti:<17} CC={np.mean([r.freq_CC for r in rs]):.3f} "
              f"DD={np.mean([r.freq_DD for r in rs]):.3f}")
    rs = [run_pair("rerun", "empathic", "self_interested",
                   dict(AGENT_TYPES["empathic"]),
                   dict(AGENT_TYPES["self_interested"]), T=T, seed=s)
          for s in SEEDS]
    print(f"  empathic vs self-interested: "
          f"exploitability={np.mean([r.exploitability for r in rs]):.2f} "
          f"gap={np.mean([r.payoff_gap for r in rs]):+.2f}")
    return True


def main():
    p = argparse.ArgumentParser(description="Re-baseline under corrected model")
    p.add_argument("--mode", choices=["validate", "all"], default="all")
    a = p.parse_args()
    ok = mode_validate(a) if a.mode == "validate" else mode_all(a)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
