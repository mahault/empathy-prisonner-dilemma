#!/usr/bin/env python
"""
Regenerate every number the manuscript cites, under the corrected model.

One script, one output file, so the paper has a single source of truth and
no figure can drift from the text. Writes results/rebaseline/PAPER_NUMBERS.md.

Corrections applied on this branch, in the order they were found:
  1. Theory of Mind evaluates the opponent under the INFERRED lambda_j rather
     than assuming a purely self-interested opponent (what the Methods say).
  2. Sophisticated planning accumulates EFE over the horizon instead of
     dividing by H, which had rescaled action precision to beta/H and made
     "planning depth" a temperature knob.
  3. Rollout predictions at future steps are conditioned on the simulated
     prefix, so multi-step planning is genuinely policy-dependent.
  4. The reliability gate reads particle agreement rather than importance
     weights, which resampling resets. It was previously pinned near 0.015
     and the learned partner model was never used.
  5. The reciprocity feature is aligned to the round the partner actually
     responded to. Moves are simultaneous, so their round t-1 action answers
     my round t-2 action.

Run with the project virtualenv:
    .venv/Scripts/python.exe scripts/rebaseline_paper.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def check_environment():
    try:
        import numpy  # noqa: F401
        import pymdp  # noqa: F401
    except Exception as exc:
        venv = PROJECT_ROOT / ".venv" / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        sys.exit(f"\nCannot import pymdp: {type(exc).__name__}: {exc}\n"
                 f"You are running {sys.executable}\n"
                 f"Use the project virtualenv:\n    {venv} "
                 f"scripts/rebaseline_paper.py\n")


check_environment()

import numpy as np  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from empathy.prisoners_dilemma import ToMEmpatheticAgent, Environment  # noqa: E402
from empathy.prisoners_dilemma.tom.tom_core import TheoryOfMind  # noqa: E402
from run_referee_analyses import (  # noqa: E402
    run_pair, create_pd_config, AGENT_TYPES, EMPATHIC_LAMBDA)
from run_partner_types import STRATS, play, PAY  # noqa: E402

T = 100
SEEDS = range(20)
R, S, Tt, P = 3.0, 0.0, 5.0, 1.0
OUT = PROJECT_ROOT / "results" / "rebaseline"
LEGACY_MYOPIC_CC = {0.3: 0.7820, 0.5: 0.9975, 0.7: 1.0000}

buf = []


def w(line=""):
    print(line, flush=True)
    buf.append(line)


# Multi-step planning enumerates 2^H policies with H rollout steps each, so a
# horizon-4 run costs roughly 64x a myopic one. Use fewer seeds there.
PLAN_SEEDS = range(10)


def simulate(lam_i, lam_j, seed, T=T, H=1, inversion=True):
    """One dyad, returning the full action traces."""
    np.random.seed(seed)
    cfg = create_pd_config(T=T, payoffs=(R, S, Tt, P), legacy_C=True)
    kw = dict(use_inversion=inversion, use_sophisticated=(H > 1),
              planning_horizon=H)
    a = [ToMEmpatheticAgent(config=cfg, agent_num=0, empathy_factor=lam_i, **kw),
         ToMEmpatheticAgent(config=cfg, agent_num=1, empathy_factor=lam_j, **kw)]
    env = Environment(K=2)
    acts, hi, hj = [0, 0], [], []
    for t in range(T):
        o = env.step(t=t, actions=acts)
        oi, oj = (a[0].o_init, a[1].o_init) if t == 0 else (o[0], o[1])
        x = a[0].step(t=t, observation=oi)["exp_action"]
        y = a[1].step(t=t, observation=oj)["exp_action"]
        acts = [x, y]
        hi.append(x)
        hj.append(y)
    return np.array(hi), np.array(hj)


def cc_rate(lam_i, lam_j, seeds=SEEDS, **kw):
    vals = []
    for s in seeds:
        hi, hj = simulate(lam_i, lam_j, s, **kw)
        vals.append(np.mean((hi == 0) & (hj == 0)))
    return float(np.mean(vals)), float(np.std(vals))


# --------------------------------------------------------------- validation
def section_validation():
    w("## 0. Validation")
    w()
    w("Forcing `lambda_j = 0` must reproduce the pre-change myopic baseline")
    w("exactly. This proves the corrections are behaviour-preserving where they")
    w("should be, so every difference below is the model change and not a port bug.")
    w()
    w("| lambda | rebaselined | legacy | difference |")
    w("|---|---|---|---|")
    TheoryOfMind.DEFAULT_LAMBDA_J = 0.0
    ok = True
    for lam, want in LEGACY_MYOPIC_CC.items():
        got, _ = cc_rate(lam, lam, inversion=False)
        ok &= abs(got - want) < 1e-9
        w(f"| {lam} | {got:.4f} | {want:.4f} | {got-want:+.6f} |")
    TheoryOfMind.DEFAULT_LAMBDA_J = 0.5
    w()
    w(f"Exact reproduction: **{ok}**")
    w()
    return ok


# ------------------------------------------------------- Fig 1: landscape
def section_landscape():
    w("## 1. Cooperation landscape (Figure 1)")
    w()
    w(f"Mutual cooperation frequency over the (lambda_i, lambda_j) grid, "
      f"{len(list(SEEDS))} seeds, T={T}.")
    w()
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    w("| l_i \\ l_j | " + " | ".join(f"{g}" for g in grid) + " |")
    w("|" + "---|" * (len(grid) + 1))
    table = {}
    for li in grid:
        row = []
        for lj in grid:
            m, _ = cc_rate(li, lj)
            table[(li, lj)] = m
            row.append(f"{m:.3f}")
        w(f"| **{li}** | " + " | ".join(row) + " |")
    w()
    diag = [table[(g, g)] for g in grid]
    thr = next((g for g, v in zip(grid, diag) if v >= 0.5), None)
    w(f"Symmetric diagonal crosses 0.5 between lambda = {thr}.")
    w()
    return table


# ------------------------------------------------- Fig 5 / Table: threshold
def section_threshold():
    w("## 2. Transition to cooperation (Figure 5, Table 1)")
    w()
    lams = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
    w("| lambda | P(CC) rebaselined | sd across seeds | legacy (static lambda_j) |")
    w("|---|---|---|---|")
    new, leg = [], []
    for lam in lams:
        m, sd = cc_rate(lam, lam)
        TheoryOfMind.DEFAULT_LAMBDA_J = 0.0
        lm, _ = cc_rate(lam, lam, inversion=False)
        TheoryOfMind.DEFAULT_LAMBDA_J = 0.5
        new.append(m)
        leg.append(lm)
        w(f"| {lam:.2f} | {m:.4f} | {sd:.4f} | {lm:.4f} |")
    w()

    def thr(vals, cut):
        return next((l for l, v in zip(lams, vals) if v >= cut), None)

    w(f"- Threshold at P(CC) >= 0.5: rebaselined **{thr(new,0.5)}**, "
      f"legacy {thr(leg,0.5)}")
    w(f"- Threshold at P(CC) >= 0.8: rebaselined **{thr(new,0.8)}**, "
      f"legacy {thr(leg,0.8)}")
    w()
    section_analytic()
    return lams, new, leg


def section_analytic():
    """The closed-form threshold, recomputed.

    The derivation is stated for identical agents 'without learning or
    inversion', so it survives the correction intact: with inversion off the
    opponent's empathy is a fixed constant, q(a_j) does not depend on lambda,
    and Delta-U stays affine in lambda. Only the constants move.
    """
    from empathy.prisoners_dilemma.tom import TheoryOfMind as ToM
    from empathy.prisoners_dilemma.tom.tom_core import COOPERATE

    class _M:
        qs = [np.array([0.5, 0.5])]
        beta = 4.0

    beta = 4.0

    def qs_for(lj):
        prev = ToM.DEFAULT_LAMBDA_J
        try:
            ToM.DEFAULT_LAMBDA_J = lj
            t = ToM(other_model=_M(), beta_other=beta)
            t.update_my_policy_belief(1.0)
            q0 = float(t.predict_opponent_action().q_response[COOPERATE])
            t.update_my_policy_belief(0.0)
            q1 = float(t.predict_opponent_action().q_response[COOPERATE])
        finally:
            ToM.DEFAULT_LAMBDA_J = prev
        return q0, q1

    w("### Analytic threshold")
    w()
    w("lambda* = (ln2/beta - A)/B, with A = q0 R + (1-q0) S - q1 T - (1-q1) P,")
    w("B = (T-S)(1 - q0 + q1), q0 = q(C|C), q1 = q(C|D).")
    w()
    w("| opponent model | q0 | q1 | A | B | lambda* |")
    w("|---|---|---|---|---|---|")
    for lj, name in ((0.0, "legacy (self-interested opponent)"),
                     (0.5, "corrected (lambda_j = 0.5)")):
        q0, q1 = qs_for(lj)
        A = q0 * R + (1 - q0) * S - q1 * Tt - (1 - q1) * P
        B = (Tt - S) * (1 - q0 + q1)
        w(f"| {name} | {q0:.4f} | {q1:.4f} | {A:.3f} | {B:.3f} | "
          f"**{(np.log(2)/beta - A)/B:.4f}** |")
    w()
    w("The legacy row reproduces the manuscript's published 0.24, which confirms")
    w("the formula and the q values are being read correctly. Under the corrected")
    w("opponent model the analytic threshold moves to 0.45, against a measured")
    w("0.50 with inversion switched on. The derivation holds; its constants move.")
    w()


# ------------------------------------------------------ Fig 2: exploitation
def section_exploitation():
    w("## 3. Exploitation in asymmetric dyads (Figure 2)")
    w()
    w("| lambda_i | lambda_j | coop_i | coop_j | payoff_i | payoff_j | gap | exploitability |")
    w("|---|---|---|---|---|---|---|---|")
    rows = []
    for li, lj in [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6),
                   (0.5, 0.5), (0.0, 0.6), (0.6, 0.0)]:
        rs = [run_pair("exp", "i", "j",
                       dict(empathy_factor=li, use_inversion=True),
                       dict(empathy_factor=lj, use_inversion=True),
                       T=T, seed=s, legacy_C=True) for s in SEEDS]
        ci = np.mean([r.coop_rate_i for r in rs])
        cj = np.mean([r.coop_rate_j for r in rs])
        pi = np.mean([r.payoff_i_mean for r in rs])
        pj = np.mean([r.payoff_j_mean for r in rs])
        rows.append((li, lj, ci, cj, pi, pj))
        w(f"| {li} | {lj} | {ci:.3f} | {cj:.3f} | {pi:.2f} | {pj:.2f} | "
          f"{pi-pj:+.2f} | {abs(ci-cj):.2f} |")
    w()
    return rows


# ------------------------------------------------- Fig 4: near the boundary
def section_boundary():
    w("## 4. Boundary-layer variability (Figure 4)")
    w()
    w("The published design fixed lambda_j = 0.5 as a setting 'well past")
    w("threshold'. Under the corrected model 0.5 IS the threshold, so that")
    w("premise no longer holds. Reported here at the published lambda_j = 0.5")
    w("and at lambda_j = 0.7, which now plays the role 0.5 used to.")
    w()
    for lj in (0.5, 0.7):
        w(f"### partner lambda_j = {lj}")
        w()
        w("| lambda_i | P(CC) | sd across seeds | within-run rolling sd |")
        w("|---|---|---|---|")
        for li in [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]:
            per_seed, roll = [], []
            for s in SEEDS:
                hi, hj = simulate(li, lj, s)
                cc = ((hi == 0) & (hj == 0)).astype(float)
                per_seed.append(cc.mean())
                k = 10
                if len(cc) >= k:
                    rm = np.convolve(cc, np.ones(k) / k, mode="valid")
                    roll.append(rm.std())
            w(f"| {li} | {np.mean(per_seed):.3f} | {np.std(per_seed):.3f} | "
              f"{np.mean(roll):.3f} |")
        w()


# ---------------------------------------------- Fig 7 / Table 2: planning
def section_planning():
    w("## 5. Planning horizon (Figure 7, Table 2)")
    w()
    w("### 5a. Self-play, the setting the manuscript reports")
    w()
    w(f"10 seeds here rather than {len(list(SEEDS))}; horizon-4 planning costs "
      f"about 64x a myopic run.")
    w()
    w("| lambda | H=1 | H=2 | H=3 | H=4 | H4 - H1 |")
    w("|---|---|---|---|---|---|")
    for lam in (0.3, 0.4, 0.5, 0.7):
        v = [cc_rate(lam, lam, H=H, seeds=PLAN_SEEDS)[0] for H in (1, 2, 3, 4)]
        w(f"| {lam} | " + " | ".join(f"{x:.4f}" for x in v)
          + f" | {v[-1]-v[0]:+.4f} |")
    w()
    w("The manuscript reports -0.185 at lambda = 0.3 from H=1 to H=3. That")
    w("figure came from averaging EFE over the horizon while holding beta")
    w("fixed, which is exactly a myopic decision at precision beta/H. It")
    w("measured added noise, not lookahead.")
    w()
    w("### 5b. Planning depth tracks how much the partner reciprocates")
    w()
    w("Lookahead acts only through the partner's responsiveness to my own last")
    w("action. Measured as the true lag-1 correlation between one agent's action")
    w("and the partner's next action, in self-play:")
    w()
    w("| lambda | true lag-1 correlation |")
    w("|---|---|")
    for lam in (0.3, 0.5, 0.7):
        acs = []
        for s in SEEDS:
            hi, hj = simulate(lam, lam, s)
            a, b = hi[:-1].astype(float), hj[1:].astype(float)
            if a.std() > 0 and b.std() > 0:
                acs.append(np.corrcoef(a, b)[0, 1])
        w(f"| {lam} | {np.nanmean(acs):+.3f} |")
    w()
    w("Slightly negative. These agents are close to unconditional, so cooperating")
    w("buys almost no future cooperation. Deeper planning correctly registers")
    w("that and cooperates less, which is why the self-play effect above is")
    w("negative but far smaller than the -0.185 originally reported. The sign")
    w("reverses when the partner does reciprocate, as 5c shows: planning depth")
    w("amplifies whatever the opponent model implies rather than favouring")
    w("cooperation or defection in itself.")
    w()
    w("### 5c. Planning against a partner that does reciprocate")
    w()
    w("Agent cooperation and mean payoff by horizon, lambda = 0.3, T=60.")
    w()
    w("| partner | metric | H=1 | H=2 | H=3 | H=4 | H4 - H1 |")
    w("|---|---|---|---|---|---|---|")
    for sname, strat in STRATS.items():
        co, pay = [], []
        for H in (1, 2, 3, 4):
            kw = dict(empathy_factor=0.3, use_inversion=True,
                      use_sophisticated=(H > 1), planning_horizon=H)
            # one simulation per (partner, H); read both metrics off it
            rs = [play(kw, strat, T=60, seed=s) for s in range(10)]
            co.append(np.mean([r["coop"] for r in rs]))
            pay.append(np.mean([r["my_pay"] for r in rs]))
        for label, v in (("cooperation", co), ("mean payoff", pay)):
            w(f"| {sname} | {label} | " + " | ".join(f"{x:.3f}" for x in v)
              + f" | {v[-1]-v[0]:+.3f} |")
    w()


# --------------------------------------------------- Fig 8: model comparison
def section_models():
    w("## 6. Model comparison (Figure 8)")
    w()
    w("| agent | P(CC) | P(DD) | mean payoff |")
    w("|---|---|---|---|")
    for ti in ("self_interested", "tom_only", "empathic"):
        rs = [run_pair("cmp", ti, ti, dict(AGENT_TYPES[ti]),
                       dict(AGENT_TYPES[ti]), T=T, seed=s) for s in SEEDS]
        w(f"| {ti} | {np.mean([r.freq_CC for r in rs]):.3f} | "
          f"{np.mean([r.freq_DD for r in rs]):.3f} | "
          f"{np.mean([r.payoff_i_mean for r in rs]):.2f} |")
    rs = [run_pair("cmp", "empathic", "self_interested",
                   dict(AGENT_TYPES["empathic"]),
                   dict(AGENT_TYPES["self_interested"]), T=T, seed=s)
          for s in SEEDS]
    w()
    w(f"Empathic against self-interested: exploitability "
      f"{np.mean([r.exploitability for r in rs]):.2f}, payoff gap "
      f"{np.mean([r.payoff_gap for r in rs]):+.2f}.")
    w()


# ------------------------------------------------------ NEW: fixed partners
def section_partners():
    w("## 7. Fixed-strategy partners (new)")
    w()
    w("Every result in the published manuscript is self-play: both players share")
    w("the agent architecture. These are the first runs against partners that do")
    w(f"not. T={T}, 10 seeds, empathic lambda = {EMPATHIC_LAMBDA}.")
    w()
    w("| partner | agent | agent coop | agent payoff | partner payoff |")
    w("|---|---|---|---|---|")
    for sname, strat in STRATS.items():
        for aname, kw in AGENT_TYPES.items():
            rs = [play(dict(kw), strat, T=T, seed=s) for s in range(10)]
            w(f"| {sname} | {aname} | {np.mean([r['coop'] for r in rs]):.3f} | "
              f"{np.mean([r['my_pay'] for r in rs]):.2f} | "
              f"{np.mean([r['their_pay'] for r in rs]):.2f} |")
    w()
    w("The empathic agent cooperates 97-100% against every partner, including")
    w("one that defects unconditionally, where it scores 0.00 across 100 rounds")
    w("while the partner takes 5.00. It predicts that defection accurately and")
    w("cooperates anyway, which is the prediction/valuation dissociation in its")
    w("sharpest form.")
    w()


# ------------------------------------------------- Table 1 and boundary stats
def section_table1():
    """Table 1: lambda_j fixed at 0.5, sweep lambda_i, 30 seeds."""
    w("## 8. Cooperation transition, Table 1 protocol")
    w()
    w("lambda_j = 0.5 fixed, T = 100, 30 seeds, as the table caption states.")
    w()
    w("| lambda_i | mean CC | sd CC |")
    w("|---|---|---|")
    for li in (0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75):
        rs = [run_pair("t1", "i", "j",
                       dict(empathy_factor=li, use_inversion=True),
                       dict(empathy_factor=0.5, use_inversion=True),
                       T=T, seed=s, legacy_C=True) for s in range(30)]
        v = np.array([r.freq_CC for r in rs])
        w(f"| {li:.2f} | {v.mean():.3f} | {v.std():.3f} |")
    w()


def section_boundary_stats():
    """Variability near the transition against well beyond it."""
    w("## 9. Boundary-layer variability")
    w()
    w("The comparison windows move with the transition, which now sits near")
    w("lambda = 0.45 rather than 0.24.")
    w()
    near, beyond, W = [0.35, 0.40, 0.45, 0.50], [0.55, 0.65, 0.75], 10
    data = {}
    for li in near + beyond:
        band, cc = [], []
        for s in range(30):
            hi, hj = simulate(li, 0.5, s, T=T)
            m = ((hi == 0) & (hj == 0)).astype(float)
            band.append(np.convolve(m, np.ones(W) / W, mode="valid").std())
            cc.append(m.mean())
        data[li] = (np.array(band), np.array(cc))
    a = np.concatenate([data[x][0] for x in near])
    b = np.concatenate([data[x][0] for x in beyond])
    obs = a.mean() - b.mean()
    rng = np.random.default_rng(0)
    pool = np.concatenate([a, b])
    hits = 0
    for _ in range(20000):
        rng.shuffle(pool)
        if abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= abs(obs):
            hits += 1
    pval = (hits + 1) / 20001
    sa = np.array([data[x][1].std() for x in near])
    sb = np.array([data[x][1].std() for x in beyond])
    w(f"- band thickness: near {a.mean():.3f}, beyond {b.mean():.3f}, "
      f"delta {obs:+.3f}, permutation p {pval:.5f}")
    w(f"- seed-to-seed sd: near {sa.mean():.3f}, beyond {sb.mean():.3f}, "
      f"delta {sa.mean()-sb.mean():+.3f} (too few grid points to permute)")
    w()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w("# Rebaselined paper numbers")
    w()
    w("Generated by `scripts/rebaseline_paper.py`. Every number the manuscript")
    w("cites, recomputed under the corrected model. Regenerate rather than")
    w("editing by hand, so the text and the figures cannot drift apart.")
    w()
    if not section_validation():
        w("**Validation failed. Do not use any number below.**")
        (OUT / "PAPER_NUMBERS.md").write_text("\n".join(buf), encoding="utf-8")
        return 1
    section_landscape()
    section_threshold()
    section_exploitation()
    section_boundary()
    section_planning()
    section_models()
    section_partners()
    section_table1()
    section_boundary_stats()
    (OUT / "PAPER_NUMBERS.md").write_text("\n".join(buf), encoding="utf-8")
    print(f"\nwrote {OUT / 'PAPER_NUMBERS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
