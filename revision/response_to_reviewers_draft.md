# Response to Reviewers — rsif-2026-0208 (draft)

Manuscript: Empathy Modeling in Active Inference Agents
Journal of the Royal Society Interface, decision 15 July 2026, revision due ~5 August 2026.

Status 2026-07-29: APPLIED, both referees. Manuscript edits for all 8 Referee 1
points and all 6 Referee 2 points are live in the Overleaf project (main.tex),
wrapped in a blue `\rev{}` macro for the tracked-changes version (set it to
black for the clean copy). A LaTeX version of this letter is in the project as
`response_to_reviewers.tex`. 12 new references added to main.bib. Compiles
with 0 errors and no undefined references.

Journal requirements checklist for resubmission:
- [x] This response document, point by point (`response_to_reviewers.tex` in the
      Overleaf project; still needs compiling to PDF, see note below)
- [ ] Clean main manuscript (tex + pdf) — set `\revhighlightfalse` in the
      preamble and recompile; regenerate the PDF, the previous one predates
      the data accessibility section
- [x] Tracked-changes version (`revision/rsif-2026-0208_tracked_changes.pdf`,
      current as of 2026-07-29 16:38)
- [x] Figures as separate files: all in `images/`, uploaded to Overleaf
- [x] Data accessibility section, added to main.tex with the repository link,
      the reproduction commands, and the pymdp fork pin
- [ ] Ethics / Competing interests / Funding / Authors' contributions:
      Ethics is written (simulation study, no approval needed); the other three
      are visible `[To be completed by all authors]` placeholders in main.tex
      and MUST be filled before submission
- [ ] ESM if any

Note on the response letter PDF: `response_to_reviewers.tex` is a standalone
document, so Overleaf will only build it if it is set as the project's main
document (Menu > Settings > Compiler > Main document). Do that when no coauthor
is editing, compile, download, then set it back to `main.tex`.

---

## Associate Editor

> The referee comments are positive overall, but both agree that substantial
> revisions are needed before the manuscript can be published.

We thank the Editor and both referees. We have made substantial changes:
two new simulation studies (a model comparison dissociating prediction from
valuation, and a planning-horizon analysis across three payoff structures),
a reframing of the introduction around the scope of the empathy construct,
and new discussion sections on second-person neuroscience, the development of
empathic priors, and a roadmap toward richer environments. Point-by-point
responses follow.

---

## Referee 1

### Point 1 — scope of the term "empathy"

> The manuscript uses empathy in a broad way, covering perspective-taking,
> concern for the other's welfare, and social alignment. But the actual
> implementation resides in an other-regarding weighting parameter in the
> expected free energy function, which appears to be closer to prosocial
> valuation or empathic concern than to full empathy. Please discuss.

We agree, and we have narrowed the framing accordingly. The model contains
two separable components, and the revised introduction now states this
explicitly rather than letting "empathy" cover both:

1. a **perspective-taking (cognitive) component**: the structurally matched
   self–other generative model, through which the agent represents the
   other's beliefs and expected outcomes (and, in the learning-enabled
   variant, inverts the other's parameters from behaviour);
2. a **valuational component**: the other-regarding weight λ, which enters
   the social expected free energy as
   G_social = (1 − λ) G_self + λ E[G_other].

λ implements empathic concern (prosocial valuation), not empathy in its
full folk-psychological sense. What our results show is precisely that the
two components are dissociable: the perspective-taking machinery alone does
not produce cooperation (see the new model comparison, Point 8), whereas λ
does. We now say in the introduction that we model empathy's valuational
component, treat perspective-taking as its enabling substrate, and flag
components we do not model (affective sharing, contagion). [Changes: §1
intro, §Discussion.]

### Point 2 — belief accuracy vs social valuation, made central earlier

> This distinction between belief accuracy and social valuation should be
> made central earlier in the paper. At present, the reader may wonder
> whether the results are driven by better opponent prediction, by
> reciprocity, or by the empathy parameter itself.

Done, in two ways. First, the distinction is now introduced in the
introduction as the paper's organizing question (prediction ≠ valuation)
rather than emerging in the results. Second, it is no longer only an
interpretation: the new model comparison (Point 8) tests it directly. An
agent with accurate opponent inversion but λ = 0 cooperates at the same
rate as a purely self-interested agent (CC frequency 0.00 for both; the
difference in mean CC across 25 seeds is +0.001), while λ = 0.6 with the
same inversion machinery yields CC ≈ 1.00. Prediction quality is held
constant; only valuation moves behaviour. [Changes: §1 intro reordered,
new §Results subsection "Model comparison", §Discussion.]

### Point 3 — second-person neuroscience

> The authors make reference to second-person neuroscience and related
> active inference approaches (e.g. Lehmann et al.), but they do not
> elaborate where they see their own findings with regard to this
> literature. Are their findings consistent with the second-person approach?

We have added a discussion paragraph situating the results in this
literature. Our findings are consistent with the second-person claim that
social cognition during interaction differs from spectatorial mindreading:
in our simulations the interaction-level phenomena (behavioural
synchronization in high-empathy dyads, rapid post-defection recovery,
increased variability near the cooperation threshold) are properties of the
coupled dyad, not of either agent's model in isolation. The model
comparison sharpens the connection: an agent that only observes and
predicts (ToM-only) never enters the reciprocal loop that the second-person
literature treats as constitutive of social engagement; the λ-weighted
agent does, because valuing the other's outcomes couples the two agents'
free-energy landscapes. We now also note the limit of the correspondence:
our agents interact through discrete game moves, not through the continuous
mutual perceptual coupling emphasized by Lehmann et al., which the richer
environments in the roadmap (Point 7) would restore. [Changes: §Discussion,
new paragraph.]

### Point 4 — how does empathy as a structural prior develop or get learned?

> How does empathy as a structural prior develop or is learned? Please expand.

In the present model λ is fixed by design, and we now say explicitly that
this is a deliberate simplification: fixing λ is what allows the
dissociation in Points 2 and 8 to be read cleanly. The revised discussion
expands on how λ could become endogenous, on three timescales:

- **Within an interaction**: λ as a control state updated from the inferred
  disposition of the partner. Our exploitability result motivates this: an
  unconditional λ = 0.6 agent paired with a self-interested one is fully
  exploited (exploitability 0.98, payoff gap −4.91 over 100 rounds), so a
  fixed high λ is not evolutionarily or developmentally stable on its own.
  Work in progress in our group implements adaptive λ regulated by the
  agent's own resource/energy state.
- **Across development**: λ as a slowly learned prior shaped by repeated
  interaction outcomes, in the spirit of hierarchical active inference
  (fast states, slow parameters).
- **Across generations**: λ as a trait under selection, where the
  cooperation payoffs documented in the symmetric dyads provide the
  selection gradient.

[Changes: §Discussion, expanded subsection "Fixed versus adaptive empathy".]

### Point 5 — relation of the learning-enabled variant to the main model

> The learning-enabled variant is conceptually important, but its relation
> to the main model could be clearer.

We have clarified this in Methods. The two variants share the identical
generative model, EFE evaluation, and decision rule; the learning-enabled
variant differs in exactly one module, the particle-filter inversion that
estimates the opponent's parameters (cooperation bias, reciprocity
sensitivity, behavioural precision, empathic weighting) from the observed
history, replacing the fixed opponent profile. A new Methods table lists
the variants and which components are active in each, and the model
comparison (Point 8) uses this same on/off structure as its experimental
manipulation, which ties the variant design directly to a result. We also
now define in the text the history feature f(h_t) ∈ {+1, −1, 0} and the
empathy shift function empathy_shift(λ_j, p) = 5λ_j − p − 1, which were
previously only in the code. [Changes: §Methods, new table + definitions.]

### Point 6 — is the planning-depth effect general or payoff-specific?

> Increasing planning depth can reduce cooperation at moderate empathy
> levels: is this a general property or a feature of this specific payoff
> structure and planning implementation?

We ran the analysis the referee is asking for: sophisticated planning at
horizons H ∈ {1, 2, 3, 4}, λ ∈ {0.3, 0.5, 0.7}, under three payoff
matrices, the paper's standard (R,S,T,P) = (3,0,5,1), a weak-temptation
matrix (3,1,4,2), and a high-temptation matrix (5,0,8,1); all satisfy
T > R > P > S and 2R > T + S. The sweep uses the exact protocol of the
paper's Table 2 (sophisticated agent paired with a myopic partner of equal
λ, mutual cooperation frequency, T = 100, 20 seeds per cell), so the
standard-payoff column reproduces Table 2 exactly (0.782/0.657/0.597 at
λ = 0.3, H = 1/2/3) and extends it to H = 4.

The answer is: the erosion is general, but not unconditional.

1. Wherever myopic (H = 1) cooperation is high, deepening the horizon
   erodes it under **all three** matrices (e.g. standard, λ = 0.3:
   0.78 → 0.56 from H = 1 to H = 4). The effect is not an artifact of one
   payoff matrix.
2. The direction can reverse. Under weak temptation at λ = 0.3, myopic
   cooperation has already collapsed (0.17) and deeper planning no longer
   erodes it, rising slightly (0.20 at H = 4). The accurate statement,
   which the revised text now uses, is that deeper planning amplifies
   whatever the payoff–empathy configuration favours: it erodes cooperation
   above the cooperation threshold and leaves it flat or mildly supported
   below.
3. The mechanism is visible in the numbers: the λ = 0.5 weak-temptation
   row coincides with the λ = 0.3 standard row (max per-seed difference of
   one round), because λ and the payoff margins enter the social EFE
   through the same weighted sum, so empathy effectively rescales the
   temptation margin. This explains why the horizon effect is
   payoff-dependent in exactly the way it is.

We also state the planning-implementation caveat the referee implies: these
results are for our sophisticated-inference rollout planner, and other
planners (e.g. tree search with different pruning) could differ. [Changes:
new §Results subsection "Planning horizon and payoff structure", §Methods,
§Discussion.]

### Point 7 — concrete roadmap to richer environments

> The manuscript itself acknowledges the need for richer environments.
> What would be a concrete road map to do this?

The revised discussion replaces the acknowledgment with a three-stage
roadmap, ordered by what each stage adds and what it tests:

1. **Spatial multi-agent grid-worlds** (e.g. the Metta cooperative-games
   environment, for which we maintain an active-inference integration).
   Adds partial observability, resource acquisition, and freely chosen
   interaction partners; tests whether λ-driven cooperation survives when
   defection is not a single labelled action but a policy pattern.
2. **Communicative tasks** (Common Ground style), where cooperation
   requires establishing shared reference. Tests the perspective-taking
   component under genuine information asymmetry, which the PD cannot.
3. **Continuous social environments** with mutual perceptual coupling,
   closing the loop with second-person neuroscience (Point 3): here
   synchronization becomes measurable in the same currency as empirical
   dyadic studies.

Stages 1 and 2 are implementable with the present agent architecture;
stage 3 requires continuous state spaces and is the natural home for
adaptive λ (Point 4). [Changes: §Discussion, subsection "Toward richer
environments".]

### Point 8 — model comparison section

> Would it be possible to add a model comparison section showing that the
> empathic active-inference agent differs from a self-interested
> active-inference agent or a ToM-only active-inference agent?

Yes. The revision adds exactly this comparison. Three agent types share the
full architecture and differ only in λ and whether opponent inversion is
active:

| type | λ | opponent inversion |
|---|---|---|
| self-interested | 0.0 | off |
| ToM-only | 0.0 | on |
| empathic | 0.6 | on |

Symmetric pairings (T = 100 rounds, 25 seeds): self-interested and ToM-only
dyads both end in mutual defection (CC 0.00, DD 0.97 and 0.95); empathic
dyads in mutual cooperation (CC 1.00). The ToM-only vs self-interested CC
difference is +0.001, i.e. accurate opponent prediction alone contributes
nothing to cooperation. Mixed pairings quantify the cost of unconditional
empathy: empathic vs either non-empathic type gives exploitability 0.98 and
a payoff gap of −4.91, while self-interested vs ToM-only is symmetric
(0.02, +0.04). These results turn the paper's central conceptual claim
(Point 2) into a measured dissociation, and the exploitability numbers
connect to the asymmetric-empathy findings already in the paper and to the
adaptive-λ discussion (Point 4). Code and raw results are in the public
repository. [Changes: new §Results subsection "Model comparison", one new
figure or table.]

---

## Referee 2

Six substantive points (the review opens with a long, accurate summary and is
positive overall).

### 1. How specific are the findings to FEP-AI vs deep Bayesian RL?

Not specific. Any architecture separating a predictive model of the partner
from the weighting of the partner's outcomes should reproduce the
dissociation. CIRL (Hadfield-Menell et al. 2016) and empowerment
(Salge & Polani 2017) are the neighbouring cases, both now cited. What active
inference adds is a common currency: partner welfare, own preferences, and
information value are all expected free energy, which is what makes the
epistemic term and λ commensurable in one quantity. A deep Bayesian RL agent
could implement the same trade-off but would reintroduce the exploration
bonus and the other-regarding term as separate design choices.

### 2. Why a bipolar scalar rather than separate self/other weights?

New analysis, `scripts/run_referee2_weights.py`. Analytically, with
G = w_self·G_self + w_other·G_other and softmax precision β,
β(w_self·G_self + w_other·G_other) = β'[(1−λ)G_self + λG_other] where
λ = w_other/(w_self+w_other) and β' = β(w_self+w_other). So on the positive
quadrant the direction sets λ and the magnitude only rescales precision.
Verified in simulation: scaling both weights by c reproduces the bipolar
model at precision cβ **identically, seed for seed**; across a 5×5 grid
cooperation is a function of the ratio.

The 2D space does buy something outside that quadrant, now reported:
negative w_other is spite and is behaviourally distinct from indifference
(spiteful dyads (0.6,−0.4) reach DD on 99.9% of rounds, pure spite (0,−1) on
100%, vs 96.5% for the self-interested agent (1,0)); negative w_self gives
self-abnegation. Neither is reachable with λ ∈ [0,1]. This also connects to
the psychopathy literature the referee cites.

### 3. Biological bases of λ

Added to Discussion with all four suggested references (Prosser et al. 2018
Bayesian psychopathy; Wolf et al. 2015 and Maurer et al. 2022 uncinate
fasciculus; Royo et al. 2025 primate tractography). Made specific rather than
decorative: low λ with intact opponent modelling, exactly the profile the new
model comparison isolates, is a first computational gloss on that phenotype.

### 4. Structure learning and enduring prosocial orientations

Added (Safron et al. 2023 value cores; Christov-Moore et al. 2023, 2025;
Pae 2026). Anchors the response to Referee 1's point 4 and locates our own
limitation: λ is exogenous, so we show what other-regarding valuation does
but not where it comes from.

### 5. Mechanism design / reputation

Addressed in Discussion (Cieśla 2025). Our dyads are anonymous with private
history, so defection costs nothing beyond the partner's response. Stated as
expectation, not result: reputation should lower the empathy threshold by
making defection costly in a way self-interested planning can represent,
which turns it into a testable question of whether the institutional and
empathic fixes are substitutes or complements.

### 6. Could deeper temporal modelling alone give quasi-Kantian dispositions?

Our existing data answers this, negatively for this setting. If patient
self-interest sufficed, longer horizons should favour policies preserving
long non-zero-sum interaction; we find the opposite, and the new payoff sweep
shows the erosion is general. The rollouts compound the temptation margin
rather than the relationship's continuation value. Framed as a scope
condition, not a refutation: our games are finite, anonymous, and without
partner choice. Goekoop & de Kleijn 2021 cited.

---

## Additional changes not requested by referees

- Methods now state the decision payoff matrix explicitly:
  (R,S,T,P) = (3,0,5,1). During this revision we found that the
  configuration files historically listed (3,1,4,2) for state inference and
  realized-payoff logging while decisions were driven by (3,0,5,1); the
  text now reports the decision payoffs and the repository has been
  updated. This also resolves a reader question about the constants in
  empathy_shift.
- f(h_t) and empathy_shift(λ_j, p) = 5λ_j − p − 1 defined in Methods
  (previously code-only).
