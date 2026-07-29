# Response to Reviewers — rsif-2026-0208 (draft)

Manuscript: Empathy Modeling in Active Inference Agents
Journal of the Royal Society Interface, decision 15 July 2026, revision due ~5 August 2026.

Status 2026-07-28: APPLIED. The manuscript edits for all 8 Referee 1 points
are live in the Overleaf project (main.tex), wrapped in a blue `\rev{}` macro
for the tracked-changes version (set it to black for the clean copy). A LaTeX
version of this letter is in the project as `response_to_reviewers.tex`.
Referee 2 still pending — the review is in Empathy_Review_2026.pdf attached
to the 15 Jul decision email and needs to be saved locally so it can be read.

Journal requirements checklist for resubmission:
- [ ] This response document (point by point)
- [ ] Clean main manuscript (tex + pdf)
- [ ] Tracked-changes version (coloured highlights are acceptable)
- [ ] Figures as separate files (PNG/TIFF/JPG/EPS), not embedded
- [ ] Data accessibility section: repository + accession for data and code
      (the analysis code and raw results are already public at
      github.com/mahault/empathy-prisonner-dilemma, `scripts/run_referee_analyses.py`
      and `results/referee_analyses/`)
- [ ] ESM if any

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

> Please see attached for my review.

[PENDING — extract Empathy_Review_2026.pdf from the decision email
(15 Jul 2026, manuscriptcentral.com) and respond point by point here.]

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
