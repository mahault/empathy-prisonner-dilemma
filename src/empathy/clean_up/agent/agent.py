"""
Unified Agent for Clean Up Task: Active Inference with Social Learning.

This agent implements the Clean Up task with a unified Agent class that combines
all capabilities (inference, planning, self-learning, ToM, social learning)
with config-driven optional components.

Hidden State Factors:
    - Self beliefs q(s_f):
        - s_{f,self}: position (9 categories)
        - s_{f,world}: (pollution × apples) = 5 × 8 = 40 categories
    - Other beliefs q(s_o):
        - s_{o,self}: expert position (9 categories)
        - s_{o,world}: what expert believes (5 × 8 = 40 categories)

    Total state space per agent: 9 × 40 = 360 states (tractable)

Parameters Being Learned:
    - Dirichlet-Categorical beliefs over spawn outcomes per pollution context
    - For each context (pollution level 0-4):
        - alpha[context] = [alpha_no_spawn, alpha_spawn]
        - p(spawn | context) = alpha_spawn / (alpha_no_spawn + alpha_spawn)
    - Learners start with uniform prior: alpha = [1.0, 1.0] (p = 0.5)
    - Experts have high concentration matching true spawn probabilities

Actions: {UP, DOWN, LEFT, RIGHT, CLEAN, EAT} = 6 discrete
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
from itertools import product

from empathy.clean_up.agent.inference.state import update_state_belief
from empathy.clean_up.agent.planning.efe import compute_batched_efe, compute_batched_efe_policies
from empathy.clean_up.agent.planning.action_selection import select_action_softmax, select_action_softmax_policies
from empathy.clean_up.agent.social.tom import TheoryOfMind
from empathy.clean_up.agent.beliefs import DirichletBeliefs
from empathy.clean_up.agent.social.update import social_dirichlet_update

# =============================================================================
# DEFAULT HYPERPARAMETERS (class constants)
# =============================================================================
# Default hyperparameters (can be overridden via config)
# These constants serve as defaults when config values are not provided

DEFAULT_ETA_0 = 0.1
DEFAULT_T_A = 2.0
DEFAULT_BETA_TOM = 2.0
DEFAULT_N_PARTICLES = 30
DEFAULT_PLANNING_HORIZON = 3  # Default to 3-step planning
DEFAULT_U_0 = 0.05
DEFAULT_KAPPA = 0.05
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_BETA = 1.0
DEFAULT_PERFORMANCE_WINDOW = 20
DEFAULT_RESAMPLE_THRESH = 0.5
DEFAULT_LAMBDA_EPIST = 0.5
DEFAULT_DIFFUSION_NOISE = 0.05
DEFAULT_TOM_MODE = "softmax"
DEFAULT_SELF_LEARNING = True


class Agent:
    """
    Unified Active Inference Agent for Clean Up task.

    This class implements a general Active Inference agent where all agents use
    EFE-based planning. Agent differences (e.g., "expert" vs "learner") are controlled
    purely through configuration (theta values, learning rates, etc.), not through
    different planning mechanisms.

    Configuration Flags:
    -------------------
    - self_learning: Update beliefs from own observations (default: True)
    - use_tom: Particle filter over other's beliefs (default: False)
    - social_enabled: Update beliefs from others (requires ToM, default: False)
    - dirichlet_initial_alpha: Initial concentration for uniform prior (default: 1.0)
    - particle_gamma_shape: Gamma prior shape for particle initialization (default: 2.0)
    - particle_gamma_scale: Gamma prior scale for particle initialization (default: 2.0)
    - learning_rate: Pseudo-count increment per observation (default: 1.0)

    Latent Variables Maintained:
    ----------------------------
    - s_t: Hidden state belief q(s_t), updated via Bayesian inference
    - beliefs_self: Dirichlet beliefs over spawn outcomes, updated via observation
    - beliefs_other: Inferred other's beliefs (particle mixture)
    - w_j: Particle weights for ToM
    - H(w): Weight entropy (ToM uncertainty)
    - u_t: Confidence score (normalized certainty)
    - r_t: Reliability filter (from ToM confidence)
    - τ_t: Trust (accuracy gate based on log-likelihood comparison)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, agent_id: int = 0):
        """
        Initialize unified agent.
        
        Args:
            config: Configuration dictionary
            agent_id: Unique agent identifier (default: 0)
        """
        # Store config as-is (no merging - defaults applied at point of use)
        self.config = config or {}
        self.agent_id = agent_id

        # Clean Up specific configuration
        self.n_positions = 9  # 3×3 grid
        self.n_pollution_cats = 5  # 0, 1, 2, 3, 4
        self.n_apple_configs = 8  # 3-bit bitmask
        self.n_world_states = self.n_pollution_cats * self.n_apple_configs  # 40
        self.n_states = self.n_positions * self.n_world_states  # 360
        
        # Action space
        self.n_actions = 6  # UP, DOWN, LEFT, RIGHT, CLEAN, EAT
        
        # Shape: [N_states] - initialized in _init_task_specific
        self.state_belief: Optional[np.ndarray] = None

        # Dirichlet beliefs over spawn probabilities per context - initialized in _init_task_specific
        self.beliefs_self: Optional[DirichletBeliefs] = None

        # Theory of Mind inference (initialized after task-specific setup)
        self.tom: Optional[TheoryOfMind] = None
        
        # Particle attributes (delegate to self.tom when available)
        self._particle_params: Optional[np.ndarray] = None
        self._particle_weights: Optional[np.ndarray] = None
        self._state_belief_other: Optional[np.ndarray] = None
        
        # Social update statistics
        self._last_social_update_stats = {
            "social_step": np.nan,
            "social_update_norm": np.nan,
            "social_update_dot": np.nan,
            "social_update_diff_norm": np.nan,
            "social_update_clipping_norm": np.nan,
        }
        
        self._transition_matrices: Optional[np.ndarray] = None
        self._last_context: Dict[str, Any] = {}
        self.last_social_metrics: Dict[str, float] = {}
        self._last_efe_values: Optional[np.ndarray] = None
        self._last_observation: Optional[int] = None
        
        # Planning context cache (for reuse within timestep)
        self._cached_planning_context: Optional[Dict[str, Any]] = None
        self._cached_context_hash: Optional[int] = None
        
        # Set default config values if not provided
        if "self_learning" not in self.config:
            self.config["self_learning"] = DEFAULT_SELF_LEARNING
        
        # Determine use_tom from tom_mode
        # If tom_mode is "off", disable ToM; otherwise enable it
        if "use_tom" not in self.config:
            tom_mode = self.config.get("tom_mode", DEFAULT_TOM_MODE)
            if tom_mode == "off":
                self.config["use_tom"] = False
            else:
                self.config["use_tom"] = True
        
        # Performance tracking for accuracy gate
        window_size = int(self.config.get("performance_window", DEFAULT_PERFORMANCE_WINDOW))
        self._performance_window_size = window_size
        self._performance_history: list = []
        self._influence_history: list = []
        self._observation_history: list = []
        self._state_history: list = []
        
        seed = self.config.get("seed", None)
        self.rng = np.random.default_rng(seed)
        
        # Initialize task-specific components
        self._init_task_specific()
        
        # Initialize ToM after task-specific setup (particles may be initialized there)
        if self.config.get("use_tom", False):
            self._init_tom()
        
        # Config overrides are now applied directly via _init_self_parameters()
        # Dirichlet beliefs are initialized with config-specified alpha values

        # Track current position and context
        self.last_agent_pos = None
        self.last_expert_pos = None
    
    def _init_task_specific(self) -> None:
        """Initialize task-specific variables."""
        # Initialize beliefs and parameters
        self.state_belief = np.ones(self.n_states, dtype=float) / self.n_states
        # Note: state_belief_other is now managed by TheoryOfMind.perspective_tracker

        # Initialize Dirichlet beliefs for spawn probabilities
        initial_alpha = self.config.get("dirichlet_initial_alpha", 1.0)
        self.beliefs_self = DirichletBeliefs(n_contexts=5, initial_alpha=initial_alpha)
        
        # Track apples eaten for performance metric (accuracy gate)
        self.apples_eaten_window = []
        self.performance_window_size = self.config.get("performance_window", DEFAULT_PERFORMANCE_WINDOW)
        
        # Initialize particles for ToM (if enabled)
        # This must be done before base class calls _init_tom()
        if self.config.get("use_tom", False):
            self._init_particles()
    
    def _init_particles(self) -> None:
        """Initialize particle filter for Theory of Mind with Dirichlet hypotheses."""
        n_particles = int(self.config.get("n_particles", DEFAULT_N_PARTICLES))
        gamma_shape = self.config.get("particle_gamma_shape", 2.0)
        gamma_scale = self.config.get("particle_gamma_scale", 2.0)

        # Initialize particles as Dirichlet parameters: shape [N_particles, 5 contexts, 2 alphas]
        particles = np.zeros((n_particles, 5, 2), dtype=float)

        # Sample Dirichlet concentrations from Gamma prior for each context
        for context in range(5):
            particles[:, context, 0] = self.rng.gamma(gamma_shape, gamma_scale, size=n_particles)  # alpha_no_spawn
            particles[:, context, 1] = self.rng.gamma(gamma_shape, gamma_scale, size=n_particles)  # alpha_spawn

        self.particle_params = particles
        self.particle_weights = np.ones(n_particles, dtype=float) / n_particles
    
    def _init_tom(self) -> None:
        """
        Initialize Theory of Mind inference.
        
        Called after _init_task_specific() to ensure particles are initialized.
        """
        if self.particle_params is None or self.particle_weights is None:
            raise ValueError(
                "Particles must be initialized in _init_task_specific() or _init_particles() "
                "before ToM can be created."
            )
        
        if self.state_belief is None:
            raise ValueError("state_belief must be initialized before ToM.")

        if self.beliefs_self is None:
            raise ValueError("beliefs_self must be initialized before ToM.")

        n_states = self.state_belief.shape[0]
        
        self.tom = TheoryOfMind(
            config=self.config,
            n_states=n_states,
            particle_params=self.particle_params,
            particle_weights=self.particle_weights,
            observation_likelihood_fn=self.observation_likelihood_matrix,
            get_transition_matrices_fn=self.get_transition_matrices,
            preferred_observations_fn=self.preferred_observations,
            predict_context_fn=self.predict_context_for_action,
            rng=self.rng,
        )
    
    # =========================================================================
    # PLANNING CONTEXT CACHE
    # =========================================================================
    
    def _compute_context_hash(self, context: Optional[Dict[str, Any]]) -> int:
        """
        Compute a hash for the context dict to check cache validity.
        
        Only hashes simple scalar values like position. Array values are
        not hashed (too expensive and beliefs change every timestep anyway).
        
        Args:
            context: Context dictionary
        
        Returns:
            hash: Integer hash for context comparison
        """
        if context is None:
            return 0
        
        # hash only position-related keys (context-dependent observation matrices)
        hashable = []
        for key, value in context.items():
            if isinstance(value, (int, float, str, bool, tuple)):
                hashable.append((key, value))
        return hash(tuple(sorted(hashable)))
    
    def _init_episode_caches(self) -> None:
        """
        Initialize caches at episode start.
        
        Called from reset_episode() to ensure clean cache state.
        """
        self._cached_planning_context = None
        self._cached_context_hash = None
    
    def _invalidate_planning_context(self) -> None:
        """
        Clear the cached planning context so cached transition matrices and context hash are invalidated.
        
        This is used when the perceived context changes (for example, agent position), forcing recomputation
        of planning-related data on the next planning call.
        """
        self._cached_planning_context = None
        self._cached_context_hash = None
    
    # =========================================================================
    # CORE PERCEPTION-ACTION LOOP
    # =========================================================================
    
    def perceive(self, observation: np.ndarray) -> None:
        """
        Update state beliefs by minimizing VFE given observation.
        
        Implements the analytical solution to variational free energy (VFE) minimization:
            q*(s_t) ∝ q(s_{t-1}) · p(o_t | s_t, θ_self)^precision
        
        The VFE is: F = -log p(o|s) + D_KL(q(s) || p(s))
        
        Args:
            observation: Observed value(s) from environment
        
        Mathematical Details (Section 8.3.3):
        ------------------------------------
        The VFE-minimizing update is:
            q*(s_t) ∝ q(s_{t-1}) · p(o_t | s_t, θ)^precision
        
        Where precision is the sensory precision (inverse temperature) parameter.
        This is the closed-form solution for discrete state spaces.
        
        Implemented as Numba kernel with signature:
            @njit(float64[:](float64[:], int64, float64[:], float64))
        """
        if self.state_belief is None or self.beliefs_self is None:
            raise ValueError("Agent state and beliefs must be initialized before perceive().")

        likelihood = self.observation_likelihood(observation, self.state_belief, self.beliefs_self)
        precision = float(self.config.get("sensory_precision", 1.0))
        posterior = update_state_belief(self.state_belief, 0, likelihood, precision=precision)
        self.state_belief = posterior
        
        # Store observation for accuracy gate computation
        self._last_observation = observation
        
        self._store_context_from_observation(observation)
    
    def act(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Select action via Expected Free Energy (EFE) minimization.
        
        Supports both single-step (H=1) and multi-step (H>1) planning:
        
        Single-step (planning_horizon=1):
            p(a) ∝ exp(-β · G(a))
            Where G(a) is one-step EFE.
        
        Multi-step (planning_horizon>1):
            p(π) ∝ exp(-β · G(π))
            Where G(π) is recursive EFE over policy π = [a₀, a₁, ..., a_{H-1}].
            Returns first action a₀ from sampled policy.
        
        The planning_horizon is configured via config["planning_horizon"] (default=1).
        When planning_horizon=1, behavior is identical to original single-step planning.
        
        All agents use EFE-based planning. Agent differences (expert vs learner)
        are controlled through configuration (theta values, learning rates, etc.),
        not through different planning mechanisms.
        
        Args:
            context: Optional context dictionary for observation model conditioning.
        
        Returns:
            action: Selected action index
        """
        # EFE-based action selection
        if self.state_belief is None or self.beliefs_self is None:
            raise ValueError("Agent state and beliefs must be initialized before act().")
        
        if context is None:
            context = self._last_context
        
        # Get planning horizon from config (default=1 for backward compatibility)
        planning_horizon = int(self.config.get("planning_horizon", DEFAULT_PLANNING_HORIZON))
        
        # Check planning context cache for transition matrices
        context_hash = self._compute_context_hash(context)
        if (self._cached_planning_context is not None and
            self._cached_context_hash == context_hash and
            "transition_matrices" in self._cached_planning_context):
            transition_matrices = self._cached_planning_context["transition_matrices"]
        else:
            transition_matrices = self.get_transition_matrices(context)
            # Update cache
            self._cached_planning_context = {
                "transition_matrices": transition_matrices,
            }
            self._cached_context_hash = context_hash
        
        preferred_obs = self.preferred_observations()
        lambda_epist = float(self.config.get("lambda_epist", DEFAULT_LAMBDA_EPIST))
        beta = float(self.config.get("beta", DEFAULT_BETA))
        seed = int(self.rng.integers(0, 2**31 - 1))
        
        n_actions = self.n_actions
        n_states = self.state_belief.shape[0]
        n_obs = preferred_obs.shape[0]
        
        # Build observation matrices for each action (needed for both single and multi-step)
        obs_matrices = np.zeros((n_actions, n_states, n_obs), dtype=float)
        for action in range(n_actions):
            context_a = self.predict_context_for_action(action, context)
            obs_matrices[action] = self.observation_likelihood_matrix(self.beliefs_self, context_a)
        
        if planning_horizon == 1:
            # ============================================================
            # SINGLE-STEP PLANNING (H=1) - Original behavior, backward compatible
            # ============================================================
            # Prepare batches for vectorized EFE computation
            state_beliefs = np.tile(self.state_belief, (n_actions, 1))
            preferred_obs_batch = np.tile(preferred_obs, (n_actions, 1))
            
            efe_values = compute_batched_efe(
                state_beliefs,
                obs_matrices,
                transition_matrices,
                preferred_obs_batch,
                lambda_epist
            )
            
            # Store EFE values for logging
            self._last_efe_values = efe_values.copy()
            
            return select_action_softmax(efe_values, beta, seed)
        else:
            # ============================================================
            # MULTI-STEP PLANNING (H>1) - Recursive EFE over policies
            # ============================================================
            # Generate all policies of length H: each policy is [a₀, a₁, ..., a_{H-1}]
            # Using itertools.product for efficient generation
            policies = np.array(list(product(range(n_actions), repeat=planning_horizon)), dtype=np.int64)
            n_policies = policies.shape[0]
            
            # Compute EFE for each policy using recursive computation
            efe_values = compute_batched_efe_policies(
                policies,
                planning_horizon,
                self.state_belief,
                obs_matrices,
                transition_matrices,
                preferred_obs,
                lambda_epist
            )
            
            # Store EFE values for logging (policies, not single actions)
            self._last_efe_values = efe_values.copy()
            self._last_policies = policies.copy()  # Store policies for debugging
            
            # Select policy via softmax and return first action
            return select_action_softmax_policies(efe_values, policies, beta, seed)
    
    def learn_from_observation(self, observation: object) -> None:
        """
        Update Dirichlet beliefs from orchard observations.

        Learns from SPAWN EVENTS only (not apple presence). Apples persist until
        eaten, so we track state changes to identify actual spawns:
        - New spawn: apple wasn't there before, is there now → update with outcome=True
        - No spawn: apple wasn't there before, still not there → update with outcome=False
        - Eaten/persisting: apple state unchanged or disappeared → no update (not a spawn event)

        This avoids false positives from apples that spawned at different pollution
        levels and prevents triple-counting by only updating for actual state changes.

        Args:
            observation: Current observation
        """
        # Track updates for logging
        updates = []
        
        if not self.config.get("self_learning", DEFAULT_SELF_LEARNING):
            self._last_self_learning_stats = {
                "self_learning_count": 0,
                "self_learning_contexts": [],
                "self_learning_outcomes": [],
                "self_learning_pollution": None,
            }
            return

        if self.beliefs_self is None:
            self._last_self_learning_stats = {
                "self_learning_count": 0,
                "self_learning_contexts": [],
                "self_learning_outcomes": [],
                "self_learning_pollution": None,
            }
            return

        # Parse observation
        obs_value, _agent_pos = self._parse_observation(observation)
        pollution = (obs_value // 81) % 5
        apples = (obs_value // 405) % 8

        learning_rate = self.config.get("learning_rate", 1.0)

        # Get previous apple state (default to current if first observation)
        prev_apples = getattr(self, '_last_apple_state', apples)

        # Learn only from spawn events (state changes from no-apple to apple)
        # and no-spawn events (was empty, still empty)
        for bit in range(3):
            had_apple = (prev_apples & (1 << bit)) != 0
            has_apple = (apples & (1 << bit)) != 0

            if not had_apple:
                # cell was empty - this is an opportunity to observe a spawn
                # outcome=True if apple appeared (spawn), False if still empty (no spawn)
                self.beliefs_self.update(pollution, outcome=has_apple, learning_rate=learning_rate)
                updates.append({"context": pollution, "outcome": has_apple})
            # If had_apple=True: apple was already there (persisting) or got eaten
            # Either way, not a spawn event at current pollution - skip update

        # Store current state for next comparison
        self._last_apple_state = apples
        
        # Store self-learning stats for logging
        self._last_self_learning_stats = {
            "self_learning_count": len(updates),
            "self_learning_contexts": [u["context"] for u in updates],
            "self_learning_outcomes": [u["outcome"] for u in updates],
            "self_learning_pollution": pollution,
        }
    
    # =========================================================================
    # SOCIAL LEARNING INTERFACE
    # =========================================================================
    
    def observe_other_action(self, other_action: int, context: Optional[Dict[str, Any]]) -> None:
        """
        Update ToM particle weights based on observed action from other agent.
        
        Implements Bayesian weight update:
            w_j^new ∝ w_j^old · p(a_other | θ^j, context)
        
        With stochastic ToM likelihood:
            p(a | θ^j, context) ∝ exp(β_ToM · Q^j(a | context))
        
        Args:
            other_action: Action taken by other agent
            context: Observable context information (positions, configs, etc.)
        """
        if not self.config.get("use_tom", False):
            return
        if self.tom is None:
            raise ValueError("ToM must be initialized before observing other actions.")
        
        if context is None:
            context = self._last_context
        else:
            self._last_context = context
        
        # Add transition builder to context so ToM can build per-particle transitions
        context = dict(context) if context else {}  # Make a copy to avoid modifying original
        context["build_transition_matrices"] = self.build_transition_matrices
        
        # Track expert's action for belief simulation
        # Needed to predict belief after action in simulate_expert_belief_update
        context["last_expert_action"] = other_action
        
        expert_position = context.get("agent_pos")
        if expert_position is None:
            raise ValueError("expert_position (agent_pos) is required for perspective-taking ToM.")
        
        # Use TheoryOfMind class to handle all updates
        # The ToM class will set up the full context internally
        debug = bool(self.config.get("debug_tom", False))
        self.tom.observe_action(
            other_action,
            self.state_belief,
            expert_position,
            context,
            debug=debug,
        )
        
        # Update particle attributes
        self.particle_params = self.tom.particle_filter.particle_params
        self.particle_weights = self.tom.particle_filter.particle_weights
    
    def compute_social_metrics(self) -> Dict[str, float]:
        """
        Compute social-learning and trust metrics derived from the agent's ToM particles, beliefs, planning and recent performance history.
        
        Returns:
            dict: Mapping of metric names to floats. If ToM is disabled, returns an empty dict.
        """
        if not self.config.get("use_tom", False):
            return {}
        if self.tom is None:
            raise ValueError("ToM must be initialized before computing social metrics.")
        
        context = self._last_context if isinstance(self._last_context, dict) else {}
        context = {
            **context,
            "predict_context_fn": self.predict_context_for_action,
            "beta": float(self.config.get("beta", DEFAULT_BETA)),
        }
        
        # Use TheoryOfMind class to compute metrics
        obs_history = np.array(self._observation_history, dtype=int)
        state_history = np.array(self._state_history, dtype=int)

        metrics = self.tom.compute_trust_metrics(
            self.beliefs_self,
            self.state_belief,
            obs_history,
            state_history,
            context,
        )
        
        self.last_social_metrics = metrics
        return metrics
    
    def social_learn(self) -> None:
        """
        Apply social parameter update (the core social learning mechanism).
        
        Implements:
            θ_self^new = θ_self^old + η_t · (E[θ_other] - θ_self^old)
        
        Where:
        - η_t = η_0 · r_t · τ_t: Effective social learning rate
        - η_0: Base social learning rate (config: eta_0, default 0.1)
        - r_t: Reliability filter from ToM confidence (particle weight concentration)
        - τ_t: Trust = r_t · τ_accuracy (reliability × accuracy gate)
          - τ_accuracy: Accuracy gate (sigmoid of log-likelihood advantage)
        - E[θ_other] = Σ_j w_j θ^j: Expected other parameters
        """
        if not self.config.get("social_enabled", False):
            return
        if not self.config.get("use_tom", False):
            return
        
        metrics = self.compute_social_metrics()
        if not metrics:
            return
        
        if self.tom is None:
            raise ValueError("ToM must be initialized for social learning.")
        alpha_other_dict = self.tom.get_expected_parameters()

        reliability = metrics.get("reliability", 1.0)
        tau_accuracy = metrics.get("tau_accuracy", 1.0)
        eta_0 = float(self.config.get("eta_0", DEFAULT_ETA_0))

        # Get contexts that were recently observed from state history
        # Extract pollution levels from recent state indices
        contexts_observed = []
        for state_idx in self._state_history[-10:]:  # Last 10 states
            world_state = state_idx // self.n_positions
            pollution = world_state % self.n_pollution_cats
            contexts_observed.append(pollution)
        contexts_observed = list(set(contexts_observed))  # Remove duplicates

        # If no history yet, update all contexts
        if not contexts_observed:
            contexts_observed = list(range(self.n_pollution_cats))

        # Capture beliefs BEFORE social update for delta tracking
        probs_before = [self.beliefs_self.get_probability(c) for c in range(self.n_pollution_cats)]

        # Social update using Dirichlet beliefs
        social_dirichlet_update(
            beliefs_self=self.beliefs_self,
            alpha_other_dict=alpha_other_dict,
            reliability=reliability,
            trust=tau_accuracy,
            eta_0=eta_0,
            contexts_observed=contexts_observed
        )

        # Capture beliefs AFTER social update
        probs_after = [self.beliefs_self.get_probability(c) for c in range(self.n_pollution_cats)]
        social_deltas = [probs_after[c] - probs_before[c] for c in range(self.n_pollution_cats)]

        # Track effective influence for social learning
        effective_influence = eta_0 * reliability * tau_accuracy

        self._last_social_update_stats = {
            "social_step": float(effective_influence),
            "eta_0": float(eta_0),
            "reliability": float(reliability),
            "tau_accuracy": float(tau_accuracy),
            "social_learning_delta": social_deltas,
            "social_learning_contexts_updated": contexts_observed,
        }
        
        # Track performance and observations for accuracy gate
        current_performance = self.get_current_performance()
        
        # Update performance history
        self._performance_history.append(float(current_performance))
        self._influence_history.append(float(effective_influence))
        
        # Update observation history
        if self._last_observation is not None:
            if isinstance(self._last_observation, (tuple, list)) and len(self._last_observation) > 0:
                obs_value = int(self._last_observation[0])
            else:
                obs_value = int(self._last_observation)
            self._observation_history.append(obs_value)
        
        # Update state history
        if self.state_belief is not None:
            self._state_history.append(int(np.argmax(self.state_belief)))
        
        # Trim to window size
        if len(self._performance_history) > self._performance_window_size:
            self._performance_history = self._performance_history[-self._performance_window_size:]
        if len(self._influence_history) > self._performance_window_size:
            self._influence_history = self._influence_history[-self._performance_window_size:]
        if len(self._observation_history) > self._performance_window_size:
            self._observation_history = self._observation_history[-self._performance_window_size:]
        if len(self._state_history) > self._performance_window_size:
            self._state_history = self._state_history[-self._performance_window_size:]
    
    def reset_episode(self, reset_tom: bool = False) -> None:
        """
        Reset per-episode state while keeping learned parameters.
        
        Args:
            reset_tom: If True, reinitialize ToM particles as well.
        """
        # Trust-related state reset (computed dynamically)
        self._last_context = {}
        
        # Reset planning caches
        self._init_episode_caches()
        self.last_social_metrics = {}
        self._last_social_update_stats = {
            "social_step": np.nan,
            "social_update_norm": np.nan,
            "social_update_dot": np.nan,
            "social_update_diff_norm": np.nan,
            "social_update_clipping_norm": np.nan,
        }
        self._transition_matrices = None
        
        # Reset performance tracking for accuracy gate
        self._performance_history = []
        self._influence_history = []
        self._observation_history = []
        self._state_history = []
        
        # Reset per-episode belief state to prevent priors from leaking across episodes
        # Reset state belief to uniform (neutral initial value)
        if self.state_belief is not None:
            self.state_belief = np.ones(self.n_states, dtype=float) / self.n_states
        
        # Clear last observation
        self._last_observation = None
        
        if reset_tom and self.tom is not None:
            # Reinitialize particles (task-specific agents handle this)
            if hasattr(self, "_init_particles"):
                self._init_particles()
                # Update ToM with new particles
                if self.particle_params is not None and self.particle_weights is not None:
                    self.tom.particle_filter.reset(self.particle_params, self.particle_weights)
            self.tom.reset()
        elif self.tom is not None:
            self.tom.perspective_tracker.reset()
        
        self._reset_task_state()
    
    # =========================================================================
    # GETTERS AND UTILITIES
    # =========================================================================
    
    def get_state_belief(self) -> np.ndarray:
        """
        Return current state belief q(s_t).
        
        Returns:
            state_belief: Array of shape [N_states] summing to 1
        """
        return self.state_belief.copy()
    
    def get_parameters(self) -> DirichletBeliefs:
        """
        Return current self beliefs.

        Returns:
            beliefs_self: DirichletBeliefs instance (deep copy)
        """
        return self.beliefs_self.copy()
    
    def get_inferred_other_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return particle representation of inferred other model.
        
        Returns:
            particle_params: Array [N_particles, 5, 2] (Dirichlet params per context)
            particle_weights: Array [N_particles]
        """
        if self.tom is not None:
            return (
                self.tom.particle_filter.particle_params.copy(),
                self.tom.particle_filter.particle_weights.copy()
            )
        # Fallback if ToM not initialized
        if self._particle_params is not None and self._particle_weights is not None:
            return self._particle_params.copy(), self._particle_weights.copy()
        raise ValueError("Particles not initialized.")
    
    def get_expected_other_parameters(self) -> np.ndarray:
        """
        Return expected parameters under q(θ_other).
        
        Returns:
            Dict with 'alphas' array [5, 2] (expected Dirichlet params per context)
        """
        if self.tom is None:
            raise ValueError("ToM must be initialized to get expected parameters.")
        return self.tom.get_expected_parameters()
    
    @property
    def particle_params(self) -> Optional[np.ndarray]:
        """Access particles via property."""
        if self.tom is None:
            return self._particle_params
        return self.tom.particle_filter.particle_params
    
    @particle_params.setter
    def particle_params(self, value: Optional[np.ndarray]) -> None:
        """Set particles via property."""
        self._particle_params = value
        if self.tom is not None:
            self.tom.particle_filter.particle_params = value
    
    @property
    def particle_weights(self) -> Optional[np.ndarray]:
        """Access weights via property."""
        if self.tom is None:
            return self._particle_weights
        return self.tom.particle_filter.particle_weights
    
    @particle_weights.setter
    def particle_weights(self, value: Optional[np.ndarray]) -> None:
        """Set weights via property."""
        self._particle_weights = value
        if self.tom is not None:
            self.tom.particle_filter.particle_weights = value
    
    @property
    def state_belief_other(self) -> Optional[np.ndarray]:
        """Access expert state belief via property."""
        if self.tom is None:
            return self._state_belief_other
        return self.tom.get_state_belief_other()
    
    @state_belief_other.setter
    def state_belief_other(self, value: Optional[np.ndarray]) -> None:
        """Set expert state belief via property."""
        self._state_belief_other = value
        if self.tom is not None and value is not None:
            self.tom.perspective_tracker.state_belief_other = value
    
    # =========================================================================
    # TASK-SPECIFIC METHODS (Clean Up Implementation)
    # =========================================================================
    
    def _reset_task_state(self) -> None:
        """Reset task-specific state for new episode."""
        self.apples_eaten_window = []
        self.last_agent_pos = None
        self.last_expert_pos = None
        # Reset self-learning history to ensure proper initialization
        if hasattr(self, '_last_apple_state'):
            delattr(self, '_last_apple_state')
        if hasattr(self, '_last_pollution_state'):
            delattr(self, '_last_pollution_state')
    
    def observation_likelihood(
        self,
        obs: int,
        state: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute observation likelihood p(o | s, θ) for all states.
        
        Args:
            obs: Observation (can be tuple (obs_value, position) or int)
            state: State belief distribution (not used, but kept for interface compatibility)
            theta: Parameters
        
        Returns:
            likelihood: Probability of observation for each state [n_states]
        """
        # Parse observation to extract obs_value and position
        obs_value, agent_pos = self._parse_observation(obs)
        
        # Use observation likelihood matrix
        # Matrix is [n_states, n_observations]
        obs_matrix = self.observation_likelihood_matrix(theta, {"agent_pos": agent_pos})
        
        # Extract column for this observation
        return obs_matrix[:, obs_value]
    
    def observation_likelihood_matrix(self, beliefs: DirichletBeliefs, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Build observation likelihood matrix A[o, s] using current beliefs.

        The observation model maps hidden states to observations.
        For Clean Up:
        - Position observations are deterministic
        - Pollution observations are deterministic (or slightly noisy)
        - Apple observations are deterministic

        Args:
            beliefs: DirichletBeliefs for spawn probabilities (not used in deterministic obs model)
            context: Optional context (not used for Clean Up)

        Returns:
            A: Observation likelihood matrix [n_states, n_observations]
        """
        _ = (beliefs, context)
        n_obs = 9 * 9 * 5 * 8  # 3,240 observations
        A = np.zeros((self.n_states, n_obs))
        
        # For each state, compute likely observations
        for s_idx in range(self.n_states):
            # Decode state
            position = s_idx % self.n_positions
            world_state = s_idx // self.n_positions
            
            pollution_cat = world_state % self.n_pollution_cats
            apple_config = world_state // self.n_pollution_cats
            
            # Deterministic observation (for now)
            # In Clean Up, observations match state exactly
            # (position, other_position, pollution, apples)
            # We don't know other agent's position from our state, so assume uniform
            
            for other_pos in range(9):
                obs_idx = (
                    position +
                    other_pos * 9 +
                    pollution_cat * 81 +
                    apple_config * 405
                )
                
                # Equal probability for all other positions (we don't model it)
                if self.n_agents == 1:
                    # Solo: other_position is always 0
                    obs_idx = (
                        position +
                        0 * 9 +
                        pollution_cat * 81 +
                        apple_config * 405
                    )
                    A[s_idx, obs_idx] = 1.0
                    break
                else:
                    # Multi-agent: uniform over other positions
                    A[s_idx, obs_idx] = 1.0 / 9.0
        
        return A
    
    def transition_model(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Simple transition model (state dynamics handled by environment).
        
        Args:
            state: Current state distribution
            action: Action taken
        
        Returns:
            state: Next state (unchanged for Clean Up)
        """
        return state.copy()
    
    def preferred_observations(self) -> np.ndarray:
        """
        Preferred observations: eating apples is rewarding.
        
        Returns:
            C: Log-prior over observations (higher = more preferred)
        """
        n_obs = 9 * 9 * 5 * 8
        C = np.zeros(n_obs)
        
        # Prefer observations where we're eating apples
        # This happens when:
        # - We're in orchard (position 6, 7, or 8)
        # - There's an apple at that position
        # We encode a preference for these states
        
        for obs_idx in range(n_obs):
            # Decode observation
            position = obs_idx % 9
            apples = (obs_idx // 405) % 8
            
            # Check if we're on an apple
            if position in [6, 7, 8]:  # Orchard
                orchard_idx = position - 6
                apple_bit = 1 << orchard_idx
                if apples & apple_bit:  # Apple present
                    C[obs_idx] = 2.0  # Prefer being on apples
        
        # Convert to log-probabilities (softmax)
        C = C - np.max(C)  # Numerical stability
        return C
    
    def get_current_performance(self) -> float:
        """
        Return performance metric for accuracy gate.
        
        For Clean Up: average apples eaten in recent window.
        Higher = better (agent is successfully harvesting).
        """
        if len(self.apples_eaten_window) == 0:
            return 0.0
        return float(np.mean(self.apples_eaten_window))

    def get_transition_matrices(
        self, context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Return transition matrices for planning using current beliefs_self.

        For ToM perspective-taking, particles can call build_transition_matrices(beliefs_j)
        directly with their own Dirichlet beliefs.

        Args:
            context: Context (not used for Clean Up)

        Returns:
            B: Transition matrices [n_actions, n_states, n_states]
        """
        # Build transitions using agent's current belief about spawn function
        return self.build_transition_matrices(self.beliefs_self)
    
    def build_transition_matrices(self, beliefs: DirichletBeliefs) -> np.ndarray:
        """
        Build transition matrices B[a, s', s] parameterized by spawn function.

        Args:
            beliefs: DirichletBeliefs for spawn probabilities per pollution level

        Returns:
            B: Transition matrices [n_actions, n_states, n_states]
        """
        B = np.zeros((self.n_actions, self.n_states, self.n_states))

        for a in range(self.n_actions):
            for s_idx in range(self.n_states):
                # Decode state
                position = s_idx % self.n_positions
                world_state = s_idx // self.n_positions
                pollution = world_state % self.n_pollution_cats
                apples = world_state // self.n_pollution_cats

                # Deterministic position transition
                if a == 0:  # UP
                    new_position = self._move_up(position)
                elif a == 1:  # DOWN
                    new_position = self._move_down(position)
                elif a == 2:  # LEFT
                    new_position = self._move_left(position)
                elif a == 3:  # RIGHT
                    new_position = self._move_right(position)
                else:  # CLEAN or EAT (don't move)
                    new_position = position

                # Pollution dynamics (simplified: deterministic approximation)
                # Pollution decreases when agent CLEANs at river (positions 0-2)
                if a == 4 and position < 3:  # CLEAN action at river
                    new_pollution = max(0, pollution - 1)
                else:
                    new_pollution = pollution  # Stays same

                # Apple spawning dynamics (Dirichlet beliefs-based)
                # Use NEW pollution level after action
                spawn_prob = beliefs.get_probability(new_pollution)
                
                # Compute distribution over next apple configurations
                # Each orchard cell (3 bits) independently spawns/stays
                for new_apples in range(self.n_apple_configs):
                    # Compute P(new_apples | apples, spawn_prob)
                    prob = 1.0
                    for bit in range(3):  # 3 orchard cells
                        has_apple = (apples & (1 << bit)) != 0
                        will_have_apple = (new_apples & (1 << bit)) != 0
                        
                        if has_apple:
                            # Apple present: assume it stays (no decay)
                            prob *= 1.0 if will_have_apple else 0.0
                        else:
                            # Apple absent: may spawn
                            prob *= spawn_prob if will_have_apple else (1.0 - spawn_prob)
                    
                    if prob > 0:  # Only add non-zero probabilities
                        new_world = new_pollution + new_apples * self.n_pollution_cats
                        new_s = new_position + new_world * self.n_positions
                        B[a, new_s, s_idx] += prob
        
        return B
    
    def predict_context_for_action(
        self, action: int, current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict next-step context for per-action A matrices.
        
        Default behavior: return current context unchanged.
        """
        return current_context
    
    def _get_transition_matrices(self) -> np.ndarray:
        if self._transition_matrices is not None:
            return self._transition_matrices
        
        n_states = self.state_belief.shape[0]
        matrices = np.zeros((self.n_actions, n_states, n_states), dtype=float)
        for action in range(self.n_actions):
            for s in range(n_states):
                basis = np.zeros(n_states, dtype=float)
                basis[s] = 1.0
                matrices[action, s] = self.transition_model(basis, action)
        self._transition_matrices = matrices
        return matrices
    
    def _store_context_from_observation(self, observation: object) -> None:
        """Hook for task-specific observation context."""
        return
    
    def track_expert_state(
        self,
        expert_position: int,
        expert_action: int,
        context: Dict[str, Any]
    ) -> None:
        """Track expert's state for perspective-taking."""
        self.last_expert_pos = expert_position
    
    def simulate_expert_observation(
        self,
        expert_position: int,
        world_belief: np.ndarray,
        particle_params: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Predicts the observation index the expert would perceive from their position given a belief over world states.
        
        If provided, uses context['world_state_idx'] as the world-state MAP index to avoid recomputing argmax over world_belief; otherwise uses argmax(world_belief). The returned integer encodes (expert_position, learner_position, pollution_category, apple_configuration) in the same observation indexing used by the environment.
        
        Parameters:
            expert_position: Expert's grid position index.
            world_belief: Marginal belief over world states (pollution × apple configurations).
            particle_params: Candidate expert parameters (unused by this simplified simulator).
            context: Optional dict. If it contains 'world_state_idx', that index is used directly.
        
        Returns:
            int: Encoded observation index the expert would observe.
        """
        # OPTIMIZATION: Use pre-computed world state if available (from particle loop)
        # This avoids redundant argmax computation for each particle.
        if context is not None and "world_state_idx" in context:
            world_state_idx = context["world_state_idx"]
        else:
            # Fallback: compute most likely world state (MAP estimate)
            world_state_idx = np.argmax(world_belief)
        
        # Decode world state
        pollution_cat = world_state_idx % self.n_pollution_cats
        apple_config = world_state_idx // self.n_pollution_cats
        
        # Encode expert's observation (they see from their own position)
        # Get learner position from last observation
        learner_pos = self.last_agent_pos if self.last_agent_pos is not None else 0
        
        obs_idx = (
            expert_position +
            learner_pos * 9 +
            pollution_cat * 81 +
            apple_config * 405
        )
        
        return int(obs_idx)
    
    def simulate_expert_belief_update(
        self,
        prev_expert_belief: np.ndarray,
        simulated_observation: int,
        particle_params: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Simulate expert's belief update by minimizing VFE.
        
        Different particles with different spawn beliefs must produce different
        belief states, leading to different action predictions.
        
        Args:
            prev_expert_belief: Expert's prior belief over states
            simulated_observation: What expert observed
            particle_params: Expert's parameters (for building their transition model)
            context: Context with previous action
        
        Returns:
            Updated belief after VFE-minimizing update
        """
        # get expert's previous action from context
        if context is None or "last_expert_action" not in context:
            # no action history, return uniform (first timestep)
            return np.ones(self.n_states) / self.n_states
        
        prev_action = context["last_expert_action"]
        
        # build transition model using expert's spawn beliefs
        # different particles have different beliefs → different transitions
        transition_matrices = self.build_transition_matrices(particle_params)
        
        # Predict belief after action: P(s'|a) = sum_s P(s'|s,a) * P(s)
        predicted_belief = transition_matrices[prev_action].T @ prev_expert_belief
        
        # Observation model (deterministic for Clean Up)
        # P(o|s) = 1 if observation matches state, 0 otherwise
        # Get learner position from context or last known position
        learner_pos = 0
        if context is not None and "agent_pos" in context:
            learner_pos = context["agent_pos"]
        elif self.last_agent_pos is not None:
            learner_pos = self.last_agent_pos
        
        obs_likelihood = np.zeros(self.n_states)
        for s in range(self.n_states):
            # Decode state s to get full state components
            # State encoding: s = position + world_state * n_positions
            position = s % self.n_positions
            world_state = s // self.n_positions
            # Decode world_state to get pollution_cat and apple_config
            pollution_cat = world_state % self.n_pollution_cats
            apple_config = world_state // self.n_pollution_cats
            
            # Encode expected observation using same encoding as simulate_expert_observation
            expected_obs = (
                position +
                learner_pos * 9 +
                pollution_cat * 81 +
                apple_config * 405
            )
            if expected_obs == simulated_observation:
                obs_likelihood[s] = 1.0
        
        # VFE-minimizing update: q*(s) ∝ p(o|s) * q(s)
        posterior = obs_likelihood * predicted_belief
        posterior_sum = posterior.sum()
        
        # Normalize (posterior_sum should always be > 0 if observation is valid)
        return posterior / (posterior_sum + 1e-10)
    
    def _move_up(self, position: int) -> int:
        """Move up (decrease row), clipping at boundary."""
        row, col = divmod(position, 3)
        new_row = max(0, row - 1)
        return new_row * 3 + col
    
    def _move_down(self, position: int) -> int:
        """Move down (increase row), clipping at boundary."""
        row, col = divmod(position, 3)
        new_row = min(2, row + 1)
        return new_row * 3 + col
    
    def _move_left(self, position: int) -> int:
        """Move left (decrease column), clipping at boundary."""
        row, col = divmod(position, 3)
        new_col = max(0, col - 1)
        return row * 3 + new_col
    
    def _move_right(self, position: int) -> int:
        """Move right (increase column), clipping at boundary."""
        row, col = divmod(position, 3)
        new_col = min(2, col + 1)
        return row * 3 + new_col
    
    def _parse_observation(self, observation: object) -> Tuple[int, int]:
        """
        Parse observation to extract obs_value and position.
        
        Args:
            observation: Observation (can be tuple (obs_value, position) or just int)
        
        Returns:
            obs_value: Observation index
            agent_pos: Agent position
        """
        if isinstance(observation, (tuple, list, np.ndarray)) and len(observation) > 1:
            obs_value = int(observation[0])
            agent_pos = int(observation[1])
        else:
            obs_value = int(observation)
            if self.last_agent_pos is None:
                raise ValueError("agent_pos required for Clean Up observation.")
            agent_pos = int(self.last_agent_pos)
        
        self.last_agent_pos = agent_pos
        self._last_context = {"agent_pos": agent_pos}
        return obs_value, agent_pos
    
    def extract_expert_position(self, obs: int) -> int:
        """
        Extract expert position from observation.
        
        Args:
            obs: Observation index
        
        Returns:
            expert_position: Position of expert (0-8)
        """
        # Decode observation
        other_position = (obs // 9) % 9
        return other_position
    
    def extract_world_belief(self) -> np.ndarray:
        """
        Extract learner's world beliefs q(s_{f,world}) from state belief.
        
        Returns:
            world_belief: Marginal belief over world states (pollution × apples)
        """
        world_belief = np.zeros(self.n_world_states)
        
        for s_idx in range(self.n_states):
            world_state = s_idx // self.n_positions
            world_belief[world_state] += self.state_belief[s_idx]
        
        # Normalize
        world_belief = world_belief / (np.sum(world_belief) + 1e-10)
        return world_belief
    
    @property
    def n_agents(self) -> int:
        """Number of agents in environment."""
        return self.config.get("n_agents", 1)


# Task-specific alias for clarity
# Note: All agents use Active Inference. Differences between agents are controlled
# via config (theta values, learning rates, etc.), not through different agent types.
CleanUpAgent = Agent
