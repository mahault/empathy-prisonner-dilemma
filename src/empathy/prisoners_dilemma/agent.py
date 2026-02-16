import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple

from pymdp.agent import Agent
from pymdp.control import sample_action
from pymdp.maths import softmax
from pymdp.utils import dirichlet_like

from empathy.prisoners_dilemma.tom import TheoryOfMind, SocialEFE, OpponentInversion
from empathy.prisoners_dilemma.tom.inversion import ObservationContext, GatedToM
from empathy.prisoners_dilemma.tom.tom_core import softmax as tom_softmax
from empathy.prisoners_dilemma.tom.opponent_simulator import OpponentSimulator
from empathy.prisoners_dilemma.tom.sophisticated_planner import SophisticatedPlanner

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

# Prisoner's Dilemma constants
COOPERATE = 0
DEFECT = 1


class EmpatheticAgent:
    def __init__(self, config: dict, agent_num: int) -> None:
        
        self.agent_num = agent_num
        
        # Pull out config parameters for cleaner access within class
        self.A              = config["A"][self.agent_num]
        self.B              = config["B"][self.agent_num]
        self.C              = config["C"][self.agent_num]
        self.D              = config["D"][self.agent_num]
        self.empathy_factor = config["empathy_factor"][self.agent_num]
        self.K              = config["K"]
        self.actions        = config["actions"]
        self.learn          = config["learn"]
        self.policy_len     = config["policy_len"]
        
        # Get number of observation categories to use in generating observation for t=0
        num_obs_categories = self.A[0][self.agent_num].shape[0]
        
        # Initialize empty containers for EFE and agents
        self.EFE    = np.zeros(self.K)
        self.agents = []
        
        # Initialize agent's observation at t=0 by generation using likelihood (A) and state prior
        sampled_obs = np.random.choice(np.arange(num_obs_categories), p=self.A[0] @ self.D[0])
        assert isinstance(sampled_obs, np.int64), "Sampled observation must be an integer."
        self.o_init = np.array([sampled_obs] * self.K)   # Duplicate for each K agents
        
        # Initialize all K active inference agents in the simulation
        for k in range(self.K):
            if self.learn:
                if config["same_pref"]:
                    self.agents.append(Agent(A=self.A, B=self.B, C=self.C, D=self.D, 
                                         pB=dirichlet_like(self.B), lr_pB=0.5, policy_len=self.policy_len))
                else:
                    self.agents.append(Agent(A=self.A, B=config["B"][k], C=config["C"][k], D=self.D, 
                                         pB=dirichlet_like(self.B), lr_pB=0.5, policy_len=self.policy_len))
                    
            else:
                # self.agents.append(Agent(A=self.A, B=self.B, C=self.C, D=self.D, 
                #                      use_states_info_gain=False,
                #                      use_param_info_gain=False,
                #                      use_utility=True,
                #                      policy_len=1))
                self.agents.append(Agent(A=self.A, B=self.B, C=self.C, D=self.D, policy_len=self.policy_len))
        
        # Get the policy list and number of policies for use later
        self.policies     = self.agents[0].policies
        self.num_policies = len(self.policies)
        self.num_actions  = [len(config["actions"])]
        
        # Initialize previous variational state posterior with state prior
        self.qs_prev = None
            
    def step(self, t: int, o: "np.ndarray") -> list:
        """ 
        Each agent step consists of the following:
        1. Perform theory of mind (ToM) by running K copies of the agent.
        2. Determine agent's overall EFE by using as weighted average of its own EFE
           and all other agent's EFE.
        3. [TODO] Use weighted VFE to determine variational state posterior.
        3. Use weighted EFE to determine variational policy posterior, action marginal,
           and chosen action.
        4. [TODO] Determine emotion state of agent from weighted VFE and EFE
        """
        if t == 0:
            o = self.o_init
        
        # Create empty container for storing step results
        step_results = {}
        
        # Construct ToM results and add to overall step results
        step_results["tom_results"] = self._theory_of_mind(o=o, qs_prev=self.qs_prev, t=t)
        
        # Store previous qs so it can be accessed later for learning
        self.qs_prev = step_results["tom_results"]
        
        # Calculate agent's expected EFE
        EFE_arr = self._extract_tom_EFE(tom_results=step_results["tom_results"])
        exp_EFE = self._expected_value_EFE(EFE_arr=EFE_arr)
        
        # Calculate agent's expected Q_pi and action
        exp_q_pi = softmax(exp_EFE)
        p_u = sample_action(
                q_pi=exp_q_pi, policies=self.policies, num_controls= self.num_actions)
        exp_action = p_u[0]
        # LOGGER.info(">>>>>>>>>>>>>>>>")
        # LOGGER.info(f"Weighted EFE: {exp_EFE}")
        # LOGGER.info(f"Actual action: {int(exp_action)}")
        # LOGGER.info("===============")
        
        # TODO: Calculate emotion state for the agent
        
        # Assemble final step results
        step_results["qs"]         = step_results["tom_results"][0]["qs"]
        step_results["exp_G"]      = exp_EFE
        step_results["exp_q_pi"]   = exp_q_pi
        step_results["exp_p_u"]    = p_u
        step_results["exp_action"] = exp_action
        
        return step_results
    
    def _theory_of_mind(self, o: "np.ndarray", qs_prev: list, t: int) -> dict:
        """ 
        Run K copies of the agent loop by inferring states, policies, and then sampling
        actions.
        """
        
        tom_results = []
        
        # Theory of mind simulation for self (agent k = 0) and others (agent k > 0)
        for k in range(self.K):
            self.agents[k].infer_states([int(o[k])])   # Each agent gets the same observation

            if self.learn:
                self._learn(o=o, t=t, k=k, qs_prev=qs_prev)
            
            self.agents[k].infer_policies()
            self.agents[k].sample_action()

            #empirical_prior = self._empirical_prior(k=k)
            
            # LOGGER.info(f"Simulation index: {k}")
            # LOGGER.info(f"Empirical_prior: {empirical_prior}")
            # LOGGER.info(f"EFE: {self.agents[k].G}")
            # LOGGER.info(f"Action: {int(self.agents[k].action.flatten()[0])}")
            
            # Add results of simulation to dictionary
            # TODO: Add variational free energy to step results
            tom_results.append({
                "qs"     : self.agents[k].qs,
                "G"      : self.agents[k].G,
                "q_pi"   : self.agents[k].q_pi,
                "action" : self.agents[k].action
            })
            
        return tom_results
    
    def _learn(self, o:"np.ndarray", t:int, k: int, qs_prev: "np.ndarray"):
        if t > 0:
            if k == self.agent_num:
                # If this is the agents model
                self.agents[k].update_B(qs_prev[k]["qs"])
            else:
                # If this is a ToM agent, update the B matrix with the action the real agent took
                self.agents[k].action = self.infer_others_action(o, k)
                #print(f"From state {qs_prev[k]["qs"]} with action {self.agents[k].action} to state {self.agents[k].qs}")
                self.agents[k].update_B(qs_prev[k]["qs"])
        else:
            pass

    def infer_others_action(self, o:"np.ndarray", k: int):
        """ 
        Infers the action of the other agents based on the current state and this agents previous action.
        This is used to update the B matrix of the ToM agents.
        """
        # Get the action of the agent at time t-1
        # Brute forced for first experiment
        if k == 1:
            if o[self.agent_num] == 0 or o[self.agent_num] == 2:
                action = np.array([0.])  # Cooperate
            else:
                action = np.array([1.])
        else:
            if o[self.agent_num] == 0 or o[self.agent_num] == 1:
                action = np.array([0.])  # Cooperate
            else:
                action = np.array([1.])
        return action
    
    def _empirical_prior(self, k: int):
        return np.einsum("ji, i -> j", 
                        self.B[0][:, :, int(self.agents[k].action.flatten()[0])], 
                        self.agents[k].qs[0].flatten())
    
    def _extract_tom_EFE(self, tom_results: list) -> "np.ndarray":
        """ Extracts all EFE calculations for ToM agents """
        
        EFE_arr = np.zeros((self.K, self.num_policies))
        for k in range(self.K):
            EFE_arr[k] = tom_results[k]["G"]
            
        return EFE_arr

    def _expected_value_EFE(self, EFE_arr: 'np.ndarray') -> "np.ndarray":
        """ 
        Computes the expected EFE over policies by weighting the EFE of all simulated agents 
        
        For example, if there are three agents, there would be three EFEs for a policy.
        These EFEs are weighted according to the self.empathy_factor
        
        EFE    = [EFE_0, EFE_1, EFE_2]   (EFEs for each agent under policy 1)
        p(EFE) = [0.9  , 0.05 , 0.05]    (Empathy factor weighting for each EFE)
        
        When the expected value, sum_k EFE_k * p(EFE_k), is taken, we get a model average of EFE
        for that particular policy. This method applies this across the entire policy space to 
        give an expected EFE across policies.
        """

        exp_EFE = np.zeros(self.num_policies) 
        
        # Loop over each agent and then calculate the weighted average of EFE For each
        # policy.
        
        for p in range(self.num_policies):
            exp_EFE[p] = EFE_arr[:, p] @ self.empathy_factor    
        return exp_EFE
    
    def _VFE(self):
        # Expected value of VFE for empathetic agent
        # - Sum_i VFE * empathy_factor
        raise NotImplementedError

    def _emotion_state(self):
        # EFE/VFE -> emotion state
        raise NotImplementedError


class ToMEmpatheticAgent:
    """
    Empathetic agent with proper Theory of Mind.

    Replaces K-copy ensemble averaging with explicit ToM best-response prediction.

    Key differences from EmpatheticAgent:
    - Uses single self_agent and other_model instead of K copies
    - empathy_factor is a scalar λ ∈ [0, 1] instead of a weight vector
    - Computes social EFE: G_social = (1-λ) * G_self + λ * E[G_other]
    - Includes particle-based opponent inference with reliability gating
    """

    def __init__(
        self,
        config: dict,
        agent_num: int,
        empathy_factor: float = 0.5,
        beta_self: float = 4.0,
        beta_other: float = 4.0,
        use_inversion: bool = True,
        n_particles: int = 30,
        reliability_threshold: float = 0.5,
        use_sophisticated: bool = False,
        planning_horizon: int = 3,
    ) -> None:
        """
        Initialize ToM-based empathetic agent.

        Args:
            config: Configuration dictionary with A, B, C, D matrices
            agent_num: Index of this agent (0 or 1)
            empathy_factor: λ ∈ [0, 1], weight on opponent's EFE
            beta_self: My action precision (inverse temperature)
            beta_other: Opponent's action precision (for ToM prediction)
            use_inversion: Whether to use particle-based opponent inference
            n_particles: Number of particles for opponent inference
            reliability_threshold: Threshold for trusting ToM predictions
            use_sophisticated: Whether to use multi-step sophisticated planning
            planning_horizon: Number of steps to plan ahead (H) when sophisticated
        """
        self.agent_num = agent_num
        self.empathy_factor = empathy_factor
        self.beta_self = beta_self
        self.beta_other = beta_other
        self.use_inversion = use_inversion
        self.use_sophisticated = use_sophisticated
        self.planning_horizon = planning_horizon

        # Extract matrices from config
        self.A = config["A"][agent_num]
        self.B = config["B"][agent_num]
        self.C = config["C"][agent_num]
        self.D = config["D"][agent_num]
        self.policy_len = config.get("policy_len", 1)
        self.learn = config.get("learn", False)

        # Initialize self agent (my own generative model)
        self.self_agent = Agent(
            A=self.A, B=self.B, C=self.C, D=self.D,
            policy_len=self.policy_len
        )

        # Initialize other_model (model of opponent)
        # Use opponent's index to potentially get different priors
        other_idx = 1 - agent_num
        A_other = config["A"][other_idx] if len(config["A"]) > 1 else self.A
        B_other = config["B"][other_idx] if len(config["B"]) > 1 else self.B
        C_other = config["C"][other_idx] if len(config["C"]) > 1 else self.C
        D_other = config["D"][other_idx] if len(config["D"]) > 1 else self.D

        self.other_model = Agent(
            A=A_other, B=B_other, C=C_other, D=D_other,
            policy_len=self.policy_len
        )

        # Initialize Theory of Mind module
        self.tom = TheoryOfMind(
            other_model=self.other_model,
            beta_other=beta_other,
            use_pragmatic_value=True,
            use_epistemic_value=False,
        )

        # Initialize Social EFE computer (inversion added after it's created below)
        self.social_efe = SocialEFE(
            tom=self.tom,
            empathy_factor=empathy_factor,
            beta_self=beta_self,
        )
        # Note: inversion reference will be set after inversion initialization

        # Initialize opponent inversion (particle-based)
        if use_inversion:
            self.inversion = OpponentInversion(
                n_particles=n_particles,
                reliability_threshold=reliability_threshold,
            )
            self.gated_tom = GatedToM(
                tom=self.tom,
                inversion=self.inversion,
            )
            # Wire inversion into SocialEFE for epistemic value computation
            self.social_efe.inversion = self.inversion
        else:
            self.inversion = None
            self.gated_tom = None

        # Track history for context
        self.action_history: list = []
        self.observation_history: list = []
        self.my_cumulative_payoff: float = 0.0
        self.other_cumulative_payoff: float = 0.0

        # Initialize observation
        num_obs_categories = self.A[0][agent_num].shape[0]
        sampled_obs = np.random.choice(
            np.arange(num_obs_categories),
            p=self.A[0] @ self.D[0]
        )
        self.o_init = int(sampled_obs)

        # Track previous state belief for learning
        self.qs_prev = None
        self.last_action: Optional[int] = None
        self.last_opponent_action: Optional[int] = None

    def step(self, t: int, observation: int) -> Dict[str, Any]:
        """
        Execute one agent step using ToM-based action selection.

        Args:
            t: Current timestep
            observation: Joint outcome observation (CC=0, CD=1, DC=2, DD=3)

        Returns:
            Dictionary with step results
        """
        if t == 0:
            observation = self.o_init

        # Store observation
        self.observation_history.append(observation)

        # 1. Extract opponent's action from observation (for inversion)
        opponent_action = self._extract_opponent_action(observation, t)

        # 2. Update opponent inversion (if enabled)
        if self.use_inversion and t > 0 and opponent_action is not None:
            context = self._build_observation_context(t)
            self.inversion.update(opponent_action, context)
            self.last_opponent_action = opponent_action

        # 3. Infer my own state beliefs
        self.self_agent.infer_states([observation])

        # 4. Build observation context for opponent prediction
        context = self._build_observation_context(t) if t > 0 else None

        # 5. Update opponent's belief about my policy from history
        if len(self.action_history) > 0:
            coop_rate = sum(1 for a in self.action_history if a == COOPERATE) / len(self.action_history)
            self.tom.update_my_policy_belief(coop_rate)
            # Also update inversion's cooperation rate for empathy feature
            if self.use_inversion and self.inversion is not None:
                self.inversion.my_cooperation_rate = coop_rate

        # 6. Compute action distribution (myopic or sophisticated)
        my_beliefs = None
        if self.self_agent.qs is not None and len(self.self_agent.qs) > 0:
            my_beliefs = self.self_agent.qs[0]

        if self.use_sophisticated:
            # Sophisticated: multi-step rollout planner
            opponent_sim = OpponentSimulator(
                tom=self.tom,
                gated_tom=self.gated_tom if self.use_inversion else None,
                context=context,
            )
            planner = SophisticatedPlanner(
                opponent_sim=opponent_sim,
                empathy_factor=self.empathy_factor,
                horizon=self.planning_horizon,
                beta_self=self.beta_self,
            )
            q_action, best_policy, plan_info = planner.plan(my_beliefs)
            G_social = plan_info["G_policies"]
            info = plan_info
        else:
            # Myopic: single-step social EFE
            # Opponent prediction is history-conditioned q(a_j|h_t),
            # same for all candidate actions (simultaneous move)
            q_response_override = None
            if self.use_inversion and t > 0:
                q_response_override = self.gated_tom.predict_opponent_action(context)
            G_social, info = self.social_efe.compute_all_actions(
                my_beliefs=my_beliefs,
                q_response_override=q_response_override,
                context=context,
            )
            q_action = tom_softmax(-G_social, temperature=1.0/self.beta_self)

        # 6. Sample action from distribution
        action = int(np.random.choice([COOPERATE, DEFECT], p=q_action))

        # 7. Store for next step
        self.action_history.append(action)
        self.last_action = action
        self.qs_prev = self.self_agent.qs

        # 8. Update cumulative payoffs
        if t > 0 and observation is not None:
            my_payoff, other_payoff = self._get_payoffs_from_observation(observation)
            self.my_cumulative_payoff += my_payoff
            self.other_cumulative_payoff += other_payoff

        # Build results
        step_results = {
            "qs": self.self_agent.qs,
            "G_social": G_social,
            "q_action": q_action,
            "exp_action": action,
            "info": info,
        }

        # Add inversion info if enabled
        if self.use_inversion:
            step_results["inversion"] = {
                "reliability": self.inversion.reliability(),
                "profile_summary": self.inversion.get_profile_summary(),
                "mean_profile": self.inversion.get_mean_profile(),
                "lambda_j_belief": self.inversion.get_lambda_j_posterior(),
            }

        return step_results

    def _extract_opponent_action(self, observation: int, t: int) -> Optional[int]:
        """
        Extract opponent's action from joint observation.

        Observation encoding (from agent 0's perspective):
        - CC=0: both cooperated → opponent cooperated (0)
        - CD=1: I cooperated, they defected → opponent defected (1)
        - DC=2: I defected, they cooperated → opponent cooperated (0)
        - DD=3: both defected → opponent defected (1)
        """
        if t == 0 or observation is None:
            return None

        if self.agent_num == 0:
            # Agent 0's perspective
            if observation in [0, 2]:  # CC or DC
                return COOPERATE
            else:  # CD or DD
                return DEFECT
        else:
            # Agent 1's perspective (observations are from agent 0's view)
            if observation in [0, 1]:  # CC or CD
                return COOPERATE
            else:  # DC or DD
                return DEFECT

    def _build_observation_context(self, t: int) -> ObservationContext:
        """Build context for opponent inversion update."""
        return ObservationContext(
            my_last_action=self.last_action,
            their_last_action=self.last_opponent_action,
            joint_outcome=self.observation_history[-1] if self.observation_history else None,
            round_number=t,
            my_cumulative_payoff=self.my_cumulative_payoff,
            their_cumulative_payoff=self.other_cumulative_payoff,
        )

    def _get_payoffs_from_observation(self, observation: int) -> Tuple[float, float]:
        """
        Get payoffs from observation.

        Standard PD payoffs:
        - CC: (3, 3) - mutual cooperation
        - CD: (0, 5) - sucker's payoff / temptation
        - DC: (5, 0) - temptation / sucker's payoff
        - DD: (1, 1) - mutual defection
        """
        payoff_map = {
            0: (3, 3),  # CC
            1: (0, 5),  # CD
            2: (5, 0),  # DC
            3: (1, 1),  # DD
        }
        my_payoff, other_payoff = payoff_map.get(observation, (0, 0))

        # Swap if we're agent 1
        if self.agent_num == 1:
            my_payoff, other_payoff = other_payoff, my_payoff

        return my_payoff, other_payoff

    def reset(self):
        """Reset agent state for new episode."""
        self.action_history = []
        self.observation_history = []
        self.my_cumulative_payoff = 0.0
        self.other_cumulative_payoff = 0.0
        self.qs_prev = None
        self.last_action = None
        self.last_opponent_action = None

        if self.use_inversion:
            self.inversion.reset()

    def get_reliability(self) -> float:
        """Get current ToM reliability score."""
        if self.use_inversion:
            return self.inversion.reliability()
        return 1.0  # Assume reliable if no inversion

    def get_opponent_profile(self) -> Dict:
        """Get current inferred opponent behavioral profile."""
        if self.use_inversion:
            return self.inversion.get_profile_summary()
        return {}
            