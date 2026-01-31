"""
Base Environment Class for Multi-Agent Active Inference.

This module provides the foundational Environment class that all task-specific
environments inherit from. Follows the standard RL environment interface.

Architecture Overview:
---------------------
Environments handle:
- Generative process (true world dynamics)
- Observation generation with true parameters
- Multi-agent coordination (positions, turns)
- Episode management (reset, step, done)

The environment knows the TRUE parameters (θ_true) and generates observations
accordingly. Agents maintain their own beliefs about these parameters.

Dependencies:
------------
- numpy: Array operations
- Optional: rendering libraries (matplotlib, pygame)
"""

from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    """
    Abstract base class for multi-agent active inference environments.
    
    Provides standard interface for:
    - Episode management (reset, step, is_done)
    - Observation generation
    - Multi-agent coordination
    - Rendering and visualization
    
    Subclasses: CleanUpEnvironment
    
    Subclasses implement reset/step/observation logic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize environment with configuration.
        
        Args:
            config: Environment configuration dictionary
        """
        self.config = config or {}
        
        # Random state for reproducibility
        self.rng = np.random.default_rng(self.config.get("seed", None))
        
        # Episode management
        self.step_count: int = 0
        self.max_steps: int = self.config.get("max_steps", 80)
        self.done: bool = False
        
        # Agent management
        self.n_agents: int = self.config.get("n_agents", 2)
        self.agent_positions: List[Tuple[int, int]] = []
        
        # Hidden state (task-specific, set by subclass)
        self.hidden_state: Any = None
        
        # True spawn probabilities (5 values, one per pollution context 0-4)
        # Set by subclass (e.g., CleanUpEnvironment)
        self.spawn_probs: Optional[np.ndarray] = None
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset environment to initial state for new episode.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            initial_observation: Dictionary with observations for each agent
            
        """
        pass
    
    @abstractmethod
    def step(
        self, 
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute one environment step with given actions.
        
        Args:
            actions: Dictionary mapping agent_id to action index
            
        Returns:
            observations: Dict mapping agent_id to observation
            rewards: Dict mapping agent_id to reward (optional, not used in AIF)
            done: Whether episode is complete
            info: Additional information (e.g., configuration status)
            
        """
        pass
    
    @abstractmethod
    def generate_observation(self, agent_id: str) -> np.ndarray:
        """
        Generate observation for specific agent based on true parameters.
        
        Uses the TRUE observation model with θ_true parameters.
        
        Args:
            agent_id: Identifier for the agent
            
        Returns:
            observation: Agent's observation
        """
        pass
    
    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.done or self.step_count >= self.max_steps
    
    def get_agent_position(self, agent_id: str) -> Tuple[int, int]:
        """
        Get current position of specified agent.
        
        Args:
            agent_id: Agent identifier (e.g., "learner", "expert")
            
        Returns:
            position: (row, col) tuple
        """
        if isinstance(self.agent_positions, dict) and agent_id in self.agent_positions:
            return self.agent_positions[agent_id]
        if isinstance(self.agent_positions, list):
            if isinstance(agent_id, int):
                return self.agent_positions[agent_id]
            if agent_id == "learner" and len(self.agent_positions) > 0:
                return self.agent_positions[0]
            if agent_id == "expert" and len(self.agent_positions) > 1:
                return self.agent_positions[1]
        raise KeyError(f"Unknown agent_id: {agent_id}")
    
    def get_observable_state(self) -> Dict[str, Any]:
        """
        Get all observable state information (positions, etc.).
        
        Returns:
            observable: Dictionary of observable quantities
        """
        return {
            "step": self.step_count,
            "agent_positions": self.agent_positions.copy(),
            "n_agents": self.n_agents,
        }
    
    def get_hidden_state(self) -> Any:
        """
        Get hidden state (for ground truth analysis only).
        
        WARNING: Agents should not have access to this during learning.
        
        Returns:
            hidden_state: Task-specific hidden variable
        """
        return self.hidden_state
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render current environment state.
        
        Args:
            mode: Rendering mode
                - "human": Display to screen
                - "rgb_array": Return RGB array
                
        Returns:
            frame: RGB array if mode="rgb_array", else None
            
        """
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        return None
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        """
        Get observation space specification.
        
        Returns:
            spec: Dictionary describing observation dimensions
        """
        return {}
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """
        Get action space specification.
        
        Returns:
            spec: Dictionary describing valid actions
        """
        return {}
