"""
Clean Up Environment: 3×3 Grid with Causal Discovery

This environment implements the primary experimental task. Agents must discover a hidden causal rule: cleaning the river increases apple spawning rate in the orchard.

Grid Layout (0-indexed):
    [River  ][River  ][River  ]  ← Cells 0-2: pollution accumulates
    [Neutral][Neutral][Neutral]  ← Cells 3-5: nothing special
    [Orchard][Orchard][Orchard]  ← Cells 6-8: apples can spawn

Core Causal Rule:
    Apple spawning rate ∝ (1 - pollution/4)
    Hidden dependency: clean river → apples spawn

State Variables:
    - Agent positions: p_0, p_1 ∈ {0..8}
    - River pollution: [0.0, 4.0] (0=clean, 4=dirty)
    - Apple presence: {0..7} (3-bit bitmask for cells 6,7,8)

Dynamics:
    - Pollution increases: pollution_{t+1} = min(4, pollution_t + 0.15)
    - CLEAN reduces: pollution_{t+1} = max(0, pollution_t - 0.8)
    - Apple spawning (stochastic): p(spawn) = 0.25 * (1 - pollution/4)
    - Apple consumption: EAT removes apple, +1 reward

Actions: {UP, DOWN, LEFT, RIGHT, CLEAN, EAT} = 6 discrete actions
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
from empathy.clean_up.environment.base import Environment


class CleanUpEnvironment(Environment):
    """
    Clean Up Environment with causal discovery task.

    The agent must discover that cleaning the river (reducing pollution)
    increases the rate at which apples spawn in the orchard.
    """

    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    CLEAN = 4
    EAT = 5

    # Grid constants
    GRID_SIZE = 3
    RIVER_CELLS = [0, 1, 2]
    NEUTRAL_CELLS = [3, 4, 5]
    ORCHARD_CELLS = [6, 7, 8]

    def __init__(
        self,
        n_agents: int = 2,
        grid_size: int = 3,
        pollution_rate: float = 0.15,
        clean_power: float = 0.8,
        spawn_rate: float = 0.25,
        apple_obs_mode: str = "full",  # "full" or "local"
        pollution_obs_mode: str = "full",  # "full" or "local"
        seed: Optional[int] = None,
        max_steps: int = 80,
        spawn_probs: Optional[np.ndarray] = None,
    ):
        """
        Initialize Clean Up environment.

        Args:
            n_agents: Number of agents (1 or 2)
            grid_size: Grid size (fixed at 3 for this task)
            pollution_rate: Rate at which pollution accumulates per step
            clean_power: Amount pollution is reduced when CLEAN action taken
            spawn_rate: Base apple spawning rate (max when pollution = 0)
            apple_obs_mode: "full" or "local" apple observability
            pollution_obs_mode: "full" or "local" pollution observability
            seed: Random seed for reproducibility
            spawn_probs: Array of 5 spawn probabilities, one per pollution context (0-4)
        """
        assert grid_size == 3, "Clean Up task requires 3x3 grid"
        assert n_agents in [1, 2], "Clean Up task supports 1 or 2 agents"

        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.max_steps = max_steps

        # Dynamics parameters
        self.pollution_rate = pollution_rate
        self.clean_power = clean_power
        self.spawn_rate = spawn_rate
        
        # True spawn probabilities (from config or computed from spawn_rate)
        # spawn_probs is an array of 5 probabilities, one per pollution context (0-4)
        if spawn_probs is not None:
            spawn_probs_input = np.array(spawn_probs, dtype=float)
            if len(spawn_probs_input) != 5:
                raise ValueError(
                    f"spawn_probs must be 5 spawn probabilities (one per pollution context 0-4), "
                    f"got {len(spawn_probs_input)} values."
                )
            self.spawn_probs = spawn_probs_input
        else:
            # Convert spawn_rate to spawn_probs format
            # spawn_rate * (1 - pollution/4) for each context
            self.spawn_probs = np.array([
                spawn_rate * (1.0 - ctx / 4.0)
                for ctx in range(5)
            ], dtype=float)

        # Observability modes (for Phase 4 - partial observability)
        self.apple_obs_mode = apple_obs_mode
        self.pollution_obs_mode = pollution_obs_mode

        # Random number generator
        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

        # State variables
        # Use "learner" and "expert" keys to match other environments
        self.agent_positions: Dict[str, int] = {}
        self.pollution_level = 0.0  # Continuous float in [0, 4]
        self.apple_bitmask = 0  # 3 bits for cells 6, 7, 8

        # Episode tracking
        self.timestep = 0
        self.done = False

        # Initialize state
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Tuple[int, int]]:
        """
        Reset environment to initial state.

        Args:
            seed: Optional random seed

        Returns:
            observations: Dict mapping agent_id ("learner", "expert") to (obs_index, position) tuples
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Initialize agent positions with string keys
        self.agent_positions = {
            "learner": int(self.rng.randint(0, self.n_cells))
        }
        if self.n_agents == 2:
            self.agent_positions["expert"] = int(self.rng.randint(0, self.n_cells))

        # Start with moderate pollution
        self.pollution_level = 2.0

        # Start with no apples
        self.apple_bitmask = 0

        # Reset episode tracking
        self.timestep = 0
        self.done = False

        # Return initial observations as (obs, position) tuples
        observations = {
            agent_id: (self.generate_observation(agent_id), self.agent_positions[agent_id])
            for agent_id in self.agent_positions
        }

        # In solo mode, duplicate learner observation as expert (for runner compatibility)
        if self.n_agents == 1:
            observations["expert"] = observations["learner"]

        return observations

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.

        Args:
            actions: Dict mapping agent_id ("learner", "expert") to action index

        Returns:
            observations: Dict mapping agent_id to (obs_index, position) tuples
            rewards: Dict mapping agent_id to reward
            done: Whether episode is complete
            info: Additional information dict
        """
        rewards = {agent_id: 0.0 for agent_id in self.agent_positions}

        # 1. Movement actions
        for agent_id in self.agent_positions:
            action = actions.get(agent_id, self.UP)  # Default to UP if missing

            if action == self.UP:
                new_pos = self._move_up(self.agent_positions[agent_id])
                self.agent_positions[agent_id] = new_pos

            elif action == self.DOWN:
                new_pos = self._move_down(self.agent_positions[agent_id])
                self.agent_positions[agent_id] = new_pos

            elif action == self.LEFT:
                new_pos = self._move_left(self.agent_positions[agent_id])
                self.agent_positions[agent_id] = new_pos

            elif action == self.RIGHT:
                new_pos = self._move_right(self.agent_positions[agent_id])
                self.agent_positions[agent_id] = new_pos

        # 2. CLEAN and EAT actions (execute after movement)
        for agent_id in self.agent_positions:
            action = actions.get(agent_id, self.UP)
            pos = self.agent_positions[agent_id]

            if action == self.CLEAN:
                # CLEAN only works on river cells
                if pos in self.RIVER_CELLS:
                    self.pollution_level = max(0.0, self.pollution_level - self.clean_power)

            elif action == self.EAT:
                # EAT only works on orchard cells with apples
                if pos in self.ORCHARD_CELLS:
                    orchard_idx = pos - 6  # 0, 1, or 2
                    apple_bit = 1 << orchard_idx

                    if self.apple_bitmask & apple_bit:  # Apple present
                        self.apple_bitmask &= ~apple_bit  # Remove apple
                        rewards[agent_id] += 1.0  # Reward for eating apple

        # 3. Pollution dynamics (accumulates over time)
        self.pollution_level = min(4.0, self.pollution_level + self.pollution_rate)

        # 4. Stochastic apple spawning (THE CAUSAL RULE)
        # Spawn rate depends on pollution: cleaner river → more apples
        # Get spawn probability from spawn_probs array based on pollution context
        pollution_context = int(min(4, max(0, self.pollution_level)))  # Bin to 0-4
        spawn_prob = self.spawn_probs[pollution_context]
        spawn_prob = max(0.0, min(1.0, spawn_prob))  # Clip to [0, 1]
        
        for orchard_idx in range(3):  # Cells 6, 7, 8
            apple_bit = 1 << orchard_idx

            # Only spawn if no apple currently present
            if not (self.apple_bitmask & apple_bit):
                if self.rng.rand() < spawn_prob:
                    self.apple_bitmask |= apple_bit  # Spawn apple

        # 5. Generate observations as (obs, position) tuples
        observations = {
            agent_id: (self.generate_observation(agent_id), self.agent_positions[agent_id])
            for agent_id in self.agent_positions
        }

        # In solo mode, duplicate learner observation as expert (for runner compatibility)
        if self.n_agents == 1:
            observations["expert"] = observations["learner"]

        # 6. Check episode termination
        self.timestep += 1
        # Episode continues indefinitely (or until external termination)
        # Typical episode length is 100 steps (set in experiment config)

        # 7. Create info dict
        total_apples_eaten = sum(1 for r in rewards.values() if r > 0)
        info = {
            "pollution_level": float(self.pollution_level),
            "apples_eaten": total_apples_eaten,
            "timestep": int(self.timestep),
        }

        return observations, rewards, self.done, info

    def generate_observation(self, agent_id: str) -> int:
        """
        Generate observation for agent.

        Observation Encoding:
        ---------------------
        Observations are encoded as a single integer index.

        Observation space dimensions:
        - Position: 9 categories (cells 0-8)
        - Other position: 9 categories (cells 0-8, or 0 if solo)
        - Pollution: 5 categories (0, 1, 2, 3, 4)
        - Apples: 8 categories (3-bit bitmask for cells 6,7,8)

        Total observation space: 9 × 9 × 5 × 8 = 3,240 possible observations

        Encoding formula:
            obs_idx = position + (other_position * 9) + (pollution * 81) + (apples * 405)

        Args:
            agent_id: ID of agent requesting observation ("learner" or "expert")

        Returns:
            obs_idx: Observation encoded as integer in [0, 3239]
        """
        agent_pos = self.agent_positions[agent_id]

        # Other agent position (0 if solo, otherwise get the other agent)
        if self.n_agents == 1:
            other_pos = 0
        else:
            other_agent_id = "expert" if agent_id == "learner" else "learner"
            other_pos = self.agent_positions.get(other_agent_id, 0)

        # Get raw values
        pollution = self.pollution_level
        apples = self.apple_bitmask

        # Apply partial observability masks
        if self.apple_obs_mode == "local":
            apples = self._mask_apples_local(apples, agent_pos)

        if self.pollution_obs_mode == "local":
            pollution = self._mask_pollution_local(pollution, agent_pos)

        # Discretize pollution to integer category [0, 1, 2, 3, 4]
        pollution_cat = int(np.clip(np.floor(pollution), 0, 4))

        # Encode as single integer
        obs_idx = (
            agent_pos +
            other_pos * 9 +
            pollution_cat * 81 +
            apples * 405
        )

        return int(obs_idx)

    def decode_observation(self, obs_idx: int) -> Dict[str, int]:
        """
        Decode observation index back to components.

        Args:
            obs_idx: Observation encoded as integer

        Returns:
            components: {
                "position": int in [0, 8],
                "other_position": int in [0, 8],
                "pollution": int in [0, 4],
                "apples": int in [0, 7] (bitmask),
            }
        """
        position = obs_idx % 9
        other_position = (obs_idx // 9) % 9
        pollution = (obs_idx // 81) % 5
        apples = (obs_idx // 405) % 8

        return {
            "position": position,
            "other_position": other_position,
            "pollution": pollution,
            "apples": apples,
        }

    def _mask_apples_local(self, apple_bitmask: int, position: int) -> int:
        """
        Mask apples to only those visible from current position.

        Local observability: agent can only see apples at their current cell
        (if in orchard).

        Args:
            apple_bitmask: Full apple bitmask (3 bits)
            position: Agent position

        Returns:
            masked_bitmask: Visible apples only
        """
        if position not in self.ORCHARD_CELLS:
            # Not in orchard - can't see any apples
            return 0
        else:
            # In orchard - can see only apple at current cell
            orchard_idx = position - 6  # 0, 1, or 2
            visible_bit = 1 << orchard_idx
            return apple_bitmask & visible_bit

    def _mask_pollution_local(self, pollution: float, position: int) -> float:
        """
        Mask pollution to only visible when on/near river.

        Local observability: agent can only see pollution when on river row
        or adjacent neutral row.

        Args:
            pollution: Full pollution level
            position: Agent position

        Returns:
            masked_pollution: Visible pollution (or 0 if not visible)
        """
        if position in self.RIVER_CELLS or position in self.NEUTRAL_CELLS:
            # On or adjacent to river - can see pollution
            return pollution
        else:
            # In orchard - can't see pollution
            return 0.0

    def _move_up(self, position: int) -> int:
        """Move agent up (decrease row), clipping at boundary."""
        row, col = divmod(position, self.grid_size)
        new_row = max(0, row - 1)
        return new_row * self.grid_size + col

    def _move_down(self, position: int) -> int:
        """Move agent down (increase row), clipping at boundary."""
        row, col = divmod(position, self.grid_size)
        new_row = min(self.grid_size - 1, row + 1)
        return new_row * self.grid_size + col

    def _move_left(self, position: int) -> int:
        """Move agent left (decrease column), clipping at boundary."""
        row, col = divmod(position, self.grid_size)
        new_col = max(0, col - 1)
        return row * self.grid_size + new_col

    def _move_right(self, position: int) -> int:
        """Move agent right (increase column), clipping at boundary."""
        row, col = divmod(position, self.grid_size)
        new_col = min(self.grid_size - 1, col + 1)
        return row * self.grid_size + new_col

    def render(self) -> str:
        """
        Render the current state as a string.

        Returns:
            state_str: Human-readable state representation
        """
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Mark apples
        for orchard_idx in range(3):
            if self.apple_bitmask & (1 << orchard_idx):
                cell = self.ORCHARD_CELLS[orchard_idx]
                row, col = divmod(cell, self.grid_size)
                grid[row][col] = 'A'

        # Mark agents
        for agent_id in range(self.n_agents):
            pos = self.agent_positions[agent_id]
            row, col = divmod(pos, self.grid_size)
            if grid[row][col] == 'A':
                grid[row][col] = f'{agent_id}A'  # Agent on apple
            else:
                grid[row][col] = str(agent_id)

        # Build string
        lines = []
        lines.append(f"Timestep: {self.timestep}")
        lines.append(f"Pollution: {self.pollution_level:.2f} / 4.0")
        lines.append("Grid:")
        lines.append("+" + "-" * (self.grid_size * 4 - 1) + "+")
        for row in range(self.grid_size):
            line = "|"
            for col in range(self.grid_size):
                cell = row * self.grid_size + col
                content = grid[row][col]

                # Add row label
                if cell in self.RIVER_CELLS:
                    bg = "R"
                elif cell in self.NEUTRAL_CELLS:
                    bg = "N"
                else:
                    bg = "O"

                line += f" {content:>2s} "
            line += "|"
            lines.append(line)
        lines.append("+" + "-" * (self.grid_size * 4 - 1) + "+")
        lines.append("R=River, N=Neutral, O=Orchard, A=Apple, 0/1=Agent")

        return "\n".join(lines)

    @property
    def n_observations(self) -> int:
        """Number of possible observations."""
        return 9 * 9 * 5 * 8  # 3,240

    @property
    def n_actions(self) -> int:
        """Number of possible actions."""
        return 6  # UP, DOWN, LEFT, RIGHT, CLEAN, EAT

    @property
    def action_names(self):
        """Action names for logging."""
        return ["UP", "DOWN", "LEFT", "RIGHT", "CLEAN", "EAT"]
