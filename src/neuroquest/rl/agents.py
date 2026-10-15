"""The agents that play ``MazeEnv``: the learned one and the baselines it is compared with.

Every agent answers ``act(obs, state) -> action index``. The baselines are allowed to look at
the full game state (they are scripted, not learned); the Q-learning agent uses only the
observation, which is all it ever saw in training.
"""

from __future__ import annotations

import random
from typing import Protocol

from neuroquest.ai.astar import a_star
from neuroquest.ai.bfs import bfs_nearest
from neuroquest.config import DifficultyParams
from neuroquest.core.state import GameState
from neuroquest.rl.env import ACTION_INDEX, ACTIONS, Obs
from neuroquest.rl.qlearning import QTable
from neuroquest.sim.profiles import PROFILES, PlayerPolicy, follow


class Agent(Protocol):
    name: str

    def act(self, obs: Obs, state: GameState) -> int: ...


class RandomAgent:
    """Baseline: a uniformly random action every tick."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, obs: Obs, state: GameState) -> int:
        return self.rng.randrange(len(ACTIONS))


class ShortestPathAgent:
    """Rule-based baseline: walk the A* shortest path to the exit; ignore crystals and the hunter."""

    name = "astar_exit"

    def act(self, obs: Obs, state: GameState) -> int:
        return ACTION_INDEX[follow(a_star(state.maze, state.player.pos, state.exit), state.player.pos)]


class GreedyCollectorAgent:
    """Rule-based baseline: nearest crystal first (BFS), then the exit; ignores the hunter."""

    name = "greedy_collector"

    def act(self, obs: Obs, state: GameState) -> int:
        targets = state.rewards if state.rewards else {state.exit}
        return ACTION_INDEX[follow(bfs_nearest(state.maze, state.player.pos, targets), state.player.pos)]


class ProfileAgent:
    """Rule-based baseline from Phase 6: one of the scripted player profiles (``cautious`` keeps a
    safety margin from the hunter and routes around it). A fresh profile is sampled at every
    episode start, as the simulation harness does."""

    def __init__(self, profile: str, params: DifficultyParams, seed: int = 0) -> None:
        self.name, self.params, self.rng = profile, params, random.Random(seed)
        self.policy: PlayerPolicy | None = None

    def act(self, obs: Obs, state: GameState) -> int:
        if self.policy is None or state.tick == 0:
            self.policy = PROFILES[self.name](self.rng)
        return ACTION_INDEX[self.policy.act(state, self.params, self.rng)]


class QAgent:
    """The learned policy: the action with the highest Q-value for the observation (ties and
    never-seen observations are broken at random), with optional ε-greedy noise."""

    name = "q_learning"

    def __init__(self, table: QTable, epsilon: float = 0.0, seed: int = 0) -> None:
        self.table, self.epsilon, self.rng = table, epsilon, random.Random(seed)

    def act(self, obs: Obs, state: GameState) -> int:
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTIONS))
        return self.table.greedy(obs, self.rng)
