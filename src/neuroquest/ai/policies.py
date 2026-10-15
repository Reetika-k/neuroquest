"""Enemy movement policies. A policy answers one question: given the game right now, which
adjacent cell does this enemy step to? ``rules.step`` validates the answer and applies it."""

from __future__ import annotations

import random
from typing import Protocol

from neuroquest.ai.astar import a_star
from neuroquest.core.entities import Enemy
from neuroquest.core.maze import Cell
from neuroquest.core.state import GameState


class EnemyPolicy(Protocol):
    name: str

    def next_cell(self, enemy: Enemy, state: GameState, rng: random.Random) -> Cell: ...


class RandomWalker:
    """Baseline: step to a uniformly random open neighbour. No knowledge of the player."""

    name = "random"

    def next_cell(self, enemy: Enemy, state: GameState, rng: random.Random) -> Cell:
        options = state.maze.neighbours(enemy.pos)
        return rng.choice(options) if options else enemy.pos


class AStarChaser:
    """Chaser: recompute the shortest path to the player and take its first step.

    Stateless by design: on a 21×15 maze A* costs well under a millisecond, enemies move at most
    every few ticks, and recomputing every move means the path is always current.
    """

    name = "astar"

    def next_cell(self, enemy: Enemy, state: GameState, rng: random.Random) -> Cell:
        path = a_star(state.maze, enemy.pos, state.player.pos)
        return path[1] if path is not None and len(path) > 1 else enemy.pos


DEFAULT_POLICIES: dict[str, EnemyPolicy] = {"astar": AStarChaser(), "random": RandomWalker()}
