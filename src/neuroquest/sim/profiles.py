"""Simulated players with distinct personalities. Each profile samples its parameters from the
episode's random generator, so no two aggressive players are identical and the three classes
overlap in feature space. The profile name is the training label.

- aggressive: goes straight for rewards, ignores enemies, sometimes charges them
- cautious:   keeps a safety margin from enemies, routes around them, waits or retreats when boxed in
- explorer:   wants to see the maze; heads for unvisited cells, exits once coverage is reached
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Protocol

from neuroquest.ai.astar import a_star
from neuroquest.ai.bfs import bfs_nearest
from neuroquest.config import DifficultyParams
from neuroquest.core.maze import Cell, Maze, manhattan
from neuroquest.core.rules import min_enemy_distance, time_left_s
from neuroquest.core.state import Action, GameState

MOVES: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


class PlayerPolicy(Protocol):
    name: str
    params: dict

    def act(self, state: GameState, params: DifficultyParams, rng: random.Random) -> Action: ...


class MaskedMaze:
    """A read-only view of a maze in which some cells count as walls (used to route around enemies)."""

    def __init__(self, maze: Maze, blocked: set[Cell]) -> None:
        self.maze = maze
        self.blocked = blocked

    def is_open(self, cell: Cell) -> bool:
        return self.maze.is_open(cell) and cell not in self.blocked

    def neighbours(self, cell: Cell) -> list[Cell]:
        return [c for c in self.maze.neighbours(cell) if c not in self.blocked]


def direction(a: Cell, b: Cell) -> Action:
    return {(-1, 0): Action.UP, (1, 0): Action.DOWN, (0, -1): Action.LEFT, (0, 1): Action.RIGHT}[(b[0] - a[0], b[1] - a[1])]


def follow(path: list[Cell] | None, pos: Cell) -> Action:
    return direction(pos, path[1]) if path is not None and len(path) > 1 else Action.STAY


def danger_cells(state: GameState, radius: int) -> set[Cell]:
    cells: set[Cell] = set()
    for enemy in state.enemies:
        r, c = enemy.pos
        for dr in range(-radius, radius + 1):
            for dc in range(-radius + abs(dr), radius - abs(dr) + 1):
                cells.add((r + dr, c + dc))
    return cells


def retreat_action(state: GameState, rng: random.Random) -> Action:
    """Step to the neighbouring cell (or stay) that maximises the distance to the nearest enemy."""
    pos = state.player.pos
    options = [(pos, Action.STAY)] + [(cell, direction(pos, cell)) for cell in state.maze.neighbours(pos)]
    rng.shuffle(options)
    best = max(options, key=lambda item: min_enemy_distance(item[0], state.enemies))
    return best[1]


class AggressivePolicy:
    name = "aggressive"

    def __init__(self, rng: random.Random) -> None:
        self.params = {
            "epsilon": round(rng.uniform(0.0, 0.12), 3),
            "engage_prob": round(rng.uniform(0.2, 0.8), 3),
            "engage_radius": rng.choice([2, 3, 4]),
            "exit_when_rewards_left": rng.choice([0, 1, 2]),
        }

    def act(self, state: GameState, params: DifficultyParams, rng: random.Random) -> Action:
        p = self.params
        pos = state.player.pos
        if rng.random() < p["epsilon"]:
            return rng.choice(MOVES)
        if state.enemies and min_enemy_distance(pos, state.enemies) <= p["engage_radius"] and rng.random() < p["engage_prob"]:
            target = min(state.enemies, key=lambda e: manhattan(pos, e.pos))
            return follow(a_star(state.maze, pos, target.pos), pos)
        targets: Iterable[Cell] = state.rewards if len(state.rewards) > p["exit_when_rewards_left"] else {state.exit}
        return follow(bfs_nearest(state.maze, pos, targets), pos)


class CautiousPolicy:
    name = "cautious"

    def __init__(self, rng: random.Random) -> None:
        self.params = {
            "epsilon": round(rng.uniform(0.0, 0.08), 3),
            "safe_radius": rng.choice([2, 3, 4]),
            "flee_radius": rng.choice([1, 2]),
            "reward_greed": round(rng.uniform(0.0, 0.6), 3),
        }
        self.target: Cell | None = None

    def act(self, state: GameState, params: DifficultyParams, rng: random.Random) -> Action:
        p = self.params
        pos = state.player.pos
        if rng.random() < p["epsilon"]:
            return rng.choice(MOVES)
        if state.enemies and min_enemy_distance(pos, state.enemies) <= p["flee_radius"]:
            return retreat_action(state, rng)
        masked = MaskedMaze(state.maze, danger_cells(state, p["safe_radius"]) - {pos})
        if self.target is not None and self.target not in state.rewards:
            self.target = None  # collected (by us) or never valid
        if self.target is None and state.rewards and rng.random() < p["reward_greed"]:
            path = bfs_nearest(masked, pos, state.rewards)
            self.target = path[-1] if path else None
        path = a_star(masked, pos, self.target) if self.target is not None else a_star(masked, pos, state.exit)
        if path is None:
            path = a_star(masked, pos, state.exit)
        if path is None:  # no safe route anywhere: keep the distance and wait for the enemies to move
            return retreat_action(state, rng)
        return follow(path, pos)


class ExplorerPolicy:
    name = "explorer"

    def __init__(self, rng: random.Random) -> None:
        self.params = {
            "epsilon": round(rng.uniform(0.0, 0.1), 3),
            "coverage_target": round(rng.uniform(0.55, 0.9), 3),
            "caution_radius": rng.choice([1, 2]),
        }
        self._maze_id: int | None = None
        self._open: set[Cell] = set()
        self._to_exit: dict[Cell, int] = {}

    def _prepare(self, state: GameState) -> None:
        if self._maze_id != id(state.maze):
            self._maze_id = id(state.maze)
            self._open = set(state.maze.open_cells())
            self._to_exit = state.maze.reachable_from(state.exit)

    def act(self, state: GameState, params: DifficultyParams, rng: random.Random) -> Action:
        p = self.params
        pos = state.player.pos
        self._prepare(state)
        if rng.random() < p["epsilon"]:
            return rng.choice(MOVES)
        if state.enemies and min_enemy_distance(pos, state.enemies) <= p["caution_radius"]:
            return retreat_action(state, rng)
        masked = MaskedMaze(state.maze, danger_cells(state, 1) - {pos})
        coverage = len(state.visited) / max(1, len(self._open))
        seconds_to_exit = self._to_exit.get(pos, 0) / state.ticks_per_second
        if coverage >= p["coverage_target"] or time_left_s(state, params) < 2.0 * seconds_to_exit + 2.0:
            return follow(a_star(masked, pos, state.exit) or a_star(state.maze, pos, state.exit), pos)
        unvisited = self._open - state.visited
        path = bfs_nearest(masked, pos, unvisited) if unvisited else None
        return follow(path or a_star(masked, pos, state.exit), pos)


PROFILES: dict[str, type] = {"aggressive": AggressivePolicy, "cautious": CautiousPolicy, "explorer": ExplorerPolicy}
