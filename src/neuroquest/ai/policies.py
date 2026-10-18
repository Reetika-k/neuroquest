"""Enemy movement policies. A policy answers one question: given the game right now, which
adjacent cell does this enemy step to? ``rules.step`` validates the answer and applies it."""

from __future__ import annotations

import random
from pathlib import Path
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


class Anticipator:
    """Chaser that heads for where the player will be *next*, as predicted by the next-move model
    (Phase 7b), and takes the first A* step toward that cell. Without a saved model, or when the
    predicted cell is unreachable, it behaves exactly like ``AStarChaser`` (``fallbacks`` counts
    those moves). The model is loaded lazily on the first move so that importing this module
    stays cheap.
    """

    name = "anticipator"

    def __init__(self, predictor=None, model_path: "Path | None" = None, meta_path: "Path | None" = None) -> None:
        self._predictor = predictor
        self._paths = (model_path, meta_path)
        self._loaded = predictor is not None
        self._cache: tuple[tuple[int, int], Cell] | None = None
        self.fallbacks = 0
        self.predictions = 0

    def _load(self):
        if not self._loaded:
            self._loaded = True
            from neuroquest.ml.next_move import NextMovePredictor  # local import: keeps ai/ free of sklearn at import time
            from neuroquest.ml.predict import ModelMismatch

            try:
                self._predictor = NextMovePredictor.load(*(p for p in self._paths if p is not None)) if any(self._paths) else NextMovePredictor.load()
            except (FileNotFoundError, ModelMismatch):
                self._predictor = None
        return self._predictor

    def _predicted_cell(self, state: GameState) -> Cell:
        """One prediction per tick: every hunter moving on the same tick sees the same player state."""
        key = (state.tick, id(state))
        if self._cache is None or self._cache[0] != key:
            self.predictions += 1
            self._cache = (key, self._predictor.predict_cell(state))
        return self._cache[1]

    def next_cell(self, enemy: Enemy, state: GameState, rng: random.Random) -> Cell:
        predictor = self._load()
        target = state.player.pos
        if predictor is not None:
            target = self._predicted_cell(state)
        path = a_star(state.maze, enemy.pos, target) if target != state.player.pos else None
        if path is None:
            self.fallbacks += 1
            path = a_star(state.maze, enemy.pos, state.player.pos)
        return path[1] if path is not None and len(path) > 1 else enemy.pos


DEFAULT_POLICIES: dict[str, EnemyPolicy] = {"astar": AStarChaser(), "random": RandomWalker(), "anticipator": Anticipator()}
