"""Things that live in the maze."""

from __future__ import annotations

from dataclasses import dataclass

from neuroquest.core.maze import Cell


@dataclass(frozen=True)
class Player:
    pos: Cell
    health: int
    invulnerable_until: int = 0  # tick until which enemy contact does no damage


@dataclass(frozen=True)
class Enemy:
    id: int
    pos: Cell
    kind: str = "astar"  # name of the movement policy in ai/policies.py
