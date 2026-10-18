"""Game state, actions, outcomes and events. Everything here is immutable and Pygame-free."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from neuroquest.core.entities import Enemy, Player
from neuroquest.core.maze import Cell, Maze


class Action(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"

    @property
    def delta(self) -> Cell:
        return _DELTAS[self]


_DELTAS: dict[Action, Cell] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}


class Outcome(Enum):
    PLAYING = "playing"
    WON = "won"
    LOST_TIME = "lost_time"
    LOST_HEALTH = "lost_health"

    @property
    def is_over(self) -> bool:
        return self is not Outcome.PLAYING


@dataclass(frozen=True)
class Event:
    """One thing that happened on tick ``t``. Events are the raw material for every dataset (Phase 6)."""

    t: int
    type: str  # move · bump · reward · exit · timeout · near_enemy · damage · defeat
    pos: Cell
    detail: dict[str, Any] = field(default_factory=dict)


RECENT_ACTIONS = 3  # how many of the player's latest actions the state remembers (Phase 7b features)


@dataclass(frozen=True)
class GameState:
    """A complete snapshot of one tick. ``step()`` returns a new one; the old one is never mutated."""

    maze: Maze
    player: Player
    rewards: frozenset[Cell]
    exit: Cell
    visited: frozenset[Cell]
    tick: int = 0
    score: int = 0
    outcome: Outcome = Outcome.PLAYING
    ticks_per_second: int = 10
    enemies: tuple[Enemy, ...] = ()
    invulnerability_ticks: int = 15
    recent_actions: tuple[Action, ...] = ()  # the player's last RECENT_ACTIONS actions, oldest first (Phase 7b)

    @property
    def elapsed_s(self) -> float:
        return self.tick / self.ticks_per_second
