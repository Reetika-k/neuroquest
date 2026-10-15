"""Configuration shared by every package.

``GameConfig`` describes the board, the clock and fixed game-feel settings (fixed for a session).
``DifficultyParams`` are the knobs the adaptive controller may change at runtime (Phase 8),
within the bounds defined in ``adapt/bounds.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

TICKS_PER_SECOND = 10
REWARD_VALUE = 10


@dataclass(frozen=True)
class GameConfig:
    """Board and clock settings. Width and height must be odd (the maze generator needs it)."""

    width: int = 21
    height: int = 15
    max_health: int = 3
    loop_fraction: float = 0.15  # share of dead ends opened into loops, so the maze is not a single corridor tree
    ticks_per_second: int = TICKS_PER_SECOND
    enemy_kind: str = "astar"  # "astar" (A* chaser) or "random" (random-walk baseline)
    enemy_min_start_distance: float = 0.5  # enemies start at least this fraction of the farthest distance from the player
    invulnerability_ticks: int = 15  # after a hit the player cannot be hit again for this many ticks (1.5 s)


@dataclass(frozen=True)
class DifficultyParams:
    """Runtime difficulty knobs."""

    enemy_count: int = 2
    enemy_move_interval: int = 4  # ticks between enemy moves; smaller is faster
    reward_density: float = 0.10  # fraction of open cells that start with a reward
    time_limit_s: int = 120
