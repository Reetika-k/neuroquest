"""Per-episode features, all computable on any prefix of an episode.

Every feature is a rate or a ratio, so a row built from the first 25 % of an episode has the
same meaning as a row built from the whole episode. That is what lets the in-game predictor
work mid-episode and what makes training on prefix snapshots honest.
"""

from __future__ import annotations

import math

from neuroquest.core.maze import Maze
from neuroquest.data.events import EpisodeLog

FEATURE_NAMES: list[str] = [
    "elapsed_s",  # seconds of play seen (observable in-game; prefix_ratio is NOT, so it is metadata only)
    "moves_per_tick",  # successful moves / ticks
    "bump_ratio",  # wall bumps / (moves + bumps)
    "idle_ratio",  # ticks with neither move nor bump / ticks
    "new_cells_per_tick",  # first visits / ticks
    "distinct_cells_ratio",  # distinct cells visited / open cells
    "revisit_ratio",  # moves into an already visited cell / moves
    "rewards_per_tick",  # rewards collected / ticks
    "reward_share",  # rewards collected / rewards available (initial + spawned by the controller)
    "damage_per_tick",  # hits taken / ticks
    "near_enemy_ratio",  # ticks within 2 cells of an enemy / ticks
    "approach_ratio",  # near-enemy moves that reduced the distance / near-enemy moves
    "progress_rate",  # (start distance to exit − current distance) / start distance
    "min_exit_distance_ratio",  # closest the player got to the exit / start distance
]


def episode_features(log: EpisodeLog, prefix_ratio: float = 1.0) -> dict[str, float]:
    """Compute the feature vector from the events up to ``prefix_ratio`` of the episode's ticks."""
    if not 0.0 < prefix_ratio <= 1.0:
        raise ValueError("prefix_ratio must be in (0, 1]")
    total_ticks = int(log.meta["ticks"])
    return features_up_to(log, max(1, math.floor(prefix_ratio * total_ticks)))


def live_features(log: EpisodeLog, tick: int) -> dict[str, float]:
    """The same features for a game still in progress, using everything logged up to ``tick``."""
    return features_up_to(log, max(1, tick))


def features_up_to(log: EpisodeLog, ticks: int) -> dict[str, float]:
    """Feature vector from the events with ``t <= ticks``, normalised by ``ticks``."""
    meta = log.meta
    events = [e for e in log.events if e.t <= ticks]

    moves = [e for e in events if e.type == "move"]
    bumps = [e for e in events if e.type == "bump"]
    rewards = [e for e in events if e.type == "reward"]
    damage = [e for e in events if e.type == "damage"]
    spawned = sum(1 for e in events if e.type == "reward_spawn")
    near = [e for e in events if e.type == "near_enemy"]
    approaches = [e for e in near if "approach" in e.detail]

    maze = Maze.from_strings(meta["maze"])
    start = (int(meta["start"][0]), int(meta["start"][1]))
    exit_cell = (int(meta["exit"][0]), int(meta["exit"][1]))
    to_exit = maze.reachable_from(exit_cell)
    positions = [start] + [e.pos for e in moves]
    d0 = max(1, to_exit.get(start, 1))
    visited = set(positions)

    return {
        "elapsed_s": ticks / float(meta["ticks_per_second"]),
        "moves_per_tick": len(moves) / ticks,
        "bump_ratio": len(bumps) / max(1, len(moves) + len(bumps)),
        "idle_ratio": max(0, ticks - len(moves) - len(bumps)) / ticks,
        "new_cells_per_tick": sum(1 for e in moves if e.detail.get("new_cell")) / ticks,
        "distinct_cells_ratio": len(visited) / max(1, int(meta["open_cells"])),
        "revisit_ratio": sum(1 for e in moves if not e.detail.get("new_cell")) / max(1, len(moves)),
        "rewards_per_tick": len(rewards) / ticks,
        "reward_share": len(rewards) / max(1, int(meta["n_rewards"]) + spawned),
        "damage_per_tick": len(damage) / ticks,
        "near_enemy_ratio": len(near) / ticks,
        "approach_ratio": sum(1 for e in approaches if e.detail["approach"]) / max(1, len(approaches)),
        "progress_rate": (d0 - to_exit.get(positions[-1], d0)) / d0,
        "min_exit_distance_ratio": min(to_exit.get(p, d0) for p in positions) / d0,
    }
