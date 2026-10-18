"""Measurements behind two claims: "A* beats random movement for chasing" and "A* finds the same
shortest paths as BFS with less work". Both write CSV rows so the numbers can be re-checked."""

from __future__ import annotations

import csv
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path

from neuroquest.ai.astar import a_star
from neuroquest.ai.bfs import bfs
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.maze import Maze
from neuroquest.core.rules import new_game, step
from neuroquest.core.state import Action


def chase_benchmark(episodes: int = 100, max_ticks: int = 600, config: GameConfig | None = None, seed0: int = 0) -> list[dict]:
    """One enemy of each kind hunts a motionless player: ticks until first contact, capped at ``max_ticks``.

    The enemy moves every tick, so for the A* chaser ``ticks_to_catch`` should equal the shortest
    path length exactly; the random walker is the baseline.
    """
    base = config or GameConfig()
    rows: list[dict] = []
    for kind in ("astar", "random"):
        for i in range(episodes):
            seed = seed0 + i
            params = DifficultyParams(enemy_count=1, enemy_move_interval=1, time_limit_s=10**6)
            state = new_game(replace(base, enemy_kind=kind), params, seed)
            rng = random.Random(seed)
            path = bfs(state.maze, state.enemies[0].pos, state.player.pos)
            path_length = len(path) - 1 if path else -1
            caught: int | None = None
            while state.tick < max_ticks and not state.outcome.is_over:
                state, events = step(state, Action.STAY, params, rng)
                if any(e.type == "damage" for e in events):
                    caught = state.tick
                    break
            rows.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "path_length": path_length,
                    "caught": caught is not None,
                    "ticks_to_catch": caught if caught is not None else "",
                    "max_ticks": max_ticks,
                }
            )
    return rows


def search_benchmark(mazes: int = 200, width: int = 21, height: int = 15, seed0: int = 0) -> list[dict]:
    """BFS and A* on the same random start/goal pair per seeded maze: path length, cells expanded, time."""
    rows: list[dict] = []
    for i in range(mazes):
        seed = seed0 + i
        rng = random.Random(seed)
        maze = Maze.generate(width, height, rng, loop_fraction=0.15)
        start, goal = rng.sample(maze.open_cells(), 2)
        bfs_stats: dict = {}
        astar_stats: dict = {}
        t0 = time.perf_counter()
        bfs_path = bfs(maze, start, goal, stats=bfs_stats)
        t1 = time.perf_counter()
        astar_path = a_star(maze, start, goal, stats=astar_stats)
        t2 = time.perf_counter()
        rows.append(
            {
                "seed": seed,
                "start": f"{start[0]}:{start[1]}",
                "goal": f"{goal[0]}:{goal[1]}",
                "bfs_length": len(bfs_path) - 1,
                "astar_length": len(astar_path) - 1,
                "bfs_expanded": bfs_stats["expanded"],
                "astar_expanded": astar_stats["expanded"],
                "bfs_ms": round((t1 - t0) * 1000, 4),
                "astar_ms": round((t2 - t1) * 1000, 4),
            }
        )
    return rows


def pursuit_benchmark(
    episodes: int = 40,
    seed0: int = 2000,
    profiles: tuple[str, ...] = ("aggressive", "cautious", "explorer"),
    kinds: tuple[str, ...] = ("astar", "anticipator"),
    config: GameConfig | None = None,
    params: DifficultyParams | None = None,
    policies=None,
) -> list[dict]:
    """Hunters of each kind pursue *moving* scripted players on the same seeded mazes (Phase 7b):
    tick of the first hit, hits per episode and the player's outcome. The seeds start at 2000,
    away from every other experiment's seeds."""
    from neuroquest.sim.harness import run_episode  # local import: sim depends on ml, benchmark must stay importable alone
    from neuroquest.sim.profiles import PROFILES

    base, params = config or GameConfig(), params or DifficultyParams()
    rows: list[dict] = []
    for kind in kinds:
        for profile in profiles:
            for i in range(episodes):
                seed = seed0 + i
                log = run_episode(PROFILES[profile], replace(base, enemy_kind=kind), params, seed, profile_name=profile, policies=policies)
                hits = [e.t for e in log.events if e.type == "damage"]
                rows.append({"kind": kind, "profile": profile, "seed": seed, "first_contact": hits[0] if hits else "", "hits": len(hits), "outcome": log.meta["outcome"], "ticks": log.meta["ticks"], "score": log.meta["score"]})
    return rows


def summarize_pursuit(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        out.setdefault(r["kind"], {}).setdefault(r["profile"], []).append(r)
    summary: dict[str, dict[str, dict]] = {}
    for kind, by_profile in out.items():
        summary[kind] = {}
        for profile, sub in by_profile.items():
            caught = [r for r in sub if r["first_contact"] != ""]
            summary[kind][profile] = {
                "episodes": len(sub),
                "caught": len(caught),
                "catch_rate": len(caught) / len(sub),
                "median_first_contact": statistics.median(r["first_contact"] for r in caught) if caught else None,
                "mean_hits": statistics.mean(r["hits"] for r in sub),
                "player_win_rate": statistics.mean(r["outcome"] == "won" for r in sub),
            }
    return summary


def format_pursuit(summary: dict[str, dict[str, dict]]) -> str:
    lines = [f"Pursuit: hunters vs moving scripted players, same seeds per cell", f"  {'kind':<13}{'profile':<12}{'caught':>8}{'median 1st hit':>16}{'mean hits':>11}{'player wins':>13}"]
    for kind, by_profile in summary.items():
        for profile, s in by_profile.items():
            med = f"{s['median_first_contact']:.0f}" if s["median_first_contact"] is not None else "-"
            lines.append(f"  {kind:<13}{profile:<12}{s['caught']:>3}/{s['episodes']:<4}{med:>16}{s['mean_hits']:>11.2f}{s['player_win_rate']:>13.0%}")
    return "\n".join(lines)


def summarize_chase(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for kind in ("astar", "random"):
        sub = [r for r in rows if r["kind"] == kind]
        caught = [r for r in sub if r["caught"]]
        out[kind] = {
            "episodes": len(sub),
            "caught": len(caught),
            "catch_rate": len(caught) / len(sub) if sub else 0.0,
            "median_ticks": statistics.median(r["ticks_to_catch"] for r in caught) if caught else None,
            "mean_ticks_per_path_cell": (
                statistics.mean(r["ticks_to_catch"] / r["path_length"] for r in caught if r["path_length"] > 0) if caught else None
            ),
        }
    return out


def summarize_search(rows: list[dict]) -> dict:
    return {
        "mazes": len(rows),
        "equal_length": sum(r["bfs_length"] == r["astar_length"] for r in rows),
        "bfs_mean_expanded": statistics.mean(r["bfs_expanded"] for r in rows),
        "astar_mean_expanded": statistics.mean(r["astar_expanded"] for r in rows),
        "bfs_mean_ms": statistics.mean(r["bfs_ms"] for r in rows),
        "astar_mean_ms": statistics.mean(r["astar_ms"] for r in rows),
        "astar_max_ms": max(r["astar_ms"] for r in rows),
    }


def format_summary(chase: dict[str, dict], search: dict) -> str:
    lines = [
        f"Chase: one enemy vs a motionless player, {chase['astar']['episodes']} seeded mazes each, enemy moves every tick",
        f"  {'kind':<8}{'caught':>8}{'rate':>8}{'median ticks':>14}{'ticks per path cell':>22}",
    ]
    for kind, s in chase.items():
        med = f"{s['median_ticks']:.0f}" if s["median_ticks"] is not None else "-"
        per = f"{s['mean_ticks_per_path_cell']:.2f}" if s["mean_ticks_per_path_cell"] is not None else "-"
        lines.append(f"  {kind:<8}{s['caught']:>8}{s['catch_rate']:>8.0%}{med:>14}{per:>22}")
    lines += [
        f"Search: BFS vs A* on {search['mazes']} seeded mazes with random start/goal",
        f"  same path length: {search['equal_length']}/{search['mazes']}",
        f"  cells expanded (mean): BFS {search['bfs_mean_expanded']:.1f}  A* {search['astar_mean_expanded']:.1f}",
        f"  time per call (mean): BFS {search['bfs_mean_ms']:.3f} ms  A* {search['astar_mean_ms']:.3f} ms  (A* max {search['astar_max_ms']:.3f} ms)",
    ]
    return "\n".join(lines)


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
