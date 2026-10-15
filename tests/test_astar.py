"""Phase 5: BFS (reference) and A* on fixed layouts, edge cases and 200 seeded mazes."""

import random
import time

import pytest

from neuroquest.ai.astar import a_star
from neuroquest.ai.bfs import bfs
from neuroquest.ai.graph import manhattan
from neuroquest.core.maze import Maze

OPEN_ROOM = Maze.from_strings(["#######", "#.....#", "#.....#", "#.....#", "#######"])
DETOUR = Maze.from_strings(["#######", "#..#..#", "#..#..#", "#.....#", "#######"])  # wall forces a detour
SEALED = Maze.from_strings(["#######", "#..#..#", "#..#..#", "#..#..#", "#######"])  # two rooms, no door
CORRIDOR = Maze.from_strings(["#########", "#.......#", "#########"])


def assert_valid_path(maze: Maze, path: list, start, goal) -> None:
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert manhattan(a, b) == 1 and maze.is_open(b)


@pytest.mark.parametrize("search", [bfs, a_star])
def test_clear_path_has_manhattan_length(search) -> None:
    path = search(OPEN_ROOM, (1, 1), (3, 5))
    assert_valid_path(OPEN_ROOM, path, (1, 1), (3, 5))
    assert len(path) - 1 == manhattan((1, 1), (3, 5)) == 6


@pytest.mark.parametrize("search", [bfs, a_star])
def test_blocked_path_goes_around_the_wall(search) -> None:
    path = search(DETOUR, (1, 2), (1, 4))
    assert_valid_path(DETOUR, path, (1, 2), (1, 4))
    assert len(path) - 1 == 6  # down to row 3, across, back up; the straight line (2) is walled


@pytest.mark.parametrize("search", [bfs, a_star])
def test_unreachable_goal_returns_none(search) -> None:
    assert search(SEALED, (1, 1), (1, 5)) is None


@pytest.mark.parametrize("search", [bfs, a_star])
def test_start_equals_goal(search) -> None:
    assert search(OPEN_ROOM, (2, 2), (2, 2)) == [(2, 2)]


@pytest.mark.parametrize("search", [bfs, a_star])
def test_wall_or_out_of_bounds_endpoints_return_none(search) -> None:
    assert search(OPEN_ROOM, (0, 0), (1, 1)) is None  # start on the border wall
    assert search(OPEN_ROOM, (1, 1), (2, 7)) is None  # goal outside the grid
    assert search(DETOUR, (1, 1), (1, 3)) is None  # goal inside a wall


@pytest.mark.parametrize("search", [bfs, a_star])
def test_corridor_end_to_end(search) -> None:
    path = search(CORRIDOR, (1, 1), (1, 7))
    assert path == [(1, c) for c in range(1, 8)]


@pytest.mark.parametrize("seed", range(200))
def test_astar_matches_bfs_length_on_seeded_mazes(seed: int) -> None:
    rng = random.Random(seed)
    maze = Maze.generate(21, 15, rng, loop_fraction=0.15)
    cells = maze.open_cells()
    start, goal = rng.sample(cells, 2)
    bfs_stats, astar_stats = {}, {}
    reference = bfs(maze, start, goal, stats=bfs_stats)
    path = a_star(maze, start, goal, stats=astar_stats)
    assert reference is not None and path is not None  # generated mazes are connected
    assert len(path) == len(reference)
    assert_valid_path(maze, path, start, goal)
    assert astar_stats["expanded"] <= bfs_stats["expanded"]


def test_astar_expands_fewer_cells_than_bfs_overall() -> None:
    total_bfs = total_astar = 0
    for seed in range(50):
        rng = random.Random(seed)
        maze = Maze.generate(21, 15, rng, loop_fraction=0.15)
        start, goal = rng.sample(maze.open_cells(), 2)
        b, a = {}, {}
        bfs(maze, start, goal, stats=b)
        a_star(maze, start, goal, stats=a)
        total_bfs += b["expanded"]
        total_astar += a["expanded"]
    assert total_astar < total_bfs


def test_astar_is_fast_enough_for_per_tick_use() -> None:
    rng = random.Random(1)
    maze = Maze.generate(31, 21, rng, loop_fraction=0.15)
    pairs = [tuple(rng.sample(maze.open_cells(), 2)) for _ in range(50)]
    t0 = time.perf_counter()
    for start, goal in pairs:
        a_star(maze, start, goal)
    per_call_ms = (time.perf_counter() - t0) / len(pairs) * 1000
    assert per_call_ms < 5.0, f"{per_call_ms:.2f} ms per call"
