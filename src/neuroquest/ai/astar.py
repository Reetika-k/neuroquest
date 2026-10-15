"""A* search: shortest paths with a heuristic that points the search toward the goal.

Every cell gets ``f = g + h``: ``g`` is the cost walked so far and ``h`` is a guess of the
remaining cost. We use the Manhattan distance, which never overestimates on a grid where
each move costs 1 (it is *admissible*), so the first time the goal is taken off the queue the
path is guaranteed shortest. Because the guess is good, A* usually expands far fewer cells
than BFS.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable

from neuroquest.ai.graph import manhattan, reconstruct_path
from neuroquest.core.maze import Cell, Maze


def a_star(
    maze: Maze,
    start: Cell,
    goal: Cell,
    *,
    heuristic: Callable[[Cell, Cell], int] = manhattan,
    stats: dict | None = None,
) -> list[Cell] | None:
    """Shortest path from ``start`` to ``goal`` (both included), or None if unreachable.

    ``stats["expanded"]`` receives the number of cells taken off the priority queue.
    """
    if not (maze.is_open(start) and maze.is_open(goal)):
        return None
    tie = itertools.count()  # keeps the heap stable when f and g are equal
    frontier: list[tuple[int, int, int, Cell]] = [(heuristic(start, goal), 0, next(tie), start)]
    came_from: dict[Cell, Cell | None] = {start: None}
    best_g: dict[Cell, int] = {start: 0}
    closed: set[Cell] = set()
    expanded = 0
    while frontier:
        _, neg_g, _, cell = heapq.heappop(frontier)
        if cell in closed:
            continue  # a stale entry: the cell was reached more cheaply later
        closed.add(cell)
        expanded += 1
        if cell == goal:
            if stats is not None:
                stats["expanded"] = expanded
            return reconstruct_path(came_from, goal)
        g = -neg_g
        for nxt in maze.neighbours(cell):
            tentative = g + 1
            if tentative < best_g.get(nxt, FAR):
                best_g[nxt] = tentative
                came_from[nxt] = cell
                # ties on f prefer the larger g (closer to the goal), which cuts expansions further
                heapq.heappush(frontier, (tentative + heuristic(nxt, goal), -tentative, next(tie), nxt))
    if stats is not None:
        stats["expanded"] = expanded
    return None


FAR = 10**9
