"""Breadth-first search: the reference shortest-path algorithm.

BFS explores cells in rings of increasing distance, so the first time it reaches the goal it
has found a shortest path. It looks everywhere equally, which is what makes it a trustworthy
yardstick for A* and also what makes it do more work.
"""

from __future__ import annotations

from collections import deque

from neuroquest.ai.graph import reconstruct_path
from neuroquest.core.maze import Cell, Maze


def bfs(maze: Maze, start: Cell, goal: Cell, stats: dict | None = None) -> list[Cell] | None:
    """Shortest path from ``start`` to ``goal`` (both included), or None if unreachable.

    ``stats["expanded"]`` receives the number of cells taken off the queue.
    """
    if not (maze.is_open(start) and maze.is_open(goal)):
        return None
    came_from: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    expanded = 0
    while queue:
        cell = queue.popleft()
        expanded += 1
        if cell == goal:
            break
        for nxt in maze.neighbours(cell):
            if nxt not in came_from:
                came_from[nxt] = cell
                queue.append(nxt)
    if stats is not None:
        stats["expanded"] = expanded
    return reconstruct_path(came_from, goal) if goal in came_from else None


def bfs_nearest(maze: Maze, start: Cell, goals, stats: dict | None = None) -> list[Cell] | None:
    """Shortest path from ``start`` to whichever of ``goals`` is closest (both ends included)."""
    goals = set(goals)
    if not maze.is_open(start) or not goals:
        return None
    if start in goals:
        return [start]
    came_from: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    expanded = 0
    while queue:
        cell = queue.popleft()
        expanded += 1
        for nxt in maze.neighbours(cell):
            if nxt not in came_from:
                came_from[nxt] = cell
                if nxt in goals:
                    if stats is not None:
                        stats["expanded"] = expanded
                    return reconstruct_path(came_from, nxt)
                queue.append(nxt)
    if stats is not None:
        stats["expanded"] = expanded
    return None
