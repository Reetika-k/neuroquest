"""Grid-as-graph helpers shared by BFS and A*.

The maze is a graph: every open cell is a node and every pair of orthogonally adjacent open
cells is an edge of cost 1 (``Maze.neighbours`` lists the edges from a node).
"""

from __future__ import annotations

from neuroquest.core.maze import Cell, manhattan

__all__ = ["manhattan", "reconstruct_path"]


def reconstruct_path(came_from: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
    """Walk the parent links back from ``goal`` to the start and return the path start → goal."""
    path = [goal]
    cell = goal
    while came_from[cell] is not None:
        cell = came_from[cell]
        path.append(cell)
    path.reverse()
    return path
