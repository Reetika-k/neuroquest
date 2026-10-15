"""The maze: a rectangular grid of open floor and wall cells.

Cells are ``(row, col)`` tuples. The outer border is always wall. Mazes are generated with a
seeded recursive backtracker (a depth-first "carve" that produces a perfect maze: exactly one
path between any two cells) and then optionally *braided*, which opens some dead ends into
loops so a pursued player has more than one way out.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

Cell = tuple[int, int]  # (row, col)

DIRECTIONS: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right


def manhattan(a: Cell, b: Cell) -> int:
    """Grid distance ignoring walls: |dr| + |dc|. Never overestimates the true walking distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class Maze:
    """Immutable grid; ``grid[row][col]`` is True for open floor and False for wall."""

    grid: tuple[tuple[bool, ...], ...]

    # ---------- basic queries ----------

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_open(self, cell: Cell) -> bool:
        """True if the cell exists and is floor (walkable)."""
        return self.in_bounds(cell) and self.grid[cell[0]][cell[1]]

    def neighbours(self, cell: Cell) -> list[Cell]:
        """Walkable cells one step up/down/left/right from ``cell`` (the graph edges for search)."""
        r, c = cell
        return [(r + dr, c + dc) for dr, dc in DIRECTIONS if self.is_open((r + dr, c + dc))]

    def open_cells(self) -> list[Cell]:
        return [(r, c) for r, row in enumerate(self.grid) for c, open_ in enumerate(row) if open_]

    def dead_ends(self) -> list[Cell]:
        """Open cells with exactly one open neighbour."""
        return [cell for cell in self.open_cells() if len(self.neighbours(cell)) == 1]

    def reachable_from(self, start: Cell) -> dict[Cell, int]:
        """Flood fill: every cell reachable from ``start`` mapped to its step distance."""
        if not self.is_open(start):
            return {}
        dist: dict[Cell, int] = {start: 0}
        queue: deque[Cell] = deque([start])
        while queue:
            cell = queue.popleft()
            for nxt in self.neighbours(cell):
                if nxt not in dist:
                    dist[nxt] = dist[cell] + 1
                    queue.append(nxt)
        return dist

    # ---------- construction ----------

    @classmethod
    def from_strings(cls, rows: Sequence[str]) -> Maze:
        """Build a maze from text: ``#`` is wall, ``.`` is floor. Used by tests and fixed layouts."""
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("maze rows must be non-empty and of equal length")
        bad = {ch for row in rows for ch in row} - {"#", "."}
        if bad:
            raise ValueError(f"unexpected characters in maze text: {sorted(bad)}")
        return cls(tuple(tuple(ch == "." for ch in row) for row in rows))

    def to_strings(self) -> list[str]:
        return ["".join("." if open_ else "#" for open_ in row) for row in self.grid]

    @classmethod
    def generate(cls, width: int, height: int, rng: random.Random, loop_fraction: float = 0.0) -> Maze:
        """Seeded perfect maze (recursive backtracker) with optional braiding.

        ``width`` and ``height`` must be odd and at least 5: floor cells sit at odd coordinates
        and walls between them at even ones, so the border is always wall.
        """
        if width < 5 or height < 5 or width % 2 == 0 or height % 2 == 0:
            raise ValueError("width and height must be odd and >= 5")
        grid = [[False] * width for _ in range(height)]
        start: Cell = (1, 1)
        grid[1][1] = True
        stack: list[Cell] = [start]
        while stack:
            r, c = stack[-1]
            candidates = []
            for dr, dc in DIRECTIONS:
                nr, nc = r + 2 * dr, c + 2 * dc  # the cell two steps away
                if 1 <= nr < height - 1 and 1 <= nc < width - 1 and not grid[nr][nc]:
                    candidates.append(((nr, nc), (r + dr, c + dc)))  # (new cell, wall between)
            if candidates:
                (nr, nc), (wr, wc) = rng.choice(candidates)
                grid[wr][wc] = True
                grid[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()
        maze = cls(tuple(tuple(row) for row in grid))
        return maze.braided(rng, loop_fraction) if loop_fraction > 0 else maze

    def braided(self, rng: random.Random, fraction: float) -> Maze:
        """Open a wall next to ``fraction`` of the dead ends, creating loops. Keeps the border intact."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1")
        grid = [list(row) for row in self.grid]
        ends = self.dead_ends()
        rng.shuffle(ends)
        for r, c in ends[: round(fraction * len(ends))]:
            options = []
            for dr, dc in DIRECTIONS:
                wall, far = (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)
                if (
                    0 < wall[0] < self.height - 1
                    and 0 < wall[1] < self.width - 1
                    and not grid[wall[0]][wall[1]]
                    and self.is_open(far)
                ):
                    options.append(wall)
            if options:
                wr, wc = rng.choice(options)
                grid[wr][wc] = True
        return Maze(tuple(tuple(row) for row in grid))
