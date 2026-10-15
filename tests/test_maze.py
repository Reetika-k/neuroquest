"""Phase 4: maze generation, queries and braiding."""

import random

import pytest

from neuroquest.core.maze import Maze

FIXTURE = [
    "#####",
    "#...#",
    "#.#.#",
    "#...#",
    "#####",
]


def test_from_strings_and_back_round_trip() -> None:
    maze = Maze.from_strings(FIXTURE)
    assert maze.to_strings() == FIXTURE
    assert (maze.width, maze.height) == (5, 5)


def test_from_strings_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        Maze.from_strings(["###", "#..#"])  # ragged
    with pytest.raises(ValueError):
        Maze.from_strings(["#x#"])  # unknown character


def test_is_open_and_in_bounds() -> None:
    maze = Maze.from_strings(FIXTURE)
    assert maze.is_open((1, 1))
    assert not maze.is_open((2, 2))  # wall
    assert not maze.is_open((0, 0))  # border wall
    assert not maze.is_open((-1, 1)) and not maze.is_open((5, 5))  # outside


def test_neighbours_are_open_orthogonal_cells_only() -> None:
    maze = Maze.from_strings(FIXTURE)
    assert sorted(maze.neighbours((1, 1))) == [(1, 2), (2, 1)]
    assert sorted(maze.neighbours((1, 2))) == [(1, 1), (1, 3)]  # (2, 2) is a wall
    assert sorted(maze.neighbours((2, 2))) == [(1, 2), (2, 1), (2, 3), (3, 2)]  # a wall cell still lists its open neighbours


def test_reachable_from_gives_walking_distances() -> None:
    maze = Maze.from_strings(FIXTURE)
    dist = maze.reachable_from((1, 1))
    assert len(dist) == len(maze.open_cells()) == 8
    assert dist[(1, 1)] == 0 and dist[(1, 3)] == 2 and dist[(3, 3)] == 4
    assert maze.reachable_from((2, 2)) == {}  # starting inside a wall


def test_generate_requires_odd_dimensions() -> None:
    for w, h in [(20, 15), (21, 14), (3, 5)]:
        with pytest.raises(ValueError):
            Maze.generate(w, h, random.Random(0))


def test_generate_is_deterministic_per_seed_and_varies_across_seeds() -> None:
    a = Maze.generate(21, 15, random.Random(7))
    b = Maze.generate(21, 15, random.Random(7))
    c = Maze.generate(21, 15, random.Random(8))
    assert a.grid == b.grid
    assert a.grid != c.grid


@pytest.mark.parametrize("seed", range(20))
def test_generated_maze_has_wall_border_and_is_fully_connected(seed: int) -> None:
    maze = Maze.generate(21, 15, random.Random(seed))
    rows = maze.to_strings()
    assert set(rows[0]) == {"#"} and set(rows[-1]) == {"#"}
    assert all(row[0] == "#" and row[-1] == "#" for row in rows)
    assert maze.is_open((1, 1))
    assert len(maze.reachable_from((1, 1))) == len(maze.open_cells())


def test_perfect_maze_has_no_loops() -> None:
    # A perfect maze is a tree: open cells = edges + 1, where edges are adjacent open pairs.
    maze = Maze.generate(21, 15, random.Random(3))
    cells = maze.open_cells()
    edges = sum(len(maze.neighbours(cell)) for cell in cells) // 2
    assert edges == len(cells) - 1


def test_braiding_opens_dead_ends_and_keeps_connectivity() -> None:
    rng = random.Random(5)
    perfect = Maze.generate(21, 15, rng)
    braided = perfect.braided(random.Random(5), fraction=0.5)
    assert len(braided.dead_ends()) < len(perfect.dead_ends())
    assert len(braided.open_cells()) > len(perfect.open_cells())
    assert len(braided.reachable_from((1, 1))) == len(braided.open_cells())
    assert set(braided.to_strings()[0]) == {"#"}  # border untouched
    with pytest.raises(ValueError):
        perfect.braided(rng, fraction=1.5)
