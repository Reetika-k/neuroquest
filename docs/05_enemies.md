# Phase 5 — Enemy intelligence

Status: **approved by the learner** · Date: 2026-10-11

## 1. The concepts, in plain language

**A graph** is a set of nodes joined by edges. Here every open floor cell is a node, and two cells are joined by an edge when they are orthogonal neighbours and both open. `Maze.neighbours(cell)` lists the edges leaving a node. Finding a route for an enemy is finding a path in this graph.

**Breadth-first search (BFS)** explores the graph in rings: all cells one step away, then two, then three. The first time it reaches the goal it has found a shortest path, because nothing closer was left unexplored. It is simple and provably right, which makes it our *reference*: whatever A* answers must agree with BFS. Its weakness is that it looks in every direction equally, including away from the goal.

**A heuristic** is an educated guess of the remaining distance. We use the Manhattan distance `|Δrow| + |Δcol|`, the length of the route if there were no walls. It can never be *larger* than the true walking distance (walls only make routes longer), and that property, called admissibility, is what makes A* exact rather than approximate.

**A\*** keeps a priority queue of cells ordered by `f = g + h`, where `g` is the cost walked so far and `h` is the heuristic. It always expands the cell that looks most promising. With an admissible heuristic, the first time the goal leaves the queue its `g` is the true shortest distance. Ties on `f` prefer the larger `g` (the cell nearer the goal), which cuts wasted expansions further. Code: `ai/astar.py`, 45 lines.

**Why A\* rather than random movement here.** A random walker has no idea where the player is; in a maze with dead ends and loops it mostly wanders. A chaser that always steps along a shortest path converges on the player at one cell per move, which is what turns the maze from a stroll into a survival problem. The numbers in section 3 are the evidence.

**How it is tested.** On fixed layouts with known answers (open room, forced detour, sealed rooms, corridor), on edge cases (start = goal, start or goal in a wall or outside the grid), and on 200 seeded random mazes where the A* path length must equal the BFS length and every step of the path must be a legal move. A timing test keeps a call under 5 ms on a 31×21 maze.

## 2. What was built

| Piece | Behaviour | Code | Tests |
|---|---|---|---|
| BFS reference | Shortest path or `None`; counts expanded cells | `ai/bfs.py` | `test_astar.py` |
| A* | Same contract; Manhattan heuristic; stable tie-breaking | `ai/astar.py` | `test_astar.py` (10 cases + 200 seeded mazes + timing) |
| `AStarChaser` | Recomputes the path to the player on every move and takes its first step. Stateless: at 0.15 ms per call there is nothing to cache | `ai/policies.py` | `test_policies.py` |
| `RandomWalker` | Baseline: a uniformly random open neighbour | `ai/policies.py` | `test_policies.py` |
| Enemies in the rules | `new_game` places `enemy_count` enemies at least half the maximum distance from the player; enemies move every `enemy_move_interval` ticks; contact costs one health and starts a 15-tick invulnerability window; `near_enemy` events record distance and whether the player's move approached the nearest enemy; policies that return an illegal cell are ignored | `core/rules.py` | `test_policies.py` (13 cases) |
| Rendering | Red enemies with eyes; the player blinks while invulnerable; HUD shows the enemy count | `ui/render.py` | `test_ui.py` |
| Commands | `play --enemies N --enemy-interval K --enemy-kind astar\|random`; `benchmark-enemies` | `cli.py` | `test_benchmark.py` |

Design note: `core/rules.py` never imports `ai` at module level. The policy registry is passed into `step()` or, by default, imported lazily inside the enemy-move helper. The protocol is the only coupling, so tests inject fake policies and the RL environment can run enemy-free.

## 3. Results (2026-10-11, `python -m neuroquest benchmark-enemies`)

**Chase benchmark.** One enemy of each kind, moving every tick, versus a motionless player at the start cell; 100 seeded 21×15 mazes per kind; give up after 600 ticks. Rows: `reports/enemy_baseline.csv`.

| Enemy | Caught | Median ticks to catch | Ticks per path cell |
|---|---|---|---|
| A* chaser | 100 / 100 | 59 | 1.00 |
| Random walker | 1 / 100 | 302 (n = 1) | 9.44 |

"Ticks per path cell" divides the ticks needed by the shortest-path length from the enemy's start to the player. A value of exactly 1.00 means the chaser never took a wasted step in any of the 100 mazes.

**Search benchmark.** BFS and A* on the same random start/goal pair in each of 200 seeded mazes. Rows: `reports/search_benchmark.csv`.

| Metric | BFS | A* |
|---|---|---|
| Same path length as the other | 200 / 200 | 200 / 200 |
| Cells expanded, mean | 73.8 | 58.2 (−21 %) |
| Time per call, mean | 0.142 ms | 0.145 ms (max 0.395 ms) |

An honest reading: on a 21×15 maze (about 150 open cells) A* does 21 % less exploration, but each expansion costs more (a heap and a heuristic call), so wall time is a tie. The benefit of A* grows with the maze; what matters for the game is that it is exact, directed, and far below the 5 ms budget. The chase table, not the timing, is the reason enemies use it.

Screenshots: `docs/screenshots/phase5-enemies.png` (two chasers placed far from the player at the start) and `phase5-first-hit.png` (the moment the first chaser reaches a motionless player; the player is drawn in the hit colour).

## 4. Limitations at this phase

- Enemies know the player's exact position at all times (no line of sight). That is intentional for a chase game; a vision-cone variant would be a later extension.
- Two enemies may occupy the same cell; they do not block each other.
- The random-walk baseline's `rng` is shared with the game; pass one `random.Random(seed)` per episode for reproducible runs (the harness in Phase 6 will).
