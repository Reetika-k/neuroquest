# Phase 4 — Minimum viable game

Status: **approved by the learner** (needs the learner's manual play-through) · Date: 2026-10-10

## What exists now

A playable maze game with placeholder graphics: a seeded maze, a player that moves on the grid and is stopped by walls, collectible rewards, an exit that wins the level, and two loss conditions (time runs out; health reaches zero, which enemies will trigger from Phase 5).

```bash
cd neuroquest
.venv/bin/python -m neuroquest play                 # default 21×15 maze, seed 1, 120 s
.venv/bin/python -m neuroquest play --seed 7 --time-limit 60 --reward-density 0.2
```

| Key | Action |
|---|---|
| Arrow keys or W A S D | move one cell per logic tick while held (10 ticks per second) |
| R | restart with the next seed (a new maze) |
| P | pause / resume |
| F12 | save a screenshot to `reports/figures/` |
| Esc | quit |

## Mechanics and where they live

| Mechanic | Rule | Code | Tests |
|---|---|---|---|
| Maze | Seeded recursive backtracker on an odd grid; border always wall; 15 % of dead ends opened into loops | `core/maze.py` | `test_maze.py` (29) |
| Start and exit | Player starts at (1, 1); the exit is the open cell farthest by walking distance | `core/rules.py::new_game` | `test_rules.py` |
| Rewards | `round(reward_density × open cells)` rewards sampled from cells other than start and exit; +10 score each, collected once | `core/rules.py::step` | `test_rules.py` |
| Movement | One cell per tick in the held direction; a wall produces a `bump` event and no move; every move records whether the cell was new | `core/rules.py::step` | `test_rules.py` |
| Win | Standing on the exit → `WON`; the state freezes (further steps return it unchanged) | `core/rules.py::step` | `test_rules.py` |
| Loss by time | `tick ≥ time_limit_s × ticks_per_second` → `LOST_TIME`; reaching the exit on that same tick still wins | `core/rules.py::step` | `test_rules.py` |
| Loss by health | `apply_damage()` lowers health; zero → `LOST_HEALTH` (used by enemies from Phase 5) | `core/rules.py::apply_damage` | `test_rules.py` |
| Rendering | Walls, floor, visited tint, rewards (circles), exit (green), player (blue), HUD, end-of-game overlay | `ui/render.py` | `test_ui.py` (pixel checks, every outcome) |
| Loop | Fixed logic tick with a time accumulator; drawing at up to 60 FPS; restart, pause, screenshot | `ui/app.py` | `test_ui.py` (headless runs) |

Every event the game emits (`move`, `bump`, `reward`, `exit`, `timeout`, `damage`, `defeat`) is already the format Phase 6 will log.

## Scripted play-through (headless, 2026-10-10)

A script drove the real `step()` and `draw()` without a window: it collected the two nearest rewards (score 20 by tick 13), walked to the exit (won at tick 58), ran a 1-second clock out (`lost_time` at tick 10) and applied three damage points (`lost_health`). Screenshots: `docs/screenshots/phase4-start.png`, `phase4-playing.png`, `phase4-won.png`, `phase4-lost-time.png`, `phase4-defeated.png`.

The manual play-through (a human winning once and losing once in the real window) is the learner's step before approval.

## Known limitations (by design at this phase)

- No enemies yet, so health only changes through `apply_damage()`; the loss-by-health path is covered by tests and the overlay, not by play.
- Placeholder shapes, no sound, no animation.
- Movement is grid-stepped at 10 cells per second while a key is held; this is the intended "survival" feel but can be tuned in `GameConfig.ticks_per_second`.
