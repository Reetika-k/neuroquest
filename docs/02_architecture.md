# NeuroQuest — Phase 2: Architecture and implementation plan

Status: **approved by the learner** · Date: 2026-10-10 · Depends on: `01_requirements.md` (approved, revision 3)

## 1. Architecture at a glance

The whole system is built around one pure function, `step(state, action)`, that advances the game by one logic tick without drawing anything. Everything else is a consumer of that function: the Pygame window plays it, the simulation harness runs it headlessly with scripted players, the RL environment wraps it, and the tests call it directly.

```
                 ┌──────────────────────── inputs ────────────────────────┐
                 │  Keyboard (ui.input)      Scripted profiles (sim)       │
                 │                           RL agent (rl)                  │
                 └───────────────┬────────────────────────┬────────────────┘
                                 ▼                        ▼
                      ┌───────────────────────────────────────────┐
                      │  core: Maze · GameState · rules.step()     │  pure Python, no Pygame
                      │  enemies ask ai.astar / ai.bfs / random    │
                      └───────┬───────────────────────┬───────────┘
            per tick          │ events                │ state
                              ▼                       ▼
                   ┌────────────────────┐    ┌──────────────────┐
                   │ data.events        │    │ ui.render (Pygame)│  60 FPS drawing + HUD
                   │ JSONL per episode  │    └──────────────────┘
                   └───────┬────────────┘
                           │ episode end / every N seconds (prefix)
                           ▼
                   ┌────────────────────┐   build dataset   ┌──────────────────────────┐
                   │ data.features      │ ────────────────▶ │ data/processed/episodes.csv│
                   │ prefix-computable  │                   │ (label = generating profile)│
                   └───────┬────────────┘                   └────────────┬─────────────┘
                           │ live features                               │ train (offline)
                           ▼                                             ▼
                   ┌────────────────────┐   load model      ┌──────────────────────────┐
                   │ ml.predict         │ ◀──────────────── │ ml.train: LR · RF · KNN   │
                   │ label + probability│                   │ + majority baseline       │
                   └───────┬────────────┘                   │ metrics.json, model.joblib │
                           │ prediction + stats             └──────────────────────────┘
                           ▼
                   ┌────────────────────┐
                   │ adapt.controller   │  rules → bounded DifficultyParams → decisions.jsonl
                   └───────┬────────────┘
                           │ new params back into core (next tick)
                           ▼
                   ┌────────────────────────────────────────────────────────┐
                   │ dashboard: reads episodes.csv, metrics.json,             │
                   │ decisions.jsonl, ab_results.csv, rl_curve.csv           │
                   │ → reports/figures/*.png + reports/dashboard.html        │
                   └────────────────────────────────────────────────────────┘
```

Extensions attach to the same spine: the next-move model (`ml.next_move`) trains on the move sequence inside the event logs; the RL environment (`rl.env`) is a thin wrapper that turns `step()` into `reset()`/`step()` with a reward function.

## 2. Design decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Game logic vs drawing | `core` is pure Python; `ui` is the only package that imports Pygame | Logic can be unit-tested headlessly, simulated thousands of times for datasets and A/B experiments, and reused by the RL environment without a window |
| Time model | Fixed logic tick (10 ticks/s); rendering at 60 FPS interpolates nothing, it just draws the latest state | Deterministic, reproducible episodes; the same tick count means the same game whether a human or a bot plays |
| Space model | Discrete grid; one cell per move; enemy speed is "move every k ticks" with k bounded | A* works on the grid graph directly; collisions are cell equality; speed becomes an integer difficulty knob |
| Events are the source of truth | Every tick writes events (move, reward, damage, near-enemy, exit); features are *derived* from events, never logged directly | Feature definitions can change later and the dataset is rebuilt from raw logs without replaying games |
| Prefix-computable features | All features are rates or averages that can be computed on any prefix of an episode; training uses snapshots at 25 / 50 / 75 / 100 % | The in-game predictor must work mid-episode; training on prefixes makes evaluation match deployment |
| Labels | Synthetic episodes carry the profile that generated them; human episodes get a rule label marked as such | Avoids circular labels (spec, revision 2) |
| Model persistence | `joblib` model + a `.meta.json` with feature names, class order, training date, metrics, data hash | The game refuses a model whose feature list differs from the current `features.py`; evaluators can trace a prediction to a training run |
| Adaptation | Rule table evaluated every 5 s of play; outputs clamped to bounds; every decision logged with inputs, the rule that fired, old and new values | Explainable and auditable; the dashboard replays the log |
| Reproducibility | Every randomness source takes an explicit seed (`random.Random(seed)` for the game, `numpy.random.default_rng` for data/ML) | Same seed → same maze, same bot behaviour, same dataset, same split |
| Headless testing | `SDL_VIDEODRIVER=dummy` for the UI smoke test; Matplotlib `Agg` backend for figures | Tests and dashboards run without a display |
| Packaging | `src/` layout, `pyproject.toml` with setuptools, `pip install -e ".[dev]"` | Tests import the package the same way the game does; no `sys.path` hacks |

## 3. Folder and file structure

```
neuroquest/
├── README.md
├── pyproject.toml                 # metadata, dependencies, optional extras, pytest config
├── requirements.txt               # exact versions frozen after Phase 3 install
├── .gitignore
├── docs/                          # 00 brief · 01 requirements · 02 architecture · DEVELOPMENT_LOG · later phases
├── data/
│   ├── raw/                       # one JSONL file per episode: events in order
│   ├── processed/                 # episodes.csv (one row per episode prefix), moves.csv (next-move rows)
│   └── schema/                    # events.schema.json · features.schema.json
├── models/                        # behaviour_clf.joblib + .meta.json · next_move.joblib · q_table.npz
├── reports/                       # metrics.json · decisions.jsonl · ab_results.csv · rl_curve.csv · figures/ · dashboard.html · test_report.md
├── src/neuroquest/
│   ├── __init__.py
│   ├── config.py                  # GameConfig, DifficultyParams (+ BOUNDS), default paths, seeds
│   ├── cli.py                     # python -m neuroquest <command>: play · simulate · build-dataset · train · dashboard · rl-train
│   ├── core/
│   │   ├── maze.py                # Maze grid, seeded generation (recursive backtracker), neighbours(), is_open()
│   │   ├── entities.py            # Player, Enemy, Reward, ExitTile dataclasses
│   │   ├── state.py               # GameState snapshot, Action enum (UP/DOWN/LEFT/RIGHT/STAY), Outcome enum
│   │   └── rules.py               # step(state, action, params, rng) -> (next_state, events); win/loss checks
│   ├── ai/
│   │   ├── graph.py               # grid-as-graph helpers, Manhattan heuristic
│   │   ├── astar.py               # a_star(maze, start, goal) -> list[cell] | None
│   │   ├── bfs.py                 # bfs(maze, start, goal) reference used in optimality tests
│   │   └── policies.py            # EnemyPolicy: AStarChaser · RandomWalker (· Anticipator, optional, uses next-move model)
│   ├── data/
│   │   ├── events.py              # Event dataclass, EventLogger (JSONL writer), reader
│   │   ├── features.py            # episode_features(events, prefix_ratio) -> dict; FEATURE_NAMES
│   │   ├── schema.py              # validate_row(), ranges, NaN handling, label set
│   │   └── dataset.py             # raw logs -> episodes.csv; synthetic generation entry point
│   ├── sim/
│   │   ├── profiles.py            # AggressivePolicy · CautiousPolicy · ExplorerPolicy (parameterised, seeded)
│   │   └── harness.py             # run_episode(policy, params, seed, adaptive) headless; batch runner; A/B tables
│   ├── ml/
│   │   ├── split.py               # stratified, seed-grouped split; temporal split for moves
│   │   ├── train.py               # LR · RF · KNN + majority baseline, CV, metrics, confusion matrix, save
│   │   ├── evaluate.py            # metrics helpers, error inspection, feature importance (RF + permutation)
│   │   ├── predict.py             # BehaviourPredictor: load model + meta, predict(features) -> (label, proba)
│   │   └── next_move.py           # sequence features, temporal split, train/eval, baselines (Phase 7b)
│   ├── adapt/
│   │   ├── bounds.py              # per-parameter limits and clamp()
│   │   ├── controller.py          # decide(stats, prediction, params) -> (new_params, Decision)
│   │   └── log.py                 # DecisionLogger (JSONL)
│   ├── rl/
│   │   ├── env.py                 # MazeEnv: reset()/step() gym-style, observation, RewardSpec, termination (Phase 9b)
│   │   ├── qlearning.py           # QTable (npz), QConfig ε schedule, train(), evaluate(), summarize(), Wilson CI
│   │   ├── agents.py              # random · A*-to-exit · greedy collector · cautious profile · QAgent
│   │   └── pipeline.py            # run_pipeline(): train, save, evaluate all agents on held-out seeds, never-seen board, files
│   ├── dashboard/
│   │   ├── figures.py             # one function per figure (Agg backend)
│   │   └── report.py              # assemble dashboard.html from reports/
│   └── ui/
│       ├── render.py              # draw GameState, HUD, prediction overlay, last decision
│       ├── input.py               # key mapping -> Action
│       └── app.py                 # game loop: fixed logic tick, 60 FPS draw, logger + predictor + controller wiring
└── tests/
    ├── test_maze.py · test_rules.py · test_astar.py · test_bfs_vs_astar.py · test_policies.py
    ├── test_events.py · test_features.py · test_schema.py · test_profiles.py · test_dataset.py
    ├── test_split.py · test_train_pipeline.py · test_predict.py · test_next_move.py
    ├── test_bounds.py · test_controller.py · test_harness.py
    ├── test_env.py · test_qlearning.py · test_figures.py · test_ui_smoke.py (dummy video driver)
```

## 4. Data flow in detail

**During play (one logic tick).** `ui.app` reads the keyboard into an `Action` → `core.rules.step()` moves the player, asks each enemy's policy for its next cell (A* every tick, or cached path until the player moves), applies collisions, rewards, damage and the timer, and returns the new `GameState` plus a list of `Event`s → `data.events.EventLogger` appends the events to the episode's JSONL file → every 50 ticks (5 s) `data.features.episode_features(events_so_far)` builds the live feature dict → `ml.predict.BehaviourPredictor` returns a label and class probabilities → `adapt.controller.decide()` combines rolling performance (damage rate, reward rate, progress) with the prediction, applies the rule table, clamps to bounds, logs a `Decision`, and returns new `DifficultyParams` that the next tick uses → `ui.render` draws the state, the HUD and an overlay showing the prediction, confidence and the last decision's reason.

**Building the dataset (offline).** `sim.harness` runs each profile policy for N seeded episodes through the same `step()` with adaptation off → JSONL logs in `data/raw/` → `data.dataset.build()` computes features at prefix ratios 0.25 / 0.5 / 0.75 / 1.0 for each episode, validates every row against `features.schema.json`, and writes `episodes.csv` with columns: `episode_id, seed, profile (label), prefix_ratio, <features…>`. Human episodes, when present, carry `profile = rule:<class>` so they can be excluded or analysed separately.

**Training (offline).** `ml.split` groups rows by `seed` so prefixes of one episode never straddle train and test, stratifies by label, holds out 20 % → `ml.train` fits the majority baseline, logistic regression, random forest and KNN inside a `Pipeline` with scaling where needed, runs 5-fold grouped CV on the training set, evaluates once on the test set, writes `reports/metrics.json` (accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, CV mean ± std per model), saves the selected model and its metadata.

**A/B experiment (offline).** `sim.harness` runs each profile with `adaptive=False` and `adaptive=True` for ≥ 30 episodes each on matched seeds and writes `reports/ab_results.csv` (profile, condition, win, ticks, damage, rewards). No statistic is interpreted beyond what the table shows.

**RL (offline, Phase 9b).** `rl.env.MazeEnv` builds on `core` with a small fixed maze; `rl.qlearning.train()` writes `models/q_table.npz` and `reports/rl_curve.csv`; `rl.qlearning.evaluate()` runs the greedy policy and the random baseline on held-out seeds.

**Dashboard (offline).** `dashboard.figures` renders: class distribution; confusion matrix; per-class precision/recall/F1 bars; feature importance; prediction timeline with confidence for the latest session; difficulty parameters over time with decision markers and reasons; fixed-vs-adaptive bars; RL learning curve; next-move accuracy. `dashboard.report` writes `reports/dashboard.html`.

## 5. Data contracts

**Event (JSONL line).** `{"t": tick, "type": "move|reward|damage|near_enemy|exit|timeout|tick", "pos": [r, c], "detail": {...}}`. `near_enemy` fires when an enemy is within 2 cells; `damage` carries the enemy id; `move` carries the action and whether the cell was newly visited.

**Episode features (one row).** Proposed starting set, all prefix-computable; the final list is fixed in Phase 6 after inspecting the data:
`moves_per_tick`, `distinct_cells_ratio` (visited / open cells), `revisit_ratio`, `rewards_per_tick`, `reward_detour_ratio` (path length taken vs shortest to rewards collected), `damage_per_tick`, `near_enemy_ratio` (ticks within 2 cells of an enemy / ticks), `approach_ratio` (ticks the player moved toward the nearest enemy / ticks near an enemy), `avoid_count_per_tick` (near_enemy episodes ended without damage), `idle_ratio`, `progress_to_exit_rate`, `prefix_ratio`.

**Decision (JSONL line).** `{"t": tick, "inputs": {"damage_rate":…, "reward_rate":…, "progress":…, "pred": "cautious", "conf": 0.81}, "rule": "R3_cautious_low_damage", "before": {...params}, "after": {...params}, "reason": "cautious and taking no damage for 15 s → +1 enemy, bounded at 5"}`.

**Difficulty parameters and bounds.** `enemy_count ∈ [1, 5]`, `enemy_move_interval ∈ [2, 8]` ticks (smaller is faster), `reward_density ∈ [0.05, 0.25]` of open cells, `time_limit ∈ [60, 240]` s. Bounds live in `adapt/bounds.py` and are tested.

## 6. Dependencies

| Package | Version (minimum) | Purpose | Phase |
|---|---|---|---|
| Python | 3.13 (python.org build on this machine) | runtime; 3.14 Homebrew is externally managed | 3 |
| pygame-ce | 2.5 | window, drawing, input (`import pygame`; the community edition installs cleanly on 3.13 and supports the optional pygbag browser build) | 3 |
| numpy | 2.x | arrays, seeded RNG | 3 |
| pandas | 2.3 | dataset build, CSV, tables | 6 |
| scikit-learn | 1.9 | models, splits, metrics, permutation importance; bundles joblib | 7 |
| matplotlib | 3.10 | dashboard figures (Agg backend) | 9 |
| pytest | 8 | tests | 3 |
| gymnasium (optional extra `rl`) | 1.x | only to register `MazeEnv` for the standard API; the env works without it | 9b |
| pygbag (optional extra `web`) | latest | browser build of the game-only part, stretch | 11 |

Installed with `pip install -e ".[dev]"` inside a `python3.13 -m venv .venv`. Exact versions are frozen into `requirements.txt` in Phase 3. Installed and smoke-tested on 2026-10-10: pygame-ce 2.5.8, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, matplotlib 3.11.2, pytest 8.4.2 (25 pinned packages).

## 7. Incremental implementation checklist

Each item is one small, testable increment. "Tests" names the pytest file that must pass before the next item starts.

**Phase 3 — environment** · venv with python3.13 · `pyproject.toml`, `.gitignore`, folder skeleton with `__init__.py` files · `python -m neuroquest --version` smoke · a trivial `test_smoke.py` · headless Pygame init under the dummy driver · freeze `requirements.txt`.

**Phase 4 — minimum viable game** · `Maze` with seeded generation, `neighbours()`, `is_open()` (`test_maze`) · `GameState`, `Action`, `step()` for player movement and wall collision (`test_rules`) · rewards, exit, win; health, timer, loss (`test_rules`) · `ui.render` placeholder rectangles, HUD; `ui.app` loop with fixed tick; restart key · manual play-through: win once, lose once, screenshots.

**Phase 5 — enemy intelligence** · `bfs()` reference (`test_astar` uses it) · `a_star()` with Manhattan heuristic; tests for clear path, blocked path, unreachable, start = goal, edges, three layouts (`test_astar`, `test_bfs_vs_astar` over 200 seeded mazes) · `RandomWalker` and `AStarChaser` policies; stateless recompute each move (measured 0.15 ms per A* call, so no cache); enemy move interval (`test_policies`) · `step()` moves enemies, damage on contact, invulnerability frames · baseline comparison script: steps-to-catch over 100 seeded episodes → `reports/enemy_baseline.csv`.

**Phase 6 — data collection** · `Event`, `EventLogger`, reader (`test_events`) · `episode_features()` with prefix support; `FEATURE_NAMES` (`test_features`) · `features.schema.json`, `validate_row()` (`test_schema`) · three profile policies with parameter noise (`test_profiles`) · `harness.run_episode()` headless (`test_harness`) · `dataset.build()` → `episodes.csv` (`test_dataset`) · generate the first dataset (3 profiles × 120 episodes × 4 prefixes), inspect class balance and feature distributions, record findings in `docs/06_dataset.md`.

**Phase 7 — machine learning** · grouped stratified split (`test_split`) · training script with majority, LR, RF, KNN; grouped 5-fold CV; test-set metrics; confusion matrix; error inspection; model + meta saved (`test_train_pipeline` on a tiny synthetic set) · `BehaviourPredictor` with feature-list check (`test_predict`) · live prediction overlay in `ui.app` · `docs/07_ml_evaluation.md` with the real numbers.

**Phase 7b — next-move model (extension, if data supports it)** · `moves.csv` rows from event logs: last 3 actions, open directions, relative nearest reward/exit/enemy → target next action · temporal split · majority-direction and repeat-last baselines vs LR / RF (`test_next_move`) · results in `docs/07b_next_move.md`.

**Phase 8 — adaptive difficulty** · bounds and `clamp()` (`test_bounds`) · rule table and `decide()` with decision logging (`test_controller`: every rule fires once, bounds never violated in 1,000 steps) · wiring in `ui.app` every 5 s · harness `adaptive` flag; A/B run ≥ 30 episodes × 3 profiles × 2 conditions → `ab_results.csv` · `docs/08_adaptation.md` with the rule table and the A/B table.

**Phase 9 — dashboard** · figures one at a time (`test_figures` checks files are written) · `report.py` → `dashboard.html` · in-game overlay shows the rule behind the last decision · `docs/09_explainability.md` with correct and incorrect prediction examples.

**Phase 9b — RL agent (extension)** · `MazeEnv` with state, actions, reward, termination (`test_env`) · tabular Q-learning with epsilon decay; `train()` writes `rl_curve.csv`; `evaluate()` vs random baseline on held-out seeds (`test_qlearning` on a 5×5 maze) · learning-curve figure · `docs/09b_rl.md`.

**Phase 10 — testing** · full suite run; integration test play → log → features → predict; scenario tests (win, loss by health, loss by time, restart); edge cases (1×1 open area, maze with no rewards, model file missing); regression tests for every bug fixed · `reports/test_report.md`.

**Phase 11 — documentation** · README, methodology, dataset description, prompt log (`DEVELOPMENT_LOG.md`), test report, limitations, setup/run instructions, demo script, recorded playthrough.

## 8. What is deliberately not in the architecture

No database (JSONL and CSV files are enough and inspectable), no web framework (the dashboard is static HTML), no deep learning (tabular data with a few hundred rows calls for the models above), no networking, no cloud services.
