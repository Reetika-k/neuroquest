# NeuroQuest — development and prompt-engineering log

One entry per step, in the eight-part format required by the project brief (`00_project_brief.md`, section 6). Only work that actually happened is recorded; nothing is reconstructed after the fact.

---

## Entry 1 — Phase 1: requirements and feasibility (2026-10-10)

**1. Problem to solve.** Agree on what NeuroQuest is, what the minimum viable product contains, which AI/ML techniques it must demonstrate honestly, how success will be measured, and what the risks are, before any code is written.

**2. Prompt given to the coding assistant.** The full project brief in `00_project_brief.md`, ending with "Begin with Phase 1: Requirements and feasibility only … Do not generate code yet … End by asking me to approve the requirements before moving to Phase 2."

**3. Generated output.** `01_requirements.md` (draft 1 produced in an earlier Claude Code session today; revision 2 produced in this session after review).

**4. What worked and what failed.** Draft 1 covered objectives, MVP vs extensions, AI/ML concepts, functional and non-functional requirements, risks, ten measurable success criteria and the deliverables per phase. Review found two weaknesses and one gap:
- The labelling strategy (FR7) derived labels from the same features by threshold rules, so a classifier would only re-learn those rules. The brief explicitly forbids claiming that such a model "learned behaviour".
- The Python-version risk was stated as a possibility rather than checked.
- The course's game project expects a public URL, which a desktop Python game does not provide; the spec did not say how the submission handles that.

**5. Error or limitation discovered.** A read-only environment check (`which -a python3`, `pip install --dry-run --no-deps …` on both interpreters, no installation) showed: the default `python3` is Homebrew Python 3.14.0 and is marked externally managed (PEP 668), so nothing can be installed into it directly; Python 3.13.7 (python.org) resolves pygame 2.6.1, pygame-ce 2.5.8, scikit-learn 1.9.1, matplotlib and pytest, and already has pandas 2.3.3 installed.

**6. Revised output.** Revision 2 of `01_requirements.md`: synthetic episodes are labelled by the *generating profile* (the simulated player policy that produced them, with randomised parameters so classes overlap), the rule-based labeller is kept only for real human episodes with its limitation stated, S4 requires held-out seeds; the Python row records the finding and the decision to build the virtual environment with python3.13 in Phase 3; a hosting-gap risk row names the mitigation (repository, screenshots, recorded playthrough; optional pygbag browser build of the game-only part).

**7. Tests performed.** None: Phase 1 contains no code. The environment check above is the only execution and changed nothing on the machine.

**8. Final improvement.** The specification now has a non-circular labelling plan, a verified interpreter decision and an explicit answer to the hosting expectation. Awaiting the learner's approval before Phase 2 (architecture and implementation plan).

**Approval (2026-10-10).** The learner approved Phase 1 and supplied a six-feature list: (1) intelligent enemy AI with A*/BFS, (2) player-behaviour prediction with logistic regression, random forest and KNN plus precision/recall/F1, (3) next-move prediction as an optional second model with temporal splitting, (4) adaptive difficulty with explainable rules and an optional learned-policy comparison, (5) a Q-learning agent compared with a random baseline, (6) an explainable dashboard with feature importance and learning progress. Revision 3 of `01_requirements.md` adopts the list: items 1, 2, 4 and 6 were already essential; 3 and 5 become planned extensions sequenced after the MVP; FR8, FR13–FR15, S11–S12 and the deliverables were added.

---

## Entry 2 — Phase 2: architecture and implementation plan (2026-10-10)

**1. Problem to solve.** Decide how the game, the data pipeline, the models, the adaptation controller, the RL environment and the dashboard fit together so that each later phase adds one testable piece without rework.

**2. Prompt given to the coding assistant.** "ok approved and need these ai features as well you can build it step by step" followed by the six-feature list, on top of the brief's Phase 2 definition (architecture, folder structure, data flow, dependencies, incremental checklist).

**3. Generated output.** `02_architecture.md`: a spine built around a pure `step()` function; package layout under `src/neuroquest/` (core, ai, data, sim, ml, adapt, rl, dashboard, ui); data flow for play, dataset building, training, A/B experiments, RL and the dashboard; data contracts for events, features, decisions and difficulty bounds; dependency table; an increment-by-increment checklist for Phases 3 to 11 including the new 7b (next-move) and 9b (RL) sub-phases.

**4. What worked and what failed.** The learner approved the architecture without changes on 2026-10-10, including the two flagged choices.

**5. Error or limitation discovered.** None at this stage. Two choices were flagged and confirmed: pygame-ce instead of classic pygame (same `import pygame`, actively maintained, installs cleanly on 3.13), and the `src/` layout with an editable install.

**6. Revised output.** No revision requested.

**7. Tests performed.** None: no code in Phase 2.

**8. Final improvement.** Approved; Phase 3 started.

---

## Entry 3 — Phase 3: environment setup (2026-10-10)

**1. Problem to solve.** A reproducible development environment that does not touch the system Python, with the package importable the same way by the game and by the tests, and a smoke test that proves every dependency works headlessly.

**2. Prompt given to the coding assistant.** "approved" (closing Phase 2), applied to the brief's Phase 3 definition: check the environment, provide install commands, create the minimal structure, add dependency and configuration files, run a smoke test, change nothing unrelated.

**3. Generated output.** `pyproject.toml` (setuptools build, dependencies with upper bounds, `dev`/`rl`/`web` extras, console script, pytest config), `.gitignore`, `README.md` (setup section), `src/neuroquest/__init__.py` (`__version__`), `__main__.py`, `cli.py` (`--version`, `check`), nine sub-package `__init__.py` files with one-line responsibilities, `tests/conftest.py` (headless drivers set before any import), `tests/test_smoke.py` (5 tests), empty `data/`, `models/`, `reports/` folders with `.gitkeep`, and `requirements.txt` frozen from the install.

**4. What worked and what failed.** Everything ran first time: `python3.13 -m venv .venv`; `pip install -e ".[dev]"` resolved 25 packages; `python -m neuroquest check` printed pygame-ce 2.5.8, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, matplotlib 3.11.2 and "pygame headless init: ok" with exit code 0; the `neuroquest` console script works; `pytest` reported 5 passed in 1.99 s. Nothing outside the `neuroquest/` folder was changed.

**5. Error or limitation discovered.** pip printed a notice that a newer pip exists (25.2 → 26.2.1); harmless, left as is. The installed numpy and matplotlib are newer than the minimums in the architecture table; the table now records the installed versions.

**6. Revised output.** None needed.

**7. Tests performed.** `python -m neuroquest --version`, `python -m neuroquest check` (headless Pygame surface fill and read-back), `pytest` (package version, `--version` flag, help on no command, `check` command, headless surface). All executed; all passed.

**8. Final improvement.** Working environment delivered. Approved by the learner on 2026-10-10; Phase 4 started.

---

## Entry 4 — Phase 4: minimum viable game (2026-10-10)

**1. Problem to solve.** A playable game with the six MVP mechanics (maze rendering, movement, walls and collisions, rewards, exit and win, loss) built as pure logic plus a thin Pygame layer, each mechanic tested before the next.

**2. Prompt given to the coding assistant.** "approved" (closing Phase 3), applied to the brief's Phase 4 definition and the architecture checklist: seeded maze → movement and walls → rewards, exit, win → health, timer, loss → rendering with placeholder graphics and the loop → play-through with screenshots.

**3. Generated output.** `config.py` (GameConfig, DifficultyParams), `core/maze.py` (Maze: generation, braiding, queries, flood fill), `core/entities.py` (Player), `core/state.py` (Action, Outcome, Event, GameState), `core/rules.py` (new_game, step, apply_damage, time helpers), `ui/input.py`, `ui/render.py`, `ui/app.py`, the `play` sub-command in `cli.py`, tests `test_maze.py`, `test_rules.py`, `test_ui.py`, `docs/04_mvp.md`, five screenshots.

**4. What worked and what failed.**
- First run of the maze and rules tests: 39 passed, 2 failed. Both failures were mistakes in the tests, not the code: the "ragged rows" fixture had rows of equal length, and one assertion expected neighbours in a different order from the documented up/down/left/right order.
- First run after adding the Pygame layer: 50 passed, 3 failed with `pygame.error: Invalid font`. A real bug: `render.py` cached `Font` objects at module level, and after one test called `pygame.quit()` the next test re-initialised Pygame and reused dead fonts. The scripted screenshot run did not show it because it used a single Pygame session.

**5. Error or limitation discovered.** The font-cache bug above would also have appeared in the real game if anything re-initialised Pygame mid-process (for example an in-process restart in a future test). Health can only fall through `apply_damage()` until enemies exist.

**6. Revised output.** The two test fixtures were corrected. Text is now drawn through `_render()`, which catches the `Invalid font` error, clears the cache and rebuilds the font; the overlay and HUD hint use it.

**7. Tests performed.** `pytest`: 53 passed in 5.33 s (test_maze 29 incl. 20 seeded connectivity cases, test_rules 13, test_ui 6, test_smoke 5). Scripted headless play-through: rewards collected (score 20 at tick 13), exit reached (won, tick 58), timeout (tick 10 at a 1-second limit), defeat (three damage points); five screenshots rendered through the real `draw()`.

**8. Final improvement.** MVP delivered with all six mechanics tested. The learner played it, remarked that it is "a very simple game" at this stage (expected: no enemies yet), and approved on 2026-10-10.

---

## Entry 5 — Phase 5: enemy intelligence (2026-10-11)

**1. Problem to solve.** Enemies that find valid routes through the maze and pursue the player, built on a search algorithm whose correctness is proven against a reference and whose advantage over a naive baseline is measured, not asserted.

**2. Prompt given to the coding assistant.** "approved" (closing Phase 4), applied to the brief's Phase 5 definition (A* and enemy behaviour; tests for clear, blocked and unreachable paths, boundaries and different layouts; comparison with a simple baseline) and the architecture checklist.

**3. Generated output.** `ai/graph.py` (path reconstruction; Manhattan distance lives in `core/maze.py` so core never imports ai), `ai/bfs.py`, `ai/astar.py`, `ai/policies.py` (`EnemyPolicy` protocol, `AStarChaser`, `RandomWalker`, registry), `ai/benchmark.py` (chase and search benchmarks, summaries, CSV), enemies in `core/entities.py`, `core/state.py`, `core/rules.py` (placement, movement interval, contact damage with invulnerability, `near_enemy` events, validation of policy moves), rendering of enemies and the hit blink in `ui/render.py`, `play` options and the `benchmark-enemies` command in `cli.py`, tests `test_astar.py`, `test_policies.py`, `test_benchmark.py`, `docs/05_enemies.md`, two screenshots, `reports/enemy_baseline.csv`, `reports/search_benchmark.csv`.

**4. What worked and what failed.** First suite run after wiring enemies: 279 passed, 1 failed. The failure was `test_game_loop_ends_when_time_runs_out`: with two A* chasers now in every default game and a motionless headless player, the player was caught (lost on health) before the 1-second clock ran out. The game was behaving correctly; the test was written for an enemy-free world and now sets `enemy_count=0`. Final run: 283 passed in 4.57 s.

**5. Error or limitation discovered.** The architecture had planned path caching for the chaser; measurement showed an A* call costs about 0.15 ms on this maze, so the chaser is stateless and recomputes every move (simpler, always current). The search benchmark also showed that on a 21×15 maze A* expands 21 % fewer cells than BFS but takes the same wall time; this is recorded as is rather than claimed as a speed-up.

**6. Revised output.** Test fixed as above; architecture checklist updated to say "stateless recompute" instead of "path caching".

**7. Tests performed.** `pytest`: 283 passed (test_astar: 6 fixed-layout cases × 2 algorithms, 200 seeded mazes, expansion comparison, timing; test_policies: 13; test_benchmark: 3; earlier suites unchanged). `benchmark-enemies`: A* chaser caught the motionless player in 100/100 mazes in exactly the shortest-path length (ticks per path cell 1.00); random walker 1/100 within 600 ticks; BFS and A* agreed on path length in 200/200 mazes; expansions 73.8 vs 58.2; time 0.142 vs 0.145 ms per call, A* max 0.395 ms.

**8. Final improvement.** Enemies with proven-optimal pathfinding and a measured baseline comparison. Approved by the learner on 2026-10-11.

---

## Entry 6 — Phase 6: gameplay data collection (2026-10-11)

**1. Problem to solve.** A reliable event and statistics pipeline: episode logs, a documented feature set that works on any prefix of an episode, a schema with validation, a transparent labelling strategy, and a reproducible synthetic dataset inspected before any model is trained.

**2. Prompt given to the coding assistant.** "approved" (closing Phase 5), applied to the brief's Phase 6 definition and the architecture checklist (events, features, schema, profiles, harness, dataset build, inspection).

**3. Generated output.** `data/events.py` (EpisodeLog JSONL, EventLogger), `data/features.py` (14 features), `data/labels.py` (profile labels; rule labeller for humans), `data/schema.py` (ranges, `validate_row`, schema file), `data/dataset.py` (rows per prefix, build, summary), `sim/profiles.py` (aggressive, cautious, explorer with randomised parameters; MaskedMaze routing around enemies), `sim/harness.py` (`run_episode`, `run_batch`), `bfs_nearest` in `ai/bfs.py`, `simulate` and `build-dataset` commands, `scripts/inspect_dataset.py`, tests `test_events.py`, `test_features.py`, `test_profiles.py`, `test_dataset.py`, `docs/06_dataset.md`, `data/processed/episodes.csv`, `data/schema/features.schema.json`, `reports/figures/dataset_features.png`.

**4. What worked and what failed.** First suite run: 301 passed, 1 failed in the feature test with a known-answer episode; the expected values were miscounted (the player is near the enemy from tick 2 on, 5 of 6 ticks, and 2 of 3 near-enemy moves approached). The code was right; the test was corrected. The simulation of 360 episodes took 13 s, far below the minutes estimated. The build produced 1,440 rows with 0 invalid.

**5. Error or limitation discovered.** Inspection of the summary showed `prefix_ratio` in the feature list. It is not observable during play (the game does not know when the episode will end), so a model using it would be unusable live. It was moved to row metadata. Inspection also showed the rule labeller agrees with the true profile in only 60 % of full episodes (cautious 25 %): intuition-written thresholds are a weak labeller, which is the reason synthetic labels come from the generating profile. `elapsed_s` is confounded with episode length at full prefix; it stays (observable) but Phase 7 will report an ablation without it.

**6. Revised output.** `prefix_ratio` removed from `FEATURE_NAMES` and the schema, kept as a column; dataset rebuilt from the unchanged logs; feature changes versus the architecture proposal documented in `06_dataset.md`.

**7. Tests performed.** `pytest`: 302 passed in 5.3 s (test_events 3, test_features 6, test_profiles 7 incl. a behavioural check that profiles differ on average over 6 seeds each, test_dataset 2). Dataset generation and inspection as recorded in `docs/06_dataset.md`: balanced classes (480 rows each), outcomes per profile, per-feature separability from 3.21 (damage_per_tick) down to 0.34 (reward_share), rule-labeller agreement 0.60 / 0.44 at full / quarter prefix.

**8. Final improvement.** A validated, reproducible, inspected dataset with ground-truth labels and a documented weak-labeller baseline. Approved by the learner on 2026-10-11.

---

## Entry 7 — Phase 7: machine learning (2026-10-11)

**1. Problem to solve.** A reproducible pipeline that splits without leakage, compares several models against meaningful baselines, evaluates once on held-out data with the right metrics, inspects its mistakes, saves the chosen model with metadata, and puts live predictions into the game.

**2. Prompt given to the coding assistant.** "approved" (closing Phase 6), applied to the brief's Phase 7 definition and the two checks promised in Phase 6 (rule labeller as a baseline; ablation without `elapsed_s`).

**3. Generated output.** `ml/split.py` (seed-grouped split), `ml/evaluate.py` (metrics, accuracy by prefix, confusion pairs, worst errors, permutation and coefficient importances), `ml/train.py` (majority, rule labeller, single-feature tree, logistic regression, random forest, KNN; grouped CV; selection before test; ablation; model + meta + report), `ml/predict.py` (loader with feature-list check, probabilities), `live_features` in `data/features.py`, the `train` command, live prediction and human-episode logging in `ui/app.py`, the prediction footer in `ui/render.py`, tests `test_ml.py` and three UI tests, a session-scoped tiny model fixture, `docs/07_ml_evaluation.md`, `models/behaviour_clf.joblib` + `.meta.json`, `reports/metrics.json`, two screenshots.

**4. What worked and what failed.** The ML tests passed first time on a tiny dataset. The UI test that needs a real model errored in fixture setup: the tiny session model had 4 training seeds, fewer than the 5 cross-validation folds (`GroupKFold` refuses). Fixed by making the fold count adapt to the number of groups and by giving the fixture 8 seeds. Final suite: 310 passed in 11.6 s. Training on the full dataset took seconds and produced the results in `docs/07_ml_evaluation.md`: logistic regression selected by CV (0.906 ± 0.015), test accuracy 0.934, macro-F1 0.933; majority 0.167, rule labeller 0.545, single-feature tree 0.618, KNN 0.902, random forest 0.924.

**5. Error or limitation discovered.** Logistic regression and random forest are tied within one CV standard deviation; the choice is defensible on interpretability and speed but not on the score alone. Nineteen test errors, eleven of them explorers predicted as aggressive early in quiet episodes, where the discriminating features (enemy proximity, damage, approach) carry no information yet; the model is overconfident there. `elapsed_s` is unimportant (ablation 0.940 without it), which closes the leakage concern from Phase 6.

**6. Revised output.** Adaptive fold count in `cv_scores` and the single-feature search; fixture size. No change to the protocol after seeing the test results (that would be selection on the test set).

**7. Tests performed.** `pytest`: 310 passed (test_ml 5: grouped split properties, metric helpers on known answers, the full protocol on a tiny dataset, predictor load/mismatch/predict, live features; test_ui +3: footer rendering, headless loop with a real tiny model producing 3 predictions and one saved human log, graceful status without a model). `neuroquest train` on the real data as recorded above; two headless screenshots show the footer predicting an explorer and an aggressive profile.

**8. Final improvement.** An honestly evaluated classifier (+0.77 macro-F1 over majority, +0.3 over the best hand rule) with inspected errors and an ablation, saved with metadata and predicting live in the game. Approved by the learner on 2026-10-11; Phase 8 chosen next, 7b deferred.

---

## Entry 8 — Phase 8: adaptive difficulty (2026-10-11)

**1. Problem to solve.** Connect performance statistics and the style prediction to a difficulty controller that is bounded, explainable and logged, and measure fixed against adaptive difficulty on several player profiles without claiming more than the numbers support.

**2. Prompt given to the coding assistant.** "approved" (closing Phase 7), applied to the brief's Phase 8 definition and the architecture checklist.

**3. Generated output.** `adapt/bounds.py`, `adapt/controller.py` (Snapshot, Decision, rule table, lever choice, AdaptiveController), `core/rules.py::reconcile`, adaptive mode in `sim/harness.py`, `sim/ab.py` (A/B runner, summary, deltas), controller wiring with a decision log and a two-line footer in `ui/app.py` / `ui/render.py`, `ab-test` command and `--fixed`, tests `test_adapt.py` and `test_ab.py`, `docs/08_adaptation.md`, `reports/ab_results.csv`, two screenshots.

**4. What worked and what failed.** The controller tests passed first time. The first A/B run exposed two problems. (a) Adaptive scores were about 50 % above fixed scores while win rates were lower: `reconcile` topped up rewards on every parameter change, refunding collected rewards; the comparison was invalid until rewards were added only on a density increase. (b) With that fixed, controller v1 made explorers lose 20 points of win rate (80 → 60 %) and did not help aggressive players: it tightened at 5 s on players who had not yet met an enemy, and early misclassifications chose the wrong lever. Three test failures along the way were mine: an assertion that unpacked a params-history row containing a `t` key into `DifficultyParams`, a summary-text check for a column label I had renamed, and two headless loop tests that now had the real model available and got a 20-second extension from R2 (they run with the controller off).

**5. Error or limitation discovered.** v2 (3 s cadence; tightening only after an encounter, never in the first 15 s, at most once per 15 s) is safe but not shown to improve outcomes: +2 / +3 / +5 points of win rate are inside ±20 points of sampling noise at 40 episodes per cell. Easing dominates (R1 fires 3–4× more often than R3). Aggressive bots die within ~10 s, so even 3-second decisions cannot do much for them.

**6. Revised output.** Reward top-up only on a density increase (with `previous` params); controller v2; the A/B rows record the rules that fired and the summary shows outcome breakdowns, which is how the v1 problem was diagnosed.

**7. Tests performed.** `pytest`: 321 passed (test_adapt 8: bounds, every rule, levers, moves, decision records, 1,000 random decisions never out of bounds, schedule and live snapshot, reconcile incl. the refund case; test_ab 3: adaptive episodes in the harness, A/B rows/CSV/summary/deltas, the game loop writing decisions). A/B experiments as recorded in `docs/08_adaptation.md`: v1 and v2, 3 profiles × 40 unseen seeds × 2 conditions each.

**8. Final improvement.** A bounded, explainable, logged controller whose first version measurably harmed one player type and whose second version does not, with the comparison reported as inconclusive on improvement. Awaiting approval before Phase 9 (dashboard and explainability).

---

## Entry 8b — Visual design pass (2026-10-11, requested by the learner between Phases 8 and 9)

**1. Problem to solve.** The learner found the placeholder graphics "so normal" and not engaging for an audience. Make the game attractive and make the AI visible, without touching rules, data, models or the controller.

**2. Prompt given to the coding assistant.** "can we work on ui side its looks so normal it's not attract the audience right now can we more interesting".

**3. Generated output.** A rewritten `ui/render.py` (dark theme, bevelled walls, fading trail, glowing portal, pulsing coins, player with eyes, hunters with spikes and glowing eyes, HUD with hearts and a time bar, footer with style-probability bars, intro banner, pause panel, end panels, particles, shake, flash, AI-vision overlay of the hunters' A* paths on key T), effect state and event-to-effect mapping in `ui/app.py`, `docs/UI_DESIGN.md`, six screenshots.

**4. What worked and what failed.** First run: one pixel test failed because the new 3-second intro banner overlapped the start cell on the small test surface. The fix was in the design: the intro is an app effect and is painted only when the app passes `fx`, never in a bare `draw()`. The first screenshots showed three problems: corridors darker than walls (reads backwards), a hit flash too strong for a still frame, and key hints colliding with the style bars in the footer. Fixed by flipping the floor/wall contrast and brightening the trail, softening the flash, and moving the controls to the intro banner and a pause overlay.

**5. Error or limitation discovered.** The pixel-test contract (base colour at each cell centre) constrained the drawings and was kept; eyes, highlights and sparks are offset from centres. No assets were added, so the look depends on a system font (Avenir Next on this Mac, with fallbacks).

**6. Revised output.** As above; `draw()` gained optional `fx` and `paused` arguments; defaults keep every existing call working.

**7. Tests performed.** `pytest`: 321 passed. Headless renders of six screens reviewed visually (start, playing, AI vision, hit, paused, escaped).

**8. Final improvement.** The game now looks like a game and shows its AI (paths, style probabilities, decisions) on screen. Phase 9 still awaits approval.

---

## Entry 8c — UI redesign to a premium sci-fi identity (2026-10-11, requested by the learner between Phases 8 and 9)

**1. Problem to solve.** The learner supplied a detailed UI/UX brief: a polished, cohesive sci-fi survival interface with a professional HUD, a maze as the centrepiece, distinctive player and enemy designs, premium collectibles and exit, an intelligence panel that presents only real AI outputs, redesigned game states, a consistent design system, and all gameplay and AI/ML functionality preserved.

**2. Prompt given to the coding assistant.** The brief "Act as a senior game UI/UX designer, Python/Pygame developer, and visual design specialist…" (eight steps, acceptance criteria), followed by "proceed" after the Step 1 inspection and proposal.

**3. Generated output.** `docs/UI_REDESIGN_PLAN.md` (inspection findings, files, steps); `ui/theme.py` (palette, type scale, spacing, radii, card/bar/glow/vignette/edge-pulse/text helpers, six vector icons); `ui/render.py` rebuilt (adaptive layout with panel reservation, board pre-rendered per maze with grid, wall edges, shadows and vignette, player light, drone sprites with pursuit and proximity feedback driven by the real distance, crystals, portal, HUD cards, bottom bar, menu, pause, end panels with a stats grid); `ui/panel.py` (intelligence panel from `Prediction`, `DifficultyParams`, `Decision` records and model metadata; humanised rule names; word wrap and elision); `ui/app.py` (menu state, I toggle, panel-aware effect positions, decision history, hit and change counters, model meta); `cli.py` (`--no-panel`, `--no-menu`); `tests/test_ui.py` (one new test for the panel, menu, pause and end screens); `docs/UI_DESIGN.md`; screenshots `ui2-*.png`.

**4. What worked and what failed.** Every step was rendered headlessly and reviewed. Defects found in the renders and fixed: the STYLE badge colliding with the controls hint (Step 4); the full decision text overflowing the bottom bar (Step 5, fixed in Step 6 by a short humanised form with the full reason in the panel); the menu's chip icon overlapping the title, a panel line overflowing its width, the PAUSED badge colliding with the STYLE badge, and a long style value spilling out of its stat card (Step 7). One test failure was my own: the new test checked the footer pixel after drawing the defeat overlay, whose veil darkens the whole frame; the check now runs before the overlays. No game-logic test changed. The real-window feel is for the learner to judge.

**5. Error or limitation discovered.** The brief's example explanation ("recent performance indicates a high success rate") describes a rule the controller does not have; the panel shows the controller's recorded reasons verbatim instead. Frame cost rose from 1.8 ms to between 1.6 and 4.9 ms depending on effects, within the 6 ms target. The look depends on system fonts (Avenir Next here) with fallbacks.

**6. Revised output.** As above, iterated per step.

**7. Tests performed.** `pytest`: 322 passed (321 existing + 1 new) in 12.9 s. Headless renders of the start menu, intro, playing, AI vision, hit, pause, panel open, victory and defeat; frame-time measurements per step.

**8. Final improvement.** A consistent premium sci-fi interface in which the AI is visible and truthful: hunters' A* paths, the classifier's probabilities and metadata, the controller's knobs within their bounds and its decisions with reasons. Phase 9 (dashboard) still awaits approval.

---

## Entry 8d — Gameplay-screen visual fixes (2026-10-11, learner's bug report after playing)

**1. Problem to solve.** In the real window the gameplay screen showed a huge cyan circle around the player, a near-black maze with walls reading as isolated blue squares, "floating" tiles around the player, and collectibles that appeared to overlap.

**2. Prompt given to the coding assistant.** The learner's brief "Act as a senior Python/Pygame game developer and professional game UI designer… fix the gameplay visuals…" with a specified palette (#0B1020 background, #141B30 floor, #273451 walls, #344461 highlights, #36E4FF player) and an eight-step procedure.

**3. Generated output.** Targeted edits in `ui/render.py` and two palette values in `ui/theme.py`; no other file changed. Screenshots `ui3-playing.png`, `ui3-zoom.png`, `ui3-panel-vision.png`; the diagnostic crop `ui2-issue-zoom.png`.

**4. What worked and what failed.** Root causes were established with evidence before changing anything. The circle was `_draw_player_light()`, a decorative glow of radius 2.6 cells (plus a 0.85-cell glow on the sprite); removed outright, since it had no gameplay role. The "black maze with blue squares" came from walls coloured almost like the frame, a visited-cell trail drawn as inset squares at up to 75 % strength, and grid lines on every floor tile; the maze now uses the specified palette, walls are one slate fill with highlights only along wall↔floor boundaries (continuous structures), floors have two tones and a faint shade along walls, and the trail is a flat translucent tint. Collectible placement was verified clean over 200 seeds (0 duplicate cells, 0 on walls, 0 shared with player/enemy/exit, including controller top-ups); the visual overlap came from crystal glows and sparks spilling into neighbouring cells, so crystals are smaller and contained. Tile alignment was verified exact (0 of 315 cells misaligned; the renderer uses the rules' grid). Enemy, portal and player glows were restricted to their own cells.

**5. Error or limitation discovered.** The headless renders of entry 8c had looked acceptable; the learner's display made the same frames read as black with floating blocks. Lesson recorded: judge gameplay visuals at the real window's contrast, and keep decorative light effects inside the sprite's cell.

**6. Revised output.** As above. Frame cost 2.05 ms with the panel and AI vision on.

**7. Tests performed.** `pytest`: 322 passed, unchanged (the pixel contracts at the cell centres held through the palette change because `PLAYER` follows the theme's cyan). Step-7 checks through the real code: wall bump event; three profile episodes to completion with adaptive difficulty and the real model (aggressive: lost on health at tick 136 after 13 rewards, 3 hits, 1 difficulty change; cautious: won at 373; explorer: won at 179); motionless player lost on time at tick 20 with a 2-second limit; three fast hunters defeated a motionless player at tick 73; the real app loop headless at 200 ticks/s produced 12 predictions, 8 logged decisions and one saved human log in 1.7 s.

**8. Final improvement.** The gameplay screen matches the start screen's identity and reads as a maze again, with all AI/ML features untouched. Phase 9 still awaits approval.

**Follow-up (same day).** The learner's screenshot of the real window showed a menu card line ("…your style every 2 s (held-out F1 0.93).") escaping its card: the card text was placed line by line without measuring, and the real window's font metrics are slightly wider than the headless render's. Fix: every card line is word-wrapped to the card width, the menu panel sizes itself to the window (up to 780 px), and pause-panel descriptions are elided to the available width; verified at 1024×768 and 900×650. The large teal, amber and red circles visible behind the menu in that screenshot are the previous build's glows (the removed player light and the old crystal/hunter glows); the current code cannot draw them, so that window predated the fix and needed a restart.

---

## Entry 8e — "Bright arcade" visual direction (2026-10-14)

**1. Problem to solve.** The learner still disliked the gameplay look ("don't like the ui of game"). Rather than iterate blind again, a two-question check established what bothered them (maze too dark and dull; too much glow and neon clutter) and the direction they wanted (bright arcade, over neon cyberpunk, flat board-game and pixel-art options).

**2. Prompt given to the coding assistant.** "don't like the ui of game we can imporve it", plus the two answers above.

**3. Generated output.** Renderer-only changes in `ui/render.py` and two palette values in `ui/theme.py`: light blue-grey corridors, deep navy walls with crisp edges and a board outline, no vignette, a light trail tint; outlined glow-free sprites (robot, hunter darts with a "!" when within 3 cells, crystals, portal); the screen-edge pulse replaced by a thin crimson board frame for proximity and hits; fewer particles; overlay and menu glows removed. `docs/UI_DESIGN.md` updated; screenshots `ui4-*.png`.

**4. What worked and what failed.** The pixel tests caught two drawing mistakes on the way: the robot's visor band reached the centre pixel at small cell sizes, and the portal's centre highlight sat exactly on the centre. Both were fixed in the drawings (the visor is clamped to end above the centre; the highlight is offset), not by weakening the tests. Frame cost fell from 2.05 ms to 0.9 ms.

**5. Error or limitation discovered.** Taste cannot be verified headlessly; the two-question check is the cheapest way found to stop guessing. The learner's final judgement is still pending.

**6. Revised output.** As above.

**7. Tests performed.** `pytest`: 322 passed. Headless renders reviewed: playing, zoom on the player, a hit with the panel and AI vision open, the end screen.

**8. Final improvement.** A brighter, cleaner, glow-free arcade look with the same truthful AI presentation. The learner moved on to the next phase on 2026-10-15 ("work on next phase"), taken as approval of Phase 8 and the UI.

---

## Entry 9 — Phase 9: dashboard and explainability (2026-10-15)

**1. Problem to solve.** Clear charts and summaries of gameplay behaviour, predictions and evaluation; an explanation of the model's limitations with real examples of correct and incorrect predictions; a way to inspect how the AI reached its decisions.

**2. Prompt given to the coding assistant.** "work on next phase", applied to the brief's Phase 9 definition and FR12/FR15.

**3. Generated output.** `dashboard/figures.py` (nine figures from recorded data: class balance and outcomes, model comparison, confusion matrix with per-class scores, permutation importance, per-class coefficients, accuracy by prefix, a prediction timeline that re-runs the saved model over a recorded episode on the live schedule, a difficulty timeline with rule markers, fixed-vs-adaptive bars), `dashboard/report.py` (a self-contained HTML page with embedded figures, example tables recomputed with the saved model on the held-out seeds, decision tables with the recorded reasons, limitations; every section degrades to a note when its input is missing), the `dashboard` command, `tests/test_dashboard.py`, `docs/09_explainability.md`, `reports/dashboard.html`, `reports/figures/dash_*.png`, `docs/screenshots/phase9-dashboard.png`.

**4. What worked and what failed.** All figures rendered from the real files first time. Reviewing them showed one weak choice: the difficulty figure used the *latest* human session, which had no changes, on an auto-scaled axis that made a flat line look odd. Changed to the session with the most changes and fixed axes within the controller's bounds.

**5. Error or limitation discovered.** In the recorded human play (57 decisions over 10 sessions) the controller only ever eased (9 times) or held; no tightening fired, because the sessions were short and the player was being hit. That is correct behaviour, but it means the dashboard's human difficulty timeline shows easing only; the simulated A/B section covers tightening.

**6. Revised output.** As above.

**7. Tests performed.** `pytest`: 324 passed (test_dashboard 2: every figure from tiny real data, the report with all inputs and with none). Real build: 7 sections, 9 figures, 0.5 MB HTML; screenshot of the rendered page reviewed.

**8. Final improvement.** An offline, shareable explainability report that pairs with the in-game panel. The learner approved it on 2026-10-15 ("i like the dashboard") and asked that the final documentation link to it with the repository-root path `neuroquest/reports/dashboard.html`, because the project will be hosted later. Added to the root README, the NeuroQuest README and this phase's document; the Phase 11 documentation set must carry the same link. Next: Phase 9b or Phase 10, the learner's choice.

---

## Entry 9b — Phase 9b: reinforcement learning agent (2026-10-15)

**1. Problem to solve.** The brief's optional Feature 5: a small maze environment with an explicit state, action space, reward function and termination, in which an agent learns to collect rewards, avoid hazards and reach the exit, compared against a random or rule-based baseline (FR14; success criterion S12: 100 held-out episodes, success rate and average return against the random baseline, with the learning curve).

**2. Prompt given to the coding assistant.** "9b" (the learner's choice at the Phase 9 gate).

**3. Generated output.** `core/rules.py`: one optional keyword, `new_game(..., layout_seed=)`, which fixes the maze, exit and rewards while the seed still varies the enemy start (default behaviour unchanged, tested). `rl/env.py` (`MazeEnv`: gym-style `reset`/`step`, the observation encoding, `RewardSpec`, termination vs truncation, `describe()`); `rl/qlearning.py` (`QTable` with npz save/load, `QConfig` with the ε schedule, `train`, `evaluate`, `summarize`, Wilson interval); `rl/agents.py` (random, A*-to-exit, greedy collector, the Phase 6 cautious profile, `QAgent`); `rl/pipeline.py` (`run_pipeline`: train, save, evaluate every agent on the same held-out seeds, test a never-seen board, write `rl_curve.csv`, `rl_eval.csv`, `rl_results.json`, `q_table.npz` + meta); CLI `rl-train` and `play --agent`; `ui/app.py` gained `agent` and `layout_seed` arguments (the agent chooses the action instead of the keyboard; the footer says so); `dashboard/figures.py` (`fig_rl_curve`, `fig_rl_eval`) and section 8 of the dashboard, with Limitations renumbered to 9 and extended; `tests/test_env.py` (8), `tests/test_qlearning.py` (6), one more dashboard test; `docs/09b_rl.md`; screenshots `phase9b-learning-curve.png`, `phase9b-agents.png`, `phase9b-agent-playing.png`.

**4. What worked and what failed.** The learner itself worked first time; the first *board* did not. On an 11×9 maze with the game's default 15 % braiding, every agent, scripted or learned, took exactly one hit: the maze is one corridor and the hunter always starts on the only route to the portal, so "avoid the hazard" was impossible and the learned agent could only reproduce the collector. A scan of 36 layouts measured, for each, how often a hunter-blind collector is hit against how often the hazard-aware cautious profile is hit; fully braided layout seed 3 (collector hit 82 %, cautious 0 %) was chosen and is documented in the code and the phase document. On it the agent learned to avoid the hunter (0.04 hits per episode against 0.82) at the cost of about 1.4 crystals, with the highest mean return of all five agents, reproduced with a second training seed. One test mistake of my own: `pytest.approx` cannot compare a dict of lists, and the npz round trip is float32, so the save/load test compares row by row with a tolerance.

**5. Error or limitation discovered.** Layout seed 5 is a recorded counter-example: there the only safe route is a 56-step detour and, with γ = 0.95, the discounted optimum really is to take the hit; the agent converged to the collector's behaviour and scored below the cautious profile. The reward function and discount define "good play"; both are stated, neither was tuned. The table does not transfer (0 wins on layout seed 4), the hunter is deterministic (100 seeds = 20 situations), and nothing is claimed about the full-size game.

**6. Revised output.** As above: the board choice moved into `RL_CONFIG`/`LAYOUT_SEED` with the reason in a comment, the counter-example into the document.

**7. Tests performed.** `pytest`: 339 passed in 10.6 s (324 existing + 15 new; the tiny-board learning test trains 400 episodes and checks the greedy policy wins ≥ 90 % and beats random). Real run `rl-train`: 20,000 episodes in 12 s, 1,753 observations visited; evaluation table in `docs/09b_rl.md`; `dashboard` rebuilt: 8 sections, 11 figures, 0.75 MB; the three figures and the agent frame reviewed.

**8. Final improvement.** The game now contains an agent that learned to play it from the rules alone, with its state, actions, reward and termination written down, its learning curve recorded, and its behaviour compared with four baselines on the same seeds, including the honest parts (no transfer, a board where the discount prefers the hit). Approved by the learner on 2026-10-15 ("approved").

---

## Entry 10 — Phase 10: testing and improvement (2026-10-15)

**1. Problem to solve.** Unit, integration, gameplay-scenario, ML-evaluation, edge-case and regression testing, with a test report that lists test case, expected result, actual result, status and improvement made, and never reports a test as passed unless it ran.

**2. Prompt given to the coding assistant.** "approved" at the Phase 9b gate, which moved the work to Phase 10 as agreed.

**3. Generated output.** Five new test files: `test_scenarios.py` (10: the real window loop driven by a scripted player with keys injected as Pygame events: win, loss by health, loss by time, R, P, I/T/F12, Esc, window close, menu, a live difficulty change), `test_integration.py` (4: play → log → dataset → predict → decisions with the real saved model; dashboard from the repository data; byte-for-byte dataset and training reproducibility), `test_edge_cases.py` (14), `test_regressions.py` (6, one per fixed bug), `test_cli.py` (3, every command into a temp folder); two cases added to existing files. Code changes: `ui/theme.py` (font cache validated and rebuilt after a Pygame re-init; dead glow/vignette/edge-pulse helpers removed), `ui/panel.py` (`wrap` breaks overlong tokens), `ui/render.py` (dead import removed), `rl/qlearning.py` (unused `save_meta` removed), `cli.py` (`dashboard --figures-dir`), `pyproject.toml` and `requirements.txt` (`coverage`). Documents: `reports/test_report.md`, `reports/pytest_results.txt`, `reports/coverage.txt`, `docs/10_testing.md`, `docs/UI_DESIGN.md` (helper list corrected).

**4. What worked and what failed.** The first run of the new tests failed eight times; seven were my own wrong assumptions about the system (the fast test clock, where the loop records decisions, the frame delay of an injected key, the `rule:` prefix on human labels, the event order on the fatal tick, the heading count, a full board having no room for a hunter), corrected in the tests. One was real: `theme.font()` cached font objects that die when Pygame's font module is re-initialised; `text()` had a retry since entry 8b but `wrap()` and `elide()` call the font directly and crashed. The cache is now validated with a cheap call. The word-wrap edge case showed an overlong token spilling out of the panel; `wrap` now breaks it. Coverage then showed dead code, and I removed it, including the HUD bar's halo option, whose branch had never executed. The suite failed in five places: the time bar uses the halo in the last ten seconds of the clock, which no test had ever rendered. The option was restored and a test renders that state now. Pygame keeps one timer per event type, which silently replaced a resume timer with a quit timer in the pause scenario; the scenario posts Esc from the agent instead.

**5. Error or limitation discovered.** "Never executed in the suite" is not "never used": check callers before deleting. Coverage reports lines, not meaning; the 26 uncovered statements are listed in the report with what they are. No test opens a real window; the look and feel remain a manual checklist for the learner. Human-play accuracy of the classifier is still unknown for lack of labelled human data.

**6. Revised output.** As above.

**7. Tests performed.** Full suite under coverage, three times after the fixes: final run 377 passed, 0 failed, 0 skipped in 50.9 s; 99.1 % line coverage (2,955 statements, 26 missed), identical under coverage's default and C-tracer cores. The real classifier training was re-run into a scratch folder: every test metric identical to the saved report. Timings: `step()` 102 µs per tick with two A* hunters on 21×15; A* 0.285 ms; `draw()` 2.07 ms with panel and paths; dashboard build 1.8 s; training 8.0 s; RL training 12 s.

**8. Final improvement.** The project now has a test report that a reader can check line by line against executed runs, a suite that drives the real game loop, and two small robustness fixes found by that suite. Approved by the learner on 2026-10-15 ("approved").

---

## Entry 11 — Phase 11: documentation and final delivery (2026-10-15)

**1. Problem to solve.** The brief's final deliverables: project overview and problem statement, system architecture, AI/ML methodology, dataset description, prompt engineering log, AI-assisted coding iterations, testing and improvement report, limitations and future enhancements, setup and execution instructions, a demonstration script, the final README, and deployment instructions where a target is suitable. The learner's standing instruction: every final document links the dashboard with the repository-root path `neuroquest/reports/dashboard.html`, because the project will be hosted.

**2. Prompt given to the coding assistant.** "approved" at the Phase 10 gate.

**3. Generated output.** `docs/CAPSTONE_DOCUMENTATION.md` (the consolidated report, 17 sections, cover page with the learner's details as on the other two capstone projects); `docs/DEMO_SCRIPT.md` (an 8-minute walkthrough with what to say, the real numbers, and what to do if something fails); `docs/DEPLOYMENT.md` (publish the repository; GitHub Pages serves the dashboard at the root-relative path; run from a clone; regenerate everything; the browser build explicitly not attempted); `docs/PROMPT_ENGINEERING.md` (the prompts in order, what worked and what did not, the twelve coding iterations that mattered); `neuroquest/README.md` rewritten as the final README with two animations rendered from the real renderer (`docs/screenshots/neuroquest-play.gif`: the cautious profile on the default board with the panel and AI vision open and two difficulty changes at 24 s and 27 s; `neuroquest-agent.gif`: the agent crossing its board), a results table, and a documentation index; the root `README.md` links the report; the status lines of the approved phase documents now say so.

**4. What worked and what failed.** All numbers in the documents were taken from the recorded files (`metrics.json`, `ab_results.csv`, `rl_results.json`, `test_report.md`) or from the Phase 10 measurements, not from memory. The first gameplay animation came out at 18.7 MB; re-rendered at 640×480, every second tick, 128 colours: 2.7 MB with the two difficulty changes still inside. No code changed in this phase; the suite from Phase 10 stands.

**5. Error or limitation discovered.** The game is desktop-only, so what can be hosted is the documentation and the dashboard; the brief's deployment item is met for that target and declined, with the reason, for a browser build. The manual window checks of the test report remain for the learner.

**6. Revised output.** As above.

**7. Tests performed.** Relative links in the README, the final documents and the test report were checked against the file system (script in Phase 11). Suite unchanged: 377 passed, 99.1 % coverage (Phase 10 run).

**8. Final improvement.** A complete, cross-linked documentation set in which every claim points at the file that supports it, ready to publish with one git push and one GitHub Pages setting. The learner approved the final delivery on 2026-10-15 ("approved"); the project is closed. Left for the learner: the manual window checks in `reports/test_report.md` section 6, publishing per `docs/DEPLOYMENT.md`, and the optional Phase 7b next-move model.
