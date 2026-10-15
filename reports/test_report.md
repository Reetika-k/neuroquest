# NeuroQuest — Test report (Phase 10)

Date: 2026-10-15 · Suite: **377 passed, 0 failed, 0 skipped in 50.9 s** · Line coverage: **99.1 %** (2,955 statements, 26 not executed) · Raw outputs: `reports/pytest_results.txt`, `reports/coverage.txt`

Environment: macOS, Python 3.13.7 (python.org), pygame-ce 2.5.8, scikit-learn 1.9.1, pytest 8.4.2, coverage 7.16.2. Every test runs headless (`SDL_VIDEODRIVER=dummy`, Matplotlib Agg) from `tests/conftest.py`, so the suite needs no display.

Reproduce:

```bash
cd neuroquest
.venv/bin/coverage run --source=src/neuroquest -m pytest   # or just .venv/bin/pytest
.venv/bin/coverage report -m
```

Rules followed in this report: every status below comes from an executed run; nothing was marked passed without running; failures met along the way are listed in section 3 with what was changed.

## 1. Test inventory

| File | Cases | Area | Phase |
|---|---|---|---|
| test_smoke.py | 5 | package version, CLI help, headless Pygame | 3 |
| test_maze.py | 29 | grid queries, seeded generation, braiding, connectivity (parametrised over seeds) | 4 |
| test_rules.py | 12 | movement, walls, rewards, exit, timer, health | 4 |
| test_ui.py | 12 | key mapping, pixel contracts, game loop, model missing / out of date, panel and overlays | 4, 8c, 10 |
| test_astar.py | 214 | A* correctness vs BFS on 200 seeded mazes, edges, speed | 5 |
| test_policies.py | 13 | chaser and random walker, enemy interval, contact damage, invulnerability | 5 |
| test_benchmark.py | 3 | chase and search benchmarks, CSV | 5 |
| test_events.py | 3 | event logs round trip, bad files rejected | 6 |
| test_features.py | 6 | features on a known episode, prefixes, schema, rule labeller | 6 |
| test_profiles.py | 8 | scripted players, determinism, batch logs | 6 |
| test_dataset.py | 2 | dataset build, human rows rule-labelled | 6 |
| test_ml.py | 5 | grouped split, metrics, training protocol, predictor, live features | 7 |
| test_adapt.py | 8 | bounds, rules, levers, decision records, 1,000-step bound check, reconcile | 8 |
| test_ab.py | 3 | adaptive episodes, A/B rows, loop writes decisions | 8 |
| test_dashboard.py | 3 | every figure from tiny real data, report with and without inputs, RL figures | 9, 9b |
| test_env.py | 8 | MazeEnv: layout seed, observation, rewards, termination, guards | 9b |
| test_qlearning.py | 6 | Q-table, ε schedule, Wilson, learning on a tiny board, pipeline files, app hook | 9b |
| **test_integration.py** | 4 | play → log → dataset → predict → decisions; dashboard from repo data; reproducibility | 10 |
| **test_scenarios.py** | 10 | win, loss by health, loss by time, restart, pause, keys, Esc, window close, menu, adaptive change | 10 |
| **test_edge_cases.py** | 14 | smallest boards, empty/over-full parameters, walled-off enemy, window sizes, text, CLI repair, BFS, panel | 10 |
| **test_regressions.py** | 6 | one test per fixed bug (see 2.6) | 10 |
| **test_cli.py** | 3 | every command through `main()` into a temporary folder | 10 |
| **Total** | **377** | | |

Phase 10 added 37 cases in the five bold files plus 2 in existing files (out-of-date model; RL progress callback) and extended 1.

## 2. Test cases: expected, actual, status, improvement

Status is the result of the final run (section header). "Improvement" names the change made during Phase 10 when a test did not pass first time or revealed something.

### 2.1 Unit tests (by area; each row summarises the file's cases)

| Test case | Expected | Actual | Status | Improvement |
|---|---|---|---|---|
| Maze generation (29) | odd sizes only, wall border, fully connected, deterministic per seed, perfect maze has no loops, braiding opens dead ends | as expected on every seed tested | pass | — |
| Rules (12) | moves/bumps/revisits produce the right events; reward collected once; exit wins and freezes; time runs out on the last tick; health floors at 0 | as expected | pass | — |
| A* (214) | path length equals BFS on 200 seeded mazes; unreachable → None; start = goal; edges; expands fewer cells; < 1 ms | as expected; 0.285 ms median on 21×15 | pass | — |
| Enemy policies (13) | chaser takes the first step of the shortest path; moves only on its interval; invalid moves ignored; one hit per invulnerability window; defeat at 0 health | as expected | pass | — |
| Events, features, schema, labels (12) | JSONL round trip; bad files rejected; features on a hand-computed episode; prefixes; every row validates; rule labeller covers all classes | as expected | pass | — |
| Profiles and harness (8) | legal actions, sampled parameters, deterministic episodes, profiles differ on average, batch writes logs | as expected | pass | — |
| ML (5) | grouped split keeps seeds together; metrics on known answers; training writes model + meta + report; predictor checks the feature list; live features equal episode features at the end | as expected | pass | — |
| Adaptation (8) | clamp; each rule fires on its condition (incl. warm-up, no-encounter, cooldown holds); lever by style; records rule/lever/before/after/reason; 1,000 random decisions never leave the bounds; reconcile spawns/removes and tops up only when density rises | as expected | pass | — |
| RL environment and learner (14) | observation encoding; reward per event; truncation vs termination; guards; Q update math; save/load; ε schedule; Wilson; learns a tiny board (win ≥ 0.9, beats random) | as expected; tiny board win 1.0 after 400 episodes | pass | progress callback now covered |
| Dashboard (3) | every figure renders from tiny real data; report builds with all inputs and degrades to notes without | as expected | pass | — |

### 2.2 Integration tests

| Test case | Expected | Actual | Status | Improvement |
|---|---|---|---|---|
| Play → log → features → predict → adapt (real saved model, scripted player, 600 ticks) | log on disk loads; predictions arrive on the live schedule and equal a replay of the saved model; the log becomes 4 valid dataset rows; every decision logged every 3 s with inputs/rule/output inside the bounds | as expected; first prediction at tick 40 reproduced exactly by replay | pass | test corrected: the window loop logs decisions to a file, not to the log's metadata; human rows carry `rule:` labels |
| Dashboard from the repository's own data | all 8 sections present, 10 headings, > 100 KB | 8 sections, 10 headings, 0.75 MB | pass | — |
| Dataset reproducibility (S3) | same seeds → identical log bytes and identical CSV bytes | identical | pass | — |
| Training reproducibility (S3) | same seed → same split, same selected model, same test metrics | identical | pass | — |

### 2.3 Gameplay scenarios (through the real window loop, headless, 200 ticks/s)

| Test case | Expected | Actual | Status | Improvement |
|---|---|---|---|---|
| Win by reaching the exit (shortest-path player, no hunters) | outcome WON; log ends with `exit`; no bumps | WON | pass | — |
| Loss by health (motionless player, 3 fast hunters) | LOST_HEALTH; 3 damage events ≥ 15 ticks apart; `defeat` on the last tick | LOST_HEALTH, hits at 36/51/66 | pass | test corrected: `near_enemy` follows `defeat` within the tick, as the tick order documents |
| Loss by time (1 s limit) | LOST_TIME after exactly 200 ticks; `timeout` last | 200 ticks | pass | test corrected (200 ticks/s) |
| Restart with R | old log saved with seed 7, new game seed 8, Esc saves the second | two logs, seeds 7 and 8 | pass | tolerance for the frame delay of an injected key |
| Pause with P, resume by timer | ticks consecutive across the pause; one gap ≥ 0.2 s; normal gaps < 0.05 s | gap 0.25 s, ticks 0…≥ 30 | pass | Pygame keeps one timer per event type, so Esc is posted by the agent after resume |
| I / T toggles and F12 | panel strip changes after I; one screenshot PNG written | pixel changed; 1 file | pass | screenshot folder patched to a temp dir |
| Esc quits | returns PLAYING; log saved with ≥ 8 ticks | 14 ticks | pass | tolerance as above |
| Window close event | loop ends, log saved | as expected | pass | — |
| Start menu waits for Enter | no tick before Enter (≥ 0.19 s) | as expected | pass | — |
| Adaptive controller eases a player who keeps getting hit (tiny model, 5 health) | an `R1_ease` decision with before ≠ after; decisions every 30 ticks; 5 damage events | ease fired; 5 hits | pass | covers the live-change path of the loop, untested before |

### 2.4 ML evaluation (re-run during this phase into a scratch folder, compared with the saved report)

| Model | Test accuracy | Test macro-F1 | Same as saved `reports/metrics.json`? |
|---|---|---|---|
| majority baseline | 0.333 | 0.167 | yes |
| rule labeller | 0.535 | 0.545 | yes |
| single-feature tree | 0.622 | 0.619 | yes |
| KNN | 0.903 | 0.902 | yes |
| random forest | 0.924 | 0.924 | yes |
| **logistic regression (selected)** | **0.934** | **0.933** | yes |

Success criterion S4 (≥ 0.20 macro-F1 over the majority baseline): met by 0.767. Accuracy by prefix 0.917 / 0.931 / 0.931 / 0.958; ablation without `elapsed_s`: 0.940. RL criterion S12: 100 held-out episodes, Q-learning win rate 1.00 [0.96, 1.00] and return +7.32 against random 0.00 and −11.05, learning curve recorded (`docs/09b_rl.md`).

### 2.5 Edge cases

| Test case | Expected | Actual | Status | Improvement |
|---|---|---|---|---|
| 5×5 board, reward density 0, 50 enemies | 7 open cells; still 1 reward; enemies clamped; no shared cells; winnable | as expected | pass | — |
| Single open cell, exit on the start | every move bumps; STAY wins at tick 1 | as expected | pass | — |
| Walled-off enemy | never moves, never hurts | as expected | pass | — |
| Reconcile with no free cell / zero enemies / lower density | no spawn when full; fills free cells only; removes all; lowering density changes nothing | as expected | pass | test corrected (a full board has no room for the first hunter either) |
| 1 health, 1 s limit; finished game is inert | LOST_TIME at tick 10; step returns the same state | as expected | pass | — |
| Draw at 2560×1440 … 200×150 with panel and paths | no crash; player cell inside the surface | as expected | pass | — |
| HUD in the last 10 s; a fresh difficulty change in the footer | renders at three sizes | as expected | pass | **new** (found by coverage; see 3.4) |
| Word-wrap and elision | no rendered line wider than the width; elided text ends with … | as expected | pass | **wrap now breaks words wider than the width** (an overlong token could spill out of the panel) |
| Dashboard decisions section | shows the session with the most changes, not the latest | as expected | pass | — |
| CV folds vs groups | never more folds than groups, minimum 2 | as expected | pass | — |
| CLI repairs even dimensions | 20×14 → 21×15; flags pass through | as expected | pass | — |
| RL env on the smallest board | 1 crystal; shortest path wins; bound 14 | as expected | pass | — |
| BFS-nearest edge cases | empty goals / wall start → None; start in goals → [start]; stats on both paths | as expected | pass | — |
| Panel with changed and recent decisions | short decision text ASCII; panel draws the recent list | as expected | pass | test corrected (the middle dot is fine; only the arrow was the problem) |

### 2.6 Regression tests (one per fixed bug)

| Bug (development-log entry) | Test | Status |
|---|---|---|
| Cached text raised "Invalid font" after Pygame re-init (8b); `font()` still handed dead fonts to `wrap`/`elide` (10) | test_regressions::text_cache_survives_pygame_quit_and_reinit | pass |
| `prefix_ratio` was a feature but is unobservable live (7) | test_regressions::prefix_ratio_is_row_metadata_not_a_model_feature | pass |
| Decision summary used an arrow glyph the font lacks (8b) | test_regressions::decision_summary_uses_ascii_only | pass |
| Intro banner covered the start cell (8b) | test_regressions::intro_banner_does_not_cover_the_player | pass |
| RL board made the hit unavoidable (9b) | test_regressions::rl_board_keeps_the_hazard_real_and_avoidable; default_board_of_the_rl_size_really_is_a_single_corridor | pass |
| Rewards refunded on any parameter change (8) | test_adapt::reconcile_spawns_and_removes_enemies_and_tops_up_rewards (existing) | pass |
| Controller tightened before any encounter (8, v2) | test_adapt::each_direction_rule_fires_on_its_condition (existing) | pass |
| GroupKFold with fewer groups than folds (7) | test_edge_cases::cv_folds_never_exceed_the_number_of_groups | pass |
| Visor band / portal highlight on the centre pixel (8e) | test_ui::draw_paints_player_exit_rewards_and_background (existing pixel contracts) | pass |
| Difficulty figure showed a flat latest session (9) | test_edge_cases::dashboard_shows_the_session_with_the_most_changes | pass |
| Q-table float32 round trip (9b) | test_qlearning::qtable_update_greedy_and_round_trip | pass |
| `layout_seed` must not change default games (9b) | test_env::layout_seed_fixes_board_but_not_enemy_start | pass |

### 2.7 Command line

| Test case | Expected | Actual | Status | Improvement |
|---|---|---|---|---|
| benchmark-enemies, simulate, build-dataset, train (tiny arguments, temp folder) | exit 0; files written; summaries printed | as expected | pass | — |
| ab-test, dashboard, rl-train with the repository data (outputs to temp) | exit 0; CSV with 2 rows; dashboard > 100 KB with figures; Q-table and results | as expected | pass | `dashboard --figures-dir` added so a test never rewrites the repository's figures |
| play --agent without a Q-table | exit 1 with a hint to run rl-train | as expected | pass | — |

## 3. What testing found, and what was changed

1. **Eight first-run failures in the new tests, seven of them my own assumptions** (the fast clock is 200 ticks/s; the window loop records decisions in a file rather than in the log; an injected key acts on the next frame; human rows carry `rule:` labels; the fatal tick lists `near_enemy` after `defeat`; the dashboard has 10 headings; a full board has no room for a hunter). The tests were corrected; the code was right.
2. **One real fragility: the font cache.** `theme.font()` cached Pygame font objects that die when the font module is quit and re-initialised. `text()` had a retry since entry 8b, but `wrap()` and `elide()` call `font().size()` directly and crashed. `font()` now validates the cached font with a cheap call and rebuilds the caches when it is dead. Only test processes re-initialise Pygame, so no player could have hit it; it was still wrong.
3. **Word-wrap could overflow.** A single token wider than the panel was kept on its own line and spilled out. `wrap()` now breaks such a token between characters; the test asserts no rendered line is wider than the width.
4. **Dead code removed; one removal reverted.** The glow, vignette and edge-pulse helpers of the sci-fi look (entry 8c) had no caller since the bright-arcade redesign (8e), and `save_meta` in the RL module was never called; all were deleted. I also removed the `glow_alpha` option of the HUD bar because coverage showed its branch never executed, and the suite failed in five places: the time bar uses it in the last ten seconds of the clock, which no test had rendered. The option was restored and a test now renders that state. Lesson recorded: "never executed in the suite" is not "never used".
5. **A CLI option for testability.** `dashboard --figures-dir` lets a test build the dashboard without rewriting `reports/figures/`.

## 4. Coverage

25 of 38 modules are at 100 % (bounds, controller, benchmark, BFS, graph, policies, config, entities, maze, rules, state, events, features, labels, schema, predict, split, train, RL pipeline and learner, harness, profiles, app, input). The others:

| Module | Statements | Missed | Coverage | What is not executed |
|---|---|---|---|---|
| ui/theme.py | 107 | 7 | 93.5 % | fallback to the bundled font when no system font exists; the text-cache retry (now shadowed by the `font()` check) and the 600-entry cache flush |
| cli.py | 199 | 6 | 97.0 % | the `play --agent` branch that opens the real window (exercised with a stubbed window during Phase 9b, outside pytest); printing invalid dataset rows when there are none |
| ai/astar.py | 36 | 1 | 97.2 % | the stats counter on the unreachable-goal path |
| dashboard/figures.py, report.py | 376 | 2 | 99.5 % | coefficient figure for a model without coefficients; "figure not available" placeholder |
| data/dataset.py, ml/evaluate.py | 84 | 2 | 97.6 % | collecting an invalid row; impurity importance for a forest (the selected model is logistic regression) |
| rl/agents.py, rl/env.py | 126 | 2 | 98.4 % | the ε-greedy random branch of the agent (ε = 0 everywhere); stepping a finished episode |
| sim/ab.py, ui/panel.py, ui/render.py | 636 | 4 | 99.4 % | one `continue` in the A/B progress path; the recent-decisions list running out of panel height; in the renderer, a hunter with no path while AI vision is on, and the blink phase of the time bar's colour in the last ten seconds |
| `__main__.py` | 2 | 2 | 0 % | only runs as `python -m neuroquest` |

## 5. Performance (measured on this Mac, Phase 10)

| Measure | Value | Budget |
|---|---|---|
| `step()` on 21×15 with 2 A* hunters | 102 µs per tick | 100 ms per tick (10 ticks/s) |
| A* start → exit on 21×15 | 0.285 ms median (BFS 0.255 ms) | < 1 ms |
| `draw()` at 1024×768 | 2.07 ms with panel and AI vision, 0.88 ms plain | 16.7 ms (60 FPS) |
| dashboard build | 1.8 s | — |
| classifier training, full protocol | 8.0 s | — |
| RL training, 20,000 episodes | 12 s | — |
| test suite | 50.9 s | — |

## 6. Manual checks in the real window (for the learner; not executed by the assistant)

The scenarios in 2.3 drive the real loop headlessly. What a headless run cannot judge is what the screen looks like and feels like, so these remain for a human:

| Check | How | Status |
|---|---|---|
| Start screen, Enter starts | `neuroquest play` | pending |
| Movement, crystal pickup (score +10, particles), hunter hit (frame flash, heart shake) | play | pending |
| Win screen, defeat screen, time-out screen; R restarts | play | pending |
| P pauses with the controls listed; T shows hunters' paths; I opens the panel; F12 saves to `reports/figures/` | play | pending |
| Footer shows the style after 3 s and "Difficulty changed" after a decision; `--fixed` shows FIXED | play, play --fixed | pending |
| The agent plays its board | `neuroquest play --agent` | pending |
| Dashboard opens and reads well in a browser | open `reports/dashboard.html` | pending |

## 7. Known gaps

- No test opens a real window; `play` is covered with a stubbed window and the loop headlessly.
- The human-play classifier accuracy is unknown (no labelled human data); this is a data gap, not a test gap, and is stated on the dashboard.
- Timing is reported, not asserted (timing assertions are flaky on shared machines); the A* speed test asserts a generous bound only.
- The 100 RL evaluation seeds cover 20 distinct hunter starts (deterministic hunter); stated in `docs/09b_rl.md`.
