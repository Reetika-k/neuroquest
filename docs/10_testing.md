# Phase 10 — Testing and improvement

Status: **approved by the learner** · Date: 2026-10-15 · Report: [`reports/test_report.md`](../reports/test_report.md) · Raw runs: `reports/pytest_results.txt`, `reports/coverage.txt`

## 1. Strategy

Tests were written phase by phase (every phase had to leave its tests green before the next began), so Phase 10 did not start from zero: 339 cases existed. This phase asked what those cases could *not* tell us and filled the gaps:

| Layer | Question it answers | Where |
|---|---|---|
| Unit | Does each piece do what its docstring says? | test_maze, test_rules, test_astar, test_policies, test_events, test_features, test_profiles, test_dataset, test_ml, test_adapt, test_env, test_qlearning, test_dashboard |
| Integration | Do the pieces work *together*, with the real saved model? | test_integration (play → log → dataset → predict → decisions; dashboard from the repository's own files; reproducibility) |
| Scenario | Does the real game loop behave for a player: win, lose both ways, restart, pause, keys, quit, menu, a live difficulty change? | test_scenarios (scripted player through `ui.app.run`, keys injected as Pygame events) |
| Edge case | What happens at the limits: smallest board, empty or absurd parameters, unreachable enemy, odd windows, overlong text, bad model? | test_edge_cases, test_ui |
| Regression | Can a fixed bug come back unnoticed? | test_regressions (one test per bug, naming the log entry) |
| Command line | Does every command run end to end? | test_cli |

Honesty rules: a test is only reported with the status of an executed run; a failing test is fixed in the code when the code is wrong and in the test when the test is wrong, and the report says which; coverage is measured, not estimated; timing is measured, not asserted.

## 2. Result

**377 passed, 0 failed, 0 skipped in 50.9 s; 99.1 % line coverage** (26 of 2,955 statements not executed, each listed and explained in the report). The suite runs headless and needs no display.

## 3. Requirement traceability

| Requirement | Evidence |
|---|---|
| FR1 movement, walls | test_rules (move, bump), test_scenarios (win walks the shortest path with no bumps) |
| FR2 rewards, score, HUD | test_rules (reward once), test_ui (pixel contracts, HUD at every size), test_edge_cases (HUD in the last 10 s) |
| FR3 win / loss / restart | test_rules, test_scenarios (win, loss by health, loss by time, restart with R) |
| FR4 A* enemies follow the player | test_astar (214), test_policies, test_benchmark |
| FR5 random baseline, A* faster | test_benchmark, `reports/enemy_baseline.csv` |
| FR6 event logs, schema, invalid rows rejected | test_events, test_features, test_dataset, test_integration |
| FR7 labels, rule labeller | test_features, test_dataset (human rows rule-labelled) |
| FR8 training protocol | test_ml, test_integration (reproducible), test_cli (train) |
| FR9 live prediction from the saved model | test_ui, test_integration (replay equals live) |
| FR10 bounded, logged adaptation | test_adapt (1,000 steps in bounds), test_ab, test_scenarios (live ease), test_integration |
| FR11 harness, fixed vs adaptive | test_ab, test_cli (ab-test) |
| FR12, FR15 dashboard | test_dashboard, test_integration (all sections from repo data), test_edge_cases |
| FR14 RL environment and agent | test_env, test_qlearning, test_regressions (board hazard), test_cli (rl-train) |
| S1–S4, S6, S7, S9, S12 | test_astar, test_benchmark, reproducibility tests, ML table in the report, test_adapt, test_ab, this suite (≥ 40 cases), RL results |
| S5, S8 | headless equivalents pass; the look itself is in the manual checklist of the report |

## 4. Improvements made in this phase

- `ui/theme.py`: the font cache validates cached fonts and rebuilds them when Pygame's font module was re-initialised (the old retry covered text rendering only; word-wrap crashed).
- `ui/panel.py`: `wrap()` breaks a token wider than the panel between characters instead of letting it spill out.
- Dead code removed: glow, vignette and edge-pulse helpers of the earlier sci-fi look; an unused RL `save_meta`. A removal of the HUD bar's halo option was reverted when the suite showed the time bar uses it in the last ten seconds, and a test now renders that state.
- `cli.py`: `dashboard --figures-dir`, so tests build the dashboard without touching `reports/figures/`.
- `coverage` added to the `dev` extra and `requirements.txt`.

## 5. What is deliberately not claimed

No test opens a real window, so the manual checklist in the report (start screen, feel of hits, screens, `play --agent`, the dashboard in a browser) is marked *pending* for the learner. Classifier accuracy on human play is unknown for lack of labelled human data.

## 6. Reproduce

```bash
cd neuroquest
.venv/bin/pytest                                             # 377 tests, ~50 s
.venv/bin/coverage run --source=src/neuroquest -m pytest && .venv/bin/coverage report -m
```
