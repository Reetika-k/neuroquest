# Prompt engineering log and AI-assisted coding iterations

This document summarises how the learner directed the coding assistant (Claude Code) and what each kind of prompt produced. The full record, one eight-part entry per step (problem, prompt, output, what worked and failed, error or limitation, revision, tests, improvement), is `docs/DEVELOPMENT_LOG.md` (16 entries). Nothing here is invented: every prompt below is quoted or paraphrased from that log and the conversation it records; every iteration named is one that happened.

## 1. How the collaboration was set up

The learner's opening prompt was a complete project brief (`docs/00_project_brief.md`): the game, five AI features, an eleven-phase workflow with a stop-and-wait gate after each phase, a nine-part report format (A–I) for every code step, code-quality rules, and two honesty rules that shaped everything after: *never hardcode predictions to simulate successful ML* and *do not fabricate test results or metrics*. The assistant was asked to explain generated code so the learner could present it independently.

That structure did most of the work. Each phase began with the assistant reading the brief's definition of the phase and the architecture checklist, and ended with an A–I report and a request for approval. The learner's replies were usually one word.

## 2. The prompts, in order

| Phase / step | The learner's prompt | What it produced |
|---|---|---|
| 1 Requirements | The full brief, ending "Begin with Phase 1 … Do not generate code yet" | `docs/01_requirements.md`: objectives, MVP vs extensions, concepts, FR1–FR15, risks, success criteria S1–S12 |
| 2 Architecture | "ok approved and need these ai features as well you can build it step by step" + a six-item list (A* enemies; style classifier with LR/RF/KNN and precision/recall/F1; optional next-move model; adaptive difficulty with an optional learned-policy comparison; Q-learning vs random; explainable dashboard) | The requirements were revised (revision 3) to adopt the list; `docs/02_architecture.md` with the pure-`step()` spine, package layout, data contracts and an incremental checklist |
| 3 Environment | "approved" | venv on Python 3.13 (the Homebrew 3.14 is externally managed), `pyproject.toml`, smoke test |
| 4 Minimum viable game | "approved"; at the gate the learner remarked that it looked like a very simple game, then approved | Maze, rules, window loop, HUD; placeholder graphics by design (the brief: working features before polish) |
| 5 Enemies | "approved" | A* with 214 tests against BFS; chaser and random-walk policies; benchmark CSVs |
| 6 Data | "approved" | Events, 14 prefix-computable features, schema validation, three scripted profiles, 1,440-row dataset with an inspection report |
| 7 Machine learning | "approved" | Grouped split, six candidates, selection by CV, one test evaluation, error inspection, ablation, live prediction in the game |
| 8 Adaptive difficulty | "approved" | Rule table, bounds, decision log, A/B harness; a v2 controller after the v1 tightened too early (see section 4) |
| UI pass 1 | "can we work on ui side its looks so normal it's not attract the audience right now can we more interesting" | A first visual pass (entry 8b) |
| UI pass 2 | A detailed brief: "Act as a senior game UI/UX designer, Python/Pygame developer, and visual design specialist…" (eight steps, acceptance criteria), then "proceed" after the inspection step | A sci-fi redesign with a design system, intelligence panel and new game states (entry 8c) |
| UI bug report | "Act as a senior Python/Pygame game developer and professional game UI designer… fix the gameplay visuals…" with a specified palette and an eight-step procedure, plus a screenshot: "see the text is going outside" | Root-caused fixes: a decorative light removed, walls made continuous, collectible glows contained, menu text wrapped (entry 8d) |
| UI direction | "don't like the ui of game we can imporve it" | The assistant asked two multiple-choice questions (what bothers you; which direction); answers: "maze too dark / dull", "too much glow / neon clutter", "bright arcade". One targeted pass (entry 8e) |
| 9 Dashboard | "work on next phase" | Nine figures and a self-contained HTML report (entry 9) |
| Documentation link | "i like the dashboard make sure you add link of this to the final documentation … link start with this neuroquest/reports/dashboard.html we later live that project that's why" | Root-relative dashboard links in every final document; recorded as a standing rule |
| 9b RL agent | "9b" (choosing between 9b and Phase 10) | Environment, Q-learning, five-agent comparison, dashboard section 8 (entry 9b) |
| 10 Testing | "approved" | 38 new tests, two robustness fixes, the test report (entry 10) |
| 11 Documentation | "approved" | This documentation set |

## 3. What worked

- **A complete brief with gates.** The phase definitions, the A–I format and the two honesty rules were written once and applied sixteen times. The assistant never had to guess what "done" meant for a phase.
- **One-word approvals.** "approved" and "work on next phase" were enough because the next step was already defined; the learner's attention went to the gates where a decision was needed ("9b").
- **Specific feedback with evidence.** The screenshot with "see the text is going outside" was fixed in one iteration. The gameplay-fix brief that named the symptoms (a huge circle, a black maze, overlapping collectibles) led to root-cause diagnosis rather than cosmetic patches.
- **A constraint stated with its reason.** "link start with this neuroquest/reports/dashboard.html we later live that project that's why" told the assistant *why*, so the same rule could be applied to documents written later without being asked again.

## 4. What did not work, and what replaced it

- **Vague aesthetic prompts.** "looks so normal", then a long professional-sounding brief, then "don't like the ui" produced two redesigns the learner rejected. Taste cannot be verified headlessly, and a long brief is not the same as a preference. What worked was two short multiple-choice questions before touching anything, after which one pass was accepted. Lesson recorded in the log: ask, do not guess.
- **Headless judgement of visuals.** The assistant's renders looked acceptable in isolation but read as black with floating blocks on the learner's display (entry 8d). Lesson: judge gameplay visuals at the real window's contrast.

## 5. AI-assisted coding iterations (the ones that mattered)

| Entry | What failed | What was changed | How it was verified |
|---|---|---|---|
| 4–6 | Several of the assistant's own test fixtures were wrong (a "ragged" maze that was not ragged; a miscounted event) | The tests were corrected; the code was right | Suite green |
| 6 → 7 | `prefix_ratio` was a feature but cannot be known during live play | Moved to row metadata; a regression test pins it | `test_regressions` |
| 7 | `GroupKFold` asked for more folds than the tiny test data had groups | `_folds()` adapts the fold count | `test_edge_cases` |
| 8 | `reconcile()` refunded collected rewards on any parameter change, inflating adaptive scores | Rewards top up only when the density is raised | `test_adapt` |
| 8 | The v1 controller tightened within 5 s, before any encounter; explorers' win rate fell from 80 % to 60 % | v2: 3 s cadence, encounter requirement, warm-up and cooldown holds | A/B re-run; `test_adapt` |
| 8b | The decision summary used an arrow glyph the Pygame font lacks | ASCII arrow | `test_regressions` |
| 8b | Cached text raised "Invalid font" after Pygame re-initialised in tests | A retry in the text helper; completed in Phase 10 by validating the font cache itself | `test_regressions` |
| 8c–8e | Footer and panel collisions and overflows; a visor band and a portal highlight landing on the pixel the tests pin | Measuring, eliding and wrapping; drawings adjusted, tests kept | `test_ui` pixel contracts |
| 9 | The difficulty figure showed the latest session, which had no changes, on an odd auto-scale | The session with the most changes, fixed axes | `test_edge_cases` |
| 9b | The first RL board was a single corridor: every agent took exactly one hit, so "avoid the hazard" was unlearnable | A scan of 36 layouts; a fully braided board where the hazard is real and avoidable | `test_regressions` pins the board's properties |
| 10 | Coverage showed a bar option never executed; the assistant deleted it; the suite failed in five places | Restored; a test renders the last ten seconds | Full suite |
| 10 | Pygame keeps one timer per event type, silently replacing a resume timer | The scenario posts the second key from the agent hook | `test_scenarios` |

## 6. Where to look

- `docs/DEVELOPMENT_LOG.md`: the eight-part entries (the primary record).
- `docs/01_requirements.md` and `docs/02_architecture.md`: what was agreed before code.
- `reports/test_report.md`: every test case with expected and actual results.
