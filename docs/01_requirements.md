# NeuroQuest — Phase 1: Requirements and Feasibility

Status: **approved 2026-10-10** · Author: Reetika Kukreja with Claude Code as coding assistant · Date: 2026-10-10 · Revision 2: labelling strategy (profile-derived labels), Python version finding, hosting risk · Revision 3 (at approval): six-feature list adopted; next-move model, RL agent, KNN baseline, per-class precision/recall, feature importance and learning-progress views added as planned extensions

## 1. Project summary

NeuroQuest is a 2D survival and strategy game (Python + Pygame) in which the player moves through a maze, collects rewards, avoids enemies and reaches the exit. Enemies navigate with A* pathfinding. The game records what the player does, a scikit-learn classifier categorises the player's behaviour (aggressive, cautious, exploration-focused) from engineered features, and a transparent adaptation policy tunes difficulty from performance and the prediction. A dashboard shows the statistics, predictions, decisions and model evaluation. The point is to demonstrate real, measurable AI/ML, with honest evaluation, inside a playable game.

## 2. Objectives

1. A playable maze game with clear win and loss conditions.
2. One real search algorithm (A*) with tests that prove optimality against a brute-force baseline.
3. One properly evaluated supervised model: documented labelling, held-out evaluation, at least two baselines, metrics and a confusion matrix.
4. An explainable adaptive-difficulty controller whose every decision is logged with its inputs and reason, compared against fixed difficulty.
5. A dashboard that lets a reviewer inspect how the AI reached its decisions.
6. A development record (prompts, iterations, tests, failures and fixes) suitable for a capstone evaluation.

## 3. Minimum viable product

| Essential (MVP) | Planned extensions, approved 2026-10-10 (built only after the MVP is stable) |
|---|---|
| Grid maze rendered in Pygame; keyboard movement; walls and collisions | Procedurally generated mazes beyond the hand-made set |
| Collectible rewards; exit tile; win condition (reach exit); loss condition (health reaches 0 or time runs out) | Sprites, animations, sound |
| Enemies that chase the player with A* on the maze graph; a random-walk baseline enemy for comparison | Multiple enemy types (patroller, ambusher) |
| Event logger → per-episode feature vector (CSV) with schema and validation | Live streaming features mid-episode |
| Profile-labelled synthetic episodes (plus a transparent rule labeller for human play) → classifier (≥ 2 baselines) → held-out evaluation → saved model → in-game prediction | Self-reported labels from real play-test participants |
| Adaptive difficulty policy with safety bounds and a decision log; fixed vs adaptive comparison on simulated player profiles | Learned (bandit/RL) adaptation policy |
| Dashboard (Matplotlib figures + a text/HTML summary) | Reinforcement-learning agent in a small maze environment (gym-style API; Gymnasium optional) compared with a random-action baseline, with a learning curve |
| | Next-move prediction: a second multiclass model on sequence features with a temporal train/test split, only if the dataset supports it |
| | Dashboard additions: per-class precision/recall, feature importance, agent learning progress, next-move accuracy |
| | Optional: a learned adaptation policy compared with the rule-based one in simulation |

## 4. AI/ML concepts demonstrated

| Area | Concept | Where it appears |
|---|---|---|
| Search | Graphs, heuristics (Manhattan distance), A* vs BFS vs random walk, admissibility and optimality | Enemy navigation (Phase 5) |
| Data | Event logging, feature engineering, dataset schema, validation, reproducible synthetic data | Gameplay data collection (Phase 6) |
| Supervised learning | Labelling strategy and its limits, train/test split without leakage, class balance, baselines (majority class, logistic regression, random forest or k-NN), accuracy, macro-F1, confusion matrix, error inspection, model persistence | Behaviour classifier (Phase 7) |
| Decision systems | Rule-based control with bounded outputs, logging of decisions, A/B comparison with simulated users | Adaptive difficulty (Phase 8) |
| Explainability | Showing inputs, prediction, confidence and the rule that fired | Dashboard (Phase 9) |
| Predictive ML (extension) | Sequence features, multiclass classification, temporal train/test split, baselines (majority direction, repeat-last-move) | Next-move model (Phase 7b) |
| Reinforcement learning (extension) | State, action space, reward function, termination, Q-learning, exploration vs exploitation, comparison with a random baseline | RL agent (Phase 9b) |

## 5. Functional requirements

| ID | Requirement |
|---|---|
| FR1 | The player moves in four directions on a grid maze; walls block movement. |
| FR2 | Rewards can be collected; the score increases; the HUD shows score, health/lives and time. |
| FR3 | Reaching the exit wins the level; health 0 or time-out loses it; the game can be restarted. |
| FR4 | Enemies compute an A* path to the player on the maze graph and follow it; paths update as the player moves. |
| FR5 | A baseline enemy (random walk) exists for comparison; a test shows A* reaches the player faster. |
| FR6 | Every episode produces an event log and a feature vector following a documented schema; invalid rows are rejected. |
| FR7 | Each synthetic episode is labelled with the profile that generated it (aggressive, cautious, exploration-focused); a separate documented, inspectable rule-based labeller exists for real human episodes, with its circularity limitation stated. |
| FR8 | A training script builds train/test sets (stratified, seeded, grouped by seed), trains logistic regression, random forest and KNN plus a majority-class baseline, reports accuracy, macro-F1, per-class precision and recall and a confusion matrix, and saves the chosen model with its metadata. |
| FR9 | The game loads the saved model and shows the predicted class (and probability where the model supports it) during play. |
| FR10 | The adaptation controller adjusts enemy speed, enemy count, reward density and time pressure within fixed bounds, from performance and the prediction; each decision is logged with inputs, rule and output. |
| FR11 | A simulation harness runs scripted player profiles with adaptive and fixed difficulty and reports outcomes (win rate, time to exit, damage). |
| FR12 | The dashboard shows current prediction and confidence, episode statistics, difficulty over time, model metrics, and the fixed-vs-adaptive comparison. |
| FR13 | (Extension) A next-move model predicts the player's next direction from recent positions and actions; it is evaluated with a temporal split against majority-direction and repeat-last-move baselines. |
| FR14 | (Extension) A maze environment with a gym-style `reset`/`step` API trains a tabular Q-learning agent to collect rewards, avoid hazards and reach the exit; it is compared with a random-action agent on success rate and average return. |
| FR15 | (Extension) The dashboard additionally shows per-class precision/recall, feature importance, the agent's learning curve and next-move accuracy. |

## 6. Non-functional requirements

- Runs locally on macOS with Python 3; no cloud services, databases or paid APIs.
- Game logic is pure Python modules independent of Pygame so they can be tested headlessly with pytest.
- Reproducibility: seeded mazes, seeded synthetic data, seeded train/test split.
- Code quality: modular, PEP 8, type hints and docstrings, logging, no hard-coded predictions.
- Performance: 60 FPS target on a 20×15 maze with up to 5 enemies; A* per enemy per step must stay below 5 ms.
- Honesty: no fabricated metrics; failed tests are reported as failed.

## 7. Risks and assumptions

| Risk | Impact | Mitigation |
|---|---|---|
| Real gameplay data will be tiny (one developer) | Classifier cannot be trained on real play alone | Reproducible synthetic dataset from parameterised simulated player profiles, documented as such; schema identical to real logs so real data can replace it |
| Label circularity: if labels are derived from the same features by threshold rules, the classifier only re-learns the rules | Over-claiming "the model learned behaviour" | For synthetic data the label is the *generating profile* (the policy that played the episode), not a threshold on the features; profiles have randomised parameters so their feature distributions overlap and the task is non-trivial. The rule-based labeller is used only for real human episodes and is documented as a limitation; self-reported labels remain an extension. Evaluation is on held-out episodes from unseen seeds and noise levels |
| Python 3.14 (Homebrew) is the default interpreter and is marked externally managed (PEP 668), so packages cannot be installed into it directly; wheel support for 3.14 was not confirmed | Setup failure in Phase 3 | Checked on 2026-10-10: Python 3.13.7 (python.org) resolves pygame 2.6.1, pygame-ce 2.5.8, scikit-learn 1.9.1, matplotlib and pytest, and already has pandas 2.3.3. Decision: Phase 3 creates the virtual environment with python3.13 |
| Pygame cannot run tests headlessly on CI | Tests depend on a display | Keep game logic display-free; use the dummy video driver for smoke tests |
| Adaptive difficulty may not measurably help | Unsupported claim | Compare with fixed difficulty on simulated profiles and report whatever the numbers show |
| Scope creep (RL, graphics) | MVP never stabilises | RL and visuals are gated behind a stable MVP |
| The course's game project expects a public URL, but this is a desktop Python game | Submission gap | The brief requires a local demonstration; the submission includes the repository, screenshots and a recorded playthrough. A browser build of the game-only part with pygbag (Pygame to WebAssembly) is a possible stretch; the ML pipeline and dashboard stay local |

Assumptions: single-player, keyboard controls, local execution, English UI, one developer with an AI coding assistant.

## 8. Measurable success criteria

| # | Criterion | How it is measured |
|---|---|---|
| S1 | A* paths are optimal | On 200 seeded random mazes, A* path length equals BFS length whenever a path exists; unreachable targets return "no path" without error |
| S2 | A* beats the baseline | Over 100 seeded episodes, the A* enemy reaches the player in fewer steps than the random-walk enemy in ≥ 90 % of cases |
| S3 | Dataset is valid and reproducible | Schema validation passes on every row; regenerating with the same seed yields identical files |
| S4 | Classifier is evaluated honestly | Held-out test set (≥ 20 %, stratified, episodes from seeds never used in training); ≥ 2 models plus a majority-class baseline; accuracy, macro-F1 and confusion matrix reported; the chosen model beats the majority baseline on macro-F1 by ≥ 0.20, and the numbers are reported whether or not the target is met |
| S5 | Predictions are live in the game | During play the HUD shows the predicted class from the saved model (not hard-coded) |
| S6 | Adaptation is bounded and explainable | 100 % of adaptation decisions are logged with inputs, rule and output; parameters never leave their bounds in 1,000 simulated steps |
| S7 | Fixed vs adaptive is compared | A table with ≥ 30 simulated episodes per condition and profile reports win rate, time and damage, with no claims beyond the numbers |
| S8 | Dashboard is complete | Shows all six items in FR12 and the rule behind the latest decision |
| S9 | Tests | ≥ 40 pytest cases covering maze, movement, collisions, win/loss, A*, logging, features, labelling, training pipeline, adaptation; all executed, results in the test report |
| S10 | Documentation | Every Phase 11 document exists; the prompt log records real iterations only |
| S11 | Next-move model (extension) | Temporal split (train on earlier episodes, test on later); accuracy and macro-F1 reported against majority-direction and repeat-last-move baselines, whether or not it beats them |
| S12 | RL agent (extension) | Over 100 evaluation episodes on held-out seeds, the trained agent's success rate and average return are reported against the random-action baseline, with the learning curve; no claim beyond the numbers |

## 9. Deliverables by phase

1. This specification (Phase 1)
2. Architecture, folder structure, data-flow and implementation checklist (Phase 2)
3. Working environment with dependency file and smoke test (Phase 3)
4. Playable MVP game (Phase 4)
5. A* enemies with tests and baseline comparison (Phase 5)
6. Event logger, dataset schema, labelling function, synthetic dataset (Phase 6)
7. Training pipeline, evaluation report, saved model, in-game prediction (Phase 7)
7b. Next-move prediction model with temporal evaluation (Phase 7b, if the dataset supports it)
8. Adaptive difficulty with decision log and fixed-vs-adaptive comparison (Phase 8)
9. Dashboard and explainability notes (Phase 9)
9b. RL maze environment, Q-learning agent, random-baseline comparison and learning curve (Phase 9b)
10. Test report (Phase 10)
11. Final documentation set, demo script and README (Phase 11)

## 10. Out of scope for the MVP

Multiplayer, online hosting, sound and sprite art, and procedural generation beyond simple seeded mazes. The RL agent, the next-move model and a learned-policy comparison are planned extensions sequenced after the MVP (section 3); they are not part of the MVP acceptance.
