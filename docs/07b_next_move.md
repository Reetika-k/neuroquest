# Phase 7b — Next-move model (extension)

Status: **approved by the learner 2026-10-17** · Date: 2026-10-16 · Data: `data/processed/moves.csv` (86,726 rows, 360 episodes) · Report: `reports/next_move_metrics.json` · Model: `models/next_move.joblib` · Pursuit benchmark: `reports/pursuit_benchmark.csv` · Dashboard section 9 in `neuroquest/reports/dashboard.html`

## 1. The idea, in plain language

The style classifier (Phase 7) says *what kind* of player someone is. This model asks a smaller, sharper question every tick: **where will the player step next?** Up, down, left, right, or stay. It is the "predictive ML" item of the brief: sequence features, multiclass classification, a temporal train/test split, and two naive baselines it has to beat (the majority direction and repeating the last move). It also has a use inside the game: a hunter that walks toward where the player *will be* rather than where they are.

The target needs no labelling strategy: the player's next action is written in the log. That makes this the one model in the project that human play could train directly, once there is enough of it.

## 2. The data: one row per tick, replayed from the logs

Each simulated episode log is replayed through the real `step()` with the actions it recorded (a move or bump event names the action; a tick with neither is a *stay*). Because the mazes and the A* hunters are seeded and deterministic, the replay reproduces the game exactly, and the row for tick *t* holds what the game showed just before the player's *t*-th action. An episode whose replay does not end with the log's outcome, tick count and score would be dropped and counted; none was (360 replayed, 0 mismatches). The 13 human and adaptive logs are skipped: their difficulty changed mid-game, so an exact replay is not available from the log alone.

To support the features, `GameState` gained one field, `recent_actions`, the last three actions, updated by `step()`; every existing test still passes.

| Feature group | Columns | Meaning |
|---|---|---|
| Last three actions | `prev1_*`, `prev2_*`, `prev3_*` (15) | one-hot of each of the last three actions; all zero when fewer have been taken |
| Neighbourhood | `open_*` (4), `visited_*` (4) | whether each adjacent cell is walkable, and whether it was already visited |
| Nearest crystal, portal, nearest hunter | `*_dr`, `*_dc`, `*_dist` (9) | sign of the row and column offset, Manhattan distance (clipped at 40, 40, 20) |
| Situation | `rewards_left`, `health`, `near_enemy` (3) | crystals still on the board, hearts, a hunter within two cells |
| Shortest paths | `path_reward_*`, `path_exit_*` (8) | first step of the shortest path to the nearest crystal and to the portal |

43 features, all computable live from the state. Targets in the data: stay 30,832 (35.6 %), right 15,979, down 14,774, left 12,691, up 12,450.

## 3. Temporal split, and why

A model that will be used *after* it is trained should be tested on episodes that come *after* its training data. Episodes are ordered by seed, the order in which they were generated: seeds 0–95 train (69,976 rows), seeds 96–119 test (16,750 rows). Because the maze is a function of the seed, the test mazes are also mazes the model never saw. The model is chosen between logistic regression and random forest on the last 20 % of the *training* episodes (a validation tail of 15,206 rows), then refitted on all training rows and evaluated once on the test rows, so the test score is not used for any choice.

**Baselines.** *Majority*: always "stay". *Repeat-last*: the previous action again. *Greedy*: the first step of the shortest path to the nearest crystal while any is left, otherwise to the portal; this is what a hand-written "predict the player" rule would be.

## 4. Results on the test episodes (seeds 96–119)

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 | Accuracy: aggressive | cautious | explorer |
|---|---|---|---|---|---|---|
| Majority ("stay") | – | 0.315 | 0.096 | 0.09 | 0.43 | 0.27 |
| Repeat the last move | – | 0.662 | 0.629 | 0.54 | 0.71 | 0.65 |
| Greedy rule | – | 0.510 | 0.484 | **0.79** | 0.41 | 0.52 |
| Logistic regression | 0.801 | 0.829 | 0.822 | 0.76 | 0.84 | 0.84 |
| **Random forest (selected)** | **0.842** | **0.853** | **0.846** | 0.78 | **0.86** | **0.88** |

Random forest on the test rows, per class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| up | 0.86 | 0.79 | 0.82 | 2,533 |
| down | 0.87 | 0.83 | 0.85 | 3,033 |
| left | 0.85 | 0.80 | 0.82 | 2,612 |
| right | 0.85 | 0.85 | 0.85 | 3,288 |
| stay | 0.85 | 0.92 | 0.89 | 5,284 |

Confusion matrix (rows = true, columns = predicted: up, down, left, right, stay):

```
up     1995   172    62    84   220
down    138  2524    74    81   216
left     53    64  2094   205   196
right    56    62   144  2797   229
stay     84    82    98   134  4886
```

**Reading the numbers honestly.**

- The forest beats the strongest naive baseline (repeat-last) by +0.22 macro-F1 and the greedy rule by +0.36; success criterion S11 asked only that the comparison be reported.
- The forest is deliberately bounded (120 trees, depth 12, at least 10 rows per leaf): the first, unbounded 300-tree forest scored 0.849 on the validation tail but pickled to 216 MB, over GitHub's 100 MB file limit. The bounded one scores 0.842 on the same validation rows and is 7.5 MB compressed; that choice was made on the validation tail, not on the test rows.
- The greedy rule is *better* for aggressive players (0.79 against 0.78). That is not a failure of the model so much as a fact about the data: aggressive players walk the shortest path to crystals by construction, so a rule that says exactly that is hard to beat. The model's gain is on cautious and explorer players, whose moves depend on the hunters and on which cells they have already seen, which no single rule captures.
- "Stay" is the easiest class (recall 0.92): a player who stood still last tick usually stands still this tick. The most informative feature by permutation is `prev1_stay` (0.085), then `open_left` (0.018), `prev1_left`, `open_up`, `prev2_stay`, `prev1_down`, `open_down`. The confusion matrix's largest off-diagonal entries are moves mistaken for stay (196–229 per direction): the moment a waiting player sets off is the hardest thing to call.
- Logistic regression is only 0.02 behind; the forest was chosen on the validation tail, before the test rows were seen.

## 5. Where it lives

`ml/next_move.py` (features, replay, dataset, split, baselines, training, `NextMovePredictor`), the commands `build-moves` and `train-next-move`, and `fig_next_move` on the dashboard.

## 6. In the game: the anticipating hunter

`ai/policies.py::Anticipator` is a hunter policy that asks the model for the player's predicted next cell and takes the first A* step toward it; when there is no saved model, or the predicted cell is unreachable, it behaves exactly like the plain A* chaser (tested). One prediction is made per tick and shared by every hunter moving on that tick; a prediction costs about 7 ms with the bounded forest (single-threaded on purpose: a forest saved with a thread pool spun it up for every call, measured at about 100 ms, and stalled the first benchmark). `neuroquest play --enemy-kind anticipator` uses it.

`neuroquest benchmark-pursuit` measures whether anticipation helps: hunters of each kind pursue the three *moving* scripted players on the same 40 seeded mazes each (seeds 2000–2039, used by no other experiment), with the default difficulty (two hunters moving every 4 ticks). The signal is the share of episodes with a hit, the mean hits per episode and the player's win rate.

| Hunter | Player | Episodes with a hit | Median tick of the first hit | Mean hits | Player win rate |
|---|---|---|---|---|---|
| A* chaser | aggressive | 40/40 | 56 | 2.62 | 25 % |
| anticipator | aggressive | 40/40 | 57 | 2.67 | 22 % |
| A* chaser | cautious | 37/40 | 176 | 1.80 | 72 % |
| anticipator | cautious | 38/40 | 178 | 1.77 | 72 % |
| A* chaser | explorer | 31/40 | 152 | 1.55 | 72 % |
| anticipator | explorer | 31/40 | 160 | 1.40 | 80 % |

**Anticipation did not help.** Paired by seed, the anticipating hunters landed the same number of hits as the plain A* hunters in 103 of 120 episodes, more in 6 and fewer in 11 (mean difference −0.04 hits); the catch rates differ by at most one episode per cell and the first hit comes no earlier. The reason is structural, not a model error: the hunters move once every four ticks while the player moves every tick, so aiming one predicted step ahead moves the A* target by at most one cell and almost never changes the hunter's next step, and the plain chaser already re-plans every move. A hunter that actually intercepts would have to roll the model forward several steps and aim at where the paths cross; that is a different, larger experiment. The honest result is recorded on the dashboard: a model that predicts the next move well (0.85) is not automatically a better hunter.

## 7. Limitations

- Trained and tested on simulated players whose decisions follow written rules; accuracy on human play is unknown. The self-labelled target means human logs can be added without any labelling step, which is the obvious next step.
- The greedy rule's strength on aggressive players is a reminder that a hand-written predictor is a serious competitor whenever the behaviour being predicted was itself written by hand.
- One temporal split; no confidence interval on the 0.846 beyond the 0.02 gap to logistic regression and the 0.84 validation score agreeing with the 0.85 test score.

## 8. Reproduce

```bash
.venv/bin/python -m neuroquest build-moves          # ~70 s: replays 360 logs → data/processed/moves.csv
.venv/bin/python -m neuroquest train-next-move      # ~30 s: temporal split, baselines, LR vs RF, saves models/next_move.joblib (7.5 MB)
.venv/bin/python -m neuroquest benchmark-pursuit    # ~1.5 min: anticipator vs A* hunters → reports/pursuit_benchmark.csv
.venv/bin/python -m neuroquest dashboard            # section 9
.venv/bin/pytest tests/test_next_move.py
```
