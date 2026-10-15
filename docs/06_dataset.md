# Phase 6 — Gameplay data collection

Status: **approved by the learner** · Date: 2026-10-11

## 1. From play to data

```
step() events (move · bump · reward · near_enemy · damage · exit · timeout)
   → EpisodeLog  (data/raw/<profile>-<seed>.jsonl: one meta line + one line per event)
   → episode_features(log, prefix)  (14 rates and ratios, computable on any prefix)
   → validate_row()  against data/schema/features.schema.json
   → data/processed/episodes.csv  (one row per episode × prefix 0.25 / 0.5 / 0.75 / 1.0)
```

Raw logs are regenerable from their seeds and are git-ignored; the CSV and the schema are tracked.

## 2. Schema

Row columns: `episode_id, label, profile, seed, outcome, ticks_total, prefix_ratio, ticks_seen` followed by the 14 features. Only the features are model inputs; the other columns are metadata for splitting and analysis. `outcome` and `ticks_total` are never features: they are only known when the episode ends.

| Feature | Meaning | Range |
|---|---|---|
| `elapsed_s` | seconds of play seen | ≥ 0 |
| `moves_per_tick` | successful moves / ticks | 0–1 |
| `bump_ratio` | wall bumps / (moves + bumps) | 0–1 |
| `idle_ratio` | ticks with neither move nor bump / ticks | 0–1 |
| `new_cells_per_tick` | first visits of a cell / ticks | 0–1 |
| `distinct_cells_ratio` | distinct cells visited / open cells | 0–1 |
| `revisit_ratio` | moves into already visited cells / moves | 0–1 |
| `rewards_per_tick` | rewards collected / ticks | 0–1 |
| `reward_share` | rewards collected / rewards available | 0–1 |
| `damage_per_tick` | hits taken / ticks | 0–1 |
| `near_enemy_ratio` | ticks within 2 cells of an enemy / ticks | 0–1 |
| `approach_ratio` | near-enemy moves that reduced the distance / near-enemy moves | 0–1 |
| `progress_rate` | (start distance to exit − current distance) / start distance | −5–1 |
| `min_exit_distance_ratio` | closest distance to the exit reached / start distance | 0–5 |

Every feature is a rate or ratio, so a row built from the first 25 % of an episode means the same thing as a row from the whole episode. That is what lets the in-game predictor run mid-episode. **`prefix_ratio` was removed from the features during inspection**: a live game cannot know what fraction of the episode has elapsed, so a model must not see it. It remains a row column for analysis.

Validation (`data/schema.py::validate_row`): every feature present, numeric, finite and within its range; label in the allowed set. The build reports invalid rows and excludes them. This dataset: 1,440 rows, 0 invalid.

## 3. Labelling strategy

**Synthetic episodes.** The label is the *profile that played the episode*: `aggressive`, `cautious` or `explorer` (`sim/profiles.py`). It is ground truth by construction, not derived from the features. Each profile samples its parameters from the episode's seed so the classes overlap:

| Profile | Behaviour | Randomised parameters |
|---|---|---|
| aggressive | nearest reward by shortest path, ignores enemies, charges an enemy within `engage_radius` with probability `engage_prob`, exits when few rewards remain | epsilon 0–0.12, engage_prob 0.2–0.8, engage_radius {2,3,4}, exit_when_rewards_left {0,1,2} |
| cautious | treats cells within `safe_radius` of an enemy as walls, retreats when an enemy is within `flee_radius`, picks a reward only with probability `reward_greed`, otherwise heads for the exit, waits when no safe route exists | epsilon 0–0.08, safe_radius {2,3,4}, flee_radius {1,2}, reward_greed 0–0.6 |
| explorer | heads for the nearest unvisited cell, retreats from enemies within `caution_radius`, goes to the exit once `coverage_target` of the maze is seen or time is short | epsilon 0–0.10, coverage_target 0.55–0.9, caution_radius {1,2} |

`epsilon` is the probability of a random move on any tick: noise that no real player lacks.

**Human episodes.** There is no ground truth. `data/labels.py::rule_label` assigns `rule:<class>` from a few documented thresholds. Its limitation is circularity (a model trained on it can only re-learn the thresholds), so it is used only for human play, reported separately, and never mixed into the synthetic training labels. Self-reported labels are the better extension.

## 4. Generation and inspection (2026-10-11)

```bash
.venv/bin/python -m neuroquest simulate --episodes 120 --seed 0     # 13 s on this laptop
.venv/bin/python -m neuroquest build-dataset
.venv/bin/python scripts/inspect_dataset.py
```

Default difficulty (2 A* enemies, interval 4, reward density 0.10, 120 s), 21×15 mazes, seeds 0–119, the same seeds for every profile so the mazes are identical across classes.

**Outcomes per profile (full episodes).**

| Profile | Won | Lost (health) | Median length |
|---|---|---|---|
| aggressive | 26 | 94 | 104 ticks (10 s) |
| cautious | 80 | 40 | 325 ticks (33 s) |
| explorer | 93 | 27 | 260 ticks (26 s) |

**Class balance:** 480 rows each (120 episodes × 4 prefixes). Nothing to rebalance.

**Per-label means (full episodes) and separability.** The separability score is the spread of the class means divided by the mean within-class standard deviation; higher means a single threshold would already do well.

| Feature | aggressive | cautious | explorer | Separability |
|---|---|---|---|---|
| damage_per_tick | 0.028 | 0.006 | 0.006 | 3.21 |
| idle_ratio | 0.091 | 0.472 | 0.275 | 2.87 |
| moves_per_tick | 0.880 | 0.508 | 0.700 | 2.82 |
| new_cells_per_tick | 0.642 | 0.272 | 0.422 | 2.68 |
| elapsed_s | 10.9 | 34.5 | 26.9 | 2.66 |
| rewards_per_tick | 0.082 | 0.032 | 0.043 | 2.53 |
| approach_ratio | 0.502 | 0.115 | 0.368 | 2.34 |
| near_enemy_ratio | 0.327 | 0.084 | 0.251 | 1.89 |
| revisit_ratio | 0.270 | 0.482 | 0.419 | 1.68 |
| distinct_cells_ratio | 0.509 | 0.630 | 0.734 | 1.03 |
| min_exit_distance_ratio | 0.303 | 0.199 | 0.127 | 0.69 |
| progress_rate | 0.641 | 0.713 | 0.802 | 0.46 |
| bump_ratio | 0.031 | 0.040 | 0.036 | 0.35 |
| reward_share | 0.639 | 0.726 | 0.730 | 0.34 |

Figure: `reports/figures/dataset_features.png` (box plots per feature and label).

**What the inspection says.**

1. The classes differ in the way the profiles were designed to differ: aggressive players take damage and approach enemies; cautious players idle and revisit; explorers discover cells at a high rate. Several single features already separate well, so Phase 7 must include a majority-class baseline *and* report whether a one-feature rule would do as well as the models. The within-class spreads (for example `approach_ratio` 0.368 ± 0.258 for explorers) show real overlap, so the task is not trivial.
2. `elapsed_s` is confounded with episode length at prefix 1.0 (aggressive episodes are short). It is legitimately observable in-game, but Phase 7 will train with and without it to show how much the models lean on it.
3. The hand-written rule labeller agrees with the true profile in only 60 % of full episodes (aggressive 0.88, cautious 0.25, explorer 0.66) and 44 % of 25 % prefixes. Cautious players visit more than half the maze while waiting out enemies, which the explorer rule mistakes for exploring. This is the circularity warning made concrete: rules written from intuition are a weak labeller and a weak classifier. In Phase 7 the rule labeller becomes a reported baseline.
4. Aggressive episodes are short (10 s median), so their 25 % prefix is about 26 ticks and noisy; the model must cope with that.

**Changes from the architecture's proposed feature list.** `reward_detour_ratio` and `avoid_count_per_tick` were dropped: both need shortest-path replay per reward and added little over `rewards_per_tick`, `near_enemy_ratio` and `approach_ratio`. `bump_ratio`, `idle_ratio` and `min_exit_distance_ratio` were added. `prefix_ratio` was removed from the features for the reason above.

## 5. Replacing synthetic data with real play

Human episodes will be logged by the game loop in Phase 7 with `profile = "human"`; `build-dataset` labels them `rule:<class>` and keeps them separate (`profile` column). The feature code, schema and validation are identical, so real rows can be analysed, used to sanity-check the model, and, with self-reported labels, eventually added to training.
