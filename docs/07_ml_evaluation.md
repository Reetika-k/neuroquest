# Phase 7 — Machine learning: the player-behaviour classifier

Status: **approved by the learner** · Date: 2026-10-11 · Data: `data/processed/episodes.csv` (sha256 c583814a…, 1,440 rows, 360 episodes) · Report: `reports/metrics.json`

## 1. The concepts, in plain language

**The task.** Given 14 numbers describing how someone has played so far, say whether they are playing like an aggressive, a cautious or an exploration-focused player. Three classes, so this is *multiclass classification*.

**Leakage, and why we split by seed.** Each episode produced four rows (its 25/50/75/100 % prefixes), and each maze seed produced three episodes (one per profile). If a prefix of an episode sat in the training set and another prefix of the same episode in the test set, the model would be tested on near-copies of what it trained on and the score would be inflated. So the split is by seed: 96 seeds (1,152 rows) to train, 24 seeds (288 rows) to test, chosen once with a fixed random seed. Cross-validation uses the same rule (`GroupKFold` with the seed as the group).

**Why choose the model before looking at the test set.** If you pick whichever model scores best on the test set, the test score is no longer an honest estimate: you have used the test set to make a decision. We therefore rank candidates by 5-fold grouped cross-validation on the training rows, pick the best, and only then evaluate everything once on the test rows.

**The candidates.**
- *Majority class*: always predicts the most common class. If a model cannot beat this, it has learned nothing.
- *Rule labeller*: the hand-written thresholds from Phase 6. It is what you would ship without machine learning.
- *Single-feature tree*: a depth-2 decision tree on one feature, the best of the 14 by CV. It answers "would one well-chosen threshold do the job?".
- *Logistic regression*: a linear model; each feature pushes the odds of each class up or down by a coefficient you can read. Fast, hard to overfit, interpretable. Features are standardised first because they live on different scales.
- *Random forest*: 300 decision trees trained on random subsets, voting. Captures interactions and non-linear thresholds without feature scaling.
- *k-nearest neighbours (k = 15)*: predicts the majority class among the 15 most similar training rows. No training step; a good sanity check that similar behaviour clusters together.

**The metrics.** *Accuracy* is the share of rows predicted right; fine here because the classes are balanced. *Precision* for a class is "of the rows predicted as this class, how many really were"; *recall* is "of the rows that really are this class, how many did we catch"; *F1* is their harmonic mean. *Macro-F1* averages F1 over the three classes with equal weight, so a model cannot hide a bad class behind two good ones. The *confusion matrix* shows which classes are mistaken for which (rows are the true class, columns the prediction).

## 2. Results on the 24 held-out seeds (288 rows)

| Model | CV macro-F1 (train, 5 folds) | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Majority class | 0.167 | 0.333 | 0.167 |
| Rule labeller (Phase 6 thresholds) | – | 0.535 | 0.545 |
| Single-feature tree (`approach_ratio`) | 0.602 | 0.622 | 0.618 |
| k-nearest neighbours | 0.886 ± 0.023 | 0.903 | 0.902 |
| Random forest | 0.904 ± 0.012 | 0.924 | 0.924 |
| **Logistic regression (selected)** | **0.906 ± 0.015** | **0.934** | **0.933** |

Success criterion S4 asked for at least +0.20 macro-F1 over the majority baseline; the gap is +0.77. The three learned models also beat the best hand-written rule by about +0.3, which is the measurable case for using machine learning here rather than thresholds.

**Selection, honestly.** Logistic regression and random forest are tied within one standard deviation of the CV estimate (0.906 ± 0.015 versus 0.904 ± 0.012). The protocol picks the higher mean; with a different split seed the forest could win. Logistic regression is also the easier model to explain (section 5) and the fastest to run in the game loop, so the choice is defensible on more than a hair's breadth.

**Logistic regression on the test rows.**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| aggressive | 0.89 | 0.99 | 0.94 | 96 |
| cautious | 0.96 | 0.97 | 0.96 | 96 |
| explorer | 0.96 | 0.84 | 0.90 | 96 |

Confusion matrix (rows = true, columns = predicted: aggressive, cautious, explorer):

```
aggressive   95   0   1
cautious      1  93   2
explorer     11   4  81
```

**Accuracy by how much of the episode the model sees:** 25 % → 0.917, 50 % → 0.931, 75 % → 0.931, 100 % → 0.958. The model is usable early and improves with evidence, which is what the in-game predictor needs.

## 3. Error inspection

Nineteen of 288 test rows are wrong. Eleven are explorers predicted as aggressive, four explorers predicted as cautious, the rest scattered. The most confident mistakes:

| Episode | Prefix | True → predicted | Confidence | What the features show |
|---|---|---|---|---|
| explorer-0020 | 0.50 | explorer → aggressive | 0.97 | 9.7 s, idle 0.02, new cells 0.67 per tick, approach 0.50 |
| explorer-0114 | 0.75 | explorer → aggressive | 0.93 | 11 s, idle 0.00, new cells 0.74 per tick |
| explorer-0020 | 0.25 | explorer → aggressive | 0.89 | 4.8 s, no damage, new cells 0.81 per tick, no enemy met |
| explorer-0113 | 0.25 | explorer → aggressive | 0.88 | 1.8 s, new cells 1.0 per tick, no enemy met |
| cautious-0088 | 1.00 | cautious → explorer | 0.82 | 85 s, idle 0.56, approach 0.70 |

The pattern is real, not a bug: an explorer who has not met an enemy yet sprints through new cells exactly like an aggressive player who has not met one yet. The features that separate those two classes (`near_enemy_ratio`, `damage_per_tick`, `approach_ratio`) carry no information until an encounter happens. Early in a quiet episode the distinction is genuinely unavailable, and the model is confidently wrong there because logistic regression's probabilities are not calibrated for that situation. The cautious-0088 case is a cautious player with a high `reward_greed` parameter who spent a long episode approaching rewards near enemies; the profile boundary is fuzzy by design.

## 4. Ablation: does the model lean on `elapsed_s`?

Phase 6 flagged that `elapsed_s` is confounded with episode length (aggressive episodes are short). Retraining logistic regression without it: CV macro-F1 0.908 ± 0.013, test accuracy 0.941, test macro-F1 0.940, slightly *better* than with it. Permutation importance of `elapsed_s` on the test set is −0.006 (removing it changes nothing). The model is not using play time as a shortcut.

## 5. What the model looks at

Permutation importance (drop in test macro-F1 when the feature is shuffled, mean ± std over 10 repeats):

| Feature | Drop | Feature | Drop |
|---|---|---|---|
| near_enemy_ratio | 0.231 ± 0.024 | reward_share | 0.062 ± 0.008 |
| damage_per_tick | 0.156 ± 0.013 | min_exit_distance_ratio | 0.051 ± 0.007 |
| idle_ratio | 0.139 ± 0.019 | approach_ratio | 0.027 ± 0.010 |
| new_cells_per_tick | 0.129 ± 0.019 | bump_ratio | 0.022 ± 0.005 |
| distinct_cells_ratio | 0.122 ± 0.021 | progress_rate | 0.016 ± 0.010 |
| moves_per_tick | 0.121 ± 0.019 | revisit_ratio, rewards_per_tick, elapsed_s | ≈ 0 |

Largest standardised coefficients per class (sign = direction): aggressive ↑ damage_per_tick (+2.1), ↓ idle_ratio (−2.0), ↑ moves_per_tick (+1.9); cautious ↓ near_enemy_ratio (−3.6), ↑ idle_ratio (+1.7); explorer ↑ near_enemy_ratio (+2.4), ↓ damage_per_tick (−1.9), ↑ distinct_cells_ratio (+1.4). Read together: aggressive players get hit and never stop; cautious players stay away from enemies and wait; explorers get close to enemies *without* taking damage while covering the maze. That matches how the profiles were designed, which is the sanity check we wanted.

## 6. In the game

`neuroquest play` now loads `models/behaviour_clf.joblib` with its metadata (the loader refuses a model whose feature list differs from the code). After 3 s of play, and every 2 s after that, the live features are computed from the episode's own event log and the footer shows the predicted style with its probabilities. Every human episode is also saved to `data/raw/human-<timestamp>-<seed>.jsonl` (`--no-log` disables it), so real play accumulates in the same format as the synthetic data. Screenshots: `docs/screenshots/phase7-prediction-explorer.png`, `phase7-prediction-aggressive.png`.

Hyperparameters were fixed, not tuned (logistic regression C = 1.0, lbfgs, 5,000 iterations; forest 300 trees, min 2 samples per leaf; k = 15). Tuning would need a nested CV to stay honest and was not worth it at these scores.

## 7. Limitations

- The model recognises *simulated* personalities. Human play will differ; the rule-labelled human rows are kept separate so the gap can be measured as real episodes accumulate.
- Early in a quiet episode, explorer and aggressive are not distinguishable from the available features, and the model is overconfident there (probabilities near 1.0 are common; calibration was not assessed).
- One split seed and one dataset of 360 episodes: the ±0.015 CV spread is the uncertainty to keep in mind when quoting 0.933.
