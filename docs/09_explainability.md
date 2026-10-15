# Phase 9 — Dashboard and explainability

Status: **approved by the learner 2026-10-15** · Date: 2026-10-15 · Output: `neuroquest/reports/dashboard.html` (self-contained; link from the repository root: [neuroquest/reports/dashboard.html](../reports/dashboard.html)) and `reports/figures/dash_*.png`

## 1. What the dashboard is

One HTML page, built by `neuroquest dashboard` from files that earlier phases recorded, nothing else:

| Section | Source | Figure |
|---|---|---|
| Overview stats | `episodes.csv`, `metrics.json` | – |
| 1 Gameplay data: rows per label, outcomes per profile | `episodes.csv` | `dash_data.png` |
| 2 Model evaluation: all models' CV and test macro-F1, confusion matrix, per-class precision/recall/F1, accuracy by prefix, ablation note | `metrics.json` | `dash_models.png`, `dash_confusion.png`, `dash_prefix.png` |
| 3 What the model looks at: permutation importance, per-class coefficients | `metrics.json` | `dash_importance.png`, `dash_coefficients.png` |
| 4 Correct and incorrect predictions | saved model re-run on the held-out seeds of the training split | tables |
| 5 A session as the model saw it: style probabilities over time | the saved model re-run over the latest human episode log on the live schedule | `dash_session.png` |
| 6 Adaptive difficulty decisions: parameters over time with rule markers, rules fired, the recorded reasons | `decisions.jsonl` | `dash_difficulty.png` |
| 7 Fixed versus adaptive | `ab_results.csv` | `dash_ab.png` |
| 8 Limitations, Reproduce | text | – |

A missing input makes its section say so; nothing is estimated or filled in. The in-game intelligence panel (key I) is the live counterpart; the dashboard is the offline, shareable one.

## 2. How the AI reached its decisions, inspectable

- **Prediction:** the panel and section 5 show the three class probabilities, and section 3 shows which features move them (permutation importance; standardised coefficients per class). Section 4 shows real rows with their features next to the prediction.
- **Difficulty:** every decision record carries the rule that fired, the lever, before/after values, the inputs snapshot and the reason string written at the time; section 6 lists them and plots the parameters over the session. The game's bottom bar and panel show the same records live.

## 3. Correct and incorrect predictions (held-out seeds, saved model)

Confident and right:

| Episode | Prefix | True | Confidence | near enemy | damage/tick | idle | new cells/tick |
|---|---|---|---|---|---|---|---|
| aggressive-0003 | 1.00 | aggressive | 1.00 | 0.43 | 0.039 | 0.05 | 0.68 |
| aggressive-0092 | 1.00 | aggressive | 1.00 | 0.46 | 0.039 | 0.09 | 0.62 |
| cautious-0084 | 0.25 | cautious | 1.00 | 0.00 | 0.000 | 0.93 | 0.01 |
| cautious-0084 | 0.50 | cautious | 1.00 | 0.00 | 0.000 | 0.93 | 0.00 |
| explorer-0024 | 0.50 | explorer | 1.00 | 0.74 | 0.000 | 0.51 | 0.26 |
| explorer-0088 | 0.75 | explorer | 1.00 | 0.67 | 0.003 | 0.60 | 0.13 |

Confident and wrong (the five most confident errors):

| Episode | Prefix | True → predicted | Confidence | near enemy | damage/tick | idle | new cells/tick |
|---|---|---|---|---|---|---|---|
| explorer-0020 | 0.50 | explorer → aggressive | 0.97 | 0.09 | 0.010 | 0.02 | 0.67 |
| explorer-0114 | 0.75 | explorer → aggressive | 0.93 | 0.12 | 0.009 | 0.00 | 0.74 |
| explorer-0020 | 0.25 | explorer → aggressive | 0.89 | 0.00 | 0.000 | 0.00 | 0.81 |
| explorer-0113 | 0.25 | explorer → aggressive | 0.88 | 0.00 | 0.000 | 0.00 | 1.00 |
| cautious-0088 | 1.00 | cautious → explorer | 0.82 | 0.09 | 0.000 | 0.56 | 0.14 |

Reading the two tables together explains the model: the correct explorers were near enemies a lot without being hurt (near 0.67–0.74, damage ≈ 0), which is what the explorer coefficients reward; the wrong explorers had barely met an enemy (near 0.00–0.12) and were sprinting through new cells, which is indistinguishable from an aggressive player who has not met one yet. The cautious error is a cautious profile with a high reward-greed parameter, a genuinely blurry boundary.

## 4. What the recorded human play shows (2026-10-15)

- 13 human episode logs in `data/raw/`; the latest (14 s, lost on health, score 30) was read as *cautious* at 100 % until 11 s, drifting toward *explorer* at the end.
- `reports/decisions.jsonl`: 57 decisions over 10 sessions: 35 plain holds, 12 warm-up holds, 1 "no encounter yet" hold, 7 easings for heavy damage and 2 more that were already at the bound. No tightening fired in human play yet: the sessions were short and the player was being hit, which is exactly when the controller is designed to ease.

## 5. Limitations (also in the dashboard)

- The classifier recognises simulated personalities; human play may differ. Human rows stay separate so the gap can be measured.
- Early in a quiet episode explorer and aggressive are not distinguishable from the features; the model is confidently wrong there, and its probabilities are not calibrated.
- One split seed, 360 episodes: quote 0.933 with the ±0.015 CV spread in mind.
- The controller is rule-based with reasoned, untuned thresholds; safe on simulated players, not shown to improve outcomes, untested on humans beyond the logs above.

## 6. Reproduce

```bash
.venv/bin/python -m neuroquest dashboard      # ~5 s; opens nothing, writes reports/dashboard.html
open reports/dashboard.html
```
