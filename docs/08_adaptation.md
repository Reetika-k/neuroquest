# Phase 8 — Adaptive difficulty

Status: **approved by the learner** · Date: 2026-10-11 · Rows: `reports/ab_results.csv` · Live decisions: `reports/decisions.jsonl`

## 1. The idea, in plain language

The controller answers two questions every 3 seconds of play.

**Which way?** Performance decides. A player who is being hurt gets an easier game; a player who is handling the enemies without a scratch gets a harder one; otherwise nothing changes. These rules are few and readable on purpose, and every decision records which rule fired and the numbers that made it fire.

**Which knob?** The predicted style decides, when the classifier is at least 60 % confident. An aggressive player is challenged with *more enemies*; a cautious player with *faster enemies*, so waiting stops paying; an explorer with *less time*, so wandering stops paying. Easing always relieves the cause of the damage: fewer enemies for an aggressive player, slower enemies for everyone else. If the style is unknown, the lever is enemy speed.

Every change is one step of one knob and is clamped to safety bounds, so the game can never become unplayable.

## 2. Inputs, rules, outputs

**Inputs (the `Snapshot`, recorded with every decision).** tick, health, hits in the last 15 s, rewards in the last 15 s, ticks near an enemy in the last 15 s, ticks since the last change, fraction of time left, progress toward the exit (0 = as far as at the start, 1 = there), predicted style and confidence.

**Rule table (first match wins).**

| Rule | Condition | Direction | Reason recorded |
|---|---|---|---|
| R1_ease_heavy_damage | health ≤ 1, or ≥ 2 hits in the last 15 s | ease | health and recent hits |
| R2_ease_behind_on_time | < 25 % of the time left and < 50 % of the way to the exit | ease (always the time limit, +20 s) | time left vs progress |
| R0_hold_warmup | cruising, but less than 15 s played | hold | too early to judge |
| R0_hold_no_encounter | cruising, but no enemy within 2 cells in the last 15 s | hold | nothing to judge yet |
| R0_hold_cooldown | cruising, but a change happened less than 15 s ago | hold | cooldown |
| R3_tighten_cruising | no hits in 15 s, full health, and (≥ 2 rewards recently or ≥ 75 % of the way), after an encounter | tighten | encounters, rewards, progress |
| R0_hold | anything else | hold | the numbers that did not trigger a rule |

"Cruising" = no hits in the last 15 s at full health with ≥ 2 recent rewards or ≥ 75 % progress.

**Levers and steps.** enemy_count ±1 · enemy_move_interval ±1 tick (smaller is faster) · time_limit_s ±20 s. Reward density is bounded and reconcilable but no rule moves it in this version.

**Bounds (`adapt/bounds.py`).** enemies 1–5 · enemy move interval 2–8 ticks · reward density 0.05–0.25 · time limit 60–240 s. A test applies 1,000 random decisions in a row and checks the parameters never leave these ranges.

**Applying a change to a live game (`core/rules.py::reconcile`).** Speed and time limit take effect on the next tick. A higher enemy count spawns enemies far from the player; a lower one removes the farthest. Rewards are topped up only when the density was raised (see section 4 for why that clause matters).

**Decision record (JSON Lines).** `t, rule, direction, lever, before, after, inputs, reason`, plus the episode id in the game's `reports/decisions.jsonl`. The footer shows the latest decision for 6 s ("Difficulty changed: enemy count 2 → 1 (R1_ease_heavy_damage: health 1/3, 2 hits in the last 15 s (style aggressive 99%))"), then the current settings.

## 3. Fixed versus adaptive (2026-10-11)

`neuroquest ab-test --episodes 40 --seed 500`: three profiles × 40 seeded mazes (seeds 500–539, never used in training) × two conditions. The same profile with the same parameters plays the same maze under both conditions. Fixed = the defaults (2 enemies, interval 4, 120 s).

**Controller v2 (current).**

| Profile | Condition | Win | Lost (health) | Lost (time) | Median ticks | Mean damage | Mean score | Changes / episode | Rules that changed something |
|---|---|---|---|---|---|---|---|---|---|
| aggressive | fixed | 20 % | 32 | 0 | 99 | 2.80 | 93.8 | – | – |
| aggressive | adaptive | 22 % | 31 | 0 | 100 | 2.77 | 95.0 | 0.68 | R1 ×27 |
| cautious | fixed | 62 % | 15 | 0 | 366 | 1.98 | 102.5 | – | – |
| cautious | adaptive | 65 % | 14 | 0 | 366 | 1.98 | 103.5 | 2.27 | R1 ×76, R3 ×15 |
| explorer | fixed | 80 % | 8 | 0 | 269 | 1.57 | 104.2 | – | – |
| explorer | adaptive | 85 % | 6 | 0 | 270 | 1.52 | 107.0 | 2.02 | R1 ×57, R3 ×24 |

Differences, adaptive − fixed: aggressive +2 points win rate, cautious +3, explorer +5; mean damage −0.02 / 0.00 / −0.05.

**What this does and does not show.** With 40 episodes per cell, the sampling noise on a win-rate difference is roughly ±20 points at 95 % confidence, so none of these differences is evidence of improvement. What the table does show is that v2 is *safe*: no profile is worse off, damage is unchanged, no episode was lost on time, and every change is traceable to a rule. It also shows the controller is mostly a relief valve in this game: R1 (ease) fires three to four times as often as R3 (tighten). Equalising outcomes across styles (fixed: 20 / 62 / 80 %) would need stronger or earlier easing for aggressive players, who die within about 10 s before more than one decision can act, and is left as the next iteration.

## 4. Failure → improvement: how v1 became v2

**v1 results (same experiment).** aggressive 20 % → 15 %, cautious 62 % → 62 % (but median length 366 → 274 ticks), explorer 80 % → **60 %** with mean damage 1.57 → 2.15. Rules fired: R3 tighten ×29 on aggressive players, ×104 on explorers.

**Why.** v1 decided every 5 s and tightened on "no damage, full health, rewards collected", which is true of *everyone at 5 s*: the enemies have not arrived yet. It therefore made the game harder for players who had not been tested, and did so before the classifier had seen an encounter, so early misreads of explorers as aggressive (the known confusion from Phase 7) picked the wrong lever (more enemies). Aggressive players, who die in about 10 s, received one decision, and it was a tightening.

**Also found while reading the first results.** Adaptive scores were about 50 % higher than fixed scores even though win rates were lower. The cause was in `reconcile`: it topped up rewards whenever *any* parameter changed, refunding rewards the player had already collected. The experiment was invalid until that was fixed (rewards are now added only when the density is raised); with the fix, scores match across conditions, as they should.

**Changes (v2).** Decisions every 3 s instead of 5 so relief arrives in time; tightening requires an encounter in the last 15 s (`near_recent ≥ 1`), is never applied in the first 15 s, and at most once per 15 s after any change; easing has no gates. The three new hold reasons (warm-up, no encounter, cooldown) are logged so the footer can say *why* nothing changed.

## 5. In the game

`neuroquest play` runs the controller by default (`--fixed` turns it off). The footer's second line shows the current settings and, for 6 s after a change, the decision and its reason. Decisions are appended to `reports/decisions.jsonl` with the episode id. Screenshots: `docs/screenshots/phase8-decision-aggressive.png`, `phase8-decision-explorer.png`.

## 6. Limitations

- The experiment uses simulated players; the controller's effect on humans is untested.
- Rule thresholds (2 hits, 15 s window, 25 % time left, 75 % progress) were set by reasoning, not tuned; tuning them against the simulation would be the next experiment, and a learned policy (a bandit over the three levers) is the optional extension from the brief.
- The style lever depends on the classifier; its early explorer/aggressive confusion still reaches the controller, now only after an encounter.
