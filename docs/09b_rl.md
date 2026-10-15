# Phase 9b — Reinforcement learning agent (extension)

Status: **approved by the learner** · Date: 2026-10-15 · Outputs: `models/q_table.npz` (+ `.meta.json`), `reports/rl_curve.csv`, `reports/rl_eval.csv`, `reports/rl_results.json`, dashboard section 8 in `neuroquest/reports/dashboard.html`

## 1. The idea, in plain language

The brief's optional extension asks for a small maze environment in which an agent *learns* to collect rewards, avoid hazards and reach the exit, with the state, actions, reward function and termination defined explicitly, and the learned agent compared with a random or rule-based baseline.

Nothing new was written for the game itself. The environment (`rl/env.py`) is a thin wrapper around the same pure `step()` function the window, the simulator and the tests use, so the agent plays the real game: the same walls, the same A* hunter, the same crystals, portal, health and clock. The agent is a **tabular Q-learning** agent: a table with one number per (observation, action) that is nudged after every tick toward "the reward I just got plus the best I expect from the next observation". It starts knowing nothing, not even where the walls are.

## 2. The environment, defined

| Element | Definition |
|---|---|
| Board | One fixed 11×9 maze (layout seed 3, fully braided so every dead end is a loop): 43 open cells, 4 crystals at (3,8), (5,3), (6,1), (7,6), portal at (7,9), start (1,1). The maze, portal and crystals are the same in every episode; the hunter's start cell comes from the episode seed (always on the far half of the board) |
| Hazard | 1 A* hunter moving every 2 ticks (the player moves every tick); 3 health; 15-tick invulnerability after a hit, as in the game |
| Clock | 200 ticks (20 s) |
| Observation | `(player_row, player_col, crystals_mask, hunter_dr, hunter_dc)`: the agent's cell; one bit per crystal still on the board; the hunter's exact offset when it is within ±2 cells, otherwise only its direction (±3 or 0 on each axis). The agent never sees the clock, its health or the walls. Upper bound 23,392 observations; 1,753 were actually visited in training |
| Actions | the game's five: up, down, left, right, stay |
| Reward per tick | −0.01 (time) · +1 per crystal · +5 at the portal · −2 per hit · −5 more when health reaches zero. Read straight off the events the rules emit |
| Termination | portal reached or health zero → *terminated*; clock run out → *truncated* (the agent cannot see the clock, so the time-out is treated as a cut-off that the learner bootstraps through, Gymnasium's convention) |

The API mirrors Gymnasium's (`reset(seed) → (obs, info)`, `step(a) → (obs, reward, terminated, truncated, info)`) without depending on the package; the optional `rl` extra in `pyproject.toml` remains unused.

**Why this board.** The first board tried was an 11×9 maze with the game's default 15 % braiding. Every agent on it, scripted or learned, took *exactly one hit*: that maze is a single corridor and the hunter, which starts on the far half, always sits on the only route to the portal. A learner cannot learn to avoid what the board does not allow it to avoid. A scan of 36 layouts (braiding 0.15 / 0.5 / 1.0 × twelve seeds, 50 episodes each) measured, per board, how often a hunter-blind collector is hit and how often the hazard-aware cautious profile is hit. On fully braided layout seed 3 the collector is hit in 82 % of episodes and the cautious profile in 0 %: the hazard is real *and* avoidable, which is the test "avoid hazards" needs.

## 3. Training

| Setting | Value |
|---|---|
| Episodes | 20,000 (episode seed = episode index, so the hunter's start cell cycles through the 20 possible cells) |
| Learning rate α | 0.2 |
| Discount γ | 0.95 |
| Exploration | ε-greedy, ε linear from 1.0 to 0.05 over the first 16,000 episodes (80 %), then 0.05 |
| Initial Q-values | 0 (optimistic, since most ticks cost −0.01) |
| Time | 12 s on this Mac (the pure `step()` costs ~15 µs) |

**Learning curve** (`reports/rl_curve.csv`, moving mean over 250 episodes; figure `docs/screenshots/phase9b-learning-curve.png`):

| Point | Win rate | Hits / episode | Return |
|---|---|---|---|
| first 250 episodes (ε ≈ 1) | 0.00 | 3.00 | −11.14 |
| episode 6,241 (ε 0.63) | first reaches 0.50 | | |
| episode 9,509 (ε 0.44) | first reaches 0.95 | | |
| last 500 episodes (ε 0.05) | 1.00 | 0.11 | +7.13 |

The three panels of the figure tell the story in order: the return climbs from the random baseline toward the scripted agents' level, the win rate follows, and the hits per episode fall from three (death every time) to about a tenth, most of that while ε is still above 0.2.

## 4. Evaluation against the baselines

All agents play the same 100 held-out episodes (seeds 100000–100099, which cover 20 distinct hunter start cells; the hunter is deterministic, so these are 20 distinct situations rather than 100). The scripted baselines read the full game state; the learned agent reads only its observation.

| Agent | What it does | Win rate [95 % CI] | Mean return | Crystals | Hits | Steps |
|---|---|---|---|---|---|---|
| random | a uniformly random action every tick | 0.00 [0.00, 0.04] | −11.05 | 0.45 | 3.00 | 50 |
| A* to exit | shortest path to the portal; ignores crystals and the hunter | 1.00 [0.96, 1.00] | +6.08 | 2.00 | 0.39 | 14 |
| greedy collector | nearest crystal first, then the portal; ignores the hunter | 1.00 [0.96, 1.00] | +7.10 | 4.00 | 0.82 | 26 |
| cautious profile | the Phase 6 scripted player that keeps a safety margin from the hunter | 1.00 [0.96, 1.00] | +6.85 | 2.17 | 0.00 | 32 |
| **Q-learning** | the learned table, greedy | **1.00 [0.96, 1.00]** | **+7.32** | 2.59 | 0.04 | 19 |

Reading the table honestly:

- **Against random** the comparison the brief asks for is not close: random play never reaches the portal and always dies; the learned agent always reaches it.
- **Against the scripted agents** the learned agent has the highest mean return, which is what it was trained to maximise: it collects 2 or 3 of the 4 crystals (2 in 41 episodes, 3 in 59, never all 4) and is hit in only 4 of 100 episodes (seeds 100028, 100048, 100079, 100091), where the hunter-blind collector is hit in 82. It learned to trade the two crystals that lie on the hunter's side of the board for safety, and to do it in 19 steps, faster than the cautious profile's 32.
- **Win rate alone does not separate the four non-random agents**; on this board every scripted agent reaches the portal too. The differences are in hits, crystals and return, which is why all four are reported.
- A second training seed (ε-greedy randomness seed 1) reached the same behaviour (win 1.00, return +7.27, 2.62 crystals, 0.07 hits), and evaluating the table with 5 % random actions still gave win 1.00, return +7.11, 0.13 hits (measured during development with the same settings; not written to the reports).

**No transfer.** The same table on a board it never saw (layout seed 4) wins 0 of 100 episodes, the same as random play. A tabular agent memorises one board; this is expected and is stated on the dashboard.

**A counter-example worth keeping.** On layout seed 5, where the only hunter-free route is a 56-step detour, the same training converged to the collector's behaviour: all four crystals, one hit every episode, return +6.75, *below* the cautious profile's +7.53. With γ = 0.95 the portal's +5 is worth about 0.3 after 56 steps and about 1.3 after 26, so the discounted optimum really is "take the hit". The reward function and discount shape what "good play" means; they are design choices and are listed above for that reason.

## 5. Watch it play

`python -m neuroquest play --agent` opens the normal window with the trained policy choosing the moves on its training board (the difficulty is fixed to the training settings, nothing is logged as human play, R restarts with a new hunter start). The footer says the agent is playing; T shows the hunter's A* path; the style classifier still runs and shows what it makes of the agent. Frame: `docs/screenshots/phase9b-agent-playing.png`.

## 6. Limitations

- One small board, one hunter, partial observation; no claim is made about the full 21×15 game. Tabular Q-learning does not scale to it (the state would include the whole maze); that would need function approximation, which is out of scope.
- The hunter is deterministic, so 100 evaluation episodes are 20 distinct situations; the Wilson interval treats them as 100 independent trials and is therefore optimistic.
- The reward numbers (−2 per hit, +1 per crystal) were chosen by reasoning, not tuned; a different ratio would produce a different trade-off between crystals and safety (section 4's counter-example).
- The optional next-move model (Phase 7b) and a learned adaptation policy for the difficulty controller are still not built; the RL agent plays the game, it does not tune it.

## 7. Reproduce

```bash
.venv/bin/python -m neuroquest rl-train            # ~15 s: trains, evaluates, writes models/ and reports/
.venv/bin/python -m neuroquest dashboard           # rebuilds reports/dashboard.html with section 8
.venv/bin/python -m neuroquest play --agent        # watch the agent
.venv/bin/pytest tests/test_env.py tests/test_qlearning.py
```
