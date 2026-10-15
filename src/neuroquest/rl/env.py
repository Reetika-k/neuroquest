"""MazeEnv: NeuroQuest's own rules behind a gym-style ``reset()`` / ``step()`` API.

The environment is a thin wrapper. Every transition is the game's pure ``core.rules.step``, so
the agent plays exactly the game a human plays, only on a small board (11×9), with one hunter
and a 20-second clock. What this module adds is the three things reinforcement learning needs
on top of a game:

**Observation** (a tuple, so it can index a Q-table)::

    (player_row, player_col, crystals_mask, enemy_dr, enemy_dc)

- ``crystals_mask``: bit *i* is set while the *i*-th crystal (fixed order) is still on the board.
- enemy offset: the exact ``(dr, dc)`` when the nearest hunter is within ±2 cells; beyond that
  only its direction, as ``(±3 or 0, ±3 or 0)``; ``(4, 4)`` when there is no hunter at all.
- The agent never sees the clock, its health or the maze walls: it must learn the walls by
  bumping into them.

**Actions**: ``0`` up · ``1`` down · ``2`` left · ``3`` right · ``4`` stay (the game's own five).

**Reward per tick**: −0.01 (time) · +1 per crystal · +5 for reaching the exit · −2 per hit ·
−5 extra when health reaches zero. These are read off the ``Event`` list the rules emit and
nothing else; the numbers live in ``RewardSpec`` so a report can print them.

**Termination**: won or lost all health → ``terminated``; out of time → ``truncated`` (the agent
cannot see the clock, so a time-out is a cut-off rather than a state it could have avoided from
what it saw; a learner keeps bootstrapping through it, as Gymnasium's convention recommends).

**Layout**: the maze, exit and crystals come from ``layout_seed`` and are the same in every
episode; the hunter's start cell comes from the episode's ``seed``. A tabular agent learns one
board. The API mirrors Gymnasium's without depending on it.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.maze import Cell, manhattan
from neuroquest.core.rules import new_game, step
from neuroquest.core.state import Action, Event, GameState, Outcome

ACTIONS: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY)
ACTION_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ACTIONS)}
Obs = tuple[int, int, int, int, int]

# The training board. Fully braided (every dead end opened into a loop) so that the hunter can be
# avoided: on the game's default 15 %-braided boards of this size the maze is one corridor and
# the hunter, which starts on the far half, always blocks the only route (measured: every agent,
# scripted or learned, took exactly one hit). Layout seed 3 was chosen from a scan of twelve
# seeds because the hunter-blind collector is hit in 82 % of episodes there while the cautious
# scripted player is never hit, so "avoid the hazard" is both necessary and possible.
RL_CONFIG = GameConfig(width=11, height=9, loop_fraction=1.0)
RL_PARAMS = DifficultyParams(enemy_count=1, enemy_move_interval=2, reward_density=0.10, time_limit_s=20)
LAYOUT_SEED = 3
ENEMY_WINDOW = 2  # the hunter's exact offset is observed within this many cells on each axis
FAR = ENEMY_WINDOW + 1  # beyond the window only the direction is observed, encoded as ±FAR
NO_ENEMY = FAR + 1


@dataclass(frozen=True)
class RewardSpec:
    """The reward function, one number per kind of event plus a per-tick cost."""

    step: float = -0.01
    crystal: float = 1.0
    exit: float = 5.0
    hit: float = -2.0
    defeat: float = -5.0

    def of(self, events: list[Event]) -> float:
        total = self.step
        for e in events:
            if e.type == "reward":
                total += self.crystal
            elif e.type == "exit":
                total += self.exit
            elif e.type == "damage":
                total += self.hit
            elif e.type == "defeat":
                total += self.defeat
        return total


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


class MazeEnv:
    """One fixed board; ``reset(seed)`` starts an episode, ``step(action)`` advances one tick."""

    def __init__(
        self,
        config: GameConfig = RL_CONFIG,
        params: DifficultyParams = RL_PARAMS,
        layout_seed: int = LAYOUT_SEED,
        rewards: RewardSpec = RewardSpec(),
    ) -> None:
        self.config, self.params, self.layout_seed, self.reward_spec = config, params, layout_seed, rewards
        template = new_game(config, params, seed=0, layout_seed=layout_seed)
        self.maze, self.exit = template.maze, template.exit
        self.crystals: tuple[Cell, ...] = tuple(sorted(template.rewards))
        self.n_actions = len(ACTIONS)
        self.state: GameState | None = None
        self._rng: random.Random | None = None

    # ---------- gym-style API ----------

    def reset(self, seed: int) -> tuple[Obs, dict]:
        self.state = new_game(self.config, self.params, seed, layout_seed=self.layout_seed)
        self._rng = random.Random(seed + 10_007)  # drives random-walk hunters only; A* hunters are deterministic
        return self.observe(self.state), self._info([])

    def step(self, action: int) -> tuple[Obs, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        if self.state.outcome.is_over:
            raise RuntimeError("the episode is over; call reset()")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in 0..{self.n_actions - 1}, got {action}")
        self.state, events = step(self.state, ACTIONS[action], self.params, self._rng)
        reward = self.reward_spec.of(events)
        terminated = self.state.outcome in (Outcome.WON, Outcome.LOST_HEALTH)
        truncated = self.state.outcome is Outcome.LOST_TIME
        return self.observe(self.state), reward, terminated, truncated, self._info(events)

    # ---------- what the agent sees ----------

    def observe(self, state: GameState) -> Obs:
        pr, pc = state.player.pos
        mask = 0
        for i, cell in enumerate(self.crystals):
            if cell in state.rewards:
                mask |= 1 << i
        if not state.enemies:
            return (pr, pc, mask, NO_ENEMY, NO_ENEMY)
        nearest = min(state.enemies, key=lambda e: manhattan(state.player.pos, e.pos))
        dr, dc = nearest.pos[0] - pr, nearest.pos[1] - pc
        if abs(dr) > ENEMY_WINDOW or abs(dc) > ENEMY_WINDOW:
            dr, dc = FAR * _sign(dr), FAR * _sign(dc)
        return (pr, pc, mask, dr, dc)

    def state_space_bound(self) -> int:
        """An upper bound on distinct observations (most are unreachable)."""
        relations = (2 * ENEMY_WINDOW + 1) ** 2 + 9 if self.params.enemy_count else 1
        return len(self.maze.open_cells()) * 2 ** len(self.crystals) * relations

    def describe(self) -> dict:
        """Everything a report needs to state what the agent faced."""
        return {
            "board": f"{self.config.width}x{self.config.height}",
            "open_cells": len(self.maze.open_cells()),
            "crystals": len(self.crystals),
            "exit": list(self.exit),
            "layout_seed": self.layout_seed,
            "hunters": self.params.enemy_count,
            "hunter_kind": self.config.enemy_kind,
            "hunter_move_interval": self.params.enemy_move_interval,
            "time_limit_ticks": self.params.time_limit_s * self.config.ticks_per_second,
            "max_health": self.config.max_health,
            "actions": [a.value for a in ACTIONS],
            "enemy_window": ENEMY_WINDOW,
            "rewards": asdict(self.reward_spec),
            "state_space_bound": self.state_space_bound(),
        }

    def _info(self, events: list[Event]) -> dict:
        assert self.state is not None
        return {
            "state": self.state,
            "events": events,
            "outcome": self.state.outcome.value,
            "won": self.state.outcome is Outcome.WON,
            "crystals": len(self.crystals) - len(self.state.rewards),
            "hits": sum(e.type == "damage" for e in events),
        }
