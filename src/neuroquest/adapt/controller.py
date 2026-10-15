"""Rule-based adaptive difficulty.

Two questions, answered in order:

1. *Which way?* Performance decides: a player who is being hurt gets an easier game, a player who
   is cruising gets a harder one, otherwise nothing changes. These rules are deliberately few and
   readable; each decision records which rule fired and the numbers that made it fire.
2. *Which knob?* The predicted play style decides, when the model is confident: an aggressive
   player is challenged with more enemies, a cautious player with faster enemies (waiting stops
   paying), an explorer with less time (wandering stops paying). Easing always relieves the cause
   of the damage: fewer enemies for aggressive players, slower enemies for everyone else.

Every change is one step of one knob, clamped to the bounds. Decisions are taken every
ADAPT_EVERY_TICKS; the game is never made harder in the first TIGHTEN_AFTER_TICKS, only after the
player has actually met an enemy recently without being hit, and at most once per
TIGHTEN_COOLDOWN_TICKS. Easing has no such gates: a player in trouble gets relief at the next
decision. (Version 2: version 1 tightened at 5 s on players who had not met an enemy yet; see docs/08.)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from neuroquest.adapt.bounds import STEP, clamp
from neuroquest.config import DifficultyParams
from neuroquest.core.maze import Cell
from neuroquest.core.rules import time_left_s
from neuroquest.core.state import Event, GameState

ADAPT_EVERY_TICKS = 30  # one decision per 3 s of play
WINDOW_TICKS = 150  # "recent" = the last 15 s
MIN_CONFIDENCE = 0.6  # below this the style is treated as unknown
TIGHTEN_AFTER_TICKS = 150  # never harder in the first 15 s
TIGHTEN_COOLDOWN_TICKS = 150  # and at most one tightening per 15 s after any change


@dataclass(frozen=True)
class Snapshot:
    """Everything a decision is based on, so the decision can be replayed and explained."""

    tick: int
    health: int
    max_health: int
    damage_recent: int
    rewards_recent: int
    near_recent: int  # ticks within 2 cells of an enemy in the window: has the player actually been challenged?
    ticks_since_change: int
    time_left_frac: float
    progress: float  # 0 = as far from the exit as at the start, 1 = on the exit
    style: str | None
    confidence: float


@dataclass(frozen=True)
class Decision:
    t: int
    rule: str
    direction: str  # ease · tighten · hold
    lever: str | None
    before: dict
    after: dict
    inputs: dict
    reason: str

    @property
    def changed(self) -> bool:
        return self.before != self.after

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if not self.changed:
            return f"{self.rule}: {self.reason}"
        name = self.lever or ""
        return f"{name.replace('_', ' ')} {self.before[name]} -> {self.after[name]}  ({self.rule}: {self.reason})"  # ASCII arrow: Pygame's default font has no glyph for the real one


def direction_rule(s: Snapshot) -> tuple[str, str, str]:
    """(direction, rule id, reason). The first rule whose condition holds wins."""
    window_s = WINDOW_TICKS // 10
    if s.health <= 1 or s.damage_recent >= 2:
        return "ease", "R1_ease_heavy_damage", f"health {s.health}/{s.max_health}, {s.damage_recent} hits in the last {window_s} s"
    if s.time_left_frac < 0.25 and s.progress < 0.5:
        return "ease", "R2_ease_behind_on_time", f"{s.time_left_frac:.0%} of the time left but only {s.progress:.0%} of the way to the exit"
    cruising = s.damage_recent == 0 and s.health == s.max_health and (s.rewards_recent >= 2 or s.progress >= 0.75)
    if cruising:
        if s.tick < TIGHTEN_AFTER_TICKS:
            return "hold", "R0_hold_warmup", f"doing well but only {s.tick / 10:.0f} s played; no tightening before {TIGHTEN_AFTER_TICKS // 10} s"
        if s.near_recent == 0:
            return "hold", "R0_hold_no_encounter", f"doing well but has not met an enemy in the last {window_s} s; nothing to judge yet"
        if s.ticks_since_change < TIGHTEN_COOLDOWN_TICKS:
            return "hold", "R0_hold_cooldown", f"doing well but the last change was {s.ticks_since_change / 10:.0f} s ago; cooldown is {TIGHTEN_COOLDOWN_TICKS // 10} s"
        return "tighten", "R3_tighten_cruising", f"near enemies on {s.near_recent} ticks in the last {window_s} s without a hit, full health, {s.rewards_recent} rewards recently, {s.progress:.0%} of the way"
    return "hold", "R0_hold", f"{s.damage_recent} recent hits, health {s.health}/{s.max_health}, {s.rewards_recent} recent rewards: no rule applies"


def lever_for(style: str | None, direction: str, rule: str) -> str:
    """Which knob to move, given the (confident) predicted style and the direction."""
    if rule == "R2_ease_behind_on_time":
        return "time_limit_s"
    if direction == "ease":
        return "enemy_count" if style == "aggressive" else "enemy_move_interval"
    return {"aggressive": "enemy_count", "cautious": "enemy_move_interval", "explorer": "time_limit_s"}.get(style or "", "enemy_move_interval")


def move(params: DifficultyParams, lever: str, direction: str) -> DifficultyParams:
    """One step of one knob in the given direction, then clamped. 'Harder' means more enemies,
    a smaller move interval (faster), fewer seconds, or fewer rewards."""
    harder = direction == "tighten"
    sign = 1 if lever == "enemy_count" else -1  # for the other knobs, a larger value is easier
    delta = STEP[lever] * (sign if harder else -sign)
    return clamp(replace(params, **{lever: getattr(params, lever) + delta}))


def decide(s: Snapshot, params: DifficultyParams) -> tuple[DifficultyParams, Decision]:
    direction, rule, reason = direction_rule(s)
    style = s.style if s.confidence >= MIN_CONFIDENCE else None
    style_note = f" (style {s.style} {s.confidence:.0%})" if s.style else " (style unknown)"
    before = asdict(params)
    if direction == "hold":
        return params, Decision(s.tick, rule, direction, None, before, before, asdict(s), reason + style_note)
    lever = lever_for(style, direction, rule)
    new = move(params, lever, direction)
    if new == params:
        rule += "_at_bound"
        reason += f"; {lever} already at its bound"
    return new, Decision(s.tick, rule, direction, lever, before, asdict(new), asdict(s), reason + style_note)


class AdaptiveController:
    """Stateful wrapper used by the game loop and the harness: decides on schedule from live data."""

    def __init__(self, every_ticks: int = ADAPT_EVERY_TICKS, window_ticks: int = WINDOW_TICKS) -> None:
        self.every_ticks = every_ticks
        self.window_ticks = window_ticks
        self._maze_id: int | None = None
        self._to_exit: dict[Cell, int] = {}
        self._start_distance = 1
        self.last_change_tick = -(10**9)

    def due(self, tick: int) -> bool:
        return tick >= self.every_ticks and tick % self.every_ticks == 0

    def snapshot(self, state: GameState, params: DifficultyParams, events: list[Event], max_health: int, prediction) -> Snapshot:
        if self._maze_id != id(state.maze):
            self._maze_id = id(state.maze)
            self._to_exit = state.maze.reachable_from(state.exit)
            self._start_distance = max(1, self._to_exit.get((1, 1), 1))
        since = state.tick - self.window_ticks
        recent = [e for e in events if e.t > since]
        return Snapshot(
            tick=state.tick,
            health=state.player.health,
            max_health=max_health,
            damage_recent=sum(1 for e in recent if e.type == "damage"),
            rewards_recent=sum(1 for e in recent if e.type == "reward"),
            near_recent=sum(1 for e in recent if e.type == "near_enemy"),
            ticks_since_change=state.tick - self.last_change_tick,
            time_left_frac=time_left_s(state, params) / max(1, params.time_limit_s),
            progress=1.0 - self._to_exit.get(state.player.pos, self._start_distance) / self._start_distance,
            style=getattr(prediction, "label", None),
            confidence=float(getattr(prediction, "confidence", 0.0)),
        )

    def step(self, state: GameState, params: DifficultyParams, events: list[Event], max_health: int, prediction) -> tuple[DifficultyParams, Decision | None]:
        if not self.due(state.tick):
            return params, None
        new_params, decision = decide(self.snapshot(state, params, events, max_health, prediction), params)
        if decision.changed:
            self.last_change_tick = state.tick
        return new_params, decision
