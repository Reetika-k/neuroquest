"""Phase 8: bounds, the rule table, lever choice, reconciliation of a live game."""

import random
from dataclasses import asdict, replace

import pytest

from neuroquest.adapt.bounds import BOUNDS, clamp, within_bounds
from neuroquest.adapt.controller import ADAPT_EVERY_TICKS, TIGHTEN_AFTER_TICKS, TIGHTEN_COOLDOWN_TICKS, AdaptiveController, Decision, Snapshot, decide, direction_rule, lever_for, move
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.rules import new_game, reconcile, step
from neuroquest.core.state import Action
from neuroquest.ml.predict import Prediction


def snap(**kw) -> Snapshot:
    base = dict(tick=50, health=3, max_health=3, damage_recent=0, rewards_recent=0, near_recent=0, ticks_since_change=10**6, time_left_frac=0.9, progress=0.1, style=None, confidence=0.0)
    base.update(kw)
    return Snapshot(**base)


# ---------- bounds ----------


def test_clamp_keeps_every_knob_inside_its_bounds_and_types() -> None:
    wild = DifficultyParams(enemy_count=99, enemy_move_interval=0, reward_density=0.9, time_limit_s=5)
    c = clamp(wild)
    assert (c.enemy_count, c.enemy_move_interval, c.reward_density, c.time_limit_s) == (5, 2, 0.25, 60)
    assert isinstance(c.enemy_count, int) and isinstance(c.enemy_move_interval, int) and isinstance(c.time_limit_s, int)
    assert within_bounds(c) and not within_bounds(wild)
    assert clamp(DifficultyParams()) == DifficultyParams()  # defaults are inside the bounds


# ---------- rules ----------


def test_each_direction_rule_fires_on_its_condition() -> None:
    assert direction_rule(snap(health=1))[:2] == ("ease", "R1_ease_heavy_damage")
    assert direction_rule(snap(damage_recent=2))[:2] == ("ease", "R1_ease_heavy_damage")
    assert direction_rule(snap(time_left_frac=0.2, progress=0.3))[:2] == ("ease", "R2_ease_behind_on_time")
    cruising = dict(rewards_recent=2, near_recent=3, tick=200)
    assert direction_rule(snap(**cruising))[:2] == ("tighten", "R3_tighten_cruising")
    assert direction_rule(snap(progress=0.8, near_recent=1, tick=200))[:2] == ("tighten", "R3_tighten_cruising")
    assert direction_rule(snap())[:2] == ("hold", "R0_hold")
    assert direction_rule(snap(rewards_recent=5, damage_recent=1))[:2] == ("hold", "R0_hold")  # one hit blocks tightening
    # v2 gates: no tightening early, without an encounter, or during the cooldown
    assert direction_rule(snap(**{**cruising, "tick": TIGHTEN_AFTER_TICKS - 1}))[:2] == ("hold", "R0_hold_warmup")
    assert direction_rule(snap(**{**cruising, "near_recent": 0}))[:2] == ("hold", "R0_hold_no_encounter")
    assert direction_rule(snap(**{**cruising, "ticks_since_change": TIGHTEN_COOLDOWN_TICKS - 1}))[:2] == ("hold", "R0_hold_cooldown")
    assert direction_rule(snap(health=1, tick=30))[:2] == ("ease", "R1_ease_heavy_damage")  # easing has no warm-up


def test_lever_depends_on_style_and_direction() -> None:
    assert lever_for("aggressive", "tighten", "R3") == "enemy_count"
    assert lever_for("cautious", "tighten", "R3") == "enemy_move_interval"
    assert lever_for("explorer", "tighten", "R3") == "time_limit_s"
    assert lever_for(None, "tighten", "R3") == "enemy_move_interval"
    assert lever_for("aggressive", "ease", "R1") == "enemy_count"
    assert lever_for("explorer", "ease", "R1") == "enemy_move_interval"
    assert lever_for("explorer", "ease", "R2_ease_behind_on_time") == "time_limit_s"


def test_move_goes_the_right_way_for_every_knob() -> None:
    p = DifficultyParams()
    assert move(p, "enemy_count", "tighten").enemy_count == 3 and move(p, "enemy_count", "ease").enemy_count == 1
    assert move(p, "enemy_move_interval", "tighten").enemy_move_interval == 3 and move(p, "enemy_move_interval", "ease").enemy_move_interval == 5
    assert move(p, "time_limit_s", "tighten").time_limit_s == 100 and move(p, "time_limit_s", "ease").time_limit_s == 140
    assert move(p, "reward_density", "ease").reward_density == pytest.approx(0.15)


def test_decide_records_rule_lever_before_after_and_reason() -> None:
    params = DifficultyParams()
    new, d = decide(snap(rewards_recent=3, near_recent=2, tick=200, style="aggressive", confidence=0.9), params)
    assert isinstance(d, Decision) and d.rule == "R3_tighten_cruising" and d.lever == "enemy_count"
    assert d.before["enemy_count"] == 2 and d.after["enemy_count"] == 3 and new.enemy_count == 3
    assert "style aggressive 90%" in d.reason and d.changed and "enemy count 2 -> 3" in d.summary()
    assert d.to_dict()["inputs"]["rewards_recent"] == 3

    same, hold = decide(snap(), params)
    assert same == params and hold.rule == "R0_hold" and hold.lever is None and not hold.changed

    low_conf, d2 = decide(snap(rewards_recent=3, near_recent=2, tick=200, style="aggressive", confidence=0.4), params)
    assert d2.lever == "enemy_move_interval"  # style ignored below MIN_CONFIDENCE

    at_bound, d3 = decide(snap(health=1, style="aggressive", confidence=0.9), DifficultyParams(enemy_count=1))
    assert at_bound.enemy_count == 1 and d3.rule.endswith("_at_bound") and not d3.changed


def test_thousand_random_decisions_never_leave_the_bounds() -> None:
    rng = random.Random(0)
    params = DifficultyParams()
    styles = [None, "aggressive", "cautious", "explorer"]
    for i in range(1000):
        s = snap(tick=30 * (i + 1), health=rng.randint(0, 3), damage_recent=rng.randint(0, 4), rewards_recent=rng.randint(0, 5), near_recent=rng.randint(0, 20), ticks_since_change=rng.randint(0, 400), time_left_frac=rng.random(), progress=rng.random(), style=rng.choice(styles), confidence=rng.random())
        params, d = decide(s, params)
        assert within_bounds(params), (i, asdict(params), d)
    for name, (lo, hi) in BOUNDS.items():
        assert lo <= getattr(params, name) <= hi


# ---------- controller schedule and snapshot ----------


def test_controller_decides_only_on_schedule_and_reads_live_data() -> None:
    config, params = GameConfig(), DifficultyParams(enemy_count=1, enemy_move_interval=100)
    state = new_game(config, params, seed=2)
    ctrl = AdaptiveController()
    events = []
    for _ in range(ADAPT_EVERY_TICKS - 1):
        state, ev = step(state, Action.STAY, params)
        events += ev
        assert ctrl.step(state, params, events, config.max_health, None) == (params, None)
    state, ev = step(state, Action.STAY, params)
    events += ev
    new, decision = ctrl.step(state, params, events, config.max_health, Prediction("cautious", {"cautious": 0.8}, 0.8))
    assert decision is not None and decision.t == ADAPT_EVERY_TICKS and decision.inputs["health"] == 3
    assert decision.inputs["style"] == "cautious" and 0.0 <= decision.inputs["progress"] <= 1.0
    assert decision.inputs["near_recent"] >= 0 and decision.inputs["ticks_since_change"] > 10**6


# ---------- reconcile ----------


def test_reconcile_spawns_and_removes_enemies_and_tops_up_rewards() -> None:
    config = GameConfig()
    state = new_game(config, DifficultyParams(enemy_count=2), seed=5)
    rng = random.Random(1)
    more, events = reconcile(state, DifficultyParams(enemy_count=4), rng)
    assert len(more.enemies) == 4 and [e.type for e in events] == ["enemy_spawn", "enemy_spawn"]
    assert len({e.id for e in more.enemies}) == 4
    distances = state.maze.reachable_from(state.player.pos)
    for enemy in more.enemies[2:]:
        assert state.maze.is_open(enemy.pos) and enemy.pos not in state.rewards and enemy.pos != state.exit
        assert distances[enemy.pos] >= 0.5 * max(distances.values())
    fewer, events = reconcile(more, DifficultyParams(enemy_count=1), rng)
    assert len(fewer.enemies) == 1 and len(events) == 3 and all(e.type == "enemy_despawn" for e in events)
    nearest = min(more.enemies, key=lambda e: abs(e.pos[0] - state.player.pos[0]) + abs(e.pos[1] - state.player.pos[1]))
    assert fewer.enemies[0].id == nearest.id

    richer, events = reconcile(state, DifficultyParams(reward_density=0.2), rng)
    target = round(0.2 * len(state.maze.open_cells()))
    assert len(richer.rewards) == target and all(e.type == "reward_spawn" for e in events)
    assert all(c not in state.visited for c in richer.rewards - state.rewards)
    poorer, events = reconcile(richer, DifficultyParams(reward_density=0.05), rng)
    assert poorer.rewards == richer.rewards and events == []  # a lower density removes nothing

    unchanged, events = reconcile(state, DifficultyParams(enemy_count=2), rng)
    assert unchanged == state and events == []

    # collecting rewards and then changing an unrelated knob must not refund them
    collected = replace(state, rewards=frozenset(list(state.rewards)[:2]))
    still, events = reconcile(collected, DifficultyParams(enemy_count=3), rng, previous=DifficultyParams(enemy_count=2))
    assert still.rewards == collected.rewards and [e.type for e in events] == ["enemy_spawn"]
    raised, events = reconcile(collected, DifficultyParams(reward_density=0.15), rng, previous=DifficultyParams(reward_density=0.10))
    assert len(raised.rewards) == round(0.15 * len(state.maze.open_cells())) and all(e.type == "reward_spawn" for e in events)
    done = replace(state, outcome=state.outcome.__class__.WON)
    assert reconcile(done, DifficultyParams(enemy_count=5), rng) == (done, [])
