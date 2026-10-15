"""Phase 4: starting a game and advancing it one tick at a time."""

import pytest

from neuroquest.config import REWARD_VALUE, DifficultyParams, GameConfig
from neuroquest.core.entities import Player
from neuroquest.core.maze import Maze
from neuroquest.core.rules import START, apply_damage, new_game, step, time_left_s
from neuroquest.core.state import Action, GameState, Outcome

ROWS = [
    "#####",
    "#...#",
    "#.#.#",
    "#...#",
    "#####",
]
PARAMS = DifficultyParams(time_limit_s=100)


def make_state(rewards=(), exit=(3, 3), player=(1, 1), health=3) -> GameState:
    return GameState(
        maze=Maze.from_strings(ROWS),
        player=Player(pos=player, health=health),
        rewards=frozenset(rewards),
        exit=exit,
        visited=frozenset({player}),
    )


# ---------- new_game ----------


def test_new_game_is_deterministic_and_well_formed() -> None:
    config, params = GameConfig(), DifficultyParams()
    a, b = new_game(config, params, seed=3), new_game(config, params, seed=3)
    assert a == b
    assert a.player == Player(START, config.max_health)
    assert a.maze.is_open(a.exit) and a.exit != START
    distances = a.maze.reachable_from(START)
    assert distances[a.exit] == max(distances.values())  # exit is as far as it gets
    expected_rewards = round(params.reward_density * len(a.maze.open_cells()))
    assert len(a.rewards) == expected_rewards
    assert all(a.maze.is_open(cell) for cell in a.rewards)
    assert START not in a.rewards and a.exit not in a.rewards
    assert a.tick == 0 and a.score == 0 and a.outcome is Outcome.PLAYING


def test_new_game_varies_with_seed() -> None:
    assert new_game(GameConfig(), DifficultyParams(), 1) != new_game(GameConfig(), DifficultyParams(), 2)


# ---------- movement ----------


def test_move_into_open_cell_updates_position_visited_and_events() -> None:
    state = make_state()
    nxt, events = step(state, Action.RIGHT, PARAMS)
    assert nxt.player.pos == (1, 2) and nxt.tick == 1
    assert (1, 2) in nxt.visited
    assert [e.type for e in events] == ["move"]
    assert events[0].detail == {"action": "right", "new_cell": True}
    assert state.player.pos == (1, 1), "the original state must not change"


def test_revisiting_a_cell_is_flagged() -> None:
    state, _ = step(make_state(), Action.RIGHT, PARAMS)
    state, _ = step(state, Action.LEFT, PARAMS)
    _, events = step(state, Action.RIGHT, PARAMS)
    assert events[0].detail["new_cell"] is False


def test_move_into_wall_bumps_and_stays() -> None:
    state, _ = step(make_state(), Action.RIGHT, PARAMS)  # now at (1, 2); (2, 2) is a wall
    nxt, events = step(state, Action.DOWN, PARAMS)
    assert nxt.player.pos == (1, 2)
    assert [e.type for e in events] == ["bump"]


def test_stay_advances_the_clock_only() -> None:
    nxt, events = step(make_state(), Action.STAY, PARAMS)
    assert nxt.tick == 1 and nxt.player.pos == (1, 1) and events == []


# ---------- rewards, exit, timer ----------


def test_reward_is_collected_once() -> None:
    state = make_state(rewards=[(1, 2)])
    nxt, events = step(state, Action.RIGHT, PARAMS)
    assert nxt.score == REWARD_VALUE and nxt.rewards == frozenset()
    assert [e.type for e in events] == ["move", "reward"]
    again, events = step(step(nxt, Action.LEFT, PARAMS)[0], Action.RIGHT, PARAMS)
    assert again.score == REWARD_VALUE and [e.type for e in events] == ["move"]


def test_reaching_the_exit_wins_and_freezes_the_game() -> None:
    state = make_state(exit=(1, 2))
    won, events = step(state, Action.RIGHT, PARAMS)
    assert won.outcome is Outcome.WON
    assert [e.type for e in events] == ["move", "exit"]
    frozen, events = step(won, Action.LEFT, PARAMS)
    assert frozen == won and events == []


def test_time_runs_out_on_the_last_tick() -> None:
    params = DifficultyParams(time_limit_s=1)  # 10 ticks at the default clock
    state = make_state()
    for _ in range(9):
        state, _ = step(state, Action.STAY, params)
    assert state.outcome is Outcome.PLAYING and time_left_s(state, params) == pytest.approx(0.1)
    state, events = step(state, Action.STAY, params)
    assert state.outcome is Outcome.LOST_TIME and state.tick == 10
    assert [e.type for e in events] == ["timeout"] and time_left_s(state, params) == 0


def test_reaching_the_exit_on_the_last_tick_still_wins() -> None:
    params = DifficultyParams(time_limit_s=1)
    state = make_state(exit=(1, 2))
    for _ in range(9):
        state, _ = step(state, Action.STAY, params)
    state, _ = step(state, Action.RIGHT, params)
    assert state.outcome is Outcome.WON


# ---------- health ----------


def test_damage_reduces_health_and_zero_loses() -> None:
    state = make_state(health=2)
    hurt, events = apply_damage(state)
    assert hurt.player.health == 1 and hurt.outcome is Outcome.PLAYING
    assert [e.type for e in events] == ["damage"]
    dead, events = apply_damage(hurt, source="trap")
    assert dead.player.health == 0 and dead.outcome is Outcome.LOST_HEALTH
    assert [e.type for e in events] == ["damage", "defeat"] and events[1].detail["source"] == "trap"
    same, events = apply_damage(dead)
    assert same == dead and events == []


def test_health_cannot_go_negative_and_zero_amount_is_ignored() -> None:
    state = make_state(health=1)
    dead, _ = apply_damage(state, amount=5)
    assert dead.player.health == 0
    assert apply_damage(state, amount=0) == (state, [])
