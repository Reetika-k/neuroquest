"""Phase 10: edge cases — the smallest boards, empty or over-full parameters, unreachable
enemies, odd window sizes, long texts, and the CLI's input repair."""

from dataclasses import asdict
import json

import numpy as np
import pygame
import pytest

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.entities import Enemy, Player
from neuroquest.core.maze import Maze
from neuroquest.core.rules import new_game, reconcile, step
from neuroquest.core.state import Action, GameState, Outcome
from neuroquest.dashboard.report import build as build_dashboard
from neuroquest.ml.train import _folds
from neuroquest.rl.agents import ShortestPathAgent
from neuroquest.rl.env import ACTIONS, MazeEnv
from neuroquest.ui.app import new_fx
from neuroquest.ui.panel import elide, wrap
from neuroquest.ui.render import PLAYER, cell_rect, draw
import random


def test_smallest_board_is_playable_with_zero_reward_density_and_too_many_enemies() -> None:
    config, params = GameConfig(width=5, height=5), DifficultyParams(enemy_count=50, reward_density=0.0, time_limit_s=10)
    state = new_game(config, params, seed=3)
    cells = set(state.maze.open_cells())
    assert len(cells) == 7 and state.player.pos == (1, 1) and state.exit in cells
    assert len(state.rewards) == 1  # density 0 still leaves one reward: the rules guarantee at least one
    assert 0 < len(state.enemies) < 50
    occupied = [state.player.pos, state.exit, *state.rewards, *(e.pos for e in state.enemies)]
    assert len(occupied) == len(set(occupied))  # nothing shares a cell at the start
    # a shortest-path player with no enemies wins on this board
    quiet = new_game(config, DifficultyParams(enemy_count=0, reward_density=0.0, time_limit_s=10), seed=3)
    agent = ShortestPathAgent()
    while not quiet.outcome.is_over:
        quiet, _ = step(quiet, ACTIONS[agent.act(None, quiet)], params)
    assert quiet.outcome is Outcome.WON and quiet.tick == quiet.maze.reachable_from((1, 1))[quiet.exit]


def test_single_open_cell_board_wins_on_the_first_tick_and_every_move_bumps() -> None:
    maze = Maze.from_strings(["###", "#.#", "###"])
    base = GameState(maze=maze, player=Player((1, 1), 3), rewards=frozenset(), exit=(1, 1), visited=frozenset({(1, 1)}))
    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        state, events = step(base, action, DifficultyParams())
        assert events[0].type == "bump" and state.player.pos == (1, 1)
    won, events = step(base, Action.STAY, DifficultyParams())
    assert won.outcome is Outcome.WON and events[-1].type == "exit"
    assert maze.reachable_from((1, 1)) == {(1, 1): 0} and maze.neighbours((1, 1)) == [] and maze.dead_ends() == []


def test_walled_off_enemy_stays_put_and_never_hurts() -> None:
    maze = Maze.from_strings(["#####", "#.#.#", "#####"])
    state = GameState(maze=maze, player=Player((1, 1), 3), rewards=frozenset(), exit=(1, 3), visited=frozenset({(1, 1)}), enemies=(Enemy(0, (1, 3)),))
    params = DifficultyParams(enemy_move_interval=1, time_limit_s=5)
    for _ in range(20):
        state, events = step(state, Action.RIGHT, params, random.Random(0))
        assert not any(e.type == "damage" for e in events)
    assert state.enemies[0].pos == (1, 3) and state.player.health == 3 and state.player.pos == (1, 1)


def test_reconcile_copes_with_no_free_cells_and_with_zero_enemies() -> None:
    rng = random.Random(0)
    # a board where rewards fill every candidate cell: no room for a hunter at the start or later, and no crash
    full = new_game(GameConfig(width=5, height=5), DifficultyParams(enemy_count=1, reward_density=1.0), seed=2)
    assert full.enemies == () and len(full.rewards) == len(full.maze.open_cells()) - 2
    still_full, events = reconcile(full, DifficultyParams(enemy_count=99, reward_density=1.0), rng, previous=DifficultyParams(enemy_count=1, reward_density=1.0))
    assert still_full.enemies == () and events == []
    # a board with some room: spawning 99 hunters fills the free cells only, each on its own cell
    state = new_game(GameConfig(width=5, height=5), DifficultyParams(enemy_count=1, reward_density=0.4), seed=2)
    assert len(state.enemies) == 1
    crowded, events = reconcile(state, DifficultyParams(enemy_count=99, reward_density=0.4), rng, previous=DifficultyParams(enemy_count=1, reward_density=0.4))
    cells = set(crowded.maze.open_cells())
    positions = [e.pos for e in crowded.enemies]
    assert 1 < len(positions) <= len(cells) and len(set(positions)) == len(positions) and set(positions) <= cells
    assert not (set(positions) & ({crowded.player.pos, crowded.exit} | crowded.rewards))
    assert all(e.type == "enemy_spawn" for e in events) and len(events) == len(positions) - 1
    empty, events = reconcile(crowded, DifficultyParams(enemy_count=0, reward_density=0.4), rng, previous=DifficultyParams(enemy_count=99, reward_density=0.4))
    assert empty.enemies == () and all(e.type == "enemy_despawn" for e in events) and len(events) == len(positions)
    same, events = reconcile(empty, DifficultyParams(enemy_count=0, reward_density=0.2), rng, previous=DifficultyParams(enemy_count=0, reward_density=0.4))
    assert same.rewards == empty.rewards and events == []  # lowering the density changes nothing


def test_one_health_and_one_second_limits_end_the_game_correctly() -> None:
    config = GameConfig(max_health=1, enemy_kind="random")
    state = new_game(config, DifficultyParams(enemy_count=0, time_limit_s=1), seed=1)
    for _ in range(10):
        state, _ = step(state, Action.STAY, DifficultyParams(enemy_count=0, time_limit_s=1))
    assert state.outcome is Outcome.LOST_TIME and state.tick == 10
    assert step(state, Action.RIGHT, DifficultyParams(time_limit_s=1)) == (state, [])  # a finished game is inert


def test_draw_survives_huge_and_tiny_windows_with_panel_and_paths() -> None:
    pygame.init()
    try:
        state = new_game(GameConfig(), DifficultyParams(), seed=4)
        font = pygame.font.Font(None, 24)
        fx = new_fx()
        fx.update(intro=False, show_paths=True, panel_open=True, adaptive=True, coins_total=len(state.rewards), seed=4)
        for size in ((1920, 1080), (2560, 1440), (400, 300), (200, 150)):
            surface = pygame.Surface(size)
            draw(surface, state, DifficultyParams(), font, None, "notice " * 40, fx)
            rect = cell_rect(state.maze, size, state.player.pos)
            assert 0 <= rect.left and rect.right <= size[0] and 0 <= rect.top and rect.bottom <= size[1]
    finally:
        pygame.quit()


def test_hud_in_the_last_ten_seconds_and_with_a_fresh_difficulty_change() -> None:
    """The time bar glows when ≤ 10 s remain and the footer highlights a just-made decision; neither
    had been rendered by a test before Phase 10 (found by coverage)."""
    from dataclasses import replace

    from neuroquest.adapt.controller import Decision

    pygame.init()
    try:
        params = DifficultyParams(time_limit_s=12)
        state = replace(new_game(GameConfig(), params, seed=1), tick=30)  # 3 s played, 9 s left
        surface = pygame.Surface((1024, 768))
        font = pygame.font.Font(None, 24)
        draw(surface, state, params, font, None, "Difficulty changed: enemy count 2 -> 1  (R1_ease_heavy_damage: health 1/3)")
        base = asdict(params)
        fx = new_fx()
        fx.update(intro=False, adaptive=True, coins_total=len(state.rewards), seed=1, last_decision=Decision(t=30, rule="R1_ease_heavy_damage", direction="ease", lever="enemy_count", before=base, after=dict(base, enemy_count=1), inputs={}, reason="health 1/3"), last_decision_until=90)
        draw(surface, state, params, font, None, "Difficulty changed: enemy count 2 -> 1", fx)
        for size in ((320, 240), (640, 480)):
            draw(pygame.Surface(size), state, params, font, None, "Difficulty changed: enemy count 2 -> 1", fx)
    finally:
        pygame.quit()


def test_wrap_and_elide_keep_text_inside_the_width() -> None:
    pygame.init()
    try:
        long = "R1_ease_heavy_damage: health 1/3, 2 hits in the last 15 s (style aggressive 99%) and a very long tail of words " * 3
        for width in (80, 160, 320):
            lines = wrap(14, long, width)
            assert lines and all(s.get_width() <= width for s in lines)
            short = elide(14, long, width)
            assert short.endswith("…") and len(short) < len(long)
            assert elide(14, "ok", width) == "ok"
        assert wrap(14, "", 100) == [] or all(s.get_width() <= 100 for s in wrap(14, "", 100))
    finally:
        pygame.quit()


def test_dashboard_shows_the_session_with_the_most_changes(tmp_path) -> None:
    base = asdict(DifficultyParams())
    quiet = [{"episode_id": "human-quiet", "t": t, "rule": "R0_hold", "direction": "hold", "lever": None, "before": base, "after": base, "inputs": {}, "reason": "nothing to do"} for t in (30, 60, 90)]
    eased = dict(base, enemy_count=1)
    busy = [{"episode_id": "human-busy", "t": 30, "rule": "R1_ease_heavy_damage", "direction": "ease", "lever": "enemy_count", "before": base, "after": eased, "inputs": {}, "reason": "health 1/3"}]
    path = tmp_path / "decisions.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in busy + quiet))  # the busy session is older, the quiet one latest
    summary = build_dashboard(tmp_path / "x.csv", tmp_path / "x.json", path, tmp_path / "x.csv", tmp_path / "noraw", tmp_path / "nomodels", tmp_path / "d.html", tmp_path / "figs", tmp_path / "x.json", tmp_path / "x.csv")
    html = (tmp_path / "d.html").read_text()
    assert "decisions" in summary["sections"] and "Shown: <code>human-busy</code>" in html and "the session with the most changes (1)" in html


def test_cv_folds_never_exceed_the_number_of_groups() -> None:
    assert _folds(np.array([1, 1, 2, 2, 3, 3])) == 3
    assert _folds(np.array([1, 2, 3, 4, 5, 6, 7])) == 5
    assert _folds(np.array([1, 1, 1])) == 2  # a single group still asks for the minimum of two folds


def test_cli_repairs_even_dimensions_and_passes_the_rest_through(monkeypatch) -> None:
    import neuroquest.ui.app as app_module
    from neuroquest.cli import main

    captured = {}
    monkeypatch.setattr(app_module, "run", lambda config, params, seed, **kw: captured.update(config=config, params=params, seed=seed, **kw) or Outcome.PLAYING)
    assert main(["play", "--width", "20", "--height", "14", "--seed", "9", "--enemies", "4", "--fixed", "--no-log"]) == 0
    assert (captured["config"].width, captured["config"].height, captured["seed"]) == (21, 15, 9)
    assert captured["params"].enemy_count == 4 and captured["adaptive"] is False and captured["log_dir"] is None and captured["agent"] is None


def test_bfs_nearest_edge_cases_and_stats() -> None:
    from neuroquest.ai.bfs import bfs, bfs_nearest

    maze = Maze.from_strings(["#####", "#...#", "#.#.#", "#####"])
    assert bfs_nearest(maze, (1, 1), set()) is None and bfs_nearest(maze, (0, 0), {(1, 1)}) is None
    assert bfs_nearest(maze, (1, 1), {(1, 1), (1, 3)}) == [(1, 1)]
    stats: dict = {}
    assert bfs_nearest(maze, (1, 1), {(2, 3)}, stats) == [(1, 1), (1, 2), (1, 3), (2, 3)] and stats["expanded"] >= 1
    unreachable = Maze.from_strings(["#####", "#.#.#", "#####"])
    stats = {}
    assert bfs_nearest(unreachable, (1, 1), {(1, 3)}, stats) is None and stats["expanded"] == 1
    assert bfs(unreachable, (1, 1), (1, 3)) is None and bfs(maze, (1, 1), (1, 1)) == [(1, 1)]


def test_short_decision_and_panel_with_changed_and_recent_decisions() -> None:
    from neuroquest.adapt.controller import Decision
    from neuroquest.ui.panel import rule_name, short_decision

    base = asdict(DifficultyParams())
    eased = dict(base, enemy_count=1)
    hold = Decision(t=30, rule="R0_hold_warmup", direction="hold", lever=None, before=base, after=base, inputs={}, reason="too early to judge")
    ease = Decision(t=60, rule="R1_ease_heavy_damage", direction="ease", lever="enemy_count", before=base, after=eased, inputs={}, reason="health 1/3, 2 hits in the last 15 s")
    tighten = Decision(t=120, rule="R3_tighten_cruising_at_bound", direction="tighten", lever="enemy_move_interval", before=eased, after=eased, inputs={}, reason="at the bound")
    assert short_decision(None) == "" and short_decision(hold) == rule_name("R0_hold_warmup") != "R0_hold_warmup"
    assert "hunters 2 -> 1" in short_decision(ease) and "→" not in short_decision(ease)  # ASCII arrow; the middle dot renders fine
    assert "(at bound)" in short_decision(tighten)
    pygame.init()
    try:
        state = new_game(GameConfig(), DifficultyParams(), seed=1)
        fx = new_fx()
        fx.update(intro=False, adaptive=True, panel_open=True, coins_total=len(state.rewards), seed=1, decisions=[hold, ease, tighten, ease, hold, ease])
        surface = pygame.Surface((1024, 768))
        draw(surface, state, DifficultyParams(enemy_count=1), pygame.font.Font(None, 24), None, None, fx)
        assert surface.get_at((1024 - 60, 384))[:3] != (0, 0, 0)
    finally:
        pygame.quit()


def test_rl_env_on_the_smallest_board() -> None:
    env = MazeEnv(GameConfig(width=5, height=5), DifficultyParams(enemy_count=0, reward_density=0.0, time_limit_s=5), layout_seed=1)
    obs, info = env.reset(seed=0)
    assert len(env.crystals) == 1 and obs[2] == 1
    agent = ShortestPathAgent()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs, info["state"]))
        done = terminated or truncated
    assert info["won"] and env.state_space_bound() == 7 * 2
