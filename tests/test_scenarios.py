"""Phase 10: gameplay scenarios driven through the real window loop (headless).

The ``agent`` hook of ``ui.app.run`` chooses the action each tick, so a scripted player can
walk the real loop; key presses are injected as Pygame events from inside that hook (or from a
Pygame timer while the game is paused and the hook is not called). Every scenario asserts on the
loop's outcome and on the files it writes, not on internal variables.
"""

import time
from pathlib import Path

import pygame
import pytest

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.state import Action, Outcome
from neuroquest.data.events import EpisodeLog
from neuroquest.rl.agents import ShortestPathAgent
from neuroquest.rl.env import ACTIONS
from neuroquest.ui import app
from neuroquest.ui.app import run

FAST = GameConfig(ticks_per_second=200)  # 20× real time: a 30 s game-clock limit is 1.5 s of wall time


def key(k: int) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k))


def shortest_path_agent():
    inner = ShortestPathAgent()
    return lambda state: ACTIONS[inner.act(None, state)]


def test_scenario_win_by_reaching_the_exit(tmp_path) -> None:
    outcome = run(FAST, DifficultyParams(enemy_count=0, time_limit_s=30), seed=1, max_ticks=2000, window=(320, 240), adaptive=False, log_dir=tmp_path, agent=shortest_path_agent())
    assert outcome is Outcome.WON
    log = EpisodeLog.load(next(tmp_path.glob("human-*.jsonl")))
    assert log.meta["outcome"] == "won" and log.events[-1].type == "exit" and log.meta["ticks"] == log.events[-1].t
    assert sum(e.type == "bump" for e in log.events) == 0  # the shortest path never walks into a wall


def test_scenario_loss_by_health(tmp_path) -> None:
    params = DifficultyParams(enemy_count=3, enemy_move_interval=1, time_limit_s=60)
    outcome = run(FAST, params, seed=1, max_ticks=5000, window=(320, 240), adaptive=False, log_dir=tmp_path, agent=lambda s: Action.STAY)
    assert outcome is Outcome.LOST_HEALTH
    log = EpisodeLog.load(next(tmp_path.glob("human-*.jsonl")))
    damage = [e for e in log.events if e.type == "damage"]
    last_tick = [e.type for e in log.events if e.t == log.meta["ticks"]]
    assert len(damage) == 3 and damage[-1].detail["health"] == 0 and "defeat" in last_tick  # near_enemy follows defeat within the tick
    assert all(b.t - a.t >= FAST.invulnerability_ticks for a, b in zip(damage, damage[1:]))  # one hit per invulnerability window


def test_scenario_loss_by_time(tmp_path) -> None:
    outcome = run(FAST, DifficultyParams(enemy_count=0, time_limit_s=1), seed=1, max_ticks=5000, window=(320, 240), adaptive=False, log_dir=tmp_path, agent=lambda s: Action.STAY)
    assert outcome is Outcome.LOST_TIME
    log = EpisodeLog.load(next(tmp_path.glob("human-*.jsonl")))
    assert log.meta["ticks"] == FAST.ticks_per_second * 1 and log.events[-1].type == "timeout"


def test_scenario_restart_with_r_starts_a_new_seeded_game_and_saves_the_old_log(tmp_path) -> None:
    seen: list[int] = []
    game = [1]

    def agent(state) -> Action:
        if seen and state.tick < seen[-1]:
            game[0] += 1
        seen.append(state.tick)
        if game[0] == 1 and state.tick == 5:
            key(pygame.K_r)
        if game[0] == 2 and state.tick == 20:
            key(pygame.K_ESCAPE)
        return Action.STAY

    outcome = run(FAST, DifficultyParams(enemy_count=0, time_limit_s=30), seed=7, max_ticks=10_000, window=(320, 240), adaptive=False, log_dir=tmp_path, agent=agent)
    assert outcome is Outcome.PLAYING and game[0] == 2
    logs = sorted(tmp_path.glob("human-*.jsonl"), key=lambda p: EpisodeLog.load(p).meta["seed"])
    assert [EpisodeLog.load(p).meta["seed"] for p in logs] == [7, 8]
    first, second = (EpisodeLog.load(p) for p in logs)
    # a key posted inside the tick loop takes effect on the next frame, which may hold a few more ticks
    assert 5 <= first.meta["ticks"] <= 60 and second.meta["ticks"] >= 20  # the first log was saved on R, the second on Esc


def test_scenario_pause_stops_the_clock_and_resumes(tmp_path) -> None:
    stamps: list[tuple[int, float]] = []

    def agent(state) -> Action:
        stamps.append((state.tick, time.perf_counter()))
        if state.tick == 4:
            key(pygame.K_p)  # pause now…
            # …resume after 250 ms. (Pygame keeps one timer per event *type*, so a second KEYDOWN
            # timer would replace this one; Esc is therefore posted by the agent once ticks resume.)
            pygame.time.set_timer(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p), 250, loops=1)
        if state.tick == 40:
            key(pygame.K_ESCAPE)
        return Action.STAY

    outcome = run(FAST, DifficultyParams(enemy_count=0, time_limit_s=30), seed=1, max_ticks=10_000, window=(320, 240), adaptive=False, agent=agent)
    assert outcome is Outcome.PLAYING
    ticks = [t for t, _ in stamps]
    assert ticks == list(range(len(ticks)))  # ticks are consecutive: nothing was skipped or repeated across the pause
    gaps = sorted(b - a for (_, a), (_, b) in zip(stamps, stamps[1:]))
    assert gaps[-1] >= 0.2 and gaps[len(gaps) // 2] < 0.05  # one gap is the pause; otherwise ticks come at the usual rate
    assert len(ticks) >= 30  # ticks resumed after the pause and ran until Esc


def test_scenario_keys_toggle_panel_and_paths_and_f12_saves_a_screenshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app, "SCREENSHOT_DIR", tmp_path / "shots")
    samples: dict[int, tuple] = {}

    def agent(state) -> Action:
        screen = pygame.display.get_surface()
        if state.tick in (3, 40):
            samples[state.tick] = screen.get_at((1024 - 60, 384))[:3]
        if state.tick == 5:
            key(pygame.K_i)
            key(pygame.K_t)
        if state.tick == 20:
            key(pygame.K_F12)
        return Action.RIGHT if state.tick % 2 else Action.STAY

    outcome = run(FAST, DifficultyParams(enemy_count=1, time_limit_s=30), seed=1, max_ticks=60, window=(1024, 768), adaptive=False, panel=False, agent=agent)
    assert outcome is Outcome.PLAYING
    shots = list((tmp_path / "shots").glob("screenshot_t*.png"))
    assert len(shots) == 1 and shots[0].stat().st_size > 1000
    assert samples[3] != samples[40]  # the right-hand strip changed: the panel opened on I


def test_scenario_escape_quits_and_saves_the_log(tmp_path) -> None:
    def agent(state) -> Action:
        if state.tick == 8:
            key(pygame.K_ESCAPE)
        return Action.RIGHT

    outcome = run(FAST, DifficultyParams(enemy_count=0, time_limit_s=30), seed=1, max_ticks=10_000, window=(320, 240), adaptive=False, log_dir=tmp_path, agent=agent)
    assert outcome is Outcome.PLAYING
    log = EpisodeLog.load(next(tmp_path.glob("human-*.jsonl")))
    assert 8 <= log.meta["ticks"] <= 60 and log.meta["outcome"] == "playing"  # Esc acts on the next frame


def test_scenario_adaptive_controller_eases_a_player_who_keeps_getting_hit(tmp_path, tiny_model_dir) -> None:
    import json

    decisions_path = tmp_path / "decisions.jsonl"
    config = GameConfig(ticks_per_second=200, max_health=5)
    params = DifficultyParams(enemy_count=3, enemy_move_interval=1, time_limit_s=60)
    outcome = run(config, params, seed=1, max_ticks=5000, window=(640, 480), adaptive=True, model_path=tiny_model_dir / "behaviour_clf.joblib", meta_path=tiny_model_dir / "behaviour_clf.meta.json", decisions_path=decisions_path, log_dir=tmp_path, agent=lambda s: Action.STAY)
    assert outcome is Outcome.LOST_HEALTH
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    changed = [d for d in decisions if d["before"] != d["after"]]
    assert changed and changed[0]["direction"] == "ease" and changed[0]["rule"].startswith("R1_ease")
    assert all(d["t"] % 30 == 0 for d in decisions)
    log = EpisodeLog.load(next(tmp_path.glob("human-*.jsonl")))
    assert sum(e.type == "damage" for e in log.events) == 5


def test_scenario_window_close_event_ends_the_loop_and_saves(tmp_path) -> None:
    def agent(state) -> Action:
        if state.tick == 6:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return Action.RIGHT

    outcome = run(FAST, DifficultyParams(enemy_count=0, time_limit_s=30), seed=1, max_ticks=10_000, window=(320, 240), adaptive=False, log_dir=tmp_path, agent=agent)
    assert outcome is Outcome.PLAYING and len(list(tmp_path.glob("human-*.jsonl"))) == 1


def test_scenario_menu_waits_for_enter(tmp_path) -> None:
    calls: list[int] = []

    def agent(state) -> Action:
        calls.append(state.tick)
        if state.tick == 3:
            key(pygame.K_ESCAPE)
        return Action.STAY

    pygame.time.set_timer(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN), 200, loops=1)
    t0 = time.perf_counter()
    outcome = run(FAST, DifficultyParams(enemy_count=0, time_limit_s=30), seed=1, max_ticks=10_000, window=(320, 240), adaptive=False, menu=True, agent=agent)
    assert outcome is Outcome.PLAYING and calls and calls[0] == 0
    assert time.perf_counter() - t0 >= 0.19  # nothing ticked until Enter arrived
