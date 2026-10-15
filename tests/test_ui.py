"""Phase 4: input mapping, headless rendering and the game loop (dummy video driver via conftest)."""

from collections import defaultdict

import pygame

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.entities import Player
from neuroquest.core.maze import Maze
from neuroquest.core.rules import apply_damage, new_game, step
from neuroquest.core.state import Action, GameState, Outcome
from neuroquest.ui.app import run
from neuroquest.ui.input import action_from_pressed
from neuroquest.ml.predict import Prediction
from neuroquest.ui.render import BG, ENEMY, EXIT, FOOTER_BG, PLAYER, REWARD, cell_rect, draw


def pressed(*keys):
    table = defaultdict(bool)
    for key in keys:
        table[key] = True
    return table


def test_no_keys_means_stay() -> None:
    assert action_from_pressed(pressed()) is Action.STAY


def test_arrows_and_wasd_map_to_actions() -> None:
    assert action_from_pressed(pressed(pygame.K_RIGHT)) is Action.RIGHT
    assert action_from_pressed(pressed(pygame.K_a)) is Action.LEFT
    assert action_from_pressed(pressed(pygame.K_s)) is Action.DOWN
    assert action_from_pressed(pressed(pygame.K_UP)) is Action.UP


def test_up_wins_when_several_keys_are_held() -> None:
    assert action_from_pressed(pressed(pygame.K_RIGHT, pygame.K_UP)) is Action.UP


def test_draw_paints_player_exit_rewards_and_background() -> None:
    pygame.init()
    try:
        state = new_game(GameConfig(), DifficultyParams(), seed=1)
        surface = pygame.Surface((640, 480))
        draw(surface, state, DifficultyParams(), pygame.font.Font(None, 24))
        size = surface.get_size()
        assert surface.get_at(cell_rect(state.maze, size, state.player.pos).center)[:3] == PLAYER
        assert surface.get_at(cell_rect(state.maze, size, state.exit).center)[:3] == EXIT
        reward = next(iter(state.rewards))
        assert surface.get_at(cell_rect(state.maze, size, reward).center)[:3] == REWARD
        enemy = state.enemies[0]
        assert surface.get_at(cell_rect(state.maze, size, enemy.pos).center)[:3] == ENEMY
        assert surface.get_at((0, 479))[:3] == FOOTER_BG
        assert surface.get_at((0, 100))[:3] == BG  # the frame around the board
    finally:
        pygame.quit()


def test_draw_handles_every_outcome() -> None:
    pygame.init()
    try:
        maze = Maze.from_strings(["#####", "#...#", "#####"])
        base = GameState(maze=maze, player=Player((1, 1), 3), rewards=frozenset(), exit=(1, 3), visited=frozenset({(1, 1)}))
        params = DifficultyParams(time_limit_s=1)
        won, _ = step(step(base, Action.RIGHT, params)[0], Action.RIGHT, params)
        timed = base
        for _ in range(10):
            timed, _ = step(timed, Action.STAY, params)
        dead, _ = apply_damage(base, amount=3)
        assert (won.outcome, timed.outcome, dead.outcome) == (Outcome.WON, Outcome.LOST_TIME, Outcome.LOST_HEALTH)
        surface = pygame.Surface((320, 240))
        for state in (base, won, timed, dead):
            draw(surface, state, params, pygame.font.Font(None, 20))
    finally:
        pygame.quit()


def test_game_loop_runs_headless_and_stops_at_max_ticks() -> None:
    outcome = run(GameConfig(ticks_per_second=200), DifficultyParams(time_limit_s=5, enemy_count=0), seed=1, max_ticks=40, window=(320, 240), adaptive=False)
    assert outcome is Outcome.PLAYING


def test_game_loop_ends_when_time_runs_out() -> None:
    params = DifficultyParams(time_limit_s=1, enemy_count=0)  # no enemies: a motionless player must lose on time, not health
    # adaptive=False: otherwise rule R2 would extend the clock for a player who is behind on time
    outcome = run(GameConfig(ticks_per_second=200), params, seed=1, max_ticks=10_000, window=(320, 240), adaptive=False)
    assert outcome is Outcome.LOST_TIME


def test_footer_renders_prediction_status_and_nothing() -> None:
    pygame.init()
    try:
        state = new_game(GameConfig(), DifficultyParams(), seed=1)
        surface = pygame.Surface((640, 480))
        font = pygame.font.Font(None, 24)
        draw(surface, state, DifficultyParams(), font, None)
        draw(surface, state, DifficultyParams(), font, "AI style: watching…")
        draw(surface, state, DifficultyParams(), font, Prediction("cautious", {"aggressive": 0.1, "cautious": 0.7, "explorer": 0.2}, 0.7))
        assert surface.get_at((0, 479))[:3] == FOOTER_BG  # footer background
    finally:
        pygame.quit()


def test_game_loop_logs_the_human_episode_and_predicts(tmp_path, tiny_model_dir) -> None:
    seen = []
    outcome = run(
        GameConfig(ticks_per_second=200), DifficultyParams(time_limit_s=5, enemy_count=0), seed=1, max_ticks=80, window=(320, 240),
        log_dir=tmp_path / "raw", model_path=tiny_model_dir / "behaviour_clf.joblib", meta_path=tiny_model_dir / "behaviour_clf.meta.json", on_prediction=seen.append,
    )
    assert outcome is Outcome.PLAYING
    logs = list((tmp_path / "raw").glob("human-*.jsonl"))
    assert len(logs) == 1 and '"profile": "human"' in logs[0].read_text().splitlines()[0]
    assert len(seen) == 3 and all(p.label in {"aggressive", "cautious", "explorer"} for p in seen)  # ticks 40, 60, 80


def test_game_loop_without_a_model_shows_a_status_instead_of_crashing(tmp_path) -> None:
    outcome = run(GameConfig(ticks_per_second=200), DifficultyParams(time_limit_s=5, enemy_count=0), seed=1, max_ticks=40, window=(320, 240), model_path=tmp_path / "missing.joblib", meta_path=tmp_path / "missing.json")
    assert outcome is Outcome.PLAYING


def test_out_of_date_model_gives_a_status_not_a_crash(tmp_path, tiny_model_dir) -> None:
    """Phase 10 edge case: a saved model whose feature list no longer matches the code."""
    import json
    import shutil

    from neuroquest.ui.app import load_predictor

    shutil.copy(tiny_model_dir / "behaviour_clf.joblib", tmp_path / "old.joblib")
    meta = json.loads((tiny_model_dir / "behaviour_clf.meta.json").read_text())
    meta["features"] = meta["features"][:-1] + ["a_feature_that_no_longer_exists"]
    (tmp_path / "old.meta.json").write_text(json.dumps(meta))
    predictor, status = load_predictor(tmp_path / "old.joblib", tmp_path / "old.meta.json")
    assert predictor is None and "out of date" in status
    predictor, status = load_predictor(tmp_path / "missing.joblib", tmp_path / "missing.json")
    assert predictor is None and "no model yet" in status
    outcome = run(GameConfig(ticks_per_second=200), DifficultyParams(time_limit_s=5, enemy_count=0), seed=1, max_ticks=40, window=(320, 240), model_path=tmp_path / "old.joblib", meta_path=tmp_path / "old.meta.json")
    assert outcome is Outcome.PLAYING


def test_panel_menu_and_end_screens_render_from_real_data(tiny_model_dir) -> None:
    """The intelligence panel, the start menu and the end screens draw without error on a wide surface,
    and the panel reserves width (the board moves left) only when open."""
    import json

    from neuroquest.adapt.controller import AdaptiveController
    from neuroquest.core.rules import apply_damage, step
    from neuroquest.ui.app import new_fx
    from neuroquest.ui.render import PANEL_WIDTH, layout, panel_width

    pygame.init()
    try:
        surface = pygame.Surface((1024, 768))
        font = pygame.font.Font(None, 24)
        state = new_game(GameConfig(), DifficultyParams(), seed=1)
        fx = new_fx()
        fx.update(adaptive=True, coins_total=len(state.rewards), panel_open=True, seed=1, model_meta=json.loads((tiny_model_dir / "behaviour_clf.meta.json").read_text()))
        assert panel_width(surface.get_size(), fx) == PANEL_WIDTH and panel_width((640, 480), fx) == 0
        assert layout(state.maze, surface.get_size(), PANEL_WIDTH)[1] < layout(state.maze, surface.get_size())[1]
        draw(surface, state, DifficultyParams(), font, "AI style: watching…", None, fx, menu=True)
        pred = Prediction("cautious", {"aggressive": 0.1, "cautious": 0.7, "explorer": 0.2}, 0.7)
        draw(surface, state, DifficultyParams(), font, pred, None, fx)
        assert surface.get_at((0, 767))[:3] == FOOTER_BG  # footer visible while playing (overlays veil it later)
        ctrl = AdaptiveController()
        events = []
        for _ in range(ctrl.every_ticks):
            state, ev = step(state, Action.STAY, DifficultyParams())
            events += ev
        _, decision = ctrl.step(state, DifficultyParams(), events, 3, pred)
        fx["decisions"] = [decision]
        draw(surface, state, DifficultyParams(), font, pred, None, fx, paused=True)
        dead, _ = apply_damage(state, amount=3)
        fx["hits"] = 3
        draw(surface, dead, DifficultyParams(), font, pred, None, fx)
    finally:
        pygame.quit()
