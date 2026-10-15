"""Phase 10: one test per bug that was found and fixed during development (the development log
entry is named in each docstring), so the fix cannot silently come undone. Bugs that already had
a dedicated test when they were fixed are not repeated here; the test report lists where they live.
"""

from dataclasses import asdict

import pygame

from neuroquest.adapt.controller import Decision
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.rules import new_game
from neuroquest.data.dataset import rows_from_log
from neuroquest.data.features import FEATURE_NAMES
from neuroquest.rl.agents import GreedyCollectorAgent, ProfileAgent
from neuroquest.rl.env import MazeEnv
from neuroquest.rl.qlearning import evaluate, summarize
from neuroquest.sim.harness import run_episode
from neuroquest.sim.profiles import PROFILES
from neuroquest.ui import theme as th
from neuroquest.ui.app import new_fx
from neuroquest.ui.render import PLAYER, cell_rect, draw


def test_text_cache_survives_pygame_quit_and_reinit() -> None:
    """Entry 8b: cached text surfaces raised 'Invalid font' after pygame.quit(); the text helper
    re-rendered, but ``font()`` itself still handed out dead fonts to ``wrap``/``elide`` (found in
    Phase 10, entry 10): the font cache is now validated and rebuilt."""
    from neuroquest.ui.panel import elide, wrap

    pygame.init()
    first = th.text(20, "hello")
    size = th.font(14).size("hello")
    pygame.quit()
    pygame.init()
    try:
        again = th.text(20, "hello")
        assert again.get_size() == first.get_size()
        assert th.font(14).size("hello") == size  # a dead cached font would raise here
        assert wrap(14, "one two three", 400) and elide(14, "one two three", 400) == "one two three"
        th.font(18, bold=True).render("x", True, (255, 255, 255))
    finally:
        pygame.quit()


def test_prefix_ratio_is_row_metadata_not_a_model_feature() -> None:
    """Entry 7: prefix_ratio was a feature but is unobservable live; it moved to row metadata."""
    assert "prefix_ratio" not in FEATURE_NAMES and "elapsed_s" in FEATURE_NAMES
    log = run_episode(PROFILES["cautious"], GameConfig(), DifficultyParams(time_limit_s=10), seed=1)
    rows = rows_from_log(log)
    assert [r["prefix_ratio"] for r in rows] == [0.25, 0.5, 0.75, 1.0]
    assert all(set(FEATURE_NAMES) <= set(r) for r in rows)


def test_decision_summary_uses_ascii_only() -> None:
    """Entry 8b: the real arrow glyph has no glyph in Pygame's default font and rendered as a box."""
    params = asdict(DifficultyParams())
    d = Decision(t=30, rule="R1_ease_heavy_damage", direction="ease", lever="enemy_count", before=params, after=asdict(DifficultyParams(enemy_count=1)), inputs={}, reason="health 1/3")
    assert d.changed and d.summary().isascii() and "2 -> 1" in d.summary()
    hold = Decision(t=30, rule="R0_hold", direction="hold", lever=None, before=params, after=params, inputs={}, reason="nothing to do")
    assert not hold.changed and hold.summary().isascii()


def test_intro_banner_does_not_cover_the_player_on_a_large_surface() -> None:
    """Entry 8b: the intro banner was drawn over the start cell on small surfaces; it is now only
    drawn when the app passes effects, and the drawing is centred away from the start cell."""
    pygame.init()
    try:
        state = new_game(GameConfig(), DifficultyParams(), seed=1)
        surface = pygame.Surface((1024, 768))
        font = pygame.font.Font(None, 24)
        draw(surface, state, DifficultyParams(), font)  # no fx: no banner
        assert surface.get_at(cell_rect(state.maze, surface.get_size(), state.player.pos).center)[:3] == PLAYER
        fx = new_fx()
        fx.update(coins_total=len(state.rewards), seed=1)
        draw(surface, state, DifficultyParams(), font, None, None, fx)  # fx at t=0: banner on
        assert surface.get_at(cell_rect(state.maze, surface.get_size(), state.player.pos).center)[:3] == PLAYER
    finally:
        pygame.quit()


def test_rl_board_keeps_the_hazard_real_and_avoidable() -> None:
    """Entry 9b: on the game's default braiding the hunter blocked the only corridor and every agent
    took exactly one hit; the training board must keep the hazard avoidable (cautious: no hits)
    and real (the hunter-blind collector is hit often)."""
    env = MazeEnv()
    seeds = range(100_000, 100_030)
    cautious = summarize(evaluate(env, ProfileAgent("cautious", env.params, seed=0), seeds))
    collector = summarize(evaluate(env, GreedyCollectorAgent(), seeds))
    assert cautious["mean_hits"] <= 0.05 and cautious["win_rate"] == 1.0
    assert collector["mean_hits"] >= 0.5


def test_default_board_of_the_rl_size_really_is_a_single_corridor() -> None:
    """Entry 9b, the finding itself: with 15 % braiding an 11×9 maze has at most one loop, so the
    hunter's start on the far half lies on the only route to the exit."""
    config = GameConfig(width=11, height=9)
    maze = new_game(config, DifficultyParams(), seed=7).maze
    cells = maze.open_cells()
    edges = sum(len(maze.neighbours(c)) for c in cells) // 2
    assert edges - (len(cells) - 1) <= 1
    braided = new_game(GameConfig(width=11, height=9, loop_fraction=1.0), DifficultyParams(), seed=3).maze
    bcells = braided.open_cells()
    assert sum(len(braided.neighbours(c)) for c in bcells) // 2 - (len(bcells) - 1) >= 3
