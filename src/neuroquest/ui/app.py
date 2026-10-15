"""The game loop: a fixed logic tick, drawing at up to 60 FPS, keyboard, restart, pause, screenshots.

The loop accumulates real time and advances the game one tick per ``1 / ticks_per_second``
seconds, however fast or slow the drawing is. That keeps a human game identical in rules and
timing to a headless simulated one.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pygame

from neuroquest.adapt.controller import AdaptiveController
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.rules import new_game, reconcile, step
from neuroquest.core.state import Action, GameState, Outcome
from neuroquest.data.events import EventLogger
from neuroquest.data.features import live_features
from neuroquest.ml.predict import PREDICT_AFTER_TICKS, PREDICT_EVERY_TICKS, BehaviourPredictor, ModelMismatch, Prediction
from neuroquest.ml.train import META_PATH, MODEL_PATH
from neuroquest.ui.input import action_from_pressed
from neuroquest.ui.render import ENEMY, EXIT, REWARD, cell_rect, draw, panel_width

SCREENSHOT_DIR = Path("reports/figures")
NOTICE_TICKS = 60  # a fresh decision stays highlighted for 6 s
FACING = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def new_fx() -> dict:
    """UI-only effect state: never part of the game state, never logged."""
    return {"t": 0.0, "particles": [], "shake": 0.0, "flash": 0.0, "heart_shake": 0.0, "trail": {}, "score_pop": 0.0, "facing": (0, 1), "show_paths": False, "panel_open": False, "intro": True, "adaptive": False, "coins_total": 0, "decisions": [], "hits": 0, "changes": 0, "seed": 0}


def burst(fx: dict, centre: tuple[int, int], colour, n: int, speed: float = 160.0, size: float = 3.0) -> None:
    for _ in range(n):
        ang = random.uniform(0, 6.283)
        v = random.uniform(0.3, 1.0) * speed
        fx["particles"].append({"x": centre[0], "y": centre[1], "vx": math.cos(ang) * v, "vy": math.sin(ang) * v - 40, "life": 1.0, "colour": colour, "size": random.uniform(1.5, size)})


def apply_events(fx: dict, events, state: GameState, surface_size: tuple[int, int]) -> None:
    """Turn this tick's game events into particles, shake, flash, trail and facing."""
    panel = panel_width(surface_size, fx)
    for e in events:
        if e.type == "move":
            fx["trail"][e.pos] = e.t
            fx["facing"] = FACING.get(e.detail.get("action", "right"), fx["facing"])
        elif e.type == "reward":
            burst(fx, cell_rect(state.maze, surface_size, e.pos, panel).center, REWARD, 8, 120.0, 2.5)
            fx["score_pop"] = 1.0
        elif e.type == "damage":
            burst(fx, cell_rect(state.maze, surface_size, e.pos, panel).center, ENEMY, 10, 160.0, 2.5)
            fx["shake"], fx["flash"], fx["heart_shake"] = 1.0, 0.6, 1.0
            fx["hits"] += 1
        elif e.type == "exit":
            c = cell_rect(state.maze, surface_size, e.pos, panel).center
            burst(fx, c, EXIT, 40, 260.0, 4.0)
            burst(fx, c, REWARD, 20, 220.0, 3.0)
        elif e.type == "defeat":
            fx["shake"] = 1.6


def advance_fx(fx: dict, dt: float) -> None:
    fx["t"] += dt
    fx["shake"] = max(0.0, fx["shake"] - dt * 3.5)
    fx["flash"] = max(0.0, fx["flash"] - dt * 3.0)
    fx["heart_shake"] = max(0.0, fx["heart_shake"] - dt * 2.5)
    fx["score_pop"] = max(0.0, fx["score_pop"] - dt * 3.0)
    alive = []
    for p in fx["particles"]:
        p["x"] += p["vx"] * dt
        p["y"] += p["vy"] * dt
        p["vy"] += 320 * dt
        p["life"] -= dt * 1.4
        if p["life"] > 0:
            alive.append(p)
    fx["particles"] = alive


def difficulty_line(params: DifficultyParams, adaptive: bool) -> str:
    mode = "adaptive" if adaptive else "fixed"
    return f"Difficulty ({mode}): enemies {params.enemy_count}  ·  enemy move every {params.enemy_move_interval} ticks  ·  time limit {params.time_limit_s}s"


def load_predictor(model_path: Path, meta_path: Path) -> tuple[BehaviourPredictor | None, str]:
    """Load the saved classifier; on failure return a status string for the footer instead of crashing."""
    try:
        return BehaviourPredictor.load(model_path, meta_path), ""
    except FileNotFoundError:
        return None, "AI style: no model yet (run: neuroquest train)"
    except ModelMismatch:
        return None, "AI style: model out of date (run: neuroquest train)"


def _new_logger(state: GameState, config: GameConfig, params: DifficultyParams, seed: int) -> EventLogger:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return EventLogger(state, config, params, episode_id=f"human-{stamp}-{seed}", profile="human", seed=seed)


def run(
    config: GameConfig,
    params: DifficultyParams,
    seed: int,
    *,
    max_ticks: int | None = None,
    window: tuple[int, int] = (1024, 768),
    log_dir: Path | None = None,
    model_path: Path = MODEL_PATH,
    meta_path: Path = META_PATH,
    on_prediction: Callable[[Prediction], None] | None = None,
    adaptive: bool = True,
    decisions_path: Path | None = None,
    panel: bool = True,
    menu: bool = False,
    agent: Callable[[GameState], Action] | None = None,
    layout_seed: int | None = None,
) -> Outcome:
    """Open the window and play until Esc, the window closes, or (in tests) ``max_ticks``/game over.

    Every episode is logged like a simulated one (``profile = "human"``) and saved to ``log_dir``
    when it ends, so real play can later replace the synthetic data. While playing, the saved
    classifier predicts the player's style from the live features and the footer shows it. With
    ``adaptive`` on (and a model available) the controller may change the difficulty every 5 s;
    each decision is appended to ``decisions_path`` as JSON Lines and shown in the footer.

    ``agent``, if given, chooses the action each tick instead of the keyboard (Phase 9b: watch
    the trained policy play); ``layout_seed`` fixes the maze, exit and rewards across restarts.
    """
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    pygame.init()
    screen = pygame.display.set_mode(window)
    pygame.display.set_caption("NeuroQuest — The Game That Learns You")
    font = pygame.font.Font(None, 26)
    fx = new_fx()
    clock = pygame.time.Clock()

    predictor, status = load_predictor(model_path, meta_path)
    base_params = params
    adaptive = adaptive and predictor is not None
    current_seed = seed
    state = new_game(config, params, current_seed, layout_seed=layout_seed)
    logger = _new_logger(state, config, params, current_seed)
    watching = "AI agent: the trained Q-learning policy is playing" if agent is not None else "AI style: watching your first moves…"
    prediction: Prediction | str | None = status or watching
    controller = AdaptiveController() if adaptive else None
    game_rng = random.Random(current_seed)

    def base_notice() -> str:
        return ("Q-LEARNING AGENT PLAYING  ·  " if agent is not None else "") + difficulty_line(params, adaptive)

    notice, notice_until = base_notice(), 0
    saved = False
    fx["adaptive"], fx["coins_total"], fx["panel_open"], fx["seed"] = adaptive, len(state.rewards), panel, current_seed
    fx["model_meta"] = predictor.meta if predictor is not None else None
    in_menu = menu

    def record(decision) -> None:
        if decisions_path is not None:
            decisions_path.parent.mkdir(parents=True, exist_ok=True)
            with decisions_path.open("a") as fh:
                fh.write(json.dumps({"episode_id": logger.log.meta["episode_id"], **decision.to_dict()}) + "\n")

    def save_log() -> None:
        nonlocal saved
        if log_dir is not None and state.tick > 0 and not saved:
            logger.finish(state).save(log_dir / f"{logger.log.meta['episode_id']}.jsonl")
            saved = True

    tick_len = 1.0 / config.ticks_per_second
    accumulator = 0.0
    paused = False
    running = True
    try:
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.25)  # cap: a stalled window must not fire 100 ticks at once
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if in_menu and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        in_menu, accumulator = False, 0.0
                        continue
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        save_log()
                        current_seed += 1
                        params = base_params
                        state = new_game(config, params, current_seed, layout_seed=layout_seed)
                        logger = _new_logger(state, config, params, current_seed)
                        prediction = status or watching
                        controller = AdaptiveController() if adaptive else None
                        game_rng = random.Random(current_seed)
                        notice, notice_until = base_notice(), 0
                        saved = False
                        accumulator, paused = 0.0, False
                        panel_open = fx["panel_open"]
                        fx = new_fx()
                        fx["adaptive"], fx["coins_total"], fx["panel_open"], fx["seed"] = adaptive, len(state.rewards), panel_open, current_seed
                        fx["model_meta"] = predictor.meta if predictor is not None else None
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_t:
                        fx["show_paths"] = not fx["show_paths"]
                    elif event.key == pygame.K_i:
                        fx["panel_open"] = not fx["panel_open"]
                    elif event.key == pygame.K_F12:
                        _screenshot(screen, state.tick)

            pressed = pygame.key.get_pressed()
            accumulator += dt
            while accumulator >= tick_len:
                accumulator -= tick_len
                if paused or in_menu or state.outcome.is_over:
                    continue
                action = agent(state) if agent is not None else action_from_pressed(pressed)
                state, events = step(state, action, params, game_rng)
                logger.extend(events)
                apply_events(fx, events, state, screen.get_size())
                if state.outcome.is_over:
                    save_log()
                    continue
                if predictor is not None and state.tick >= PREDICT_AFTER_TICKS and state.tick % PREDICT_EVERY_TICKS == 0:
                    prediction = predictor.predict(live_features(logger.log, state.tick))
                    if on_prediction is not None:
                        on_prediction(prediction)
                if controller is not None:
                    current = prediction if isinstance(prediction, Prediction) else None
                    new_params, decision = controller.step(state, params, logger.log.events, config.max_health, current)
                    if decision is not None:
                        record(decision)
                        fx["decisions"] = (fx["decisions"] + [decision])[-6:]
                        if decision.changed:
                            fx["last_decision"], fx["last_decision_until"] = decision, state.tick + NOTICE_TICKS
                            fx["changes"] += 1
                        if decision.changed:
                            state, extra = reconcile(state, new_params, game_rng, previous=params)
                            logger.extend(extra)
                            params = new_params
                            notice, notice_until = "Difficulty changed: " + decision.summary(), state.tick + NOTICE_TICKS
                if state.tick >= notice_until:
                    notice = base_notice()

            advance_fx(fx, dt)
            draw(screen, state, params, font, prediction, notice, fx, paused=paused, menu=in_menu)
            pygame.display.flip()
            if max_ticks is not None and (state.tick >= max_ticks or state.outcome.is_over):
                running = False
    finally:
        save_log()
        pygame.quit()
    return state.outcome


def _screenshot(screen: pygame.Surface, tick: int) -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / f"screenshot_t{tick}.png"
    pygame.image.save(screen, str(path))
    print(f"saved {path}")
    return path
