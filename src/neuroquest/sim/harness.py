"""Headless episodes: a player policy plays the real ``step()`` with no window, producing an
``EpisodeLog``. Used to build datasets (Phase 6) and to compare difficulty settings (Phase 8)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from neuroquest.adapt.controller import AdaptiveController
from neuroquest.ai.policies import EnemyPolicy
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.rules import new_game, reconcile, step
from neuroquest.data.events import EpisodeLog, EventLogger
from neuroquest.data.features import live_features
from neuroquest.ml.predict import PREDICT_AFTER_TICKS, PREDICT_EVERY_TICKS, BehaviourPredictor
from neuroquest.sim.profiles import PROFILES, PlayerPolicy

PolicyFactory = Callable[[random.Random], PlayerPolicy]


def run_episode(
    policy_factory: PolicyFactory,
    config: GameConfig,
    params: DifficultyParams,
    seed: int,
    *,
    max_ticks: int | None = None,
    profile_name: str | None = None,
    adaptive: bool = False,
    predictor: BehaviourPredictor | None = None,
    controller: AdaptiveController | None = None,
    policies: Mapping[str, EnemyPolicy] | None = None,
) -> EpisodeLog:
    """Play one seeded episode to the end (or ``max_ticks``) and return its log.

    With ``adaptive=True`` the saved classifier predicts the style on the usual schedule and the
    controller may change the difficulty every 5 s; every decision and parameter change is
    recorded in the log's meta (``decisions``, ``params_history``, ``params_final``).
    ``policies`` overrides the enemy policy registry (used by benchmarks that inject a model).
    """
    policy_rng = random.Random(seed)
    game_rng = random.Random(seed + 10_007)  # enemies' randomness, independent of the player's
    policy = policy_factory(policy_rng)
    name = profile_name or policy.name
    state = new_game(config, params, seed)
    logger = EventLogger(state, config, params, episode_id=f"{name}-{seed:04d}", profile=name, seed=seed, policy_params=policy.params)
    if adaptive:
        predictor = predictor or BehaviourPredictor.load()
        controller = controller or AdaptiveController()
    prediction = None
    decisions: list[dict] = []
    history = [{"t": 0, **asdict(params)}]
    while not state.outcome.is_over and (max_ticks is None or state.tick < max_ticks):
        action = policy.act(state, params, policy_rng)
        state, events = step(state, action, params, game_rng, policies)
        logger.extend(events)
        if adaptive and not state.outcome.is_over:
            if state.tick >= PREDICT_AFTER_TICKS and state.tick % PREDICT_EVERY_TICKS == 0:
                prediction = predictor.predict(live_features(logger.log, state.tick))
            new_params, decision = controller.step(state, params, logger.log.events, config.max_health, prediction)
            if decision is not None:
                decisions.append(decision.to_dict())
                if decision.changed:
                    state, extra = reconcile(state, new_params, game_rng, previous=params)
                    logger.extend(extra)
                    params = new_params
                    history.append({"t": state.tick, **asdict(params)})
    log = logger.finish(state)
    log.meta.update({"adaptive": adaptive, "decisions": decisions, "params_history": history, "params_final": asdict(params)})
    return log


def run_batch(
    profiles: Sequence[str],
    episodes: int,
    config: GameConfig,
    params: DifficultyParams,
    seed0: int,
    out_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Run ``episodes`` seeded episodes per profile (same seeds across profiles) and save JSONL logs."""
    paths: list[Path] = []
    for name in profiles:
        factory = PROFILES[name]
        t0 = time.perf_counter()
        outcomes: dict[str, int] = {}
        for i in range(episodes):
            seed = seed0 + i
            log = run_episode(factory, config, params, seed, profile_name=name)
            paths.append(log.save(out_dir / f"{name}-{seed:04d}.jsonl"))
            outcomes[log.meta["outcome"]] = outcomes.get(log.meta["outcome"], 0) + 1
        if progress:
            progress(f"{name:<11}{episodes} episodes in {time.perf_counter() - t0:.1f}s  outcomes {outcomes}")
    return paths
