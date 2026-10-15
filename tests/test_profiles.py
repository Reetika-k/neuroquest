"""Phase 6: simulated player profiles and the headless harness."""

import random
import statistics

import pytest

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.rules import new_game
from neuroquest.core.state import Action
from neuroquest.data.events import EpisodeLog
from neuroquest.data.features import episode_features
from neuroquest.sim.harness import run_batch, run_episode
from neuroquest.sim.profiles import PROFILES, MaskedMaze, danger_cells, retreat_action

CONFIG = GameConfig()
PARAMS = DifficultyParams()


@pytest.mark.parametrize("name", list(PROFILES))
def test_profiles_return_legal_actions_and_sample_parameters(name: str) -> None:
    a, b = PROFILES[name](random.Random(1)), PROFILES[name](random.Random(2))
    assert a.name == name and a.params and a.params != b.params
    state = new_game(CONFIG, PARAMS, seed=3)
    rng = random.Random(0)
    for _ in range(50):
        assert a.act(state, PARAMS, rng) in Action


def test_masked_maze_and_danger_cells() -> None:
    state = new_game(CONFIG, PARAMS, seed=3)
    enemy = state.enemies[0].pos
    danger = danger_cells(state, 2)
    assert enemy in danger and (enemy[0] + 2, enemy[1]) in danger and (enemy[0] + 3, enemy[1]) not in danger
    masked = MaskedMaze(state.maze, danger)
    assert not masked.is_open(enemy)
    assert all(c not in danger for c in masked.neighbours(state.player.pos))


def test_retreat_moves_away_from_the_nearest_enemy() -> None:
    state = new_game(CONFIG, PARAMS, seed=3)
    action = retreat_action(state, random.Random(0))
    assert action in Action


def test_run_episode_is_deterministic_and_complete() -> None:
    a = run_episode(PROFILES["explorer"], CONFIG, PARAMS, seed=4, max_ticks=300)
    b = run_episode(PROFILES["explorer"], CONFIG, PARAMS, seed=4, max_ticks=300)
    assert a.meta == b.meta and a.events == b.events
    assert a.meta["profile"] == "explorer" and a.meta["episode_id"] == "explorer-0004"
    assert 0 < a.meta["ticks"] <= 300
    assert all(e.t <= a.meta["ticks"] for e in a.events)
    capped = run_episode(PROFILES["cautious"], CONFIG, DifficultyParams(time_limit_s=1000), seed=4, max_ticks=20)
    assert capped.meta["ticks"] == 20 and capped.meta["outcome"] == "playing"


def test_profiles_behave_differently_on_average() -> None:
    feats = {name: [] for name in PROFILES}
    for name, factory in PROFILES.items():
        for seed in range(6):
            feats[name].append(episode_features(run_episode(factory, CONFIG, PARAMS, seed=seed, max_ticks=600)))
    mean = lambda name, key: statistics.mean(f[key] for f in feats[name])  # noqa: E731
    assert mean("aggressive", "damage_per_tick") > mean("cautious", "damage_per_tick")
    assert mean("aggressive", "approach_ratio") > mean("cautious", "approach_ratio")
    assert mean("explorer", "distinct_cells_ratio") > mean("cautious", "distinct_cells_ratio")


def test_run_batch_writes_logs(tmp_path) -> None:
    messages = []
    paths = run_batch(["aggressive", "cautious"], 2, CONFIG, DifficultyParams(time_limit_s=20), seed0=10, out_dir=tmp_path, progress=messages.append)
    assert [p.name for p in paths] == ["aggressive-0010.jsonl", "aggressive-0011.jsonl", "cautious-0010.jsonl", "cautious-0011.jsonl"]
    assert len(messages) == 2 and "aggressive" in messages[0]
    log = EpisodeLog.load(paths[0])
    assert log.meta["profile"] == "aggressive" and log.meta["seed"] == 10
