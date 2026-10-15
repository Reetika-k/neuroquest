"""Phase 6: features computed from a scripted episode with known answers."""

import math

import pytest

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.entities import Enemy, Player
from neuroquest.core.maze import Maze
from neuroquest.core.rules import step
from neuroquest.core.state import Action, GameState
from neuroquest.data.events import EventLogger
from neuroquest.data.features import FEATURE_NAMES, episode_features
from neuroquest.data.labels import LABELS, rule_label
from neuroquest.data.schema import FEATURE_SPECS, validate_row, write_schema

ROWS = ["#######", "#.....#", "#.#.#.#", "#.....#", "#######"]  # start (1,1), exit (3,5): distance 6
PARAMS = DifficultyParams(enemy_count=1, enemy_move_interval=100, time_limit_s=100)  # enemy never moves


def scripted_log(actions, enemy=(1, 5)):
    maze = Maze.from_strings(ROWS)
    state = GameState(maze=maze, player=Player((1, 1), 3), rewards=frozenset({(1, 3)}), exit=(3, 5), visited=frozenset({(1, 1)}), enemies=(Enemy(0, enemy),))
    logger = EventLogger(state, GameConfig(), PARAMS, episode_id="s", profile="test", seed=0)
    for action in actions:
        state, events = step(state, action, PARAMS)
        logger.extend(events)
    return logger.finish(state)


def test_feature_values_on_a_known_episode() -> None:
    # enemy at (1,5). right → (1,2) d=3; right → (1,3) d=2 (near, approaching; reward); right → (1,4) d=1 (approaching);
    # left → (1,3) d=2 (retreating); stay ×2 at d=2. Near-enemy ticks: 2..6; near moves: 3, of which 2 approached.
    log = scripted_log([Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.LEFT, Action.STAY, Action.STAY])
    f = episode_features(log)
    assert list(f) == FEATURE_NAMES
    assert f["moves_per_tick"] == pytest.approx(4 / 6)
    assert f["idle_ratio"] == pytest.approx(2 / 6)
    assert f["bump_ratio"] == 0.0
    assert f["new_cells_per_tick"] == pytest.approx(3 / 6)  # (1,2), (1,3), (1,4) new; (1,3) revisited
    assert f["revisit_ratio"] == pytest.approx(1 / 4)
    assert f["distinct_cells_ratio"] == pytest.approx(4 / 13)
    assert f["rewards_per_tick"] == pytest.approx(1 / 6) and f["reward_share"] == 1.0
    assert f["damage_per_tick"] == 0.0
    assert f["near_enemy_ratio"] == pytest.approx(5 / 6)
    assert f["approach_ratio"] == pytest.approx(2 / 3)
    assert f["progress_rate"] == pytest.approx((6 - 4) / 6)  # ends at (1,3), 4 steps from the exit
    assert f["min_exit_distance_ratio"] == pytest.approx(3 / 6)  # closest point was (1,4)


def test_prefix_uses_only_the_first_part() -> None:
    log = scripted_log([Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.LEFT, Action.STAY, Action.STAY])
    half = episode_features(log, 0.5)  # ticks 1..3
    assert half["elapsed_s"] == pytest.approx(0.3)
    assert half["moves_per_tick"] == 1.0 and half["idle_ratio"] == 0.0
    assert half["reward_share"] == 1.0
    with pytest.raises(ValueError):
        episode_features(log, 0.0)


def test_bumps_are_counted() -> None:
    log = scripted_log([Action.UP, Action.UP, Action.RIGHT])  # two bumps into the top wall, then a move
    f = episode_features(log)
    assert f["bump_ratio"] == pytest.approx(2 / 3) and f["moves_per_tick"] == pytest.approx(1 / 3)


def test_damage_is_counted() -> None:
    log = scripted_log([Action.STAY, Action.STAY], enemy=(1, 1))  # enemy on the player: hit on tick 1 only (invulnerable after)
    f = episode_features(log)
    assert f["damage_per_tick"] == pytest.approx(1 / 2)


def test_schema_matches_features_and_validation_works(tmp_path) -> None:
    assert list(FEATURE_SPECS) == FEATURE_NAMES
    log = scripted_log([Action.RIGHT])
    row = {"label": "cautious", **episode_features(log)}
    assert validate_row(row) == []
    assert validate_row({**row, "label": "rule:explorer"}) == []
    assert any("bad label" in p for p in validate_row({**row, "label": "wizard"}))
    assert any("not finite" in p for p in validate_row({**row, "moves_per_tick": math.nan}))
    assert any("outside" in p for p in validate_row({**row, "reward_share": 1.5}))
    assert any("missing" in p for p in validate_row({k: v for k, v in row.items() if k != "idle_ratio"}))
    assert any("not numeric" in p for p in validate_row({**row, "idle_ratio": "x"}))
    schema = write_schema(tmp_path / "schema" / "features.schema.json")
    assert schema.exists() and '"name": "approach_ratio"' in schema.read_text()


def test_rule_label_covers_all_classes() -> None:
    base = {name: 0.0 for name in FEATURE_NAMES}
    assert rule_label(base) == "cautious"
    assert rule_label({**base, "approach_ratio": 0.8, "near_enemy_ratio": 0.2}) == "aggressive"
    assert rule_label({**base, "damage_per_tick": 0.05}) == "aggressive"
    assert rule_label({**base, "distinct_cells_ratio": 0.7}) == "explorer"
    assert set(LABELS) == {"aggressive", "cautious", "explorer"}
