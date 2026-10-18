"""Phase 7b: recent actions in the state, next-move features, exact replay of logs, the temporal
split, baselines and models, the predictor, the anticipating hunter and the pursuit benchmark."""

import json
from pathlib import Path

import pandas as pd
import pytest

from neuroquest.ai.benchmark import format_pursuit, pursuit_benchmark, summarize_pursuit
from neuroquest.ai.policies import Anticipator, AStarChaser
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.entities import Enemy, Player
from neuroquest.core.maze import Maze
from neuroquest.core.rules import new_game, step
from neuroquest.core.state import Action, GameState
from neuroquest.data.events import EpisodeLog
from neuroquest.ml.next_move import (
    MOVE_ACTIONS,
    MOVE_FEATURES,
    NextMovePredictor,
    actions_from_log,
    build_moves,
    format_next_move_report,
    greedy_predictions,
    move_features,
    repeat_last_predictions,
    replay_rows,
    temporal_split,
    train_next_move,
)
from neuroquest.ml.predict import ModelMismatch


@pytest.fixture(scope="module")
def tiny_moves(tmp_path_factory, tiny_model_dir) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("moves") / "moves.csv"
    _, report = build_moves(tiny_model_dir.parent / "raw", out)
    return out, report


@pytest.fixture(scope="module")
def tiny_next_move(tmp_path_factory, tiny_moves) -> tuple[dict, Path]:
    root = tmp_path_factory.mktemp("nm")
    report = train_next_move(tiny_moves[0], model_dir=root / "models", report_path=root / "report.json", seed=1, fast=True, test_fraction=0.25)
    return report, root / "models"


def test_step_tracks_the_last_three_actions() -> None:
    state = new_game(GameConfig(), DifficultyParams(enemy_count=0), seed=1)
    assert state.recent_actions == ()
    for a in (Action.RIGHT, Action.STAY, Action.DOWN, Action.LEFT):
        state, _ = step(state, a, DifficultyParams(enemy_count=0))
    assert state.recent_actions == (Action.STAY, Action.DOWN, Action.LEFT)


def test_move_features_names_and_values_on_a_hand_made_board() -> None:
    maze = Maze.from_strings(["#####", "#...#", "#.#.#", "#...#", "#####"])
    state = GameState(maze=maze, player=Player((1, 1), 3), rewards=frozenset({(3, 1)}), exit=(3, 3), visited=frozenset({(1, 1), (1, 2)}), enemies=(Enemy(0, (1, 3)),), recent_actions=(Action.RIGHT, Action.LEFT))
    f = move_features(state)
    assert list(f) == MOVE_FEATURES and len(MOVE_FEATURES) == 43
    assert f["prev1_left"] == 1 and f["prev2_right"] == 1 and sum(f[f"prev3_{a}"] for a in MOVE_ACTIONS) == 0  # only two actions so far
    assert (f["open_up"], f["open_down"], f["open_left"], f["open_right"]) == (0, 1, 0, 1)
    assert f["visited_right"] == 1 and f["visited_down"] == 0
    assert (f["reward_dr"], f["reward_dc"], f["reward_dist"]) == (1, 0, 2) and (f["exit_dr"], f["exit_dc"], f["exit_dist"]) == (1, 1, 4)
    assert (f["enemy_dr"], f["enemy_dc"], f["enemy_dist"], f["near_enemy"]) == (0, 1, 2, 1)
    assert f["path_reward_down"] == 1 and f["path_exit_down"] + f["path_exit_right"] == 1 and f["rewards_left"] == 1 and f["health"] == 3
    empty = move_features(GameState(maze=maze, player=Player((3, 3), 1), rewards=frozenset(), exit=(3, 3), visited=frozenset()))
    assert empty["reward_dist"] == 40 and empty["enemy_dist"] == 20 and sum(empty[f"path_exit_{d}"] for d in ("up", "down", "left", "right")) == 0


def test_replay_reproduces_every_tiny_log(tiny_model_dir) -> None:
    logs = sorted((tiny_model_dir.parent / "raw").glob("*.jsonl"))
    assert logs
    for path in logs:
        log = EpisodeLog.load(path)
        actions = actions_from_log(log)
        assert len(actions) == log.meta["ticks"] and set(actions) <= set(MOVE_ACTIONS)
        rows, ok = replay_rows(log)
        assert ok and len(rows) == log.meta["ticks"]
        assert rows[0]["t"] == 1 and rows[-1]["t"] == log.meta["ticks"] and all(r["target"] in MOVE_ACTIONS for r in rows)


def test_build_moves_and_temporal_split(tiny_moves) -> None:
    csv, report = tiny_moves
    df = pd.read_csv(csv)
    assert report["replay_failed"] == [] and report["episodes"] == 24 and report["rows"] == len(df) > 0
    assert list(df.columns) == ["episode_id", "profile", "seed", "t", "target"] + MOVE_FEATURES
    assert set(df["target"]) <= set(MOVE_ACTIONS) and df[MOVE_FEATURES].notna().all().all()
    train, test = temporal_split(df, 0.25)
    assert len(train) + len(test) == len(df) and train["seed"].max() < test["seed"].min()
    assert set(train["episode_id"]).isdisjoint(set(test["episode_id"]))
    with pytest.raises(ValueError):
        temporal_split(df, 1.0)


def test_build_moves_skips_human_logs_and_counts_replay_mismatches(tiny_model_dir, tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = sorted((tiny_model_dir.parent / "raw").glob("*.jsonl"))[:3]
    good = EpisodeLog.load(source[0])
    good.save(raw / "good.jsonl")
    human = EpisodeLog.load(source[1])
    human.meta["profile"] = "human"
    human.save(raw / "human-x.jsonl")
    tampered = EpisodeLog.load(source[2])
    tampered.meta["score"] += 10  # the log claims a score the replay cannot reproduce
    tampered.save(raw / "tampered.jsonl")
    df, report = build_moves(raw, tmp_path / "moves.csv")
    assert report["episodes"] == 1 and report["skipped"] == ["human-x.jsonl"] and report["replay_failed"] == ["tampered.jsonl"]
    assert len(df) == good.meta["ticks"] == report["rows"]


def test_baselines_are_what_they_say() -> None:
    row = {f: 0.0 for f in MOVE_FEATURES}
    df = pd.DataFrame([dict(row, prev1_up=1.0, rewards_left=1.0, path_reward_left=1.0, path_exit_down=1.0), dict(row, rewards_left=0.0, path_exit_down=1.0), dict(row)])
    assert repeat_last_predictions(df, "stay") == ["up", "stay", "stay"]
    assert greedy_predictions(df) == ["left", "down", "stay"]


def test_train_next_move_reports_baselines_and_saves_the_model(tiny_next_move) -> None:
    report, model_dir = tiny_next_move
    assert set(report["models"]) == {"majority", "repeat_last", "greedy", "logistic_regression", "random_forest"}
    assert report["selected"]["model"] in {"logistic_regression", "random_forest"}
    for name, m in report["models"].items():
        assert 0.0 <= m["test"]["accuracy"] <= 1.0 and 0.0 <= m["test"]["macro_f1"] <= 1.0 and len(m["test"]["confusion_matrix"]) == 5
        assert set(m["per_profile_accuracy"]) == {"aggressive", "cautious", "explorer"}
    assert "temporal" in report["split"]["by"] and report["split"]["train_seeds"][1] < report["split"]["test_seeds"][0]
    assert set(report["selected"]["importances"]["permutation"]) == set(MOVE_FEATURES)
    meta = json.loads((model_dir / "next_move.meta.json").read_text())
    assert meta["features"] == MOVE_FEATURES and set(meta["classes"]) <= set(MOVE_ACTIONS)
    assert "selected:" in format_next_move_report(report)


def test_predictor_predicts_legal_actions_and_cells(tiny_next_move, tmp_path) -> None:
    _, model_dir = tiny_next_move
    predictor = NextMovePredictor.load(model_dir / "next_move.joblib", model_dir / "next_move.meta.json")
    state = new_game(GameConfig(), DifficultyParams(), seed=3)
    for _ in range(10):
        action, proba = predictor.predict(state)
        assert action in MOVE_ACTIONS and sum(proba.values()) == pytest.approx(1.0)
        cell = predictor.predict_cell(state)
        assert state.maze.is_open(cell) and (cell == state.player.pos or cell in state.maze.neighbours(state.player.pos))
        state, _ = step(state, Action(action), DifficultyParams())
    meta = json.loads((model_dir / "next_move.meta.json").read_text())
    meta["features"] = meta["features"][1:]
    (tmp_path / "bad.json").write_text(json.dumps(meta))
    with pytest.raises(ModelMismatch):
        NextMovePredictor.load(model_dir / "next_move.joblib", tmp_path / "bad.json")


def test_anticipator_moves_legally_and_falls_back_without_a_model(tiny_next_move, tmp_path) -> None:
    _, model_dir = tiny_next_move
    predictor = NextMovePredictor.load(model_dir / "next_move.joblib", model_dir / "next_move.meta.json")
    smart = Anticipator(predictor=predictor)
    blind = Anticipator(model_path=tmp_path / "missing.joblib", meta_path=tmp_path / "missing.json")
    plain = AStarChaser()
    state = new_game(GameConfig(), DifficultyParams(enemy_count=2), seed=4)
    import random

    rng = random.Random(0)
    for _ in range(30):
        enemy = state.enemies[0]
        target = smart.next_cell(enemy, state, rng)
        assert target == enemy.pos or target in state.maze.neighbours(enemy.pos)
        assert blind.next_cell(enemy, state, rng) == plain.next_cell(enemy, state, rng)  # no model: identical to A*
        state, _ = step(state, rng.choice(list(Action)), DifficultyParams(enemy_count=2), rng)
    assert smart.predictions == 30 and blind.predictions == 0 and blind.fallbacks == 30


def test_pursuit_benchmark_rows_and_summary(tiny_next_move) -> None:
    _, model_dir = tiny_next_move
    predictor = NextMovePredictor.load(model_dir / "next_move.joblib", model_dir / "next_move.meta.json")
    policies = {"astar": AStarChaser(), "anticipator": Anticipator(predictor=predictor)}
    rows = pursuit_benchmark(episodes=1, seed0=9000, profiles=("aggressive", "cautious"), params=DifficultyParams(time_limit_s=20), policies=policies)
    assert len(rows) == 4 and {r["kind"] for r in rows} == {"astar", "anticipator"}
    assert all(r["first_contact"] == "" or 0 < r["first_contact"] <= r["ticks"] for r in rows)
    summary = summarize_pursuit(rows)
    assert set(summary) == {"astar", "anticipator"} and summary["astar"]["aggressive"]["episodes"] == 1
    assert 0.0 <= summary["anticipator"]["cautious"]["catch_rate"] <= 1.0
    assert "Pursuit" in format_pursuit(summary)
