"""Phase 7: split without leakage, the training protocol on a tiny dataset, persistence, prediction."""

import json
from pathlib import Path

import pandas as pd
import pytest

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.data.dataset import build
from neuroquest.data.events import EpisodeLog
from neuroquest.data.features import FEATURE_NAMES, episode_features, live_features
from neuroquest.data.labels import LABELS
from neuroquest.ml.evaluate import accuracy_by_prefix, classification_metrics, confusion_pairs
from neuroquest.ml.predict import BehaviourPredictor, ModelMismatch
from neuroquest.ml.split import grouped_split
from neuroquest.ml.train import format_report, train
from neuroquest.sim.harness import run_batch


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ml")
    raw = root / "raw"
    run_batch(list(LABELS), 8, GameConfig(), DifficultyParams(time_limit_s=40), seed0=0, out_dir=raw)
    csv = root / "processed" / "episodes.csv"
    build(raw, csv)
    return csv


def test_grouped_split_keeps_seeds_and_episodes_together(tiny_dataset: Path) -> None:
    df = pd.read_csv(tiny_dataset)
    train_df, test_df = grouped_split(df, test_fraction=0.25, seed=1)
    assert set(train_df["seed"]).isdisjoint(set(test_df["seed"]))
    assert set(train_df["episode_id"]).isdisjoint(set(test_df["episode_id"]))
    assert len(train_df) + len(test_df) == len(df)
    assert set(test_df["label"]) == set(LABELS) and set(train_df["label"]) == set(LABELS)
    again, _ = grouped_split(df, test_fraction=0.25, seed=1)
    assert again.equals(train_df)
    with pytest.raises(ValueError):
        grouped_split(df, test_fraction=1.5)


def test_metrics_helpers_on_known_answers() -> None:
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    m = classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["b"]["recall"] == 1.0 and m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert confusion_pairs(y_true, y_pred) == [{"true": "a", "predicted": "b", "count": 1}, {"true": "c", "predicted": "a", "count": 1}]
    df = pd.DataFrame({"label": y_true, "prefix_ratio": [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]})
    assert accuracy_by_prefix(df, y_pred) == {"0.5": 1.0, "1.0": pytest.approx(1 / 3)}


def test_train_protocol_writes_model_meta_and_report(tiny_dataset: Path, tmp_path: Path) -> None:
    report = train(tiny_dataset, model_dir=tmp_path / "models", report_path=tmp_path / "reports" / "metrics.json", seed=3, fast=True, test_fraction=0.25)
    assert set(report["models"]) == {"rule_labeller", "single_feature_tree", "majority", "logistic_regression", "random_forest", "knn"}
    assert report["selected"]["model"] in {"logistic_regression", "random_forest", "knn"}
    for name in ("majority", "logistic_regression", "random_forest", "knn"):
        m = report["models"][name]
        assert 0.0 <= m["cv"]["macro_f1_mean"] <= 1.0 and 0.0 <= m["test"]["macro_f1"] <= 1.0
        assert len(m["test"]["confusion_matrix"]) == 3
    assert report["models"]["majority"]["test"]["macro_f1"] < report["models"][report["selected"]["model"]]["test"]["macro_f1"]
    assert set(report["selected"]["importances"]["permutation"]) == set(FEATURE_NAMES)
    assert "elapsed_s" not in report["ablation_without_elapsed_s"] and 0.0 <= report["ablation_without_elapsed_s"]["test"]["macro_f1"] <= 1.0
    assert (tmp_path / "models" / "behaviour_clf.joblib").exists()
    meta = json.loads((tmp_path / "models" / "behaviour_clf.meta.json").read_text())
    assert meta["features"] == FEATURE_NAMES and set(meta["classes"]) == set(LABELS) and meta["model"] == report["selected"]["model"]
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text())["selected"]["model"] == meta["model"]
    assert "selected:" in format_report(report)


def test_predictor_loads_checks_features_and_predicts(tiny_dataset: Path, tmp_path: Path) -> None:
    train(tiny_dataset, model_dir=tmp_path / "models", report_path=tmp_path / "metrics.json", seed=3, fast=True, test_fraction=0.25)
    predictor = BehaviourPredictor.load(tmp_path / "models" / "behaviour_clf.joblib", tmp_path / "models" / "behaviour_clf.meta.json")
    df = pd.read_csv(tiny_dataset)
    pred = predictor.predict(df.iloc[0][FEATURE_NAMES].to_dict())
    assert pred.label in LABELS and set(pred.proba) == set(LABELS)
    assert sum(pred.proba.values()) == pytest.approx(1.0) and pred.confidence == pytest.approx(max(pred.proba.values()))
    assert pred.summary().startswith(pred.label)
    meta = json.loads((tmp_path / "models" / "behaviour_clf.meta.json").read_text())
    meta["features"] = meta["features"][:-1]
    bad = tmp_path / "bad.meta.json"
    bad.write_text(json.dumps(meta))
    with pytest.raises(ModelMismatch):
        BehaviourPredictor.load(tmp_path / "models" / "behaviour_clf.joblib", bad)


def test_live_features_match_episode_features_at_the_end(tiny_dataset: Path) -> None:
    log = EpisodeLog.load(next((tiny_dataset.parent.parent / "raw").glob("*.jsonl")))
    assert live_features(log, log.meta["ticks"]) == episode_features(log, 1.0)
    early = live_features(log, 5)
    assert early["elapsed_s"] == pytest.approx(0.5)
