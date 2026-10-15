"""Phase 10: the whole system end to end, with the real saved model where it exists.

play (real loop, scripted player) → event log on disk → dataset row with schema validation →
live prediction from the saved classifier → adaptive decision log → dashboard build. Plus the
reproducibility checks behind success criterion S3: same seed, same bytes.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from neuroquest.adapt.bounds import BOUNDS
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.state import Outcome
from neuroquest.dashboard.report import build as build_dashboard
from neuroquest.data.dataset import build as build_dataset
from neuroquest.data.events import EpisodeLog
from neuroquest.data.features import FEATURE_NAMES, live_features
from neuroquest.data.labels import LABELS
from neuroquest.data.schema import validate_row
from neuroquest.ml.predict import BehaviourPredictor
from neuroquest.ml.train import train
from neuroquest.rl.agents import GreedyCollectorAgent
from neuroquest.rl.env import ACTIONS
from neuroquest.sim.harness import run_batch
from neuroquest.ui.app import run

ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL, REAL_META = ROOT / "models" / "behaviour_clf.joblib", ROOT / "models" / "behaviour_clf.meta.json"
needs_real_model = pytest.mark.skipif(not REAL_MODEL.exists(), reason="run `neuroquest train` first")


@needs_real_model
def test_play_log_features_predict_adapt_end_to_end(tmp_path) -> None:
    collector = GreedyCollectorAgent()
    predictions = []
    decisions_path = tmp_path / "decisions.jsonl"
    outcome = run(
        GameConfig(ticks_per_second=200), DifficultyParams(), seed=11, max_ticks=600, window=(640, 480),
        log_dir=tmp_path / "raw", model_path=REAL_MODEL, meta_path=REAL_META, on_prediction=predictions.append,
        adaptive=True, decisions_path=decisions_path, agent=lambda s: ACTIONS[collector.act(None, s)],
    )
    assert outcome in (Outcome.PLAYING, Outcome.WON, Outcome.LOST_HEALTH)

    # 1. the episode is on disk and loads back
    log_path = next((tmp_path / "raw").glob("human-*.jsonl"))
    log = EpisodeLog.load(log_path)
    assert log.meta["profile"] == "human" and log.meta["seed"] == 11
    assert log.meta["ticks"] >= 30 and {e.type for e in log.events} >= {"move", "reward", "near_enemy"}

    # 2. live predictions came from the saved model on the live schedule (3 s, then every 2 s)
    assert predictions and all(p.label in LABELS for p in predictions)
    assert all(abs(sum(p.proba.values()) - 1) < 1e-6 and p.confidence == max(p.proba.values()) for p in predictions)
    predictor = BehaviourPredictor.load(REAL_MODEL, REAL_META)
    replay = predictor.predict(live_features(log, 40))
    assert replay.label == predictions[0].label and replay.proba == pytest.approx(predictions[0].proba)

    # 3. the log becomes valid dataset rows (rule-labelled, since it is human play)
    df, report = build_dataset(tmp_path / "raw", tmp_path / "episodes.csv")
    assert report["episodes"] == 1 and report["invalid"] == [] and len(df) == 4
    assert all(validate_row(r) == [] for r in df.to_dict("records")) and set(FEATURE_NAMES) <= set(df.columns)
    assert all(label.startswith("rule:") and label[5:] in LABELS for label in df["label"]) and df["profile"].eq("human").all()  # human rows are marked as rule-labelled

    # 4. every controller decision was logged with inputs, rule and output, inside the bounds
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    assert decisions and [d["t"] for d in decisions] == list(range(30, 30 * len(decisions) + 1, 30))  # one every 3 s, none skipped
    assert decisions[-1]["t"] <= log.meta["ticks"]
    for d in decisions:
        assert {"episode_id", "t", "rule", "direction", "lever", "before", "after", "inputs", "reason"} <= set(d)
        assert d["episode_id"] == log.meta["episode_id"] and d["reason"]
        for knob, (lo, hi) in BOUNDS.items():
            assert lo <= d["after"][knob] <= hi


@needs_real_model
def test_dashboard_builds_from_the_repository_data(tmp_path) -> None:
    summary = build_dashboard(
        ROOT / "data/processed/episodes.csv", ROOT / "reports/metrics.json", ROOT / "reports/decisions.jsonl", ROOT / "reports/ab_results.csv",
        ROOT / "data/raw", ROOT / "models", tmp_path / "dash.html", tmp_path / "figs", ROOT / "reports/rl_results.json", ROOT / "reports/rl_curve.csv",
    )
    present = set(summary["sections"])
    assert {"data", "model", "interpretability", "examples", "session"} <= present
    html = (tmp_path / "dash.html").read_text()
    assert html.count("<h2>") == 10 and (tmp_path / "dash.html").stat().st_size > 100_000  # sections 1–8, Limitations, Reproduce
    missing = {"decisions", "ab", "rl"} - present
    assert not missing, f"sections missing because their input files are absent: {missing}"


def test_dataset_build_is_byte_for_byte_reproducible(tmp_path) -> None:
    raw = tmp_path / "raw"
    run_batch(["cautious"], 3, GameConfig(), DifficultyParams(time_limit_s=20), seed0=300, out_dir=raw)
    again = tmp_path / "raw2"
    run_batch(["cautious"], 3, GameConfig(), DifficultyParams(time_limit_s=20), seed0=300, out_dir=again)
    assert [p.read_bytes() for p in sorted(raw.glob("*.jsonl"))] == [p.read_bytes() for p in sorted(again.glob("*.jsonl"))]
    build_dataset(raw, tmp_path / "a.csv")
    build_dataset(again, tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_training_is_reproducible_for_the_same_seed(tiny_model_dir, tmp_path) -> None:
    data = tiny_model_dir.parent / "processed" / "episodes.csv"
    a = train(data, model_dir=tmp_path / "a", report_path=tmp_path / "a.json", seed=3, fast=True, test_fraction=0.25)
    b = train(data, model_dir=tmp_path / "b", report_path=tmp_path / "b.json", seed=3, fast=True, test_fraction=0.25)
    assert a["selected"] == b["selected"] and a["split"] == b["split"]
    assert {k: v["test"] for k, v in a["models"].items()} == {k: v["test"] for k, v in b["models"].items()}
    assert pd.read_csv(data).shape[0] > 0
