"""Phase 6: raw logs → validated CSV."""

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.data.dataset import PREFIXES, build, rows_from_log, summary
from neuroquest.data.events import EpisodeLog
from neuroquest.data.features import FEATURE_NAMES
from neuroquest.data.schema import ROW_COLUMNS
from neuroquest.sim.harness import run_batch


def test_build_dataset_from_a_small_batch(tmp_path) -> None:
    raw = tmp_path / "raw"
    run_batch(["aggressive", "cautious", "explorer"], 2, GameConfig(), DifficultyParams(time_limit_s=30), seed0=0, out_dir=raw)
    df, report = build(raw, tmp_path / "processed" / "episodes.csv")
    assert report["episodes"] == 6 and report["rows"] == 6 * len(PREFIXES) and report["invalid"] == []
    assert list(df.columns) == ROW_COLUMNS
    assert set(df["label"]) == {"aggressive", "cautious", "explorer"}
    assert (df["label"] == df["profile"]).all()
    assert df.groupby("episode_id")["prefix_ratio"].apply(lambda s: sorted(s) == list(PREFIXES)).all()
    assert (tmp_path / "processed" / "episodes.csv").exists()
    assert (tmp_path / "schema" / "features.schema.json").exists()
    text = summary(df)
    assert "label balance" in text and "feature means per label" in text


def test_human_logs_get_rule_labels(tmp_path) -> None:
    raw = tmp_path / "raw"
    run_batch(["explorer"], 1, GameConfig(), DifficultyParams(time_limit_s=30), seed0=0, out_dir=raw)
    path = next(raw.glob("*.jsonl"))
    log = EpisodeLog.load(path)
    log.meta["profile"] = "human"
    rows = rows_from_log(log)
    assert all(r["label"].startswith("rule:") for r in rows) and all(r["profile"] == "human" for r in rows)
    assert all(name in rows[0] for name in FEATURE_NAMES)
