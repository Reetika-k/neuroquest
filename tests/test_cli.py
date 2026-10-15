"""Phase 10: every command runs end to end through ``cli.main`` with small arguments, writing only
into a temporary folder; the window-opening ``play`` is covered with a stubbed window elsewhere."""

from pathlib import Path

import pytest

from neuroquest.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_simulate_build_train_commands(tmp_path, tiny_model_dir, capsys) -> None:
    assert main(["benchmark-enemies", "--episodes", "2", "--mazes", "3", "--out", str(tmp_path / "bench")]) == 0
    assert (tmp_path / "bench" / "enemy_baseline.csv").exists() and (tmp_path / "bench" / "search_benchmark.csv").exists()
    assert main(["simulate", "--profiles", "cautious,explorer", "--episodes", "1", "--seed", "700", "--time-limit", "10", "--out", str(tmp_path / "raw")]) == 0
    assert len(list((tmp_path / "raw").glob("*.jsonl"))) == 2
    assert main(["build-dataset", "--raw", str(tiny_model_dir.parent / "raw"), "--out", str(tmp_path / "episodes.csv")]) == 0
    assert (tmp_path / "episodes.csv").exists()
    assert main(["train", "--data", str(tmp_path / "episodes.csv"), "--model-dir", str(tmp_path / "models"), "--report", str(tmp_path / "metrics.json"), "--seed", "1", "--fast"]) == 0
    assert (tmp_path / "models" / "behaviour_clf.joblib").exists() and (tmp_path / "metrics.json").exists()
    out = capsys.readouterr().out
    assert "A*" in out and "episode logs written" in out and "model saved to" in out


@pytest.mark.skipif(not (ROOT / "models" / "behaviour_clf.joblib").exists(), reason="run `neuroquest train` first")
def test_ab_dashboard_and_rl_commands_with_repository_data(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(ROOT)  # default paths (models/, data/, reports/) resolve against the repository; outputs go to tmp
    assert main(["ab-test", "--profiles", "cautious", "--episodes", "1", "--seed", "800", "--out", str(tmp_path / "ab.csv")]) == 0
    assert (tmp_path / "ab.csv").read_text().count("\n") == 3  # header + fixed + adaptive
    assert main(["dashboard", "--out", str(tmp_path / "dash.html"), "--figures-dir", str(tmp_path / "figs")]) == 0
    assert (tmp_path / "dash.html").stat().st_size > 100_000 and list((tmp_path / "figs").glob("dash_*.png"))
    assert main(["rl-train", "--episodes", "30", "--eval-episodes", "2", "--model-dir", str(tmp_path / "rl-models"), "--report-dir", str(tmp_path / "rl-reports")]) == 0
    assert (tmp_path / "rl-models" / "q_table.npz").exists() and (tmp_path / "rl-reports" / "rl_results.json").exists()
    out = capsys.readouterr().out
    assert "adaptive − fixed" in out and "dashboard:" in out and "Q-learning" in out


def test_play_without_a_q_table_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["play", "--agent"]) == 1
    assert "rl-train" in capsys.readouterr().out
