"""Phase 9: every figure and the report build from small real inputs; missing inputs degrade gracefully."""

import json

import pandas as pd

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.dashboard import figures as F
from neuroquest.dashboard.report import build
from neuroquest.data.events import EpisodeLog
from neuroquest.ml.predict import BehaviourPredictor
from neuroquest.rl.env import MazeEnv
from neuroquest.rl.pipeline import run_pipeline
from neuroquest.sim.ab import ab_compare
from neuroquest.sim.harness import run_episode
from neuroquest.sim.profiles import PROFILES


def test_figures_from_tiny_real_data(tmp_path, tiny_model_dir) -> None:
    root = tiny_model_dir.parent
    df = pd.read_csv(root / "processed" / "episodes.csv")
    metrics = json.loads((root / "metrics.json").read_text())
    out = tmp_path / "figs"
    paths = [
        F.fig_class_balance(df, out / "a.png"),
        F.fig_model_comparison(metrics, out / "b.png"),
        F.fig_confusion(metrics, out / "c.png"),
        F.fig_importance(metrics, out / "d.png"),
        F.fig_accuracy_by_prefix(metrics, out / "e.png"),
    ]
    coef = F.fig_coefficients(metrics, out / "f.png")
    if coef is not None:
        paths.append(coef)
    predictor = BehaviourPredictor.load(tiny_model_dir / "behaviour_clf.joblib", tiny_model_dir / "behaviour_clf.meta.json")
    log = EpisodeLog.load(next((root / "raw").glob("*.jsonl")))
    rows = F.prediction_timeline(log, predictor)
    assert rows and all(set(r) >= {"t", "label", "confidence", "aggressive", "cautious", "explorer"} for r in rows)
    paths.append(F.fig_prediction_timeline(rows, "t", out / "g.png"))
    adaptive = run_episode(PROFILES["aggressive"], GameConfig(), DifficultyParams(), seed=7, max_ticks=300, adaptive=True, predictor=predictor)
    paths.append(F.fig_difficulty_timeline(adaptive.meta["decisions"], 10, "t", out / "h.png"))
    ab = ab_compare(["aggressive", "cautious"], 1, GameConfig(), DifficultyParams(time_limit_s=40), seed0=900, predictor=predictor)
    paths.append(F.fig_ab(ab, out / "i.png"))
    assert all(p is not None and p.exists() and p.stat().st_size > 1000 for p in paths)
    assert F.fig_prediction_timeline([], "t", out / "z.png") is None and F.fig_difficulty_timeline([], 10, "t", out / "z.png") is None and F.fig_ab([], out / "z.png") is None


def test_report_builds_with_all_inputs_and_degrades_without(tmp_path, tiny_model_dir) -> None:
    root = tiny_model_dir.parent
    predictor = BehaviourPredictor.load(tiny_model_dir / "behaviour_clf.joblib", tiny_model_dir / "behaviour_clf.meta.json")
    adaptive = run_episode(PROFILES["explorer"], GameConfig(), DifficultyParams(), seed=8, max_ticks=300, adaptive=True, predictor=predictor)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("".join(json.dumps({"episode_id": "human-test", **d}) + "\n" for d in adaptive.meta["decisions"]))
    ab_csv = tmp_path / "ab.csv"
    pd.DataFrame(ab_compare(["aggressive", "cautious"], 1, GameConfig(), DifficultyParams(time_limit_s=40), seed0=901, predictor=predictor)).to_csv(ab_csv, index=False)
    rl_env = MazeEnv(GameConfig(width=7, height=7, loop_fraction=1.0), DifficultyParams(enemy_count=0, reward_density=0.1, time_limit_s=10), layout_seed=1)
    run_pipeline(rl_env, episodes=120, seed=0, eval_episodes=6, model_dir=tmp_path / "models", report_dir=tmp_path / "rl")
    summary = build(root / "processed" / "episodes.csv", root / "metrics.json", decisions, ab_csv, root / "raw", tiny_model_dir, tmp_path / "dash.html", tmp_path / "figs", rl_results_path=tmp_path / "rl" / "rl_results.json", rl_curve_csv=tmp_path / "rl" / "rl_curve.csv")
    assert set(summary["sections"]) == {"data", "model", "interpretability", "examples", "session", "decisions", "ab", "rl"}
    html = (tmp_path / "dash.html").read_text()
    assert html.count("<h2>") >= 10 and html.count("<img") >= 10 and "Confident and right" in html and "Q-learning" in html
    assert summary["examples"]["correct"] and all(r["true"] == r["predicted"] for r in summary["examples"]["correct"])
    assert all(r["true"] != r["predicted"] for r in summary["examples"]["wrong"])

    empty = build(tmp_path / "none.csv", tmp_path / "none.json", tmp_path / "none.jsonl", tmp_path / "none.csv", tmp_path / "noraw", tmp_path / "nomodels", tmp_path / "empty.html", tmp_path / "figs2", rl_results_path=tmp_path / "none-rl.json", rl_curve_csv=tmp_path / "none-rl.csv")
    assert empty["sections"] == [] and "No dataset found" in (tmp_path / "empty.html").read_text() and "No RL results" in (tmp_path / "empty.html").read_text()


def test_rl_figures_from_tiny_real_data(tmp_path) -> None:
    env = MazeEnv(GameConfig(width=7, height=7, loop_fraction=1.0), DifficultyParams(enemy_count=0, reward_density=0.1, time_limit_s=10), layout_seed=1)
    report = run_pipeline(env, episodes=80, seed=0, eval_episodes=4, model_dir=tmp_path / "models", report_dir=tmp_path / "rl")
    curve = pd.read_csv(tmp_path / "rl" / "rl_curve.csv")
    p1 = F.fig_rl_curve(curve, tmp_path / "curve.png", references={"random": -1.0})
    p2 = F.fig_rl_eval(report["evaluation"]["agents"], tmp_path / "eval.png")
    assert all(p is not None and p.exists() and p.stat().st_size > 1000 for p in (p1, p2))
    assert F.fig_rl_curve(None, tmp_path / "z.png") is None and F.fig_rl_eval({}, tmp_path / "z.png") is None
