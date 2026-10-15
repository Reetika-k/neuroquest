"""Phase 8: adaptive episodes in the harness, the A/B runner, and the game loop with the controller."""

from neuroquest.adapt.bounds import within_bounds
from neuroquest.adapt.controller import ADAPT_EVERY_TICKS
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.state import Outcome
from neuroquest.ml.predict import BehaviourPredictor
from neuroquest.sim.ab import ab_compare, deltas, summarize_ab
from neuroquest.sim.harness import run_episode
from neuroquest.sim.profiles import PROFILES
from neuroquest.ui.app import run


def predictor_from(model_dir):
    return BehaviourPredictor.load(model_dir / "behaviour_clf.joblib", model_dir / "behaviour_clf.meta.json")


def test_adaptive_episode_records_decisions_and_stays_in_bounds(tiny_model_dir) -> None:
    log = run_episode(PROFILES["aggressive"], GameConfig(), DifficultyParams(), seed=7, max_ticks=400, adaptive=True, predictor=predictor_from(tiny_model_dir))
    m = log.meta
    assert m["adaptive"] is True and m["params_history"][0]["t"] == 0
    assert all(d["t"] % ADAPT_EVERY_TICKS == 0 for d in m["decisions"])
    assert len(m["decisions"]) == m["ticks"] // ADAPT_EVERY_TICKS or m["outcome"] != "playing"
    last = {k: v for k, v in m["params_history"][-1].items() if k != "t"}
    assert DifficultyParams(**m["params_final"]) == DifficultyParams(**last)
    assert within_bounds(DifficultyParams(**m["params_final"]))
    for d in m["decisions"]:
        assert d["rule"].startswith("R") and "reason" in d and set(d["before"]) == set(d["after"])
    fixed = run_episode(PROFILES["aggressive"], GameConfig(), DifficultyParams(), seed=7, max_ticks=400, adaptive=False)
    assert fixed.meta["adaptive"] is False and fixed.meta["decisions"] == [] and len(fixed.meta["params_history"]) == 1


def test_ab_compare_rows_csv_summary_and_deltas(tmp_path, tiny_model_dir) -> None:
    rows = ab_compare(["aggressive", "cautious"], 1, GameConfig(), DifficultyParams(time_limit_s=60), seed0=900, predictor=predictor_from(tiny_model_dir), out_csv=tmp_path / "ab.csv")
    assert len(rows) == 4 and {r["condition"] for r in rows} == {"fixed", "adaptive"}
    assert all(r["changes"] == 0 for r in rows if r["condition"] == "fixed")
    assert (tmp_path / "ab.csv").read_text().splitlines()[0].startswith("profile,seed,condition,outcome,won,ticks")
    text = summarize_ab(rows)
    assert "win" in text and "rules fired" in text and "aggressive" in text and "adaptive" in text
    d = deltas(rows)
    assert set(d) == {"aggressive", "cautious"} and set(d["aggressive"]) == {"win_rate", "median_ticks", "mean_damage"}


def test_game_loop_with_controller_writes_decisions(tmp_path, tiny_model_dir) -> None:
    outcome = run(
        GameConfig(ticks_per_second=200), DifficultyParams(time_limit_s=5), seed=1, max_ticks=120, window=(320, 240),
        model_path=tiny_model_dir / "behaviour_clf.joblib", meta_path=tiny_model_dir / "behaviour_clf.meta.json",
        adaptive=True, decisions_path=tmp_path / "decisions.jsonl",
    )
    assert outcome in (Outcome.PLAYING, Outcome.LOST_HEALTH)
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert 1 <= len(lines) <= 120 // ADAPT_EVERY_TICKS and '"rule": "R' in lines[0] and '"episode_id": "human-' in lines[0]
