"""Phase 9b: the Q-table, the training loop, the baselines, the pipeline's files and the app hook."""

import json
import random

import pandas as pd
import pytest

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.state import Action
from neuroquest.rl.agents import GreedyCollectorAgent, ProfileAgent, QAgent, RandomAgent, ShortestPathAgent
from neuroquest.rl.env import LAYOUT_SEED, RL_CONFIG, RL_PARAMS, MazeEnv
from neuroquest.rl.pipeline import format_report, load_q_table, run_pipeline
from neuroquest.rl.qlearning import QConfig, QTable, evaluate, summarize, train, wilson

TINY = GameConfig(width=7, height=7, loop_fraction=1.0)
NO_HUNTER = DifficultyParams(enemy_count=0, reward_density=0.1, time_limit_s=10)


def test_qtable_update_greedy_and_round_trip(tmp_path) -> None:
    table = QTable(5)
    rng = random.Random(0)
    assert table.values((1, 1, 0, 4, 4)) == [0.0] * 5 and len(table) == 0  # reading creates nothing
    assert 0 <= table.greedy((1, 1, 0, 4, 4), rng) < 5
    err = table.update((1, 1, 0, 4, 4), 2, 1.0, (1, 2, 0, 4, 4), terminal=False, alpha=0.5, gamma=0.9)
    assert err == pytest.approx(1.0) and table.values((1, 1, 0, 4, 4))[2] == pytest.approx(0.5)
    table.update((1, 2, 0, 4, 4), 0, 2.0, (1, 3, 0, 4, 4), terminal=True, alpha=0.5, gamma=0.9)
    # non-terminal target bootstraps from the next state's best value: 1 + 0.9 * 1.0
    table.update((1, 1, 0, 4, 4), 2, 1.0, (1, 2, 0, 4, 4), terminal=False, alpha=0.5, gamma=0.9)
    assert table.values((1, 1, 0, 4, 4))[2] == pytest.approx(0.5 + 0.5 * (1.9 - 0.5))
    assert table.greedy((1, 1, 0, 4, 4), rng) == 2
    path = table.save(tmp_path / "q.npz")
    loaded = QTable.load(path)
    assert set(loaded.q) == set(table.q) and loaded.n_actions == 5
    assert all(loaded.q[k] == pytest.approx(table.q[k], abs=1e-6) for k in table.q)  # float32 on disk


def test_epsilon_schedule_and_wilson() -> None:
    cfg = QConfig(episodes=100, epsilon_start=1.0, epsilon_end=0.1, explore_fraction=0.5)
    assert cfg.epsilon(0) == 1.0 and cfg.epsilon(25) == pytest.approx(0.55) and cfg.epsilon(50) == pytest.approx(0.1) and cfg.epsilon(99) == pytest.approx(0.1)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.32
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.68 < lo < 0.75
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(1 - hi, abs=1e-9) and wilson(0, 0) == (0.0, 0.0)


def test_q_learning_learns_a_tiny_board_and_beats_random() -> None:
    env = MazeEnv(TINY, NO_HUNTER, layout_seed=1)
    messages: list[str] = []
    table, curve = train(env, QConfig(episodes=400, seed=0), progress=messages.append, log_every=100)
    assert len(messages) == 4 and messages[-1].startswith("episode    400")
    assert len(curve) == 400 and set(curve[0]) == {"episode", "epsilon", "return", "won", "steps", "crystals", "hits", "states"}
    seeds = range(500, 520)
    q = summarize(evaluate(env, QAgent(table, seed=0), seeds))
    rnd = summarize(evaluate(env, RandomAgent(0), seeds))
    best = summarize(evaluate(env, ShortestPathAgent(), seeds))
    assert q["win_rate"] >= 0.9 and q["win_rate"] > rnd["win_rate"] and q["mean_return"] > rnd["mean_return"]
    assert q["mean_steps"] <= 2 * best["mean_steps"]  # learned route is not wildly longer than the shortest path
    assert q["episodes"] == 20 and set(q["outcomes"]) <= {"won", "lost_time", "lost_health"}


def test_baselines_act_within_the_action_space() -> None:
    env = MazeEnv(RL_CONFIG, RL_PARAMS, layout_seed=LAYOUT_SEED)
    obs, info = env.reset(seed=0)
    for agent in (RandomAgent(1), ShortestPathAgent(), GreedyCollectorAgent(), ProfileAgent("cautious", env.params), QAgent(QTable(5))):
        a = agent.act(obs, info["state"])
        assert 0 <= a < env.n_actions
    rows = evaluate(env, GreedyCollectorAgent(), range(3))
    assert len(rows) == 3 and all(r["agent"] == "greedy_collector" and r["crystals"] <= len(env.crystals) for r in rows)


def test_pipeline_writes_every_artifact(tmp_path) -> None:
    env = MazeEnv(TINY, NO_HUNTER, layout_seed=1)
    report = run_pipeline(env, episodes=120, seed=0, eval_episodes=6, model_dir=tmp_path / "models", report_dir=tmp_path / "reports")
    for key in ("q_table", "meta", "curve", "eval", "results"):
        assert (tmp_path / report["artifacts"][key].replace(str(tmp_path) + "/", "")).exists()
    assert set(report["evaluation"]["agents"]) == {"random", "astar_exit", "greedy_collector", "cautious", "q_learning"}
    assert report["evaluation"]["distinct_hunter_starts"] == 0 and report["training"]["states_visited"] > 0
    assert len(pd.read_csv(tmp_path / "reports" / "rl_curve.csv")) == 120
    assert len(pd.read_csv(tmp_path / "reports" / "rl_eval.csv")) == 5 * 6
    saved = json.loads((tmp_path / "reports" / "rl_results.json").read_text())
    assert saved["env"]["board"] == "7x7" and saved["other_layout"]["layout_seed"] == 2
    assert "Q-learning" in format_report(report) and len(load_q_table(tmp_path / "models" / "q_table.npz")) == report["training"]["states_visited"]
    with pytest.raises(FileNotFoundError):
        load_q_table(tmp_path / "missing.npz")


def test_app_accepts_an_agent_and_a_layout_seed() -> None:
    from neuroquest.ui.app import run

    seen: list[int] = []

    def agent(state) -> Action:
        seen.append(state.tick)
        return Action.RIGHT

    config = GameConfig(width=11, height=9, loop_fraction=1.0, ticks_per_second=200)
    outcome = run(config, DifficultyParams(enemy_count=0, time_limit_s=2), seed=1, max_ticks=15, window=(320, 240), adaptive=False, agent=agent, layout_seed=LAYOUT_SEED)
    assert seen and seen[0] == 0 and outcome.value in {"playing", "won", "lost_time"}
