"""The Phase 9b experiment end to end: train on the board, save the table, evaluate the learned
policy and every baseline on the same held-out seeds, check a board it never saw, write files.

Outputs: ``models/q_table.npz`` + ``q_table.meta.json``, ``reports/rl_curve.csv`` (one row per
training episode), ``reports/rl_eval.csv`` (one row per evaluation episode and agent) and
``reports/rl_results.json`` (the summary the dashboard and the report read).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from neuroquest.rl.agents import GreedyCollectorAgent, ProfileAgent, QAgent, RandomAgent, ShortestPathAgent
from neuroquest.rl.env import MazeEnv
from neuroquest.rl.qlearning import QConfig, QTable, evaluate, summarize, train

AGENT_LABELS = {"random": "random", "astar_exit": "A* to exit", "greedy_collector": "greedy collector", "cautious": "cautious profile", "q_learning": "Q-learning"}


def run_pipeline(
    env: MazeEnv | None = None,
    *,
    episodes: int = 20_000,
    seed: int = 0,
    eval_episodes: int = 100,
    eval_seed0: int = 100_000,
    model_dir: Path = Path("models"),
    report_dir: Path = Path("reports"),
    progress: Callable[[str], None] | None = None,
) -> dict:
    env = env or MazeEnv()
    cfg = QConfig(episodes=episodes, seed=seed)
    t0 = time.perf_counter()
    table, curve = train(env, cfg, progress=progress)
    seconds = time.perf_counter() - t0

    report_dir.mkdir(parents=True, exist_ok=True)
    curve_path = report_dir / "rl_curve.csv"
    pd.DataFrame(curve).to_csv(curve_path, index=False)
    table_path = table.save(model_dir / "q_table.npz")
    tail = curve[-max(1, min(500, episodes // 10)) :]
    training = {
        **asdict(cfg),
        "seconds": round(seconds, 1),
        "states_visited": len(table),
        "final_window": len(tail),
        "final_window_win_rate": round(sum(r["won"] for r in tail) / len(tail), 4),
        "final_window_mean_return": round(sum(r["return"] for r in tail) / len(tail), 3),
    }
    meta_path = model_dir / "q_table.meta.json"
    meta_path.write_text(json.dumps({"env": env.describe(), "training": training}, indent=2))

    seeds = range(eval_seed0, eval_seed0 + eval_episodes)
    agents = [RandomAgent(seed), ShortestPathAgent(), GreedyCollectorAgent(), ProfileAgent("cautious", env.params, seed), QAgent(table, seed=seed)]
    rows_all: list[dict] = []
    results: dict[str, dict] = {}
    for agent in agents:
        rows = evaluate(env, agent, seeds)
        rows_all += rows
        results[agent.name] = summarize(rows)
    eval_path = report_dir / "rl_eval.csv"
    pd.DataFrame(rows_all).to_csv(eval_path, index=False)
    starts = {env.reset(seed=s)[1]["state"].enemies[0].pos for s in seeds} if env.params.enemy_count else set()

    other_env = MazeEnv(env.config, env.params, env.layout_seed + 1, env.reward_spec)
    other = {
        "layout_seed": other_env.layout_seed,
        "q_learning": summarize(evaluate(other_env, QAgent(table, seed=seed), seeds)),
        "random": summarize(evaluate(other_env, RandomAgent(seed), seeds)),
    }

    report = {
        "env": env.describe(),
        "training": training,
        "evaluation": {"seed_start": eval_seed0, "episodes": eval_episodes, "distinct_hunter_starts": len(starts), "agents": results},
        "other_layout": other,
        "artifacts": {"q_table": str(table_path), "meta": str(meta_path), "curve": str(curve_path), "eval": str(eval_path)},
    }
    results_path = report_dir / "rl_results.json"
    results_path.write_text(json.dumps(report, indent=2))
    report["artifacts"]["results"] = str(results_path)
    return report


def load_q_table(path: Path = Path("models/q_table.npz")) -> QTable:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `neuroquest rl-train` first")
    return QTable.load(path)


def format_report(report: dict) -> str:
    e, t, ev = report["env"], report["training"], report["evaluation"]
    lines = [
        f"board {e['board']} · {e['open_cells']} open cells · {e['crystals']} crystals · {e['hunters']} {e['hunter_kind']} hunter every {e['hunter_move_interval']} ticks · {e['time_limit_ticks']} ticks",
        f"trained {t['episodes']} episodes in {t['seconds']:.0f}s (α={t['alpha']}, γ={t['gamma']}, ε {t['epsilon_start']}→{t['epsilon_end']}); {t['states_visited']} observations visited; last {t['final_window']} training episodes: win rate {t['final_window_win_rate']:.2f}, mean return {t['final_window_mean_return']:+.2f}",
        f"evaluation: {ev['episodes']} held-out seeds from {ev['seed_start']} ({ev['distinct_hunter_starts']} distinct hunter starts)",
        f"  {'agent':<18}{'win rate [95% CI]':<22}{'return':>8}{'crystals':>10}{'hits':>7}{'steps':>7}",
    ]
    for name, a in ev["agents"].items():
        lo, hi = a["win_rate_ci95"]
        lines.append(f"  {AGENT_LABELS.get(name, name):<18}{a['win_rate']:.2f} [{lo:.2f}, {hi:.2f}]      {a['mean_return']:>+8.2f}{a['mean_crystals']:>10.2f}{a['mean_hits']:>7.2f}{a['mean_steps']:>7.0f}")
    o = report["other_layout"]
    lines.append(f"other board (layout seed {o['layout_seed']}, never trained on): Q-learning win rate {o['q_learning']['win_rate']:.2f}, random {o['random']['win_rate']:.2f}")
    return "\n".join(lines)
