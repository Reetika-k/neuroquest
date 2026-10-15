"""Fixed versus adaptive difficulty, measured. The same profiles play the same seeded mazes twice,
once with the difficulty fixed at the defaults and once with the controller on, and the outcomes
are tabulated. The numbers are reported as they are; the interpretation lives in docs/08."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from neuroquest.ai.benchmark import write_csv
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.ml.predict import BehaviourPredictor
from neuroquest.sim.harness import run_episode
from neuroquest.sim.profiles import PROFILES

CONDITIONS = ("fixed", "adaptive")


def ab_compare(
    profiles: Sequence[str],
    episodes: int,
    config: GameConfig,
    params: DifficultyParams,
    seed0: int,
    predictor: BehaviourPredictor | None = None,
    out_csv: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    predictor = predictor or BehaviourPredictor.load()
    rows: list[dict] = []
    for name in profiles:
        for i in range(episodes):
            seed = seed0 + i
            for condition in CONDITIONS:
                log = run_episode(PROFILES[name], config, params, seed, profile_name=name, adaptive=(condition == "adaptive"), predictor=predictor)
                m = log.meta
                decisions = m.get("decisions", [])
                rows.append(
                    {
                        "profile": name,
                        "seed": seed,
                        "condition": condition,
                        "outcome": m["outcome"],
                        "won": m["outcome"] == "won",
                        "ticks": m["ticks"],
                        "score": m["score"],
                        "damage": sum(1 for e in log.events if e.type == "damage"),
                        "rewards_collected": m["rewards_collected"],
                        "decisions": len(decisions),
                        "changes": sum(1 for d in decisions if d["before"] != d["after"]),
                        "rules_fired": " ".join(f"{r}:{n}" for r, n in sorted(Counter(d["rule"] for d in decisions if d["before"] != d["after"]).items())),
                        "final_enemy_count": m["params_final"]["enemy_count"],
                        "final_enemy_move_interval": m["params_final"]["enemy_move_interval"],
                        "final_time_limit_s": m["params_final"]["time_limit_s"],
                    }
                )
        if progress:
            progress(f"{name:<11}{episodes} seeds × 2 conditions done")
    if out_csv is not None:
        write_csv(rows, out_csv)
    return rows


def summarize_ab(rows: list[dict]) -> str:
    lines = [f"  {'profile':<11}{'condition':<10}{'n':>4}{'win':>6}{'lost hp':>9}{'lost time':>11}{'median ticks':>14}{'mean damage':>13}{'mean score':>12}{'changes/ep':>12}  rules fired (changes)"]
    for name in sorted({r["profile"] for r in rows}):
        for condition in CONDITIONS:
            sub = [r for r in rows if r["profile"] == name and r["condition"] == condition]
            if not sub:
                continue
            outcomes = Counter(r["outcome"] for r in sub)
            rules: Counter = Counter()
            for r in sub:
                for item in str(r.get("rules_fired", "")).split():
                    rule, n = item.rsplit(":", 1)
                    rules[rule] += int(n)
            lines.append(
                f"  {name:<11}{condition:<10}{len(sub):>4}{outcomes.get('won', 0) / len(sub):>6.0%}{outcomes.get('lost_health', 0):>9}{outcomes.get('lost_time', 0):>11}"
                f"{statistics.median(r['ticks'] for r in sub):>14.0f}{statistics.mean(r['damage'] for r in sub):>13.2f}"
                f"{statistics.mean(r['score'] for r in sub):>12.1f}{statistics.mean(r['changes'] for r in sub):>12.2f}  "
                + ", ".join(f"{k} ×{v}" for k, v in rules.most_common())
            )
    return "\n".join(lines)


def deltas(rows: list[dict]) -> dict[str, dict[str, float]]:
    """adaptive − fixed per profile for win rate, median ticks, mean damage."""
    out: dict[str, dict[str, float]] = {}
    for name in sorted({r["profile"] for r in rows}):
        fixed = [r for r in rows if r["profile"] == name and r["condition"] == "fixed"]
        adapt = [r for r in rows if r["profile"] == name and r["condition"] == "adaptive"]
        if fixed and adapt:
            out[name] = {
                "win_rate": statistics.mean(r["won"] for r in adapt) - statistics.mean(r["won"] for r in fixed),
                "median_ticks": statistics.median(r["ticks"] for r in adapt) - statistics.median(r["ticks"] for r in fixed),
                "mean_damage": statistics.mean(r["damage"] for r in adapt) - statistics.mean(r["damage"] for r in fixed),
            }
    return out
