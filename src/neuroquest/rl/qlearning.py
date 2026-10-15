"""Tabular Q-learning, the training loop, evaluation and the Q-table's file format.

Q-learning keeps one number per (observation, action): the return the agent expects if it takes
that action and then behaves greedily. After every tick the entry moves a fraction ``alpha``
toward the better estimate ``reward + gamma × max_a Q(next, a)`` (just ``reward`` when the
episode ended). Exploration is ε-greedy: with probability ε a random action, otherwise the best
known one; ε falls linearly from ``epsilon_start`` to ``epsilon_end`` over the first
``explore_fraction`` of the episodes and stays there.

Nothing here knows about mazes: ``train`` works on anything with the ``MazeEnv`` API.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Obs = tuple[int, ...]


class QTable:
    """``dict`` from observation tuple to a list of action values (plain lists: faster than tiny arrays)."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.q: dict[Obs, list[float]] = {}

    def __len__(self) -> int:
        return len(self.q)

    def values(self, obs: Obs) -> list[float]:
        """The action values for ``obs``; zeros if it has never been seen (no entry is created)."""
        return self.q.get(obs) or [0.0] * self.n_actions

    def greedy(self, obs: Obs, rng: random.Random) -> int:
        row = self.q.get(obs)
        if row is None:
            return rng.randrange(self.n_actions)
        best = max(row)
        choices = [i for i, v in enumerate(row) if v == best]
        return choices[0] if len(choices) == 1 else rng.choice(choices)

    def update(self, obs: Obs, action: int, reward: float, next_obs: Obs, terminal: bool, alpha: float, gamma: float) -> float:
        """One Q-learning step; returns the temporal-difference error."""
        row = self.q.get(obs)
        if row is None:
            row = self.q[obs] = [0.0] * self.n_actions
        target = reward if terminal else reward + gamma * max(self.q.get(next_obs) or (0.0,))
        error = target - row[action]
        row[action] += alpha * error
        return error

    # ---------- files ----------

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.q.keys()), dtype=np.int16).reshape(len(self.q), -1)
        values = np.array(list(self.q.values()), dtype=np.float32).reshape(len(self.q), self.n_actions)
        np.savez_compressed(path, keys=keys, values=values, n_actions=self.n_actions)
        return path

    @classmethod
    def load(cls, path: Path) -> QTable:
        data = np.load(path)
        table = cls(int(data["n_actions"]))
        for key, row in zip(data["keys"], data["values"]):
            table.q[tuple(int(k) for k in key)] = [float(v) for v in row]
        return table


@dataclass(frozen=True)
class QConfig:
    episodes: int = 20_000
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    explore_fraction: float = 0.8  # ε reaches epsilon_end after this share of the episodes
    seed: int = 0  # exploration randomness; episode seeds are 0, 1, 2, … (the hunter's start cell)

    def epsilon(self, episode: int) -> float:
        horizon = max(1, int(self.episodes * self.explore_fraction))
        frac = min(1.0, episode / horizon)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac


def train(env, cfg: QConfig, *, progress: Callable[[str], None] | None = None, log_every: int = 1000) -> tuple[QTable, list[dict]]:
    """Run ``cfg.episodes`` ε-greedy episodes on ``env``; return the table and one curve row per episode."""
    rng = random.Random(cfg.seed)
    table = QTable(env.n_actions)
    curve: list[dict] = []
    t0 = time.perf_counter()
    for i in range(cfg.episodes):
        eps = cfg.epsilon(i)
        obs, _ = env.reset(seed=i)
        total, steps, hits, done = 0.0, 0, 0, False
        while not done:
            action = rng.randrange(env.n_actions) if rng.random() < eps else table.greedy(obs, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            table.update(obs, action, reward, next_obs, terminated, cfg.alpha, cfg.gamma)
            obs, total, steps, hits = next_obs, total + reward, steps + 1, hits + info["hits"]
            done = terminated or truncated
        curve.append({"episode": i, "epsilon": round(eps, 4), "return": round(total, 3), "won": int(info["won"]), "steps": steps, "crystals": info["crystals"], "hits": hits, "states": len(table)})
        if progress and (i + 1) % log_every == 0:
            recent = curve[-log_every:]
            progress(f"episode {i + 1:>6}  ε={eps:.2f}  win rate {sum(r['won'] for r in recent) / len(recent):.2f}  mean return {sum(r['return'] for r in recent) / len(recent):+.2f}  states {len(table)}  {time.perf_counter() - t0:.0f}s")
    return table, curve


def evaluate(env, agent, seeds: Iterable[int]) -> list[dict]:
    """Play one episode per seed with ``agent`` and return one row per episode."""
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        total, steps, hits, done = 0.0, 0, 0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs, info["state"]))
            total, steps, hits = total + reward, steps + 1, hits + info["hits"]
            done = terminated or truncated
        rows.append({"agent": agent.name, "seed": seed, "return": round(total, 3), "won": int(info["won"]), "outcome": info["outcome"], "steps": steps, "crystals": info["crystals"], "hits": hits})
    return rows


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion (well-behaved near 0 and 1)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(r["won"] for r in rows)
    lo, hi = wilson(wins, n)
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    mean = lambda key: (sum(r[key] for r in rows) / n) if n else 0.0  # noqa: E731
    return {
        "episodes": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "win_rate_ci95": [round(lo, 4), round(hi, 4)],
        "mean_return": round(mean("return"), 3),
        "mean_crystals": round(mean("crystals"), 3),
        "mean_hits": round(mean("hits"), 3),
        "mean_steps": round(mean("steps"), 1),
        "outcomes": outcomes,
    }
