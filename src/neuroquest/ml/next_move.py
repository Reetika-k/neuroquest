"""The next-move model (Phase 7b): predict the player's next action from what the game shows.

Each row is one tick of one episode: what the player could see at that moment, and, as the
target, the action they took next. The features are the last three actions, which neighbouring
cells are open or already visited, where the nearest crystal, the portal and the nearest hunter
are (direction and distance), the shortest-path direction to the nearest crystal and to the
portal, crystals left, health and whether a hunter is near. Rows are built by *replaying* each
episode log through the real ``step()``; the seeds make the replay exact, and an episode whose
replay does not reproduce the log's outcome, tick count and score is dropped and counted.

Evaluation uses a **temporal split**: episodes are ordered by seed, the order in which they were
generated; the last 20 % are the test set. The model is chosen on the last 20 % of the *training*
episodes (a validation set), then refitted on all training rows and evaluated once on the test
rows. Baselines: the majority direction, repeating the last move, and a hand-written greedy rule
(shortest path to the nearest crystal, otherwise to the portal).
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuroquest.ai.astar import a_star
from neuroquest.ai.bfs import bfs_nearest
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.maze import Cell, manhattan
from neuroquest.core.rules import NEAR_DISTANCE, new_game, step
from neuroquest.core.state import RECENT_ACTIONS, Action, GameState
from neuroquest.data.events import EpisodeLog
from neuroquest.ml.evaluate import classification_metrics, importances
from neuroquest.ml.predict import ModelMismatch

MOVE_ACTIONS: tuple[str, ...] = ("up", "down", "left", "right", "stay")
DIRECTIONS: dict[str, Cell] = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
FAR = 40  # distance clip for the crystal and the portal
FAR_ENEMY = 20
MOVE_FEATURES: list[str] = (
    [f"prev{k}_{a}" for k in range(1, RECENT_ACTIONS + 1) for a in MOVE_ACTIONS]
    + [f"open_{d}" for d in DIRECTIONS]
    + [f"visited_{d}" for d in DIRECTIONS]
    + [f"{what}_{part}" for what in ("reward", "exit", "enemy") for part in ("dr", "dc", "dist")]
    + ["rewards_left", "health", "near_enemy"]
    + [f"path_reward_{d}" for d in DIRECTIONS]
    + [f"path_exit_{d}" for d in DIRECTIONS]
)
META_COLUMNS = ["episode_id", "profile", "seed", "t", "target"]

NEXT_MOVE_MODEL_PATH = Path("models/next_move.joblib")
NEXT_MOVE_META_PATH = Path("models/next_move.meta.json")
NEXT_MOVE_REPORT_PATH = Path("reports/next_move_metrics.json")
MOVES_CSV = Path("data/processed/moves.csv")


def _sign(x: int) -> float:
    return float((x > 0) - (x < 0))


def _direction(a: Cell, b: Cell) -> str:
    return {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}[(b[0] - a[0], b[1] - a[1])]


def move_features(state: GameState) -> dict[str, float]:
    """The feature row for the player's *next* decision, from the state before it."""
    pos = state.player.pos
    f: dict[str, float] = {}
    recent = [a.value for a in state.recent_actions]
    for k in range(1, RECENT_ACTIONS + 1):
        last = recent[-k] if len(recent) >= k else None
        for a in MOVE_ACTIONS:
            f[f"prev{k}_{a}"] = float(last == a)
    for name, (dr, dc) in DIRECTIONS.items():
        cell = (pos[0] + dr, pos[1] + dc)
        f[f"open_{name}"] = float(state.maze.is_open(cell))
    for name, (dr, dc) in DIRECTIONS.items():
        cell = (pos[0] + dr, pos[1] + dc)
        f[f"visited_{name}"] = float(state.maze.is_open(cell) and cell in state.visited)
    nearest_reward = min(state.rewards, key=lambda c: manhattan(pos, c)) if state.rewards else None
    nearest_enemy = min((e.pos for e in state.enemies), key=lambda c: manhattan(pos, c)) if state.enemies else None
    for what, target, clip in (("reward", nearest_reward, FAR), ("exit", state.exit, FAR), ("enemy", nearest_enemy, FAR_ENEMY)):
        if target is None:
            f[f"{what}_dr"], f[f"{what}_dc"], f[f"{what}_dist"] = 0.0, 0.0, float(clip)
        else:
            dr, dc = target[0] - pos[0], target[1] - pos[1]
            f[f"{what}_dr"], f[f"{what}_dc"], f[f"{what}_dist"] = _sign(dr), _sign(dc), float(min(clip, abs(dr) + abs(dc)))
    f["rewards_left"] = float(len(state.rewards))
    f["health"] = float(state.player.health)
    f["near_enemy"] = float(nearest_enemy is not None and manhattan(pos, nearest_enemy) <= NEAR_DISTANCE)
    reward_path = bfs_nearest(state.maze, pos, state.rewards) if state.rewards else None
    reward_dir = _direction(pos, reward_path[1]) if reward_path is not None and len(reward_path) > 1 else None
    for name in DIRECTIONS:
        f[f"path_reward_{name}"] = float(reward_dir == name)
    exit_path = a_star(state.maze, pos, state.exit)
    exit_dir = _direction(pos, exit_path[1]) if exit_path is not None and len(exit_path) > 1 else None
    for name in DIRECTIONS:
        f[f"path_exit_{name}"] = float(exit_dir == name)
    return f


# ---------- rows from logs ----------


def actions_from_log(log: EpisodeLog) -> list[str]:
    """The action taken on every tick 1..ticks: a move or bump event names it; otherwise the player stayed."""
    by_tick: dict[int, str] = {}
    for e in log.events:
        if e.type in ("move", "bump"):
            by_tick[e.t] = str(e.detail["action"])
    return [by_tick.get(t, "stay") for t in range(1, int(log.meta["ticks"]) + 1)]


def replay_rows(log: EpisodeLog) -> tuple[list[dict], bool]:
    """Replay the episode through ``step()`` and emit one row per tick; ``ok`` says the replay
    reproduced the log's outcome, tick count and score (if not, the rows must not be trusted)."""
    config = GameConfig(**log.meta["config"])
    params = DifficultyParams(**log.meta["params"])
    seed = int(log.meta["seed"])
    state = new_game(config, params, seed)
    rng = random.Random(seed + 10_007)  # the harness's game rng; unused by A* hunters
    rows: list[dict] = []
    for t, name in enumerate(actions_from_log(log), start=1):
        if state.outcome.is_over:
            break
        rows.append({"episode_id": log.meta["episode_id"], "profile": log.meta.get("profile", ""), "seed": seed, "t": t, "target": name, **move_features(state)})
        state, _ = step(state, Action(name), params, rng)
    ok = state.tick == int(log.meta["ticks"]) and state.outcome.value == log.meta["outcome"] and state.score == int(log.meta["score"])
    return rows, ok


def build_moves(raw_dir: Path, out_csv: Path = MOVES_CSV) -> tuple[pd.DataFrame, dict]:
    """Every replayable simulated log → ``moves.csv``. Human and adaptive episodes are skipped: their
    parameters changed mid-game, so an exact replay is not available from the log alone."""
    report: dict = {"episodes": 0, "skipped": [], "replay_failed": [], "rows": 0}
    rows: list[dict] = []
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        log = EpisodeLog.load(path)
        if log.meta.get("profile") == "human" or log.meta.get("adaptive"):
            report["skipped"].append(path.name)
            continue
        episode_rows, ok = replay_rows(log)
        if not ok:
            report["replay_failed"].append(path.name)
            continue
        rows += episode_rows
        report["episodes"] += 1
    df = pd.DataFrame(rows, columns=META_COLUMNS + MOVE_FEATURES)
    report["rows"] = int(len(df))
    report["targets"] = {k: int(v) for k, v in df["target"].value_counts().items()} if len(df) else {}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df, report


# ---------- split, baselines, models ----------


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier episodes (lower seeds, generated first) train; the last ``test_fraction`` of seeds test."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    seeds = sorted(df["seed"].unique())
    cut = seeds[max(1, int(round(len(seeds) * (1 - test_fraction))))] if len(seeds) > 1 else seeds[0] + 1
    train = df[df["seed"] < cut].reset_index(drop=True)
    test = df[df["seed"] >= cut].reset_index(drop=True)
    return train, test


def _onehot_name(row, prefix: str, names: Sequence[str]) -> str | None:
    for name in names:
        if row[f"{prefix}_{name}"] >= 0.5:
            return name
    return None


def repeat_last_predictions(df: pd.DataFrame, fallback: str) -> list[str]:
    return [_onehot_name(r, "prev1", MOVE_ACTIONS) or fallback for _, r in df.iterrows()]


def greedy_predictions(df: pd.DataFrame) -> list[str]:
    """Shortest path to the nearest crystal while any is left, otherwise to the portal; stay if no path."""
    out = []
    for _, r in df.iterrows():
        d = _onehot_name(r, "path_reward", DIRECTIONS) if r["rewards_left"] > 0 else None
        out.append(d or _onehot_name(r, "path_exit", DIRECTIONS) or "stay")
    return out


def build_next_move_models(seed: int, fast: bool = False) -> dict[str, object]:
    """Fixed hyperparameters (no tuning on the test rows). The forest is deliberately bounded:
    an unbounded 300-tree forest on 70,000 rows pickled to 216 MB and predicted in 19 ms; with
    120 trees, depth 12 and 10 rows per leaf it is 6.6 MB compressed and 7 ms, for −0.007
    validation macro-F1 (measured on the validation tail, not the test rows)."""
    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=1.0))]),
        "random_forest": RandomForestClassifier(n_estimators=60 if fast else 120, max_depth=12, min_samples_leaf=10, random_state=seed, n_jobs=-1),
    }


def _per_profile_accuracy(df: pd.DataFrame, pred: Sequence[str]) -> dict[str, float]:
    correct = (df["target"].to_numpy() == np.asarray(pred))
    return {str(p): float(correct[df["profile"].to_numpy() == p].mean()) for p in sorted(df["profile"].unique())}


def train_next_move(
    data_csv: Path = MOVES_CSV,
    model_dir: Path = Path("models"),
    report_path: Path = NEXT_MOVE_REPORT_PATH,
    seed: int = 42,
    fast: bool = False,
    test_fraction: float = 0.2,
) -> dict:
    """Temporal split → baselines → candidates chosen on a validation tail of the training rows →
    refit → one evaluation on the test rows → save the chosen model, its metadata and the report."""
    df = pd.read_csv(data_csv)
    df = df[df["target"].isin(MOVE_ACTIONS)].reset_index(drop=True)
    train_df, test_df = temporal_split(df, test_fraction)
    fit_df, val_df = temporal_split(train_df, test_fraction)
    labels = list(MOVE_ACTIONS)
    X = lambda d: d[MOVE_FEATURES].to_numpy()  # noqa: E731
    y_tr, y_te = train_df["target"].to_numpy(), test_df["target"].to_numpy()

    report: dict = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "data": {"path": str(data_csv), "rows": int(len(df)), "episodes": int(df["episode_id"].nunique()), "targets": {k: int(v) for k, v in df["target"].value_counts().items()}},
        "split": {
            "by": "seed order (temporal)", "test_fraction": test_fraction,
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "train_seeds": [int(train_df["seed"].min()), int(train_df["seed"].max())], "test_seeds": [int(test_df["seed"].min()), int(test_df["seed"].max())],
            "validation_rows": int(len(val_df)),
        },
        "features": list(MOVE_FEATURES),
        "models": {},
    }

    majority = DummyClassifier(strategy="most_frequent").fit(X(train_df), y_tr)
    majority_label = str(majority.predict(X(test_df)[:1])[0])
    baselines = {
        "majority": (f"always '{majority_label}' (most frequent action in the training rows)", list(majority.predict(X(test_df)))),
        "repeat_last": ("repeat the previous action", repeat_last_predictions(test_df, majority_label)),
        "greedy": ("shortest path to the nearest crystal, else to the portal", greedy_predictions(test_df)),
    }
    for name, (kind, pred) in baselines.items():
        report["models"][name] = {"kind": kind, "test": classification_metrics(y_te, pred, labels), "per_profile_accuracy": _per_profile_accuracy(test_df, pred)}

    candidates = build_next_move_models(seed, fast)
    validation: dict[str, float] = {}
    for name, model in candidates.items():
        model.fit(X(fit_df), fit_df["target"].to_numpy())
        validation[name] = classification_metrics(val_df["target"].to_numpy(), model.predict(X(val_df)), labels)["macro_f1"]
    chosen = max(candidates, key=lambda n: validation[n])

    fitted: dict[str, object] = {}
    for name, model in build_next_move_models(seed, fast).items():
        model.fit(X(train_df), y_tr)
        fitted[name] = model
        pred = model.predict(X(test_df))
        report["models"][name] = {
            "kind": type(model.steps[-1][1] if hasattr(model, "steps") else model).__name__,
            "validation_macro_f1": validation[name],
            "test": classification_metrics(y_te, pred, labels),
            "per_profile_accuracy": _per_profile_accuracy(test_df, pred),
        }
    model = fitted[chosen]
    sample = test_df.sample(n=min(3000, len(test_df)), random_state=seed) if len(test_df) > 0 else test_df
    report["selected"] = {
        "model": chosen,
        "reason": "highest macro-F1 on the validation tail of the training episodes (the test episodes were not used for selection)",
        "importances": importances(model, X(sample), sample["target"].to_numpy(), MOVE_FEATURES, seed, n_repeats=2 if fast else 5),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = model_dir / NEXT_MOVE_MODEL_PATH.name, model_dir / NEXT_MOVE_META_PATH.name
    joblib.dump(model, model_path, compress=3)  # tree ensembles compress about 3×; the file must stay far below GitHub's 100 MB limit
    meta = {
        "model": chosen, "features": list(MOVE_FEATURES), "classes": [str(c) for c in model.classes_],
        "trained_at": report["trained_at"], "sklearn": report["sklearn"], "data": report["data"], "split": report["split"],
        "test": {k: report["models"][chosen]["test"][k] for k in ("accuracy", "macro_f1")},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    report["artifacts"] = {"model": str(model_path), "meta": str(meta_path)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def format_next_move_report(report: dict) -> str:
    s = report["split"]
    lines = [
        f"temporal split: train seeds {s['train_seeds'][0]}–{s['train_seeds'][1]} ({s['train_rows']} rows), test seeds {s['test_seeds'][0]}–{s['test_seeds'][1]} ({s['test_rows']} rows)",
        f"  {'model':<22}{'validation F1':>15}{'test acc':>10}{'test macro-F1':>15}  per-profile accuracy",
    ]
    for name, m in report["models"].items():
        val = m.get("validation_macro_f1")
        per = "  ".join(f"{k} {v:.2f}" for k, v in m["per_profile_accuracy"].items())
        lines.append(f"  {name:<22}{(f'{val:.3f}' if val is not None else '-'):>15}{m['test']['accuracy']:>10.3f}{m['test']['macro_f1']:>15.3f}  {per}")
    lines.append(f"selected: {report['selected']['model']} ({report['selected']['reason']})")
    perm = report["selected"]["importances"]["permutation"]
    top = sorted(perm.items(), key=lambda kv: -kv[1]["mean"])[:5]
    lines.append("top permutation importances: " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in top))
    return "\n".join(lines)


# ---------- live use ----------


class NextMovePredictor:
    """Load the saved next-move model and predict the player's next action from a game state."""

    def __init__(self, model, meta: dict) -> None:
        if list(meta.get("features", [])) != list(MOVE_FEATURES):
            raise ModelMismatch("next-move model features differ from the current code; retrain with `neuroquest train-next-move`")
        self.model, self.meta = model, meta
        self.classes = [str(c) for c in model.classes_]
        # Live use predicts one row at a time: a forest trained with n_jobs=-1 would spin up a thread
        # pool for every call (about 100 ms each, measured); single-threaded it takes a few ms.
        final = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(final, "n_jobs"):
            final.n_jobs = 1

    @classmethod
    def load(cls, model_path: Path = NEXT_MOVE_MODEL_PATH, meta_path: Path = NEXT_MOVE_META_PATH) -> NextMovePredictor:
        return cls(joblib.load(model_path), json.loads(Path(meta_path).read_text()))

    def predict(self, state: GameState) -> tuple[str, dict[str, float]]:
        f = move_features(state)
        proba = self.model.predict_proba([[f[name] for name in MOVE_FEATURES]])[0]
        best = int(proba.argmax())
        return self.classes[best], {c: float(p) for c, p in zip(self.classes, proba)}

    def predict_cell(self, state: GameState) -> Cell:
        """The cell the player is predicted to occupy next (their own cell for 'stay' or a bump)."""
        action, _ = self.predict(state)
        if action == "stay":
            return state.player.pos
        dr, dc = DIRECTIONS[action]
        cell = (state.player.pos[0] + dr, state.player.pos[1] + dc)
        return cell if state.maze.is_open(cell) else state.player.pos
