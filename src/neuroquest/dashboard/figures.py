"""One function per figure. Every figure is drawn from recorded data (dataset rows, the training
report, decision logs, A/B rows, episode logs); nothing is estimated here. Matplotlib's Agg backend
renders to PNG files without a display."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neuroquest.data.features import FEATURE_NAMES, live_features  # noqa: E402
from neuroquest.data.labels import LABELS  # noqa: E402
from neuroquest.ml.predict import PREDICT_AFTER_TICKS, PREDICT_EVERY_TICKS  # noqa: E402

COLOURS = {"aggressive": "#ff385c", "cautious": "#36e4ff", "explorer": "#18e0aa"}
INK, MUTED, BG, PANEL = "#e8eeff", "#8a94b8", "#0f1428", "#1a2240"

plt.rcParams.update(
    {
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#2e3860", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": "#2a3458", "grid.alpha": 0.6, "font.size": 10, "legend.facecolor": PANEL, "legend.edgecolor": "#2e3860",
    }
)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---------- data ----------


def fig_class_balance(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    counts = df["label"].value_counts().reindex(LABELS).fillna(0)
    axes[0].bar(counts.index, counts.values, color=[COLOURS[c] for c in counts.index])
    axes[0].set_title("Rows per label (all prefixes)")
    axes[0].set_ylabel("rows")
    full = df[df["prefix_ratio"] == 1.0]
    outcomes = full.groupby(["profile", "outcome"]).size().unstack(fill_value=0).reindex(LABELS)
    bottom = np.zeros(len(outcomes))
    palette = {"won": COLOURS["explorer"], "lost_health": COLOURS["aggressive"], "lost_time": "#ffb830", "playing": MUTED}
    for outcome in outcomes.columns:
        axes[1].bar(outcomes.index, outcomes[outcome], bottom=bottom, label=outcome, color=palette.get(outcome, MUTED))
        bottom += outcomes[outcome].values
    axes[1].set_title("Episode outcomes per profile")
    axes[1].legend()
    return _save(fig, out)


# ---------- model evaluation ----------


def fig_model_comparison(metrics: dict, out: Path) -> Path:
    names, cv, cv_std, test = [], [], [], []
    for name, m in metrics["models"].items():
        names.append(name.replace("_", " "))
        c = m.get("cv", {})
        cv.append(c.get("macro_f1_mean", np.nan))
        cv_std.append(c.get("macro_f1_std", 0.0))
        test.append(m["test"]["macro_f1"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - 0.2, cv, 0.4, yerr=cv_std, label="CV macro-F1 (train, 5 grouped folds)", color="#8f6eff", capsize=3)
    ax.bar(x + 0.2, test, 0.4, label="test macro-F1 (held-out seeds)", color=COLOURS["cautious"])
    sel = metrics["selected"]["model"]
    for i, name in enumerate(metrics["models"]):
        if name == sel:
            ax.annotate("selected", (i + 0.2, test[i] + 0.02), ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison (baselines and candidates)")
    ax.legend(loc="lower right")
    return _save(fig, out)


def fig_confusion(metrics: dict, out: Path) -> Path:
    sel = metrics["selected"]["model"]
    t = metrics["models"][sel]["test"]
    cm = np.array(t["confusion_matrix"])
    labels = t["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion matrix — {sel.replace('_', ' ')} (test)")
    ax = axes[1]
    per = t["per_class"]
    x = np.arange(len(labels))
    for k, (metric, offset) in enumerate((("precision", -0.25), ("recall", 0.0), ("f1", 0.25))):
        ax.bar(x + offset, [per[lab][metric] for lab in labels], 0.25, label=metric, color=["#8f6eff", COLOURS["cautious"], COLOURS["explorer"]][k])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class precision / recall / F1 (test)")
    ax.legend(loc="lower right")
    return _save(fig, out)


def fig_importance(metrics: dict, out: Path) -> Path:
    imp = metrics["selected"]["importances"]["permutation"]
    names = sorted(imp, key=lambda n: imp[n]["mean"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.barh(names, [imp[n]["mean"] for n in names], xerr=[imp[n]["std"] for n in names], color="#8f6eff", capsize=2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("drop in test macro-F1 when the feature is shuffled")
    ax.set_title("Permutation importance (held-out seeds)")
    return _save(fig, out)


def fig_coefficients(metrics: dict, out: Path) -> Path | None:
    coef = metrics["selected"]["importances"].get("coefficients")
    if not coef:
        return None
    fig, axes = plt.subplots(1, len(coef), figsize=(12, 4.2), sharey=True)
    for ax, (cls, row) in zip(np.atleast_1d(axes), coef.items()):
        names = FEATURE_NAMES
        vals = [row[n] for n in names]
        ax.barh(names, vals, color=[COLOURS[cls] if v >= 0 else MUTED for v in vals])
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(f"{cls}: standardised coefficients")
    return _save(fig, out)


def fig_accuracy_by_prefix(metrics: dict, out: Path) -> Path:
    acc = metrics["selected"]["accuracy_by_prefix"]
    xs = [float(k) for k in acc]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(xs, list(acc.values()), marker="o", color=COLOURS["explorer"], lw=2)
    for x, y in zip(xs, acc.values()):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(xs, [f"{int(x * 100)} %" for x in xs])
    ax.set_ylim(0.8, 1.0)
    ax.set_xlabel("share of the episode seen")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy grows with evidence")
    return _save(fig, out)


# ---------- sessions ----------


def prediction_timeline(log, predictor) -> list[dict]:
    """Re-run the saved model over a recorded episode on the live schedule: the same predictions the
    game would have shown (real model, real events)."""
    rows = []
    for tick in range(PREDICT_AFTER_TICKS, int(log.meta["ticks"]) + 1, PREDICT_EVERY_TICKS):
        pred = predictor.predict(live_features(log, tick))
        rows.append({"t": tick / log.meta["ticks_per_second"], "label": pred.label, "confidence": pred.confidence, **pred.proba})
    return rows


def fig_prediction_timeline(rows: list[dict], title: str, out: Path) -> Path | None:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for cls in LABELS:
        ax.plot([r["t"] for r in rows], [r.get(cls, 0.0) for r in rows], label=cls, color=COLOURS[cls], lw=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("seconds of play")
    ax.set_ylabel("class probability")
    ax.set_title(title)
    ax.legend(loc="center right")
    return _save(fig, out)


def fig_difficulty_timeline(decisions: list[dict], ticks_per_second: int, title: str, out: Path) -> Path | None:
    if not decisions:
        return None
    ts = [d["t"] / ticks_per_second for d in decisions]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, (key, label, colour, lim) in zip(axes, (("enemy_count", "hunters", COLOURS["aggressive"], (0.5, 5.5)), ("enemy_move_interval", "enemy move interval (ticks)", "#8f6eff", (1.5, 8.5)), ("time_limit_s", "time limit (s)", COLOURS["cautious"], (50, 250)))):
        ax.step([0] + ts, [decisions[0]["before"][key]] + [d["after"][key] for d in decisions], where="post", color=colour, lw=2)
        ax.set_ylabel(label)
        ax.set_ylim(*lim)
        for d, t in zip(decisions, ts):
            if d["before"] != d["after"] and d["after"][key] != d["before"][key]:
                ax.annotate(d["rule"].replace("_", " "), (t, d["after"][key]), textcoords="offset points", xytext=(4, 6), fontsize=8, color=INK)
    axes[-1].set_xlabel("seconds of play")
    axes[0].set_title(title)
    return _save(fig, out)


# ---------- experiments ----------


def fig_ab(rows: list[dict], out: Path) -> Path | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    profiles = [p for p in LABELS if p in set(df["profile"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(profiles))
    for ax, (metric, title, agg) in zip(axes, (("won", "Win rate", "mean"), ("damage", "Mean hits taken", "mean"), ("ticks", "Median episode length (ticks)", "median"))):
        for i, condition in enumerate(("fixed", "adaptive")):
            sub = df[df["condition"] == condition].groupby("profile")[metric].agg(agg).reindex(profiles)
            ax.bar(x + (i - 0.5) * 0.38, sub.values, 0.38, label=condition, color=[MUTED, COLOURS["cautious"]][i])
        ax.set_xticks(x, profiles)
        ax.set_title(title)
        if metric == "won":
            ax.set_ylim(0, 1)
    axes[0].legend()
    fig.suptitle(f"Fixed vs adaptive difficulty — {df.groupby(['profile', 'condition']).size().min()} seeded episodes per cell", color=INK, fontweight="bold")
    return _save(fig, out)


# ---------- reinforcement learning (Phase 9b) ----------

RL_COLOURS = {"random": MUTED, "astar_exit": "#8f6eff", "greedy_collector": COLOURS["aggressive"], "cautious": COLOURS["cautious"], "q_learning": COLOURS["explorer"]}
RL_LABELS = {"random": "random", "astar_exit": "A* to exit", "greedy_collector": "greedy collector", "cautious": "cautious profile", "q_learning": "Q-learning"}


def fig_rl_curve(curve: pd.DataFrame | None, out: Path, window: int = 250, references: dict[str, float] | None = None) -> Path | None:
    """Return, win rate and hits per training episode (moving means), ε dotted; dashed lines mark
    the baselines' evaluation mean return so the learner's progress has a scale."""
    if curve is None or len(curve) == 0:
        return None
    w = max(1, min(window, len(curve) // 4 or 1))
    roll = curve[["return", "won", "hits"]].rolling(w, min_periods=1).mean()
    x = curve["episode"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title, colour) in zip(axes, (("return", "Return per episode", COLOURS["explorer"]), ("won", "Win rate", COLOURS["cautious"]), ("hits", "Hits taken per episode", COLOURS["aggressive"]))):
        ax.plot(x, roll[key], color=colour, lw=2)
        ax.set_title(f"{title} (moving mean, {w} episodes)")
        ax.set_xlabel("training episode")
    for name, value in (references or {}).items():
        axes[0].axhline(value, ls="--", lw=1, color=RL_COLOURS.get(name, MUTED))
        axes[0].annotate(RL_LABELS.get(name, name), (float(x.iloc[-1]), value), ha="right", va="bottom", fontsize=8, color=RL_COLOURS.get(name, MUTED))
    axes[1].set_ylim(0, 1.02)
    eps_ax = axes[1].twinx()
    eps_ax.plot(x, curve["epsilon"], color=MUTED, lw=1, ls=":")
    eps_ax.set_ylabel("ε (dotted)", color=MUTED)
    eps_ax.set_ylim(0, 1.02)
    eps_ax.grid(False)
    return _save(fig, out)


def fig_rl_eval(agents: dict[str, dict], out: Path) -> Path | None:
    """Every agent on the same held-out seeds: win rate with its 95 % interval, return, crystals, hits."""
    if not agents:
        return None
    names = [n for n in RL_LABELS if n in agents] + [n for n in agents if n not in RL_LABELS]
    x = np.arange(len(names))
    colours = [RL_COLOURS.get(n, MUTED) for n in names]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, (key, title) in zip(axes, (("win_rate", "Win rate (95 % CI)"), ("mean_return", "Mean return"), ("mean_crystals", "Crystals collected"), ("mean_hits", "Hits taken"))):
        vals = [agents[n][key] for n in names]
        if key == "win_rate":
            lo = [agents[n][key] - agents[n]["win_rate_ci95"][0] for n in names]
            hi = [agents[n]["win_rate_ci95"][1] - agents[n][key] for n in names]
            ax.bar(x, vals, color=colours, yerr=[lo, hi], capsize=3, ecolor=INK)
            ax.set_ylim(0, 1.05)
        else:
            ax.bar(x, vals, color=colours)
        ax.set_xticks(x, [RL_LABELS.get(n, n) for n in names], rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
    n = next(iter(agents.values()))["episodes"]
    fig.suptitle(f"Agents on the training board — {n} held-out episodes each", color=INK, fontweight="bold")
    return _save(fig, out)
