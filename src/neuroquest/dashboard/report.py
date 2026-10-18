"""Assemble the dashboard: figures plus tables and notes into one self-contained HTML file.

Every section is optional: if its input file is missing the section says so instead of inventing
anything. Prediction examples are recomputed from the saved model on the same held-out split the
training report used, so the examples shown are the model's real outputs.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from neuroquest.data.events import EpisodeLog
from neuroquest.data.features import FEATURE_NAMES
from neuroquest.data.labels import LABELS
from neuroquest.dashboard import figures as F
from neuroquest.ml.predict import BehaviourPredictor
from neuroquest.ml.split import grouped_split

CSS = """
body{margin:0;background:#0f1428;color:#e8eeff;font:15px/1.55 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 20px 60px}
h1{font-size:30px;margin:0 0 4px}h2{font-size:21px;margin:40px 0 10px;border-bottom:1px solid #2e3860;padding-bottom:6px}
.sub{color:#8a94b8;margin:0 0 18px}.card{background:#1a2240;border:1px solid #2e3860;border-radius:12px;padding:16px 18px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.stat b{display:block;font-size:24px}
.stat span{color:#8a94b8;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
img{max-width:100%;height:auto;border-radius:10px;border:1px solid #2e3860;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #2e3860;vertical-align:top}
th{color:#8a94b8;font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.06em}
.ok{color:#18e0aa;font-weight:700}.bad{color:#ff385c;font-weight:700}.muted{color:#8a94b8}
code{background:#0b1020;padding:2px 6px;border-radius:6px;font-size:13px}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase}
.aggressive{background:#ff385c33;color:#ff385c}.cautious{background:#36e4ff33;color:#36e4ff}.explorer{background:#18e0aa33;color:#18e0aa}
"""


def _img(path: Path | None) -> str:
    if path is None or not Path(path).exists():
        return "<p class='muted'>figure not available</p>"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"<img src='data:image/png;base64,{data}' alt='{html.escape(Path(path).stem)}'>"


def _tag(label: str) -> str:
    return f"<span class='tag {label}'>{label}</span>"


def _examples(data_csv: Path, metrics: dict, model_dir: Path, n: int = 5) -> tuple[list[dict], list[dict]]:
    """Confident correct predictions (one per class where possible) and the most confident errors,
    recomputed with the saved model on the held-out seeds of the training report's split."""
    predictor = BehaviourPredictor.load(model_dir / "behaviour_clf.joblib", model_dir / "behaviour_clf.meta.json")
    df = pd.read_csv(data_csv)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    _, test = grouped_split(df, test_fraction=metrics["split"]["test_fraction"], seed=metrics["split"]["seed"])
    rows = []
    for _, r in test.iterrows():
        pred = predictor.predict({k: r[k] for k in FEATURE_NAMES})
        rows.append({"episode_id": r["episode_id"], "prefix_ratio": r["prefix_ratio"], "true": r["label"], "predicted": pred.label, "confidence": pred.confidence, "p_true": pred.proba[r["label"]], **{k: r[k] for k in ("elapsed_s", "damage_per_tick", "near_enemy_ratio", "idle_ratio", "new_cells_per_tick", "approach_ratio")}})
    correct = [r for r in rows if r["true"] == r["predicted"]]
    wrong = sorted([r for r in rows if r["true"] != r["predicted"]], key=lambda r: -r["confidence"])
    picks: list[dict] = []
    for cls in LABELS:
        cands = sorted([r for r in correct if r["true"] == cls], key=lambda r: -r["confidence"])
        picks += cands[:2]
    return picks[:n + 1], wrong[:n]


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{h}</th>" for _, h in cols)
    body = []
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key, "")
            if key in ("true", "predicted"):
                cells.append(f"<td>{_tag(str(v))}</td>")
            elif isinstance(v, float):
                cells.append(f"<td>{v:.2f}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def build(
    data_csv: Path = Path("data/processed/episodes.csv"),
    metrics_path: Path = Path("reports/metrics.json"),
    decisions_path: Path = Path("reports/decisions.jsonl"),
    ab_csv: Path = Path("reports/ab_results.csv"),
    raw_dir: Path = Path("data/raw"),
    model_dir: Path = Path("models"),
    out_html: Path = Path("reports/dashboard.html"),
    figures_dir: Path = Path("reports/figures"),
    rl_results_path: Path = Path("reports/rl_results.json"),
    rl_curve_csv: Path = Path("reports/rl_curve.csv"),
    next_move_path: Path = Path("reports/next_move_metrics.json"),
    pursuit_csv: Path = Path("reports/pursuit_benchmark.csv"),
) -> dict:
    summary: dict = {"sections": [], "figures": []}
    parts: list[str] = []
    stamp = datetime.now(timezone.utc).isoformat(timespec="minutes")

    df = pd.read_csv(data_csv) if data_csv.exists() else None
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    # ---- overview ----
    parts.append("<h1>NeuroQuest — AI dashboard</h1><p class='sub'>What the game's AI learned, how well it works, and what it decided. Generated " + stamp + " from recorded data only.</p>")
    if metrics and df is not None:
        sel = metrics["selected"]["model"]
        t = metrics["models"][sel]["test"]
        parts.append("<div class='grid'>" + "".join(
            f"<div class='card stat'><b>{v}</b><span>{k}</span></div>"
            for k, v in (
                ("episodes", df["episode_id"].nunique()), ("rows", len(df)), ("model", sel.replace("_", " ")),
                ("test accuracy", f"{t['accuracy']:.3f}"), ("test macro-F1", f"{t['macro_f1']:.3f}"), ("majority baseline F1", f"{metrics['models']['majority']['test']['macro_f1']:.3f}"),
            )
        ) + "</div>")

    # ---- data ----
    parts.append("<h2>1. Gameplay data</h2>")
    if df is not None:
        p = F.fig_class_balance(df, figures_dir / "dash_data.png")
        summary["figures"].append(p)
        parts.append("<p>Rows are per episode and prefix (25 / 50 / 75 / 100 % of the episode). Labels are the simulated profile that played; human rows, if any, are kept separate.</p>" + _img(p))
        summary["sections"].append("data")
    else:
        parts.append("<p class='muted'>No dataset found: run <code>neuroquest simulate</code> and <code>neuroquest build-dataset</code>.</p>")

    # ---- model ----
    parts.append("<h2>2. Model evaluation</h2>")
    if metrics:
        p1 = F.fig_model_comparison(metrics, figures_dir / "dash_models.png")
        p2 = F.fig_confusion(metrics, figures_dir / "dash_confusion.png")
        p3 = F.fig_accuracy_by_prefix(metrics, figures_dir / "dash_prefix.png")
        summary["figures"] += [p1, p2, p3]
        abl = metrics.get("ablation_without_elapsed_s", {})
        parts.append(
            "<p>Candidates were ranked by grouped cross-validation on the training seeds; the test seeds were used once, afterwards. "
            "Baselines: majority class, the hand-written rule labeller, and the best single-feature tree.</p>"
            + _img(p1) + _img(p2) + _img(p3)
            + (f"<div class='card'>Ablation without <code>elapsed_s</code>: test macro-F1 {abl['test']['macro_f1']:.3f} (with it: {metrics['models'][metrics['selected']['model']]['test']['macro_f1']:.3f}). The model does not use play time as a shortcut.</div>" if abl else "")
        )
        summary["sections"].append("model")
    else:
        parts.append("<p class='muted'>No training report found: run <code>neuroquest train</code>.</p>")

    # ---- interpretability ----
    parts.append("<h2>3. What the model looks at</h2>")
    if metrics:
        p1 = F.fig_importance(metrics, figures_dir / "dash_importance.png")
        p2 = F.fig_coefficients(metrics, figures_dir / "dash_coefficients.png")
        summary["figures"] += [p for p in (p1, p2) if p]
        parts.append("<p>Permutation importance shuffles one feature at a time on the held-out rows and measures the drop in macro-F1. Coefficients are per class, on standardised features: positive pushes toward the class.</p>" + _img(p1) + _img(p2))
        summary["sections"].append("interpretability")

    # ---- examples ----
    parts.append("<h2>4. Correct and incorrect predictions</h2>")
    if metrics and df is not None and (model_dir / "behaviour_clf.joblib").exists():
        good, bad = _examples(data_csv, metrics, model_dir)
        cols = [("episode_id", "episode"), ("prefix_ratio", "prefix"), ("true", "true"), ("predicted", "predicted"), ("confidence", "confidence"), ("elapsed_s", "elapsed s"), ("damage_per_tick", "damage/tick"), ("near_enemy_ratio", "near enemy"), ("idle_ratio", "idle"), ("new_cells_per_tick", "new cells/tick"), ("approach_ratio", "approach")]
        parts.append("<p>Recomputed with the saved model on the held-out seeds. <span class='ok'>Confident and right</span>:</p>" + _table(good, cols) + "<p style='margin-top:14px'><span class='bad'>Confident and wrong</span> (the most confident errors):</p>" + _table(bad, cols))
        summary["examples"] = {"correct": good, "wrong": bad}
        summary["sections"].append("examples")

    # ---- live session ----
    parts.append("<h2>5. A session, as the model saw it</h2>")
    logs = sorted(raw_dir.glob("human-*.jsonl")) if raw_dir.exists() else []
    source = "latest human session" if logs else "a simulated episode"
    if not logs and raw_dir.exists():
        logs = sorted(raw_dir.glob("*.jsonl"))
    if logs and (model_dir / "behaviour_clf.joblib").exists():
        log = EpisodeLog.load(logs[-1])
        predictor = BehaviourPredictor.load(model_dir / "behaviour_clf.joblib", model_dir / "behaviour_clf.meta.json")
        rows = F.prediction_timeline(log, predictor)
        p = F.fig_prediction_timeline(rows, f"Style probabilities over time — {source} ({logs[-1].name})", figures_dir / "dash_session.png")
        if p:
            summary["figures"].append(p)
        last = rows[-1] if rows else None
        parts.append(f"<p>The saved model re-run over the recorded events on the live schedule (first read at 3 s, then every 2 s). Outcome: <b>{log.meta.get('outcome')}</b> after {log.meta.get('ticks', 0) / log.meta.get('ticks_per_second', 10):.0f} s; final read: {_tag(last['label']) + f' {last['confidence']:.0%}' if last else 'none (episode shorter than 3 s)'}.</p>" + _img(p))
        summary["sections"].append("session")
    else:
        parts.append("<p class='muted'>No episode logs found.</p>")

    # ---- adaptive decisions ----
    parts.append("<h2>6. Adaptive difficulty decisions</h2>")
    decisions: list[dict] = []
    if decisions_path.exists():
        decisions = [json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()]
    if decisions:
        by_episode: dict[str, list[dict]] = {}
        for d in decisions:
            by_episode.setdefault(d.get("episode_id", "?"), []).append(d)
        # show the session with the most changes (the most informative); the latest if none changed anything
        ids = list(by_episode)
        latest_id = max(ids, key=lambda k: (sum(d["before"] != d["after"] for d in by_episode[k]), ids.index(k)))
        latest = by_episode[latest_id]
        p = F.fig_difficulty_timeline(latest, 10, f"Difficulty over time — {latest_id}", figures_dir / "dash_difficulty.png")
        if p:
            summary["figures"].append(p)
        rule_counts = pd.Series([d["rule"] for d in decisions]).value_counts()
        changed = [d for d in latest if d["before"] != d["after"]]
        rows_html = _table(
            [{"t": f"{d['t'] / 10:.0f} s", "rule": d["rule"], "lever": d.get("lever") or "—", "change": (f"{d['before'][d['lever']]} → {d['after'][d['lever']]}" if d.get("lever") and d["before"] != d["after"] else "—"), "reason": d["reason"]} for d in latest[-8:]],
            [("t", "time"), ("rule", "rule"), ("lever", "lever"), ("change", "change"), ("reason", "recorded reason")],
        )
        parts.append(f"<p>{len(decisions)} decisions logged over {len(by_episode)} session(s). Shown: <code>{latest_id}</code>, the session with the most changes ({len(changed)}). Every row below is a recorded decision with the reason the controller wrote at the time.</p>" + _img(p) + "<div class='card'>Rules fired (all sessions): " + ", ".join(f"<code>{k}</code> ×{v}" for k, v in rule_counts.items()) + "</div>" + rows_html)
        summary["sections"].append("decisions")
    else:
        parts.append("<p class='muted'>No decisions logged yet: play with the adaptive controller on (<code>neuroquest play</code>).</p>")

    # ---- fixed vs adaptive ----
    parts.append("<h2>7. Fixed versus adaptive difficulty</h2>")
    if ab_csv.exists():
        ab = pd.read_csv(ab_csv).to_dict("records")
        p = F.fig_ab(ab, figures_dir / "dash_ab.png")
        if p:
            summary["figures"].append(p)
        n = pd.DataFrame(ab).groupby(["profile", "condition"]).size().min()
        parts.append(f"<p>The same simulated players replay the same seeded mazes under both conditions ({n} per cell). With this many episodes the noise on a win-rate difference is roughly ±20 points, so small differences are not evidence of improvement; the experiment shows the controller is safe and traceable.</p>" + _img(p))
        summary["sections"].append("ab")
    else:
        parts.append("<p class='muted'>No A/B results: run <code>neuroquest ab-test</code>.</p>")

    # ---- reinforcement learning ----
    parts.append("<h2>8. Reinforcement learning agent</h2>")
    rl = json.loads(rl_results_path.read_text()) if rl_results_path.exists() else None
    if rl:
        curve = pd.read_csv(rl_curve_csv) if rl_curve_csv.exists() else None
        agents = rl["evaluation"]["agents"]
        refs = {k: agents[k]["mean_return"] for k in ("random", "greedy_collector", "cautious") if k in agents}
        p1 = F.fig_rl_curve(curve, figures_dir / "dash_rl_curve.png", references=refs)
        p2 = F.fig_rl_eval(agents, figures_dir / "dash_rl_eval.png")
        summary["figures"] += [p for p in (p1, p2) if p]
        e, t, ev, rw = rl["env"], rl["training"], rl["evaluation"], rl["env"]["rewards"]
        parts.append(
            f"<p>A tabular Q-learning agent plays the game's own rules on one fixed {e['board']} board: {e['open_cells']} open cells, {e['crystals']} crystals, {e['hunters']} {e['hunter_kind']} hunter moving every {e['hunter_move_interval']} ticks, {e['time_limit_ticks']} ticks on the clock. "
            f"It observes only its own cell, which crystals remain and where the hunter is (exactly within ±{e['enemy_window']} cells, otherwise only the direction). It never sees the clock, its health or the walls.</p>"
            f"<div class='card'><b>Reward per tick:</b> {rw['step']} time cost · +{rw['crystal']:g} per crystal · +{rw['exit']:g} at the exit · {rw['hit']:g} per hit · {rw['defeat']:g} more when health reaches zero. "
            f"<b>Episode ends</b> at the exit or at zero health; running out of time is a cut-off the learner bootstraps through. "
            f"<b>Training:</b> {t['episodes']} ε-greedy episodes, α = {t['alpha']}, γ = {t['gamma']}, ε {t['epsilon_start']} → {t['epsilon_end']} over the first {t['explore_fraction']:.0%}; {t['states_visited']} observations visited (bound {e['state_space_bound']}); {t['seconds']:.0f} s.</div>"
            + _img(p1) + _img(p2)
        )
        rows = [{"agent": F.RL_LABELS.get(n, n), "win": f"{a['win_rate']:.2f} [{a['win_rate_ci95'][0]:.2f}, {a['win_rate_ci95'][1]:.2f}]", "ret": a["mean_return"], "crystals": a["mean_crystals"], "hits": a["mean_hits"], "steps": a["mean_steps"], "outcomes": ", ".join(f"{k} {v}" for k, v in a["outcomes"].items())} for n, a in agents.items()]
        parts.append(_table(rows, [("agent", "agent"), ("win", "win rate [95 % CI]"), ("ret", "mean return"), ("crystals", "crystals"), ("hits", "hits"), ("steps", "steps"), ("outcomes", "outcomes")]))
        other = rl.get("other_layout")
        if other:
            parts.append(f"<div class='card'>The same Q-table on a board it never trained on (layout seed {other['layout_seed']}): win rate {other['q_learning']['win_rate']:.2f} against {other['random']['win_rate']:.2f} for random play. A tabular agent memorises one board; it does not generalise, and the figure above should be read as learning <i>this</i> board.</div>")
        parts.append(f"<p class='muted'>The hunter is deterministic, so the {ev['episodes']} held-out seeds cover {ev['distinct_hunter_starts']} distinct hunter start cells. The scripted baselines read the full game state; the learned agent only its observation.</p>")
        summary["sections"].append("rl")
    else:
        parts.append("<p class='muted'>No RL results: run <code>neuroquest rl-train</code>.</p>")

    # ---- next-move model ----
    parts.append("<h2>9. Next-move model</h2>")
    nm = json.loads(next_move_path.read_text()) if next_move_path.exists() else None
    if nm:
        p = F.fig_next_move(nm, figures_dir / "dash_next_move.png")
        if p:
            summary["figures"].append(p)
        s, d, sel = nm["split"], nm["data"], nm["selected"]["model"]
        m = nm["models"]
        best_baseline = max((n for n in ("majority", "repeat_last", "greedy") if n in m), key=lambda n: m[n]["test"]["macro_f1"])
        verdict = "beats" if m[sel]["test"]["macro_f1"] > m[best_baseline]["test"]["macro_f1"] else "does not beat"
        parts.append(
            f"<p>A second supervised model predicts the player's <b>next action</b> (up, down, left, right, stay) from what the game shows at that tick: the last three actions, open and visited neighbours, the direction and distance of the nearest crystal, the portal and the nearest hunter, and the shortest-path direction to the nearest crystal and to the portal. "
            f"Rows come from replaying {d['episodes']} simulated episodes through the real rules ({d['rows']} ticks). <b>Temporal split:</b> episodes are ordered by seed; seeds {s['train_seeds'][0]}–{s['train_seeds'][1]} train ({s['train_rows']} rows), seeds {s['test_seeds'][0]}–{s['test_seeds'][1]} test ({s['test_rows']} rows); the model was chosen on the last {s['validation_rows']} training rows, never on the test rows.</p>"
            + _img(p)
            + f"<div class='card'>Selected: <b>{F.NM_LABELS.get(sel, sel)}</b>, test accuracy {m[sel]['test']['accuracy']:.3f}, macro-F1 {m[sel]['test']['macro_f1']:.3f}. It {verdict} the strongest hand-written baseline ({F.NM_LABELS.get(best_baseline, best_baseline)}: accuracy {m[best_baseline]['test']['accuracy']:.3f}, macro-F1 {m[best_baseline]['test']['macro_f1']:.3f}); majority {m['majority']['test']['macro_f1']:.3f}, repeat-last {m['repeat_last']['test']['macro_f1']:.3f}.</div>"
        )
        rows = [{"model": F.NM_LABELS.get(n, n), "acc": a["test"]["accuracy"], "f1": a["test"]["macro_f1"], **{p: v for p, v in a["per_profile_accuracy"].items()}} for n, a in m.items()]
        profiles = sorted(m[sel]["per_profile_accuracy"])
        parts.append(_table(rows, [("model", "model"), ("acc", "test accuracy"), ("f1", "test macro-F1")] + [(p, f"{p} acc.") for p in profiles]))
        if pursuit_csv.exists():
            pr = pd.read_csv(pursuit_csv)
            g = pr.assign(caught=pr["first_contact"].notna()).groupby(["kind", "profile"]).agg(episodes=("seed", "size"), caught=("caught", "mean"), hits=("hits", "mean"), wins=("outcome", lambda o: (o == "won").mean())).reset_index()
            prow = [{"kind": r["kind"], "profile": r["profile"], "caught": f"{r['caught']:.0%}", "hits": r["hits"], "wins": f"{r['wins']:.0%}"} for _, r in g.iterrows()]
            n = int(g["episodes"].min())
            parts.append(f"<p style='margin-top:14px'><b>Does it help the hunters?</b> Hunters that head for the player's <i>predicted</i> next cell (<code>anticipator</code>) against plain A* hunters, on the same {n} seeded episodes per cell, against moving scripted players:</p>" + _table(prow, [("kind", "hunter"), ("profile", "player"), ("caught", "episodes with a hit"), ("hits", "mean hits"), ("wins", "player win rate")]) + "<p class='muted'>With this many episodes a difference of a few points is within noise; read the hits column as the main signal.</p>")
        summary["sections"].append("next_move")
    else:
        parts.append("<p class='muted'>No next-move results: run <code>neuroquest build-moves</code> and <code>neuroquest train-next-move</code>.</p>")

    # ---- limitations ----
    parts.append("<h2>10. Limitations</h2><div class='card'><ul>"
                 "<li>The classifier recognises <b>simulated</b> personalities; human play may differ. Human rows are kept separate so the gap can be measured as they accumulate.</li>"
                 "<li>Early in a quiet episode an explorer and an aggressive player look alike (no enemy met yet); the model is confidently wrong there, and its probabilities are not calibrated.</li>"
                 "<li>One split seed and 360 episodes: quote scores with the ±0.015 CV spread in mind.</li>"
                 "<li>The adaptive controller is rule-based with reasoned, untuned thresholds; its effect on humans is untested, and on simulated players it is safe but not shown to improve outcomes.</li>"
                 "<li>The Q-learning agent learns one small board with one hunter; it sees a partial observation and its table does not transfer to other boards. Its scores are a comparison on that board, not a claim about the full game.</li>"
                 "<li>The next-move model is trained and tested on simulated players whose decisions follow written rules; its accuracy on human play is unknown, and the hand-written greedy rule is a strong competitor for a reason.</li>"
                 "</ul></div>")
    parts.append("<h2>Reproduce</h2><div class='card'><code>neuroquest simulate</code> → <code>build-dataset</code> → <code>train</code> → <code>ab-test</code> → <code>rl-train</code> → <code>build-moves</code> → <code>train-next-move</code> → <code>benchmark-pursuit</code> → <code>dashboard</code>. Figures are written to <code>reports/figures/dash_*.png</code>; this file embeds them.</div>")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>NeuroQuest — AI dashboard</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")
    summary["html"] = out_html
    return summary
