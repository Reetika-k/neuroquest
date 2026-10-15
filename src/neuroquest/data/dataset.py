"""Build the tabular dataset from raw episode logs: one row per (episode, prefix)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuroquest.data.events import EpisodeLog
from neuroquest.data.features import FEATURE_NAMES, episode_features
from neuroquest.data.labels import LABELS, rule_label
from neuroquest.data.schema import ROW_COLUMNS, validate_row, write_schema

PREFIXES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def rows_from_log(log: EpisodeLog, prefixes: tuple[float, ...] = PREFIXES) -> list[dict]:
    meta = log.meta
    rows = []
    for prefix in prefixes:
        features = episode_features(log, prefix)
        profile = meta.get("profile", "human")
        label = profile if profile in LABELS else f"rule:{rule_label(features)}"
        rows.append(
            {
                "episode_id": meta["episode_id"],
                "label": label,
                "profile": profile,
                "seed": int(meta["seed"]),
                "outcome": meta["outcome"],
                "ticks_total": int(meta["ticks"]),
                "prefix_ratio": prefix,
                "ticks_seen": int(features["elapsed_s"] * meta["ticks_per_second"]),
                **features,
            }
        )
    return rows


def build(raw_dir: Path, out_csv: Path, prefixes: tuple[float, ...] = PREFIXES) -> tuple[pd.DataFrame, dict]:
    """Read every ``*.jsonl`` log, compute rows, validate, write the CSV and the schema file."""
    paths = sorted(raw_dir.glob("*.jsonl"))
    rows: list[dict] = []
    invalid: list[tuple[str, list[str]]] = []
    for path in paths:
        for row in rows_from_log(EpisodeLog.load(path), prefixes):
            problems = validate_row(row)
            if problems:
                invalid.append((f"{row['episode_id']}@{row['prefix_ratio']}", problems))
            else:
                rows.append(row)
    df = pd.DataFrame(rows, columns=ROW_COLUMNS)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_schema(out_csv.parent.parent / "schema" / "features.schema.json")
    report = {"episodes": len(paths), "rows": len(df), "invalid": invalid, "prefixes": list(prefixes)}
    return df, report


def summary(df: pd.DataFrame) -> str:
    """Human-readable inspection: class balance, outcomes per profile, feature means per label."""
    lines = [f"rows: {len(df)}  episodes: {df['episode_id'].nunique()}  prefixes: {sorted(df['prefix_ratio'].unique())}"]
    lines.append("label balance (rows): " + ", ".join(f"{k}={v}" for k, v in df["label"].value_counts().sort_index().items()))
    full = df[df["prefix_ratio"] == 1.0]
    lines.append("outcomes per profile (full episodes):")
    for profile, sub in full.groupby("profile"):
        counts = sub["outcome"].value_counts().to_dict()
        lines.append(f"  {profile:<11}" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"  median ticks={int(sub['ticks_total'].median())}")
    means = full.groupby("label")[FEATURE_NAMES[1:]].mean().round(3)
    lines.append("feature means per label (full episodes):")
    lines.append(means.T.to_string())
    return "\n".join(lines)
