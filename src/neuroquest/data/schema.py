"""The dataset contract: what each feature means, its allowed range, and row validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from neuroquest.data.features import FEATURE_NAMES
from neuroquest.data.labels import LABELS

FEATURE_SPECS: dict[str, dict] = {
    "elapsed_s": {"min": 0.0, "max": 36000.0, "description": "seconds of play seen"},
    "moves_per_tick": {"min": 0.0, "max": 1.0, "description": "successful moves per tick"},
    "bump_ratio": {"min": 0.0, "max": 1.0, "description": "wall bumps / (moves + bumps)"},
    "idle_ratio": {"min": 0.0, "max": 1.0, "description": "ticks with no move or bump / ticks"},
    "new_cells_per_tick": {"min": 0.0, "max": 1.0, "description": "first visits per tick"},
    "distinct_cells_ratio": {"min": 0.0, "max": 1.0, "description": "distinct cells visited / open cells"},
    "revisit_ratio": {"min": 0.0, "max": 1.0, "description": "moves into already visited cells / moves"},
    "rewards_per_tick": {"min": 0.0, "max": 1.0, "description": "rewards collected per tick"},
    "reward_share": {"min": 0.0, "max": 1.0, "description": "rewards collected / rewards available"},
    "damage_per_tick": {"min": 0.0, "max": 1.0, "description": "hits taken per tick"},
    "near_enemy_ratio": {"min": 0.0, "max": 1.0, "description": "ticks within 2 cells of an enemy / ticks"},
    "approach_ratio": {"min": 0.0, "max": 1.0, "description": "near-enemy moves that reduced the distance / near-enemy moves"},
    "progress_rate": {"min": -5.0, "max": 1.0, "description": "(start distance to exit − current) / start distance; negative means farther away"},
    "min_exit_distance_ratio": {"min": 0.0, "max": 5.0, "description": "closest distance to exit reached / start distance"},
}

ROW_COLUMNS = ["episode_id", "label", "profile", "seed", "outcome", "ticks_total", "prefix_ratio", "ticks_seen", *FEATURE_NAMES]

assert list(FEATURE_SPECS) == FEATURE_NAMES, "schema and feature list out of sync"


def write_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": 1,
        "labels": list(LABELS),
        "label_note": "synthetic rows: the profile that played the episode; human rows: 'rule:<class>' from data/labels.py",
        "columns": ROW_COLUMNS,
        "features": [{"name": name, **spec} for name, spec in FEATURE_SPECS.items()],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def validate_row(row: Mapping) -> list[str]:
    """Return a list of problems (empty when the row is valid)."""
    problems: list[str] = []
    for name, spec in FEATURE_SPECS.items():
        if name not in row:
            problems.append(f"missing {name}")
            continue
        value = row[name]
        try:
            value = float(value)
        except (TypeError, ValueError):
            problems.append(f"{name} is not numeric: {value!r}")
            continue
        if math.isnan(value) or math.isinf(value):
            problems.append(f"{name} is not finite")
        elif not spec["min"] <= value <= spec["max"]:
            problems.append(f"{name}={value} outside [{spec['min']}, {spec['max']}]")
    label = row.get("label")
    if label not in LABELS and not (isinstance(label, str) and label.startswith("rule:") and label[5:] in LABELS):
        problems.append(f"bad label {label!r}")
    return problems
