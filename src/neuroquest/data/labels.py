"""Labels.

Synthetic episodes are labelled with the *profile that played them* (see ``sim/profiles.py``):
the label is ground truth by construction and is not derived from the features.

Human episodes have no such ground truth. ``rule_label`` gives them a transparent, inspectable
label from a few thresholds. Its limitation is circularity: a model trained on rule labels can
only re-learn the rules. It is therefore used for human play only, reported separately, and
never mixed into the synthetic training labels. Self-reported labels are the better extension.
"""

from __future__ import annotations

from collections.abc import Mapping

LABELS: tuple[str, ...] = ("aggressive", "cautious", "explorer")

RULE_THRESHOLDS = {
    "aggressive_approach_ratio": 0.5,  # moves toward enemies at least half the time when near them …
    "aggressive_near_enemy_ratio": 0.05,  # … and is near them in at least 5 % of ticks
    "aggressive_damage_per_tick": 0.02,  # or takes a hit every 50 ticks or more often
    "explorer_distinct_cells_ratio": 0.5,  # visits at least half the maze …
    "explorer_new_cells_per_tick": 0.35,  # … or spends most moves discovering new cells
}


def rule_label(features: Mapping[str, float]) -> str:
    """Threshold rules for human episodes. Returns one of LABELS."""
    t = RULE_THRESHOLDS
    aggressive = (
        features["approach_ratio"] >= t["aggressive_approach_ratio"]
        and features["near_enemy_ratio"] >= t["aggressive_near_enemy_ratio"]
    ) or features["damage_per_tick"] >= t["aggressive_damage_per_tick"]
    if aggressive:
        return "aggressive"
    if (
        features["distinct_cells_ratio"] >= t["explorer_distinct_cells_ratio"]
        or features["new_cells_per_tick"] >= t["explorer_new_cells_per_tick"]
    ):
        return "explorer"
    return "cautious"
