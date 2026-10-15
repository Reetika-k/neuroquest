"""Train/test splitting without leakage.

Rows from the same episode (its four prefixes) and from the same maze seed (one episode per
profile on that maze) must land on the same side of the split, otherwise the test set contains
near-copies of training rows and the score is a lie. We therefore split by *seed*: every seed
carries all three labels, so grouping by seed also keeps the classes balanced on both sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_split(df: pd.DataFrame, group_col: str = "seed", test_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ``test_fraction`` of the groups (deterministic for a given ``seed``)."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    groups = np.array(sorted(df[group_col].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = max(1, round(test_fraction * len(groups)))
    test_groups = set(groups[:n_test].tolist())
    is_test = df[group_col].isin(test_groups)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
