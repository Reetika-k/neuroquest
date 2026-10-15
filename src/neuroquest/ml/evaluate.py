"""Metrics, error inspection and feature importance. Nothing here trains a model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1/support and the confusion matrix (rows = true)."""
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for lab, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def accuracy_by_prefix(df: pd.DataFrame, y_pred: Sequence[str]) -> dict[str, float]:
    """Does the model get better the more of the episode it sees?"""
    correct = (df["label"].to_numpy() == np.asarray(y_pred))
    return {str(p): float(correct[df["prefix_ratio"].to_numpy() == p].mean()) for p in sorted(df["prefix_ratio"].unique())}


def confusion_pairs(y_true: Sequence[str], y_pred: Sequence[str]) -> list[dict]:
    """Which classes get mistaken for which, most frequent first."""
    counts = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return [{"true": t, "predicted": p, "count": n} for (t, p), n in counts.most_common()]


def worst_errors(df: pd.DataFrame, y_pred: Sequence[str], proba: np.ndarray, classes: Sequence[str], n: int = 10, show: Sequence[str] = ()) -> list[dict]:
    """The most confidently wrong predictions, with a few features for context."""
    classes = list(classes)
    rows = []
    for i, (true, pred) in enumerate(zip(df["label"], y_pred)):
        if true == pred:
            continue
        row = {
            "episode_id": str(df.iloc[i]["episode_id"]),
            "prefix_ratio": float(df.iloc[i]["prefix_ratio"]),
            "true": true,
            "predicted": pred,
            "confidence": float(proba[i][classes.index(pred)]),
            "p_true": float(proba[i][classes.index(true)]),
        }
        for name in show:
            row[name] = round(float(df.iloc[i][name]), 3)
        rows.append(row)
    rows.sort(key=lambda r: -r["confidence"])
    return rows[:n]


def importances(model, X: np.ndarray, y: Sequence[str], feature_names: Sequence[str], seed: int, n_repeats: int = 10) -> dict:
    """Permutation importance on held-out data (model-agnostic) plus impurity importance when available."""
    out: dict = {}
    perm = permutation_importance(model, X, list(y), scoring="f1_macro", n_repeats=n_repeats, random_state=seed)
    out["permutation"] = {name: {"mean": float(m), "std": float(s)} for name, m, s in zip(feature_names, perm.importances_mean, perm.importances_std)}
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(final, "feature_importances_"):
        out["impurity"] = {name: float(v) for name, v in zip(feature_names, final.feature_importances_)}
    if hasattr(final, "coef_"):
        out["coefficients"] = {cls: {name: float(v) for name, v in zip(feature_names, row)} for cls, row in zip(final.classes_, final.coef_)}
    return out
