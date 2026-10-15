"""Train, compare and save the player-behaviour classifier.

Protocol: seed-grouped 80/20 split → every candidate is scored by grouped 5-fold cross-validation
on the training rows → the best candidate (by CV macro-F1) is chosen *before* looking at the test
rows → every candidate is then evaluated once on the test rows and reported. Baselines that need
no training (majority class, the rule labeller, a single-feature tree) are reported alongside.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from neuroquest.data.features import FEATURE_NAMES
from neuroquest.data.labels import LABELS, rule_label
from neuroquest.ml.evaluate import accuracy_by_prefix, classification_metrics, confusion_pairs, importances, worst_errors
from neuroquest.ml.split import grouped_split

MODEL_PATH = Path("models/behaviour_clf.joblib")
META_PATH = Path("models/behaviour_clf.meta.json")
REPORT_PATH = Path("reports/metrics.json")
CANDIDATES = ("logistic_regression", "random_forest", "knn")


def build_models(seed: int, fast: bool = False) -> dict[str, object]:
    """The candidates, with fixed hyperparameters (no tuning on the test set)."""
    trees = 60 if fast else 300
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, C=1.0))]),
        "random_forest": RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "knn": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15))]),
    }


def _folds(groups: np.ndarray, n_splits: int = 5) -> int:
    """Never ask for more folds than there are groups (tiny test datasets)."""
    return max(2, min(n_splits, len(np.unique(groups))))


def cv_scores(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> dict:
    n_splits = _folds(groups, n_splits)
    cv = GroupKFold(n_splits=n_splits)
    f1 = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1_macro")
    acc = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="accuracy")
    return {"macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std()), "accuracy_mean": float(acc.mean()), "folds": n_splits}


def best_single_feature_tree(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[int, float]:
    """Grouped-CV macro-F1 of a depth-2 decision tree on each feature alone; returns the best (index, score)."""
    best = (0, -1.0)
    for j in range(X.shape[1]):
        score = cross_val_score(DecisionTreeClassifier(max_depth=2, random_state=seed), X[:, [j]], y, groups=groups, cv=GroupKFold(_folds(groups)), scoring="f1_macro").mean()
        if score > best[1]:
            best = (j, float(score))
    return best


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _params(model) -> dict:
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    return {k: v for k, v in final.get_params().items() if isinstance(v, (int, float, str, bool, type(None)))}


def train(
    data_csv: Path,
    model_dir: Path = Path("models"),
    report_path: Path = REPORT_PATH,
    seed: int = 42,
    fast: bool = False,
    test_fraction: float = 0.2,
) -> dict:
    """Run the whole protocol, save the chosen model and its metadata, write the report, return it."""
    df = pd.read_csv(data_csv)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)  # human rule-labelled rows are never trained on
    train_df, test_df = grouped_split(df, test_fraction=test_fraction, seed=seed)
    X_tr, y_tr, g_tr = train_df[FEATURE_NAMES].to_numpy(), train_df["label"].to_numpy(), train_df["seed"].to_numpy()
    X_te, y_te = test_df[FEATURE_NAMES].to_numpy(), test_df["label"].to_numpy()
    labels = list(LABELS)

    report: dict = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "data": {"path": str(data_csv), "sha256": _sha256(data_csv), "rows": int(len(df)), "episodes": int(df["episode_id"].nunique())},
        "split": {
            "by": "seed", "seed": seed, "test_fraction": test_fraction,
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "train_seeds": int(train_df["seed"].nunique()), "test_seeds": int(test_df["seed"].nunique()),
        },
        "features": list(FEATURE_NAMES),
        "models": {},
    }

    # --- baselines that need no training ---
    rule_pred_te = [rule_label(row) for _, row in test_df[FEATURE_NAMES].iterrows()]
    report["models"]["rule_labeller"] = {"kind": "thresholds from data/labels.py", "test": classification_metrics(y_te, rule_pred_te, labels)}
    j, stump_cv = best_single_feature_tree(X_tr, y_tr, g_tr, seed)
    stump = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(X_tr[:, [j]], y_tr)
    report["models"]["single_feature_tree"] = {
        "kind": f"depth-2 tree on {FEATURE_NAMES[j]} alone (best of {len(FEATURE_NAMES)} by CV)",
        "feature": FEATURE_NAMES[j],
        "cv": {"macro_f1_mean": stump_cv},
        "test": classification_metrics(y_te, stump.predict(X_te[:, [j]]), labels),
    }

    # --- trained candidates: CV first, then one look at the test set ---
    models = build_models(seed, fast=fast)
    fitted: dict[str, object] = {}
    for name, model in models.items():
        cv = cv_scores(model, X_tr, y_tr, g_tr)
        model.fit(X_tr, y_tr)
        fitted[name] = model
        report["models"][name] = {"kind": type(model.steps[-1][1] if hasattr(model, "steps") else model).__name__, "hyperparameters": _params(model), "cv": cv, "test": classification_metrics(y_te, model.predict(X_te), labels)}

    chosen = max(CANDIDATES, key=lambda n: report["models"][n]["cv"]["macro_f1_mean"])
    model = fitted[chosen]
    proba = model.predict_proba(X_te)
    pred = model.predict(X_te)
    report["selected"] = {
        "model": chosen,
        "reason": "highest grouped-CV macro-F1 on the training rows (test rows were not used for selection)",
        "accuracy_by_prefix": accuracy_by_prefix(test_df, pred),
        "confusion_pairs": confusion_pairs(y_te, pred),
        "worst_errors": worst_errors(test_df, pred, proba, model.classes_, n=10, show=("elapsed_s", "damage_per_tick", "idle_ratio", "new_cells_per_tick", "approach_ratio")),
        "importances": importances(model, X_te, y_te, FEATURE_NAMES, seed, n_repeats=5 if fast else 10),
    }

    # --- ablation: the same model type without elapsed_s ---
    keep = [i for i, n in enumerate(FEATURE_NAMES) if n != "elapsed_s"]
    ablated = build_models(seed, fast=fast)[chosen]
    cv_abl = cv_scores(ablated, X_tr[:, keep], y_tr, g_tr)
    ablated.fit(X_tr[:, keep], y_tr)
    report["ablation_without_elapsed_s"] = {"model": chosen, "cv": cv_abl, "test": classification_metrics(y_te, ablated.predict(X_te[:, keep]), labels)}

    # --- persist ---
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_PATH.name
    meta_path = model_dir / META_PATH.name
    joblib.dump(model, model_path)
    meta = {
        "model": chosen,
        "features": list(FEATURE_NAMES),
        "classes": [str(c) for c in model.classes_],
        "trained_at": report["trained_at"],
        "sklearn": report["sklearn"],
        "data": report["data"],
        "split": report["split"],
        "hyperparameters": report["models"][chosen]["hyperparameters"],
        "cv": report["models"][chosen]["cv"],
        "test": {k: report["models"][chosen]["test"][k] for k in ("accuracy", "macro_f1")},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    report["artifacts"] = {"model": str(model_path), "meta": str(meta_path)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def format_report(report: dict) -> str:
    """A compact table of CV and test scores for every model."""
    lines = [f"split by seed: {report['split']['train_rows']} train rows ({report['split']['train_seeds']} seeds) / {report['split']['test_rows']} test rows ({report['split']['test_seeds']} seeds)"]
    lines.append(f"  {'model':<22}{'CV macro-F1':>14}{'test acc':>10}{'test macro-F1':>15}")
    for name, m in report["models"].items():
        cv = m.get("cv", {}).get("macro_f1_mean")
        cv_txt = f"{cv:.3f}" if cv is not None else "-"
        lines.append(f"  {name:<22}{cv_txt:>14}{m['test']['accuracy']:>10.3f}{m['test']['macro_f1']:>15.3f}")
    sel = report["selected"]
    lines.append(f"selected: {sel['model']} ({sel['reason']})")
    lines.append("accuracy by prefix: " + ", ".join(f"{k}: {v:.3f}" for k, v in sel["accuracy_by_prefix"].items()))
    abl = report["ablation_without_elapsed_s"]
    lines.append(f"ablation without elapsed_s: CV macro-F1 {abl['cv']['macro_f1_mean']:.3f}, test macro-F1 {abl['test']['macro_f1']:.3f}")
    return "\n".join(lines)
