"""Inspect data/processed/episodes.csv before any training (Phase 6).

Prints class balance, outcomes, per-label feature means and spreads, a separability score per
feature, and the agreement between the rule labeller and the true profile labels. Writes
reports/figures/dataset_features.png. Usage: .venv/bin/python scripts/inspect_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from neuroquest.data.features import FEATURE_NAMES  # noqa: E402
from neuroquest.data.labels import LABELS, rule_label  # noqa: E402

csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/episodes.csv")
df = pd.read_csv(csv)
full = df[df["prefix_ratio"] == 1.0]
quarter = df[df["prefix_ratio"] == 0.25]

print(f"rows {len(df)}  episodes {df['episode_id'].nunique()}  labels {df['label'].value_counts().to_dict()}")
print("\nper-label mean ± std (full episodes):")
stats = full.groupby("label")[FEATURE_NAMES].agg(["mean", "std"]).round(3)
for name in FEATURE_NAMES:
    cells = "  ".join(f"{lab}: {stats.loc[lab, (name, 'mean')]:.3f}±{stats.loc[lab, (name, 'std')]:.3f}" for lab in LABELS)
    print(f"  {name:<24}{cells}")

print("\nseparability per feature (spread of class means / mean within-class std; higher = easier):")
sep = {}
for name in FEATURE_NAMES:
    means = full.groupby("label")[name].mean()
    within = full.groupby("label")[name].std().mean()
    sep[name] = (means.max() - means.min()) / within if within > 0 else float("inf")
for name, score in sorted(sep.items(), key=lambda kv: -kv[1]):
    print(f"  {name:<24}{score:.2f}")

print("\nrule labeller vs true profile (how often the threshold rules agree with the generating profile):")
for prefix, sub in (("1.00", full), ("0.25", quarter)):
    agree = sub.apply(lambda r: rule_label(r) == r["label"], axis=1)
    per = sub.assign(ok=agree).groupby("label")["ok"].mean().round(2).to_dict()
    print(f"  prefix {prefix}: overall {agree.mean():.2f}  per label {per}")

print("\nelapsed_s by label and prefix (median seconds):")
print(df.pivot_table(index="label", columns="prefix_ratio", values="elapsed_s", aggfunc="median").round(1).to_string())

fig, axes = plt.subplots(3, 5, figsize=(16, 9))
colours = {"aggressive": "#e23d4f", "cautious": "#4353d9", "explorer": "#1aa36b"}
for ax, name in zip(axes.flat, FEATURE_NAMES):
    data = [full.loc[full["label"] == lab, name] for lab in LABELS]
    box = ax.boxplot(data, tick_labels=[lab[:4] for lab in LABELS], patch_artist=True, widths=0.6)
    for patch, lab in zip(box["boxes"], LABELS):
        patch.set_facecolor(colours[lab]); patch.set_alpha(0.55)
    ax.set_title(name, fontsize=10)
    ax.tick_params(labelsize=8)
for ax in axes.flat[len(FEATURE_NAMES):]:
    ax.axis("off")
fig.suptitle("Feature distributions per label, full episodes (3 profiles × 120 seeded episodes)", fontsize=12)
fig.tight_layout()
out = Path("reports/figures/dataset_features.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)
print(f"\nfigure written to {out}")
