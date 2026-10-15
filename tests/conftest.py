"""Shared pytest configuration.

Sets headless drivers *before* any test imports pygame or matplotlib, so the whole
suite runs without a display (locally and on CI).
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A real (small) trained model for tests that need one; trained once per session."""
    from pathlib import Path

    from neuroquest.config import DifficultyParams, GameConfig
    from neuroquest.data.dataset import build
    from neuroquest.data.labels import LABELS
    from neuroquest.ml.train import train
    from neuroquest.sim.harness import run_batch

    root = Path(tmp_path_factory.mktemp("tiny-model"))
    run_batch(list(LABELS), 8, GameConfig(), DifficultyParams(time_limit_s=30), seed0=100, out_dir=root / "raw")
    build(root / "raw", root / "processed" / "episodes.csv")
    train(root / "processed" / "episodes.csv", model_dir=root / "models", report_path=root / "metrics.json", seed=1, fast=True, test_fraction=0.25)
    return root / "models"
