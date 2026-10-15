"""Load the saved classifier and predict a player's style from live features."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib

from neuroquest.data.features import FEATURE_NAMES
from neuroquest.ml.train import META_PATH, MODEL_PATH


PREDICT_AFTER_TICKS = 30  # wait 3 s of play before the first prediction
PREDICT_EVERY_TICKS = 20  # then refresh every 2 s


class ModelMismatch(RuntimeError):
    """The saved model was trained on a different feature list than the current code computes."""


@dataclass(frozen=True)
class Prediction:
    label: str
    proba: dict[str, float]  # class → probability
    confidence: float  # probability of the predicted class

    def summary(self) -> str:
        return f"{self.label} {self.confidence:.0%}"


class BehaviourPredictor:
    def __init__(self, model, meta: dict) -> None:
        if list(meta.get("features", [])) != list(FEATURE_NAMES):
            raise ModelMismatch(f"model features {meta.get('features')} != current {FEATURE_NAMES}; retrain with `neuroquest train`")
        self.model = model
        self.meta = meta
        self.features = list(meta["features"])
        self.classes = [str(c) for c in model.classes_]

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> BehaviourPredictor:
        return cls(joblib.load(model_path), json.loads(Path(meta_path).read_text()))

    def predict(self, features: Mapping[str, float]) -> Prediction:
        X = [[float(features[name]) for name in self.features]]
        proba = self.model.predict_proba(X)[0]
        best = int(proba.argmax())
        return Prediction(label=self.classes[best], proba={c: float(p) for c, p in zip(self.classes, proba)}, confidence=float(proba[best]))
