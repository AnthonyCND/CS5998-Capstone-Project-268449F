"""Saving and loading trained models together with their fitted pipeline.

A saved model is one file in ``models/`` holding everything needed to score new
engines: the fitted preprocessing (imputation medians and per-condition sensor
statistics), the trained model and the RUL cap. Loading it gives a
``TrainedModel`` that predicts from raw CMAPSS rows, which is what the scoring
API will use.

The CNN-BiLSTM is stored as its weights and architecture settings, not as a
pickled module, so a file stays loadable if the model code is reorganised.
Files are written with joblib (pickle); only load files this project created.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline

FORMAT_VERSION = 1
KINDS = ("random_forest", "xgboost", "cnn_bilstm")


@dataclass
class TrainedModel:
    """A fitted pipeline and model that predict RUL in cycles from raw CMAPSS rows."""

    kind: str
    pipeline: FeaturePipeline
    model: object
    rul_cap: float
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown model kind {self.kind!r}; expected one of {list(KINDS)}")

    def predict(self, df: pd.DataFrame, *, last_only: bool = True) -> pd.DataFrame:
        """Return ``unit``, ``cycle`` and ``predicted_rul`` for each engine's last cycle.

        With ``last_only=False``, returns a prediction for every cycle.
        """
        if self.kind == "cnn_bilstm":
            from src.models.cnn_bilstm import predict

            X, index = self.pipeline.transform_sequences(df, last_only=last_only)
            pred = predict(self.model, X, self.rul_cap)
        else:
            X, index = self.pipeline.transform(df, last_only=last_only)
            pred = np.clip(self.model.predict(X), 0, None)
        out = index[["unit", "cycle"]].reset_index(drop=True)
        out["predicted_rul"] = pred
        return out


def save(trained: TrainedModel, path: str | Path) -> Path:
    """Write a trained model and its pipeline to one file; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = trained.model
    if trained.kind == "cnn_bilstm":
        model = {"state_dict": model.state_dict(), "config": model.config}
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": trained.kind,
        "pipeline": trained.pipeline,
        "model": model,
        "rul_cap": trained.rul_cap,
        "info": trained.info,
    }
    joblib.dump(payload, path)
    return path


def load(path: str | Path) -> TrainedModel:
    """Read a file written by ``save``."""
    payload = joblib.load(path)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} has format version {payload.get('format_version')}, expected {FORMAT_VERSION}")
    model = payload["model"]
    if payload["kind"] == "cnn_bilstm":
        from src.models.cnn_bilstm import CNNBiLSTM

        net = CNNBiLSTM(**model["config"])
        net.load_state_dict(model["state_dict"])
        model = net.eval()
    return TrainedModel(payload["kind"], payload["pipeline"], model, payload["rul_cap"], payload["info"])
