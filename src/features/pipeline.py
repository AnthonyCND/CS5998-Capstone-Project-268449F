"""Preprocessing fitted on training engines and shared by every model.

Fits imputation medians and per-condition sensor statistics on training
engines, then turns any split into model inputs: rolling-window features for
the tree models or windowed sequences for the CNN-BiLSTM.
"""

import numpy as np
import pandas as pd

from src.data.load import SENSORS
from src.data.quality import fit_medians, impute
from src.features.conditions import fit_condition_stats
from src.features.windows import make_sequences, rolling_features


def kept_sensors(drop: list[int]) -> list[str]:
    """Return sensor column names, leaving out the dropped sensor numbers."""
    return [s for s in SENSORS if int(s.split("_")[1]) not in drop]


class FeaturePipeline:
    """Imputation, condition normalisation and windowing fitted on training engines."""

    def __init__(self, sensors: list[str], window: int):
        self.sensors, self.window = sensors, window

    def fit(self, train: pd.DataFrame) -> "FeaturePipeline":
        self.medians = fit_medians(train)
        self.stats = fit_condition_stats(impute(train, self.medians), self.sensors)
        return self

    def transform(self, df: pd.DataFrame, *, last_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return rolling-window (features, index); index holds unit, cycle and rul."""
        features = rolling_features(impute(df, self.medians), self.stats, self.window, last_only=last_only)
        keys = [c for c in ("unit", "cycle", "rul") if c in features]
        return features.drop(columns=keys), features[keys]

    def transform_sequences(self, df: pd.DataFrame, *, last_only: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
        """Return (sequences, index): windows of shape (n, window, sensors)."""
        return make_sequences(impute(df, self.medians), self.stats, self.window, last_only=last_only)
