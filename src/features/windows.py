"""Sliding-window inputs for the models.

Every cycle gets one window of the ``window`` most recent cycles of its own
engine, ending at that cycle, so no window uses future cycles or another
engine's readings. Where an engine has fewer earlier cycles than the window
needs (the start of every training engine, and six FD002 test engines), the
window is padded at the start by repeating the engine's first reading.

Both builders normalise sensors within their operating condition first, using
statistics fitted on training units, so windows never mix condition levels.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.features.conditions import ConditionStats, normalise_by_condition


def _windows(values: np.ndarray, window: int) -> np.ndarray:
    """Return a (cycles, window, sensors) view of one engine's padded readings."""
    padded = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values])
    return sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)


def _engine_windows(
    df: pd.DataFrame, stats: ConditionStats, window: int, last_only: bool
) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Yield (index rows, windows) for each engine, in unit and cycle order."""
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    sensors = stats.mean.columns.tolist()
    data = normalise_by_condition(df, stats).sort_values(["unit", "cycle"], ignore_index=True)
    if data[sensors].isna().any().any():
        raise ValueError("Sensor values are missing; impute before building windows")

    keys = ["unit", "cycle", *(["rul"] if "rul" in data else [])]
    for _, engine in data.groupby("unit", sort=False):
        windows = _windows(engine[sensors].to_numpy(dtype=float), window)
        index = engine[keys]
        if last_only:
            windows, index = windows[-1:], index.iloc[-1:]
        yield index, windows


def make_sequences(
    df: pd.DataFrame, stats: ConditionStats, window: int, *, last_only: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build windows of normalised sensor readings for the CNN-BiLSTM.

    Returns a float32 array of shape (windows, ``window``, sensors) and a frame
    with the ``unit``, ``cycle`` (and ``rul``, if present) each window ends at.
    With ``last_only``, each engine gets one window ending at its last cycle,
    as used for test engines.
    """
    parts = list(_engine_windows(df, stats, window, last_only))
    X = np.concatenate([w for _, w in parts]).astype(np.float32)
    index = pd.concat([i for i, _ in parts], ignore_index=True)
    return X, index


def rolling_features(
    df: pd.DataFrame, stats: ConditionStats, window: int, *, last_only: bool = False
) -> pd.DataFrame:
    """Build one row of window statistics per cycle for the tree models.

    For each sensor: the latest normalised reading (named after the sensor),
    and its mean (``_mean``) and standard deviation (``_std``) over the window.
    Rows also carry ``unit``, ``cycle`` and ``rul``, if present.
    """
    sensors = stats.mean.columns.tolist()
    frames = []
    for index, windows in _engine_windows(df, stats, window, last_only):
        values = np.hstack([windows[:, -1, :], windows.mean(axis=1), windows.std(axis=1)])
        columns = [*sensors, *(f"{s}_mean" for s in sensors), *(f"{s}_std" for s in sensors)]
        features = pd.DataFrame(values, columns=columns, index=index.index)
        frames.append(pd.concat([index, features], axis=1))
    return pd.concat(frames, ignore_index=True)
