"""Evaluation metrics for RUL predictions.

Errors are ``predicted - true``: positive means the model predicted more life
than the engine had left (late), negative means early.
"""

import numpy as np


def _errors(y_true, y_pred) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or y_true.size == 0:
        raise ValueError(f"Expected two non-empty 1D arrays of equal length, got {y_true.shape} and {y_pred.shape}")
    return y_pred - y_true


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in cycles."""
    return float(np.sqrt(np.mean(_errors(y_true, y_pred) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 scoring function (Saxena et al., 2008); lower is better.

    Sums ``exp(-d/13) - 1`` for early predictions and ``exp(d/10) - 1`` for late
    ones, so late predictions, which risk running an engine to failure, cost
    more than early ones of the same size. It is a sum, so it grows with the
    number of engines: compare it only across results on the same test set.
    """
    d = _errors(y_true, y_pred)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13), np.exp(d / 10)) - 1))
