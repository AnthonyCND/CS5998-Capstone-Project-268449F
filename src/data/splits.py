"""Train/validation splits by engine.

Engines differ in their baseline sensor levels (see the FD001 EDA), so a model
validated on other cycles of an engine it trained on can recognise that engine
and overstate its accuracy. Every split here keeps each engine's cycles wholly
in training or wholly in validation, like scikit-learn's GroupKFold, but with
a seeded shuffle so folds are reproducible and not ordered by unit number.

Split the raw training data first, then fit imputation medians, condition
statistics and any scaler on each fold's training engines only.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd


def engine_folds(units, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Assign engines to ``n_splits`` validation folds of near-equal size.

    Returns one ``(train_units, val_units)`` pair per fold, both sorted. Each
    engine is in exactly one validation fold.
    """
    units = np.unique(np.asarray(units))
    if not 2 <= n_splits <= len(units):
        raise ValueError(f"n_splits must be between 2 and {len(units)} engines, got {n_splits}")
    shuffled = np.random.default_rng(seed).permutation(units)
    folds = np.array_split(shuffled, n_splits)
    return [(np.setdiff1d(units, val), np.sort(val)) for val in folds]


def iter_folds(
    df: pd.DataFrame, n_splits: int, seed: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``(train, val)`` row subsets of ``df`` for each engine fold."""
    for train_units, val_units in engine_folds(df["unit"], n_splits, seed):
        train = df[df["unit"].isin(train_units)].reset_index(drop=True)
        val = df[df["unit"].isin(val_units)].reset_index(drop=True)
        yield train, val
