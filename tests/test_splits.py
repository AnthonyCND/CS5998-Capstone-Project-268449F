import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.data.load import load_split
from src.data.splits import engine_folds, iter_folds

UNITS = np.arange(1, 11)


def test_each_engine_validated_exactly_once():
    folds = engine_folds(UNITS, 3, seed=0)
    validated = np.concatenate([val for _, val in folds])
    assert sorted(validated) == UNITS.tolist()


def test_train_and_val_engines_are_disjoint_and_complete():
    for train, val in engine_folds(UNITS, 3, seed=0):
        assert not set(train) & set(val)
        assert sorted([*train, *val]) == UNITS.tolist()


def test_fold_sizes_differ_by_at_most_one():
    sizes = [len(val) for _, val in engine_folds(UNITS, 3, seed=0)]
    assert sorted(sizes) == [3, 3, 4]


def test_repeated_units_count_once():
    rows = np.repeat(UNITS, 5)  # one entry per cycle, as in a data frame column
    folds = engine_folds(rows, 5, seed=0)
    assert [len(val) for _, val in folds] == [2, 2, 2, 2, 2]


def test_same_seed_gives_same_folds():
    first, second = engine_folds(UNITS, 5, seed=42), engine_folds(UNITS, 5, seed=42)
    for (train_a, val_a), (train_b, val_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)


def test_folds_are_shuffled_not_by_unit_order():
    vals = [val.tolist() for _, val in engine_folds(UNITS, 5, seed=42)]
    assert vals != [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


@pytest.mark.parametrize("n_splits", [1, 11])
def test_invalid_fold_count_is_rejected(n_splits):
    with pytest.raises(ValueError, match="n_splits"):
        engine_folds(UNITS, n_splits, seed=0)


def test_iter_folds_keeps_every_cycle_of_an_engine_together():
    df = pd.DataFrame({"unit": np.repeat(UNITS, 3), "cycle": np.tile([1, 2, 3], 10)})
    for train, val in iter_folds(df, 5, seed=0):
        assert len(train) + len(val) == len(df)
        assert not set(train["unit"]) & set(val["unit"])
        assert (val.groupby("unit").size() == 3).all()


CONFIG = load_config()
REAL_DATA = resolve_path(CONFIG["data"]["raw_dir"])


@pytest.mark.skipif(not (REAL_DATA / "train_FD002.txt").exists(), reason="CMAPSS not downloaded")
@pytest.mark.parametrize("subset, per_fold", [("FD001", {20}), ("FD002", {52})])
def test_real_cmapss_folds(subset, per_fold):
    train = load_split(subset, "train")
    n_splits = CONFIG["validation"]["n_splits"]
    folds = list(iter_folds(train, n_splits, CONFIG["seed"]))
    assert len(folds) == n_splits
    assert {val["unit"].nunique() for _, val in folds} == per_fold
    assert sum(len(val) for _, val in folds) == len(train)
