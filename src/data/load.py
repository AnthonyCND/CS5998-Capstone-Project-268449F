"""Load raw CMAPSS text files into pandas DataFrames."""

from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OP_SETTINGS, *SENSORS]

VALID_SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _raw_dir(raw_dir: str | Path | None) -> Path:
    if raw_dir is None:
        raw_dir = load_config()["data"]["raw_dir"]
    return resolve_path(raw_dir)


def _check_subset(subset: str) -> None:
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}, expected one of {VALID_SUBSETS}")


def load_split(subset: str, split: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the ``train`` or ``test`` trajectories of a subset.

    Returns one row per (unit, cycle) with the 26 named CMAPSS columns.
    """
    _check_subset(subset)
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    path = _raw_dir(raw_dir) / f"{split}_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected {len(COLUMNS)} columns, found {df.shape[1]}")

    df.columns = COLUMNS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df.sort_values(["unit", "cycle"], ignore_index=True)


def load_rul(subset: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load ground-truth RUL for the last cycle of each test unit."""
    _check_subset(subset)
    path = _raw_dir(raw_dir) / f"RUL_{subset}.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
    rul.insert(0, "unit", range(1, len(rul) + 1))
    return rul


def load_subset(
    subset: str, raw_dir: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train trajectories, test trajectories and test RUL for a subset."""
    train = load_split(subset, "train", raw_dir)
    test = load_split(subset, "test", raw_dir)
    rul = load_rul(subset, raw_dir)

    if rul["unit"].nunique() != test["unit"].nunique():
        raise ValueError(
            f"{subset}: {test['unit'].nunique()} test units but {len(rul)} RUL values"
        )
    return train, test, rul
