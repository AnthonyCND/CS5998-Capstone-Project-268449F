"""Remaining Useful Life (RUL) target labelling.

Uses a piecewise-linear target: RUL stays flat at ``cap`` while the engine is
healthy, then decreases linearly to 0 at failure. Pass ``cap=None`` for the
plain linear (uncapped) RUL.
"""

import pandas as pd


def _apply_cap(rul: pd.Series, cap: int | None) -> pd.Series:
    if cap is None:
        return rul
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    return rul.clip(upper=cap)


def add_train_rul(df: pd.DataFrame, *, cap: int | None) -> pd.DataFrame:
    """Return a copy of training trajectories with a ``rul`` column.

    Training engines run to failure, so RUL at each cycle is the unit's last
    cycle minus the current cycle.
    """
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = _apply_cap(last_cycle - out["cycle"], cap)
    return out


def add_test_rul(df: pd.DataFrame, rul: pd.DataFrame, *, cap: int | None) -> pd.DataFrame:
    """Return a copy of test trajectories with a ``rul`` column.

    Test engines stop before failure; ``rul`` (from ``load_rul``) gives the true
    RUL at each unit's last recorded cycle, and earlier cycles count up from it.
    """
    missing = set(df["unit"]) - set(rul["unit"])
    if missing:
        raise ValueError(f"No ground-truth RUL for test units {sorted(missing)}")

    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    rul_at_end = out["unit"].map(rul.set_index("unit")["rul"])
    out["rul"] = _apply_cap(rul_at_end + last_cycle - out["cycle"], cap)
    return out
