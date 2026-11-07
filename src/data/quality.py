"""Data quality checks and missing-value imputation for CMAPSS telemetry.

Imputation forward-fills within each engine, then falls back to per-channel
medians. Fit the medians on training units only and reuse them for validation
and test units, so no information leaks from those units into training.
"""

import pandas as pd

from src.data.load import OP_SETTINGS, SENSORS

CHANNELS = [*OP_SETTINGS, *SENSORS]


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return the missing-value count and percentage of each channel with gaps."""
    counts = df[CHANNELS].isna().sum()
    report = pd.DataFrame({"missing": counts, "percent": 100 * counts / len(df)})
    return report[report["missing"] > 0]


def find_cycle_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per run of skipped cycles, including skips at a unit's start.

    Columns: ``unit``, ``after_cycle`` (last cycle seen before the gap, 0 if the
    unit starts late) and ``missing_cycles``. These rows are absent from the data
    rather than NaN, so ``impute`` cannot fill them.
    """
    ordered = df.sort_values(["unit", "cycle"])
    previous = ordered.groupby("unit")["cycle"].shift(fill_value=0)
    skipped = ordered["cycle"] - previous - 1
    has_gap = skipped > 0
    return pd.DataFrame(
        {
            "unit": ordered.loc[has_gap, "unit"],
            "after_cycle": previous[has_gap],
            "missing_cycles": skipped[has_gap],
        }
    ).reset_index(drop=True)


def fit_medians(train: pd.DataFrame) -> pd.Series:
    """Compute per-channel medians from training units."""
    medians = train[CHANNELS].median()
    empty = medians.index[medians.isna()].tolist()
    if empty:
        raise ValueError(f"Channels with no observed values: {empty}")
    return medians


def impute(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Return a copy, sorted by unit and cycle, with missing channel values filled.

    Each gap takes the unit's most recent earlier reading, so values never cross
    engines or come from later cycles. Gaps with no earlier reading take the
    channel median.
    """
    out = df.sort_values(["unit", "cycle"], ignore_index=True)
    out[CHANNELS] = out.groupby("unit")[CHANNELS].ffill().fillna(medians)
    return out
