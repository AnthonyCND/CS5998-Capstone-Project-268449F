"""Operating-condition labelling and per-condition sensor normalisation.

FD002 engines switch between six operating conditions, which set each sensor's
level far more than wear does. Z-scoring every sensor within its condition
removes that effect and leaves the degradation trend. FD001 has one condition,
so the same code reduces to an ordinary z-score.

Fit the statistics on training units only and reuse them for validation and
test units, so no information leaks from those units into training. Normalise
before building rolling windows, because consecutive cycles usually come from
different conditions.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

# Rows lie within 0.0053 of a whole number in FD001 and FD002 (see the EDA).
MAX_OFFSET = 0.1


class ConditionStats(NamedTuple):
    """Per-condition sensor means and standard deviations, indexed by condition.

    ``std`` is NaN for a sensor that is constant within a condition.
    """

    mean: pd.DataFrame
    std: pd.DataFrame


def label_conditions(df: pd.DataFrame) -> pd.Series:
    """Return each row's operating condition: ``op_setting_1`` rounded to an integer.

    In CMAPSS, rounded ``op_setting_1`` alone separates the operating conditions
    (0, 10, 20, 25, 35 and 42 in FD002; 0 in FD001), so no fitted model is needed.
    """
    setting = df["op_setting_1"]
    rounded = setting.round()
    far = (setting - rounded).abs() > MAX_OFFSET
    if far.any():
        raise ValueError(
            f"{int(far.sum())} rows have op_setting_1 more than {MAX_OFFSET} "
            "from a whole number, so their condition is unclear"
        )
    return rounded.astype(int).rename("condition")


def fit_condition_stats(train: pd.DataFrame, sensors: list[str]) -> ConditionStats:
    """Compute each sensor's mean and standard deviation within each condition."""
    grouped = train[sensors].groupby(label_conditions(train))
    mean = grouped.mean()
    std = grouped.std(ddof=0)
    # Exact check: a computed std can be ~1e-15 for a constant column.
    constant = grouped.nunique() == 1
    return ConditionStats(mean=mean, std=std.mask(constant))


def normalise_by_condition(df: pd.DataFrame, stats: ConditionStats) -> pd.DataFrame:
    """Return a copy with each sensor z-scored using its condition's training statistics.

    Sensors that were constant within a condition become 0 for that condition.
    Raises ``ValueError`` if a row's condition never appeared in training.
    """
    conditions = label_conditions(df)
    unseen = sorted(set(conditions) - set(stats.mean.index))
    if unseen:
        raise ValueError(f"Conditions not seen in training: {unseen}")

    sensors = stats.mean.columns.tolist()
    mean = stats.mean.loc[conditions].to_numpy()
    std = stats.std.loc[conditions].to_numpy()
    z = (df[sensors].to_numpy() - mean) / std

    out = df.copy()
    out[sensors] = np.where(np.isnan(std), 0.0, z)
    return out
