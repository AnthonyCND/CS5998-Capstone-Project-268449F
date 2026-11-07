import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.data.load import load_subset
from src.features.conditions import (
    fit_condition_stats,
    label_conditions,
    normalise_by_condition,
)
from src.features.labels import add_train_rul

SENSORS = ["sensor_2", "sensor_5"]


@pytest.fixture
def train():
    """Two conditions (op_setting_1 near 0 and near 10) at different sensor levels.

    sensor_2 varies within each condition; sensor_5 is constant within each.
    """
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "op_setting_1": [0.002, 10.001, -0.003, 9.998, 0.001, 10.004, 0.0, 9.996],
            "sensor_2": [1.0, 101.0, 3.0, 103.0, 1.0, 101.0, 3.0, 103.0],
            "sensor_5": [7.0, 50.0, 7.0, 50.0, 7.0, 50.0, 7.0, 50.0],
        }
    )


def test_label_conditions_rounds_op_setting_1(train):
    assert label_conditions(train).tolist() == [0, 10, 0, 10, 0, 10, 0, 10]


def test_label_conditions_rejects_ambiguous_setting(train):
    train.loc[0, "op_setting_1"] = 4.6
    with pytest.raises(ValueError, match="op_setting_1"):
        label_conditions(train)


def test_fit_condition_stats_per_condition(train):
    stats = fit_condition_stats(train, SENSORS)
    assert stats.mean.loc[0, "sensor_2"] == 2.0
    assert stats.mean.loc[10, "sensor_2"] == 102.0
    assert stats.std.loc[0, "sensor_2"] == 1.0
    assert stats.std.loc[10, "sensor_2"] == 1.0


def test_constant_sensor_has_no_std(train):
    stats = fit_condition_stats(train, SENSORS)
    assert stats.std["sensor_5"].isna().all()


def test_normalise_removes_condition_level(train):
    out = normalise_by_condition(train, fit_condition_stats(train, SENSORS))
    assert out["sensor_2"].tolist() == [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]


def test_constant_sensor_becomes_zero(train):
    out = normalise_by_condition(train, fit_condition_stats(train, SENSORS))
    assert (out["sensor_5"] == 0.0).all()


def test_test_units_use_training_statistics(train):
    stats = fit_condition_stats(train, SENSORS)
    test = train.iloc[[0, 1]].copy()
    test["sensor_2"] = [12.0, 92.0]  # condition 0 mean 2, condition 10 mean 102, std 1
    out = normalise_by_condition(test, stats)
    assert out["sensor_2"].tolist() == [10.0, -10.0]


def test_normalise_rejects_unseen_condition(train):
    stats = fit_condition_stats(train, SENSORS)
    test = train.iloc[[0]].copy()
    test["op_setting_1"] = 42.0
    with pytest.raises(ValueError, match="42"):
        normalise_by_condition(test, stats)


def test_normalise_keeps_other_columns_and_input(train):
    original = train.copy()
    out = normalise_by_condition(train, fit_condition_stats(train, SENSORS))
    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(out.drop(columns=SENSORS), train.drop(columns=SENSORS))


def test_missing_values_stay_missing(train):
    stats = fit_condition_stats(train, SENSORS)
    train.loc[0, "sensor_2"] = np.nan
    assert np.isnan(normalise_by_condition(train, stats).loc[0, "sensor_2"])


CONFIG = load_config()
REAL_DATA = resolve_path(CONFIG["data"]["raw_dir"])


@pytest.mark.skipif(not (REAL_DATA / "train_FD002.txt").exists(), reason="CMAPSS not downloaded")
@pytest.mark.parametrize("subset, n_conditions", [("FD001", 1), ("FD002", 6)])
def test_real_cmapss_normalisation(subset, n_conditions):
    prep = CONFIG["preprocessing"]
    dropped = set(prep["drop_sensors"][subset])
    sensors = [f"sensor_{i}" for i in range(1, 22) if i not in dropped]
    train, test, _ = load_subset(subset)

    stats = fit_condition_stats(train, sensors)
    assert len(stats.mean) == n_conditions
    out = normalise_by_condition(train, stats)

    # Within every condition, each kept sensor now has mean 0 and std 1.
    by_condition = out[sensors].groupby(label_conditions(train))
    np.testing.assert_allclose(by_condition.mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(by_condition.std(ddof=0), 1.0, atol=1e-9)

    # The degradation signal is recovered (EDA: sensor 11 correlates about -0.77).
    rul = add_train_rul(train, cap=prep["rul_cap"])["rul"]
    assert out["sensor_11"].corr(rul) < -0.7

    # Test engines only use conditions seen in training.
    assert normalise_by_condition(test, stats)[sensors].notna().all().all()
