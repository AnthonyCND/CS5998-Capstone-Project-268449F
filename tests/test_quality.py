import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.data.load import load_subset
from src.data.quality import CHANNELS, find_cycle_gaps, fit_medians, impute, missing_report


@pytest.fixture
def telemetry():
    """Unit 1 and unit 2, three cycles each, with every channel set to 1.0."""
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2, 2], "cycle": [1, 2, 3, 1, 2, 3]})
    for channel in CHANNELS:
        df[channel] = 1.0
    return df


def test_missing_report_lists_only_channels_with_gaps(telemetry):
    telemetry.loc[[0, 1, 2], "sensor_2"] = np.nan
    report = missing_report(telemetry)
    assert report.index.tolist() == ["sensor_2"]
    assert report.loc["sensor_2", "missing"] == 3
    assert report.loc["sensor_2", "percent"] == pytest.approx(50.0)


def test_missing_report_empty_for_complete_data(telemetry):
    assert missing_report(telemetry).empty


def test_find_cycle_gaps_reports_skips():
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 5, 3, 4]})
    gaps = find_cycle_gaps(df)
    assert gaps.to_dict("records") == [
        {"unit": 1, "after_cycle": 2, "missing_cycles": 2},
        {"unit": 2, "after_cycle": 0, "missing_cycles": 2},
    ]


def test_find_cycle_gaps_empty_for_consecutive_cycles(telemetry):
    assert find_cycle_gaps(telemetry).empty


def test_impute_forward_fills_from_earlier_cycle(telemetry):
    telemetry.loc[0, "sensor_2"] = 5.0
    telemetry.loc[1, "sensor_2"] = np.nan
    medians = fit_medians(telemetry)
    assert impute(telemetry, medians).loc[1, "sensor_2"] == 5.0


def test_impute_does_not_fill_across_units(telemetry):
    telemetry["sensor_2"] = [7.0, 7.0, 7.0, np.nan, 3.0, 3.0]
    medians = pd.Series(100.0, index=CHANNELS)
    assert impute(telemetry, medians).loc[3, "sensor_2"] == 100.0


def test_impute_does_not_fill_backwards(telemetry):
    telemetry["sensor_2"] = [np.nan, 9.0, 9.0, 1.0, 1.0, 1.0]
    medians = pd.Series(100.0, index=CHANNELS)
    assert impute(telemetry, medians).loc[0, "sensor_2"] == 100.0


def test_impute_orders_by_cycle_before_filling(telemetry):
    shuffled = telemetry.iloc[[1, 0, 2, 3, 4, 5]].copy()
    shuffled["sensor_2"] = [np.nan, 4.0, 1.0, 1.0, 1.0, 1.0]  # cycle 2 is NaN, cycle 1 is 4.0
    medians = pd.Series(100.0, index=CHANNELS)
    out = impute(shuffled, medians)
    assert out["cycle"].tolist() == [1, 2, 3, 1, 2, 3]
    assert out.loc[1, "sensor_2"] == 4.0


def test_test_units_use_medians_from_training(telemetry):
    medians = fit_medians(telemetry)  # every training median is 1.0
    test = telemetry.copy()
    test[CHANNELS] = 50.0
    test.loc[0, "sensor_2"] = np.nan
    assert impute(test, medians).loc[0, "sensor_2"] == 1.0


def test_impute_does_not_modify_input(telemetry):
    telemetry.loc[1, "sensor_2"] = np.nan
    impute(telemetry, fit_medians(telemetry))
    assert np.isnan(telemetry.loc[1, "sensor_2"])


def test_fit_medians_rejects_empty_channel(telemetry):
    telemetry["sensor_2"] = np.nan
    with pytest.raises(ValueError, match="sensor_2"):
        fit_medians(telemetry)


REAL_DATA = resolve_path(load_config()["data"]["raw_dir"])


@pytest.mark.skipif(not (REAL_DATA / "train_FD001.txt").exists(), reason="CMAPSS not downloaded")
@pytest.mark.parametrize("subset", ["FD001", "FD002"])
def test_real_cmapss_is_complete(subset):
    train, test, _ = load_subset(subset)
    for df in (train, test):
        assert missing_report(df).empty
        assert find_cycle_gaps(df).empty
        pd.testing.assert_frame_equal(impute(df, fit_medians(train)), df)
