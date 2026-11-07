import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.data.load import load_subset
from src.features.conditions import fit_condition_stats, normalise_by_condition
from src.features.labels import add_train_rul
from src.features.windows import make_sequences, rolling_features

SENSORS = ["sensor_2", "sensor_3"]
WINDOW = 3


@pytest.fixture
def engines():
    """Unit 1 with 5 cycles and unit 2 with 2 cycles, in one condition."""
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 4, 5, 1, 2],
            "op_setting_1": 0.0,
            "sensor_2": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0],
            "sensor_3": [5.0, 4.0, 3.0, 2.0, 1.0, 8.0, 6.0],
        }
    )


@pytest.fixture
def stats(engines):
    return fit_condition_stats(engines, SENSORS)


@pytest.fixture
def normalised(engines, stats):
    return normalise_by_condition(engines, stats)[SENSORS].to_numpy()


def test_one_window_per_cycle(engines, stats):
    X, index = make_sequences(engines, stats, WINDOW)
    assert X.shape == (7, WINDOW, 2)
    assert X.dtype == np.float32
    assert index[["unit", "cycle"]].values.tolist() == engines[["unit", "cycle"]].values.tolist()


def test_window_ends_at_its_cycle(engines, stats, normalised):
    X, _ = make_sequences(engines, stats, WINDOW)
    np.testing.assert_allclose(X[4], normalised[2:5], rtol=1e-6)  # unit 1, cycles 3 to 5


def test_short_history_is_padded_with_first_reading(engines, stats, normalised):
    X, _ = make_sequences(engines, stats, WINDOW)
    np.testing.assert_allclose(X[0], normalised[[0, 0, 0]], rtol=1e-6)
    np.testing.assert_allclose(X[1], normalised[[0, 0, 1]], rtol=1e-6)


def test_windows_do_not_cross_engines(engines, stats, normalised):
    X, _ = make_sequences(engines, stats, WINDOW)
    np.testing.assert_allclose(X[5], normalised[[5, 5, 5]], rtol=1e-6)  # unit 2, cycle 1


def test_last_only_gives_one_window_per_engine(engines, stats, normalised):
    X, index = make_sequences(engines, stats, WINDOW, last_only=True)
    assert X.shape == (2, WINDOW, 2)
    assert index[["unit", "cycle"]].values.tolist() == [[1, 5], [2, 2]]
    np.testing.assert_allclose(X[1], normalised[[5, 5, 6]], rtol=1e-6)


def test_input_order_does_not_matter(engines, stats):
    X, index = make_sequences(engines, stats, WINDOW)
    X_shuffled, index_shuffled = make_sequences(engines.sample(frac=1, random_state=0), stats, WINDOW)
    np.testing.assert_array_equal(X, X_shuffled)
    pd.testing.assert_frame_equal(index, index_shuffled)


def test_rul_is_carried_through(engines, stats):
    labelled = add_train_rul(engines, cap=None)
    _, index = make_sequences(labelled, stats, WINDOW)
    assert index["rul"].tolist() == [4, 3, 2, 1, 0, 1, 0]


def test_rolling_features_summarise_each_window(engines, stats, normalised):
    features = rolling_features(engines, stats, WINDOW)
    assert features.columns.tolist() == [
        "unit", "cycle",
        "sensor_2", "sensor_3", "sensor_2_mean", "sensor_3_mean", "sensor_2_std", "sensor_3_std",
    ]
    row = features.iloc[1]  # unit 1, cycle 2: window is cycles 1, 1, 2
    window = normalised[[0, 0, 1]]
    np.testing.assert_allclose(row[["sensor_2", "sensor_3"]], window[-1])
    np.testing.assert_allclose(row[["sensor_2_mean", "sensor_3_mean"]], window.mean(axis=0))
    np.testing.assert_allclose(row[["sensor_2_std", "sensor_3_std"]], window.std(axis=0))


def test_rolling_features_last_only(engines, stats):
    features = rolling_features(engines, stats, WINDOW, last_only=True)
    assert features[["unit", "cycle"]].values.tolist() == [[1, 5], [2, 2]]


def test_missing_values_are_rejected(engines, stats):
    engines.loc[2, "sensor_2"] = np.nan
    with pytest.raises(ValueError, match="impute"):
        make_sequences(engines, stats, WINDOW)


def test_window_must_be_positive(engines, stats):
    with pytest.raises(ValueError, match="window"):
        rolling_features(engines, stats, 0)


CONFIG = load_config()
REAL_DATA = resolve_path(CONFIG["data"]["raw_dir"])


@pytest.mark.skipif(not (REAL_DATA / "train_FD002.txt").exists(), reason="CMAPSS not downloaded")
@pytest.mark.parametrize("subset", ["FD001", "FD002"])
def test_real_cmapss_shapes(subset):
    prep = CONFIG["preprocessing"]
    window = prep["window_size"]
    dropped = set(prep["drop_sensors"][subset])
    sensors = [f"sensor_{i}" for i in range(1, 22) if i not in dropped]
    train, test, _ = load_subset(subset)
    stats = fit_condition_stats(train, sensors)

    X_train, index_train = make_sequences(train, stats, window)
    assert X_train.shape == (len(train), window, len(sensors))
    assert np.isfinite(X_train).all()

    # One window per test engine, including those shorter than the window.
    X_test, index_test = make_sequences(test, stats, window, last_only=True)
    assert X_test.shape == (test["unit"].nunique(), window, len(sensors))
    assert (index_test["cycle"].to_numpy() == test.groupby("unit")["cycle"].max().to_numpy()).all()

    features = rolling_features(train, stats, window)
    assert features.shape == (len(train), 2 + 3 * len(sensors))
