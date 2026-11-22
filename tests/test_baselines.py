import numpy as np
import pandas as pd
import pytest

from src.data.quality import CHANNELS
from src.features.labels import add_train_rul
from src.features.pipeline import FeaturePipeline, kept_sensors
from src.models.baselines import build_model, fit_predict

SENSORS = ["sensor_2", "sensor_3"]
SMALL = {"random_forest": {"n_estimators": 10}, "xgboost": {"n_estimators": 10}}


def engines(units, cycles=20, level=0.0):
    """Engines whose sensor_2 rises linearly towards failure, all in one condition."""
    rows = []
    for unit in units:
        for cycle in range(1, cycles + 1):
            row = {c: 1.0 for c in CHANNELS}
            row.update(unit=unit, cycle=cycle, op_setting_1=0.0)
            row["sensor_2"] = level + cycle
            row["sensor_3"] = level + 2.0 * cycle
            rows.append(row)
    return add_train_rul(pd.DataFrame(rows), cap=10)


def test_kept_sensors_drops_by_number():
    sensors = kept_sensors([1, 5, 6, 10, 16, 18, 19])
    assert len(sensors) == 14
    assert "sensor_1" not in sensors and "sensor_11" in sensors and "sensor_10" not in sensors


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="lstm"):
        build_model("lstm", {}, seed=0)


def test_pipeline_statistics_come_from_training_engines_only():
    train, other = engines([1, 2]), engines([3], level=1000.0)
    pipeline = FeaturePipeline(SENSORS, window=3).fit(train)
    assert pipeline.stats.mean.loc[0, "sensor_2"] == pytest.approx(10.5)
    features, _ = pipeline.transform(other)
    assert (features["sensor_2"] > 100).all()  # far from training levels, not re-centred


def test_transform_separates_features_from_index():
    pipeline = FeaturePipeline(SENSORS, window=3).fit(engines([1]))
    features, index = pipeline.transform(engines([1]))
    assert index.columns.tolist() == ["unit", "cycle", "rul"]
    assert features.shape == (20, 6)  # latest, mean and std of two sensors


def test_pipeline_uses_the_requested_statistics():
    pipeline = FeaturePipeline(SENSORS, window=3, summaries=["min", "max"]).fit(engines([1]))
    features, _ = pipeline.transform(engines([1]))
    assert features.columns.tolist() == [*SENSORS, *(f"{s}_min" for s in SENSORS), *(f"{s}_max" for s in SENSORS)]


@pytest.mark.parametrize("name", sorted(SMALL))
def test_fit_predict_last_only_gives_one_non_negative_prediction_per_engine(name):
    pred, index = fit_predict(name, SMALL[name], 0, engines([1, 2, 3]), engines([4, 5]), SENSORS, 3, last_only=True)
    assert index["unit"].tolist() == [4, 5]
    assert len(pred) == 2 and (pred >= 0).all()


@pytest.mark.parametrize("name", sorted(SMALL))
def test_fit_predict_learns_the_trend(name):
    pred, index = fit_predict(name, SMALL[name], 0, engines([1, 2, 3]), engines([4]), SENSORS, 3)
    assert np.corrcoef(pred, index["rul"])[0, 1] > 0.9
