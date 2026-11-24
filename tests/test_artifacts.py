import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline
from src.models.artifacts import TrainedModel, load, save
from src.models.baselines import fit
from src.models.cnn_bilstm import train_model
from tests.test_baselines import SENSORS, SMALL, engines

PARAMS = {"conv_channels": [4], "kernel_size": 3, "lstm_hidden": 4, "lstm_layers": 1, "dense": 4, "dropout": 0.0}
TRAINING = {"batch_size": 16, "learning_rate": 0.01, "max_epochs": 2, "patience": 2}
CAP = 10.0
WINDOW = 3


@pytest.fixture
def train():
    return engines([1, 2, 3])


@pytest.fixture
def new_engines():
    return engines([7, 8], cycles=12).drop(columns="rul")


def trained_baseline(name, train):
    pipeline, model = fit(name, SMALL[name], 0, train, SENSORS, WINDOW)
    return TrainedModel(name, pipeline, model, CAP, {"subset": "test"})


def trained_cnn(train):
    pipeline = FeaturePipeline(SENSORS, WINDOW).fit(train)
    X, index = pipeline.transform_sequences(train)
    model, _ = train_model(X, index["rul"].to_numpy(), PARAMS, TRAINING, CAP, seed=0, epochs=2)
    return TrainedModel("cnn_bilstm", pipeline, model, CAP, {"subset": "test"})


@pytest.mark.parametrize("name", sorted(SMALL))
def test_saved_baseline_predicts_the_same(tmp_path, train, new_engines, name):
    trained = trained_baseline(name, train)
    loaded = load(save(trained, tmp_path / f"{name}.joblib"))
    pd.testing.assert_frame_equal(loaded.predict(new_engines), trained.predict(new_engines))
    assert loaded.kind == name and loaded.rul_cap == CAP and loaded.info == {"subset": "test"}


def test_saved_cnn_predicts_the_same(tmp_path, train, new_engines):
    trained = trained_cnn(train)
    loaded = load(save(trained, tmp_path / "cnn.joblib"))
    pd.testing.assert_frame_equal(loaded.predict(new_engines), trained.predict(new_engines))
    assert loaded.model.config == trained.model.config
    assert not loaded.model.training  # ready for inference (dropout off)


def test_cnn_is_stored_as_weights_not_a_module(tmp_path, train):
    payload = joblib.load(save(trained_cnn(train), tmp_path / "cnn.joblib"))
    assert set(payload["model"]) == {"state_dict", "config"}


def test_saved_pipeline_keeps_training_statistics(tmp_path, train):
    trained = trained_baseline("random_forest", train)
    loaded = load(save(trained, tmp_path / "rf.joblib"))
    pd.testing.assert_frame_equal(loaded.pipeline.stats.mean, trained.pipeline.stats.mean)
    pd.testing.assert_series_equal(loaded.pipeline.medians, trained.pipeline.medians)


def test_predict_gives_one_non_negative_row_per_engine(train, new_engines):
    out = trained_baseline("xgboost", train).predict(new_engines)
    assert out.columns.tolist() == ["unit", "cycle", "predicted_rul"]
    assert out[["unit", "cycle"]].values.tolist() == [[7, 12], [8, 12]]
    assert (out["predicted_rul"] >= 0).all()


def test_predict_every_cycle(train, new_engines):
    out = trained_baseline("random_forest", train).predict(new_engines, last_only=False)
    assert len(out) == len(new_engines)


def test_unknown_kind_is_rejected(train):
    pipeline = FeaturePipeline(SENSORS, WINDOW).fit(train)
    with pytest.raises(ValueError, match="kind"):
        TrainedModel("lstm", pipeline, None, CAP)


def test_other_format_version_is_rejected(tmp_path, train):
    path = save(trained_baseline("random_forest", train), tmp_path / "rf.joblib")
    payload = joblib.load(path)
    payload["format_version"] = 99
    joblib.dump(payload, path)
    with pytest.raises(ValueError, match="format version"):
        load(path)


def test_save_creates_missing_folders(tmp_path, train):
    path = save(trained_baseline("random_forest", train), tmp_path / "a" / "b" / "rf.joblib")
    assert path.exists()


def test_baselines_run_subset_saves_each_model(tmp_path, monkeypatch, train):
    from src.models import baselines

    test = engines([7, 8], cycles=12)
    monkeypatch.setattr(baselines, "load_subset", lambda subset: (train.drop(columns="rul"), test.drop(columns="rul"), None))
    monkeypatch.setattr(baselines, "add_test_rul", lambda df, rul, cap: test)
    config = {
        "seed": 0,
        "preprocessing": {"rul_cap": CAP, "window_size": WINDOW, "rolling_stats": ["mean", "std"],
                          "drop_sensors": {"FDX": [i for i in range(1, 22) if f"sensor_{i}" not in SENSORS]}},
        "validation": {"n_splits": 3},
        "models": SMALL,
    }
    baselines.run_subset("FDX", config, models_dir=tmp_path, tag="_x")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["random_forest_FDX_x.joblib", "xgboost_FDX_x.joblib"]
    assert load(tmp_path / "xgboost_FDX_x.joblib").info["subset"] == "FDX"
    assert np.isfinite(load(tmp_path / "random_forest_FDX_x.joblib").predict(test)["predicted_rul"]).all()
