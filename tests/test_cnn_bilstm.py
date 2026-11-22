import numpy as np
import pytest
import torch

from src.evaluation.metrics import nasa_score
from src.models.cnn_bilstm import CNNBiLSTM, predict, train_model

PARAMS = {"conv_channels": [8], "kernel_size": 3, "lstm_hidden": 8, "lstm_layers": 1, "dense": 8, "dropout": 0.0}
TRAINING = {"batch_size": 32, "learning_rate": 0.01, "max_epochs": 30, "patience": 5}
CAP = 125.0


def trend_data(n=256, window=10, sensors=3, seed=0):
    """Windows whose level sets the RUL: higher readings mean less life left."""
    rng = np.random.default_rng(seed)
    rul = rng.uniform(0, CAP, n)
    level = 1 - rul / CAP
    X = level[:, None, None] + 0.05 * rng.standard_normal((n, window, sensors))
    return X.astype(np.float32), rul


def test_forward_maps_windows_to_one_value_each():
    model = CNNBiLSTM(n_sensors=14, conv_channels=(32, 64), lstm_layers=2)
    assert model(torch.zeros(5, 30, 14)).shape == (5,)


def test_predictions_are_in_cycles_and_non_negative():
    X, _ = trend_data(n=20)
    model = CNNBiLSTM(n_sensors=3, **PARAMS)
    with torch.no_grad():
        model.head[-1].bias.fill_(-1.0)  # push raw outputs below 0
    assert (predict(model, X, CAP) >= 0).all()


def test_training_learns_a_simple_trend():
    X, y = trend_data()
    X_val, y_val = trend_data(n=64, seed=1)
    model, history = train_model(X, y, PARAMS, TRAINING, CAP, seed=0, X_val=X_val, y_val=y_val)
    assert np.corrcoef(predict(model, X_val, CAP), y_val)[0, 1] > 0.9
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_best_epoch_weights_are_restored():
    X, y = trend_data()
    X_val, y_val = trend_data(n=64, seed=1)
    model, history = train_model(X, y, PARAMS, TRAINING, CAP, seed=0, X_val=X_val, y_val=y_val)
    best = history["best_epoch"]
    assert history["val_nasa_score"][best - 1] == min(history["val_nasa_score"])
    assert nasa_score(y_val, predict(model, X_val, CAP)) == pytest.approx(min(history["val_nasa_score"]), rel=1e-5)


def test_fixed_epochs_without_validation():
    X, y = trend_data(n=64)
    _, history = train_model(X, y, PARAMS, TRAINING, CAP, seed=0, epochs=3)
    assert len(history["train_loss"]) == 3 and "best_epoch" not in history


def test_needs_validation_or_epochs():
    X, y = trend_data(n=16)
    with pytest.raises(ValueError, match="epochs"):
        train_model(X, y, PARAMS, TRAINING, CAP, seed=0)


def test_same_seed_gives_same_model():
    X, y = trend_data(n=64)
    first, _ = train_model(X, y, PARAMS, TRAINING, CAP, seed=3, epochs=2)
    second, _ = train_model(X, y, PARAMS, TRAINING, CAP, seed=3, epochs=2)
    np.testing.assert_allclose(predict(first, X, CAP), predict(second, X, CAP))
