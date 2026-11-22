"""Hybrid 1D-CNN and stacked BiLSTM model for RUL prediction.

The convolution layers mix the sensor channels at each point in the window,
and the bidirectional LSTM layers learn how those patterns change over the
window. The model predicts RUL divided by ``rul_cap``, so its target lies in
[0, 1]; predictions are scaled back to cycles.

Validation follows the baselines: in each engine-level fold, preprocessing is
fitted on the fold's training engines, and the epoch with the lowest
validation NASA score is kept (PDD Section 5). The final model is retrained on
all training engines for the median of the folds' best epoch counts, then
scored on each test engine's last window against uncapped and capped RUL.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import load_config, resolve_path
from src.data.load import load_subset
from src.data.splits import iter_folds
from src.features.labels import add_test_rul, add_train_rul
from src.features.pipeline import FeaturePipeline, kept_sensors
from src.models.baselines import scores, summary_table


class CNNBiLSTM(nn.Module):
    """Conv1d layers over sensor channels, then stacked BiLSTM, then a dense head."""

    def __init__(
        self,
        n_sensors: int,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dense: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        convs, in_channels = [], n_sensors
        for out_channels in conv_channels:
            convs += [nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"), nn.ReLU()]
            in_channels = out_channels
        self.conv = nn.Sequential(*convs)
        self.lstm = nn.LSTM(
            in_channels,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * lstm_hidden, dense), nn.ReLU(), nn.Linear(dense, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, window, sensors) to (batch,) predictions."""
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(features)
        return self.head(output[:, -1]).squeeze(-1)


def predict(model: nn.Module, X: np.ndarray, cap: float, batch_size: int = 1024) -> np.ndarray:
    """Return RUL predictions in cycles, clipped at 0."""
    model.eval()
    with torch.no_grad():
        out = [model(torch.from_numpy(X[i : i + batch_size])) for i in range(0, len(X), batch_size)]
    return np.clip(torch.cat(out).numpy() * cap, 0, None)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    training: dict,
    cap: float,
    seed: int,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int | None = None,
) -> tuple[nn.Module, dict]:
    """Train on (X, y in cycles) and return the model and its history.

    With validation data, trains up to ``training["max_epochs"]`` and keeps the
    weights from the epoch with the lowest validation NASA score, stopping after
    ``training["patience"]`` epochs without improvement. Without it, trains for
    exactly ``epochs``.
    """
    validating = X_val is not None
    if not validating and not epochs:
        raise ValueError("Give validation data or a fixed number of epochs")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = CNNBiLSTM(X.shape[2], **params)
    optimiser = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy((y / cap).astype(np.float32)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=training["batch_size"], shuffle=True, generator=generator)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_rmse": [], "val_nasa_score": []}
    best_score, best_state, best_epoch = np.inf, None, 0
    for epoch in range(1, (training["max_epochs"] if validating else epochs) + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(xb)
        history["train_loss"].append(total / len(dataset))

        if validating:
            val = scores(y_val, predict(model, X_val, cap))
            history["val_rmse"].append(val["rmse"])
            history["val_nasa_score"].append(val["nasa_score"])
            if val["nasa_score"] < best_score:
                best_score, best_epoch = val["nasa_score"], epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= training["patience"]:
                break

    if validating:
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
    return model, history


def run_subset(subset: str, config: dict) -> tuple[dict, pd.DataFrame]:
    """Cross-validate, retrain on all training engines and test on one subset."""
    prep, seed = config["preprocessing"], config["seed"]
    cap, window = prep["rul_cap"], prep["window_size"]
    params, training = config["cnn_bilstm"]["architecture"], config["cnn_bilstm"]["training"]
    sensors = kept_sensors(prep["drop_sensors"][subset])
    train, test, rul = load_subset(subset)
    train = add_train_rul(train, cap=cap)
    test = add_test_rul(test, rul, cap=None)

    folds = []
    for i, (fold_train, fold_val) in enumerate(iter_folds(train, config["validation"]["n_splits"], seed), 1):
        start = time.time()
        pipeline = FeaturePipeline(sensors, window).fit(fold_train)
        X_tr, idx_tr = pipeline.transform_sequences(fold_train)
        X_val, idx_val = pipeline.transform_sequences(fold_val)
        model, history = train_model(
            X_tr, idx_tr["rul"].to_numpy(), params, training, cap, seed, X_val=X_val, y_val=idx_val["rul"].to_numpy()
        )
        best = history["best_epoch"]
        folds.append(
            {
                "rmse": history["val_rmse"][best - 1],
                "nasa_score": history["val_nasa_score"][best - 1],
                "best_epoch": best,
                "history": history,
            }
        )
        print(f"{subset} fold {i}: best epoch {best}, val RMSE {folds[-1]['rmse']:.2f} ({time.time() - start:.0f}s)")

    epochs = int(np.median([f["best_epoch"] for f in folds]))
    pipeline = FeaturePipeline(sensors, window).fit(train)
    X_tr, idx_tr = pipeline.transform_sequences(train)
    model, history = train_model(X_tr, idx_tr["rul"].to_numpy(), params, training, cap, seed, epochs=epochs)
    X_test, idx_test = pipeline.transform_sequences(test, last_only=True)
    pred, true = predict(model, X_test, cap), idx_test["rul"].to_numpy()

    metrics = ("rmse", "nasa_score")
    results = {
        "subset": subset,
        "rul_cap": cap,
        "window_size": window,
        "sensors": sensors,
        "models": {
            "cnn_bilstm": {
                "params": params,
                "training": training,
                "validation_folds": folds,
                "validation_mean": {k: float(np.mean([f[k] for f in folds])) for k in metrics},
                "validation_std": {k: float(np.std([f[k] for f in folds])) for k in metrics},
                "final_epochs": epochs,
                "final_train_loss": history["train_loss"],
                "test_uncapped": scores(true, pred),
                "test_capped": scores(np.minimum(true, cap), pred),
            }
        },
    }
    predictions = pd.DataFrame({"model": "cnn_bilstm", "unit": idx_test["unit"], "true_rul": true, "predicted_rul": pred})
    return results, predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the CNN-BiLSTM model.")
    parser.add_argument("--subset", nargs="+", help="Subsets to run (default: all in the config)")
    args = parser.parse_args(argv)

    config = load_config()
    out_dir = resolve_path(config["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for subset in args.subset or config["data"]["subsets"]:
        results, predictions = run_subset(subset, config)
        Path(out_dir / f"cnn_bilstm_{subset}.json").write_text(json.dumps(results, indent=2))
        predictions.to_csv(out_dir / f"cnn_bilstm_{subset}_test_predictions.csv", index=False)
        print(f"\n{subset} (test scores against uncapped RUL unless marked capped)")
        print(summary_table(results).to_string())


if __name__ == "__main__":
    main()
