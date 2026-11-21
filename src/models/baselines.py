"""Random Forest and XGBoost baselines on rolling-window features.

Every fitted step (imputation medians, condition statistics, the model) is fitted
on the training engines of a fold and only applied to its validation engines.
The final model is refitted on all training engines and scored on the test set.

Training and validation targets are capped at ``rul_cap``. Test scores are
reported against NASA's true RUL (uncapped, the primary target) and against the
capped RUL.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import load_config, resolve_path
from src.data.load import SENSORS, load_subset
from src.data.quality import fit_medians, impute
from src.data.splits import iter_folds
from src.evaluation.metrics import nasa_score, rmse
from src.features.conditions import fit_condition_stats
from src.features.labels import add_test_rul, add_train_rul
from src.features.windows import rolling_features

MODELS = {"random_forest": RandomForestRegressor, "xgboost": XGBRegressor}


def kept_sensors(drop: list[int]) -> list[str]:
    """Return sensor column names, leaving out the dropped sensor numbers."""
    return [s for s in SENSORS if int(s.split("_")[1]) not in drop]


def build_model(name: str, params: dict, seed: int):
    """Create an unfitted model from its config parameters."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(MODELS)}")
    return MODELS[name](**params, random_state=seed)


class FeaturePipeline:
    """Imputation, condition normalisation and windowing fitted on training engines."""

    def __init__(self, sensors: list[str], window: int):
        self.sensors, self.window = sensors, window

    def fit(self, train: pd.DataFrame) -> "FeaturePipeline":
        self.medians = fit_medians(train)
        self.stats = fit_condition_stats(impute(train, self.medians), self.sensors)
        return self

    def transform(self, df: pd.DataFrame, *, last_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (features, index) where index holds unit, cycle and rul."""
        features = rolling_features(impute(df, self.medians), self.stats, self.window, last_only=last_only)
        keys = [c for c in ("unit", "cycle", "rul") if c in features]
        return features.drop(columns=keys), features[keys]


def fit_predict(name, params, seed, train, other, sensors, window, *, last_only=False):
    """Fit the pipeline and model on ``train``; return predictions and index for ``other``."""
    pipeline = FeaturePipeline(sensors, window).fit(train)
    X_train, train_index = pipeline.transform(train)
    model = build_model(name, params, seed).fit(X_train, train_index["rul"])
    X_other, other_index = pipeline.transform(other, last_only=last_only)
    return np.clip(model.predict(X_other), 0, None), other_index


def scores(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred)}


def run_subset(subset: str, config: dict) -> tuple[dict, pd.DataFrame]:
    """Cross-validate and test every baseline on one subset."""
    prep, seed = config["preprocessing"], config["seed"]
    cap, window = prep["rul_cap"], prep["window_size"]
    sensors = kept_sensors(prep["drop_sensors"][subset])
    train, test, rul = load_subset(subset)
    train = add_train_rul(train, cap=cap)
    test = add_test_rul(test, rul, cap=None)

    results = {"subset": subset, "rul_cap": cap, "window_size": window, "sensors": sensors, "models": {}}
    predictions = []
    for name, params in config["models"].items():
        folds = []
        for fold_train, fold_val in iter_folds(train, config["validation"]["n_splits"], seed):
            pred, index = fit_predict(name, params, seed, fold_train, fold_val, sensors, window)
            folds.append(scores(index["rul"], pred))

        pred, index = fit_predict(name, params, seed, train, test, sensors, window, last_only=True)
        true = index["rul"].to_numpy()
        results["models"][name] = {
            "params": params,
            "validation_folds": folds,
            "validation_mean": {k: float(np.mean([f[k] for f in folds])) for k in folds[0]},
            "validation_std": {k: float(np.std([f[k] for f in folds])) for k in folds[0]},
            "test_uncapped": scores(true, pred),
            "test_capped": scores(np.minimum(true, cap), pred),
        }
        predictions.append(pd.DataFrame({"model": name, "unit": index["unit"], "true_rul": true, "predicted_rul": pred}))
    return results, pd.concat(predictions, ignore_index=True)


def summary_table(results: dict) -> pd.DataFrame:
    rows = {
        name: {
            "val_rmse": r["validation_mean"]["rmse"],
            "test_rmse": r["test_uncapped"]["rmse"],
            "test_score": r["test_uncapped"]["nasa_score"],
            "test_rmse_capped": r["test_capped"]["rmse"],
            "test_score_capped": r["test_capped"]["nasa_score"],
        }
        for name, r in results["models"].items()
    }
    return pd.DataFrame(rows).T.round(2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the tree-model baselines.")
    parser.add_argument("--subset", nargs="+", help="Subsets to run (default: all in the config)")
    args = parser.parse_args(argv)

    config = load_config()
    out_dir = resolve_path(config["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for subset in args.subset or config["data"]["subsets"]:
        results, predictions = run_subset(subset, config)
        Path(out_dir / f"baselines_{subset}.json").write_text(json.dumps(results, indent=2))
        predictions.to_csv(out_dir / f"baselines_{subset}_test_predictions.csv", index=False)
        print(f"\n{subset} (test scores against uncapped RUL unless marked capped)")
        print(summary_table(results).to_string())


if __name__ == "__main__":
    main()
