"""Random Forest and XGBoost baselines on rolling-window features.

Every fitted step (imputation medians, condition statistics, the model) is fitted
on the training engines of a fold and only applied to its validation engines.
The final model is refitted on all training engines, scored on the test set,
and saved with its fitted pipeline to ``models/`` (see src/models/artifacts.py).

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
from src.data.load import load_subset
from src.data.splits import iter_folds
from src.evaluation.metrics import nasa_score, rmse
from src.features.labels import add_test_rul, add_train_rul
from src.features.pipeline import FeaturePipeline, kept_sensors
from src.models.artifacts import TrainedModel, save
from src.features.windows import DEFAULT_SUMMARIES

MODELS = {"random_forest": RandomForestRegressor, "xgboost": XGBRegressor}


def build_model(name: str, params: dict, seed: int):
    """Create an unfitted model from its config parameters."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(MODELS)}")
    return MODELS[name](**params, random_state=seed)


def fit(name, params, seed, train, sensors, window, *, summaries=DEFAULT_SUMMARIES):
    """Fit the pipeline and model on ``train``; return (pipeline, model)."""
    pipeline = FeaturePipeline(sensors, window, summaries).fit(train)
    X_train, train_index = pipeline.transform(train)
    model = build_model(name, params, seed).fit(X_train, train_index["rul"])
    return pipeline, model


def predict(pipeline, model, df, *, last_only=False):
    """Return (predictions clipped at 0, index) for ``df``."""
    X, index = pipeline.transform(df, last_only=last_only)
    return np.clip(model.predict(X), 0, None), index


def fit_predict(name, params, seed, train, other, sensors, window, *, summaries=DEFAULT_SUMMARIES, last_only=False):
    """Fit the pipeline and model on ``train``; return predictions and index for ``other``."""
    pipeline, model = fit(name, params, seed, train, sensors, window, summaries=summaries)
    return predict(pipeline, model, other, last_only=last_only)


def scores(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred)}


def run_subset(
    subset: str, config: dict, *, models_dir: Path | None = None, tag: str = ""
) -> tuple[dict, pd.DataFrame]:
    """Cross-validate and test every baseline on one subset.

    With ``models_dir``, each final model is saved there as
    ``<model>_<subset><tag>.joblib``.
    """
    prep, seed = config["preprocessing"], config["seed"]
    cap, window, summaries = prep["rul_cap"], prep["window_size"], list(prep["rolling_stats"])
    sensors = kept_sensors(prep["drop_sensors"][subset])
    train, test, rul = load_subset(subset)
    train = add_train_rul(train, cap=cap)
    test = add_test_rul(test, rul, cap=None)

    results = {"subset": subset, "rul_cap": cap, "window_size": window, "rolling_stats": summaries,
               "sensors": sensors, "models": {}}
    predictions = []
    for name, params in config["models"].items():
        folds = []
        for fold_train, fold_val in iter_folds(train, config["validation"]["n_splits"], seed):
            pred, index = fit_predict(name, params, seed, fold_train, fold_val, sensors, window, summaries=summaries)
            folds.append(scores(index["rul"], pred))

        pipeline, model = fit(name, params, seed, train, sensors, window, summaries=summaries)
        pred, index = predict(pipeline, model, test, last_only=True)
        true = index["rul"].to_numpy()
        results["models"][name] = {
            "params": params,
            "validation_folds": folds,
            "validation_mean": {k: float(np.mean([f[k] for f in folds])) for k in folds[0]},
            "validation_std": {k: float(np.std([f[k] for f in folds])) for k in folds[0]},
            "test_uncapped": scores(true, pred),
            "test_capped": scores(np.minimum(true, cap), pred),
        }
        if models_dir is not None:
            info = {"subset": subset, "sensors": sensors, "window_size": window, "rolling_stats": summaries,
                    "params": params, "test_uncapped": results["models"][name]["test_uncapped"]}
            save(TrainedModel(name, pipeline, model, cap, info), Path(models_dir) / f"{name}_{subset}{tag}.joblib")
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
    parser.add_argument(
        "--rolling-stats",
        nargs="+",
        metavar="STAT",
        help="Window statistics to use instead of the config's (e.g. mean std min max); "
        "results are saved with the statistics in the file name",
    )
    args = parser.parse_args(argv)

    config = load_config()
    suffix = ""
    if args.rolling_stats:
        config["preprocessing"]["rolling_stats"] = args.rolling_stats
        suffix = "_" + "_".join(args.rolling_stats)
    out_dir = resolve_path(config["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = resolve_path(config["models_dir"])
    for subset in args.subset or config["data"]["subsets"]:
        results, predictions = run_subset(subset, config, models_dir=models_dir, tag=suffix)
        Path(out_dir / f"baselines_{subset}{suffix}.json").write_text(json.dumps(results, indent=2))
        predictions.to_csv(out_dir / f"baselines_{subset}{suffix}_test_predictions.csv", index=False)
        print(f"\n{subset} (test scores against uncapped RUL unless marked capped)")
        print(summary_table(results).to_string())


if __name__ == "__main__":
    main()
