"""Model training and report generation for the NASA CMAPSS RUL project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import prepare_datasets


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_rul_distribution(train_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_df["rul"], bins=30, color="#2c7be5", alpha=0.85, edgecolor="white")
    ax.set_title("Training RUL Distribution")
    ax.set_xlabel("Remaining Useful Life")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_actual_vs_pred(pred_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pred_df["sample_index"], pred_df["actual_rul"], label="Actual", linewidth=2)
    ax.plot(pred_df["sample_index"], pred_df["predicted_rul"], label="Predicted", linewidth=2)
    ax.set_title("Actual vs Predicted RUL")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("RUL")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_plot(pred_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pred_df["residual"], bins=30, color="#e55353", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Residual Error Distribution")
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> None:
    top = feature_importance.sort_values("importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.3)))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#00a896")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_feature_importance_df(model: RandomForestRegressor, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def calculate_regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics shared by the main model and the baseline."""

    corr = 0.0
    if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
        corr = float(np.corrcoef(actual, predicted)[0, 1])

    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "correlation_coefficient": corr,
    }


def train_and_evaluate(
    train_path: str | Path,
    test_path: str | Path,
    rul_path: str | Path,
    model_output_path: str | Path = "model.pkl",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Train a RandomForestRegressor and generate dashboard/report artifacts."""

    reports_path = ensure_directory(reports_dir)

    data = prepare_datasets(train_path, test_path, rul_path, window=5)
    raw_train = data["raw_train"]
    raw_test = data["raw_test"]
    train_df = data["train_processed"]
    test_df = data["test_processed"]
    feature_columns = data["feature_columns"]

    X_train = train_df.loc[:, feature_columns]
    y_train = train_df["rul"].astype(float)
    X_test = test_df.loc[:, feature_columns]
    y_test = test_df["rul"].astype(float)

    model = RandomForestRegressor(
        n_estimators=40,
        random_state=42,
        n_jobs=-1,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
    )
    model.fit(X_train, y_train)

    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)

    predictions = np.clip(model.predict(X_test), a_min=0.0, a_max=None)
    baseline_predictions = np.clip(baseline_model.predict(X_test), a_min=0.0, a_max=None)

    model_metrics = calculate_regression_metrics(y_test, predictions)
    baseline_metrics = calculate_regression_metrics(y_test, baseline_predictions)

    pred_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_test)),
            "engine_id": raw_test["engine_id"].to_numpy(),
            "cycle": raw_test["cycle"].to_numpy(),
            "actual_rul": y_test.to_numpy(),
            "predicted_rul": predictions,
            "random_forest_predicted_rul": predictions,
            "baseline_predicted_rul": baseline_predictions,
        }
    )
    pred_df["residual"] = pred_df["actual_rul"] - pred_df["predicted_rul"]
    pred_df["random_forest_residual"] = pred_df["actual_rul"] - pred_df["random_forest_predicted_rul"]
    pred_df["baseline_residual"] = pred_df["actual_rul"] - pred_df["baseline_predicted_rul"]

    feature_importance_df = build_feature_importance_df(model, feature_columns)
    comparison_df = pd.DataFrame(
        [
            {"model": "Random Forest", **model_metrics},
            {"model": "Linear Baseline", **baseline_metrics},
        ]
    )

    rul_plot_path = reports_path / "rul_distribution.png"
    actual_pred_path = reports_path / "actual_vs_pred.png"
    residual_plot_path = reports_path / "residual_plot.png"
    feature_importance_path = reports_path / "feature_importance.png"

    save_rul_distribution(train_df, rul_plot_path)
    save_actual_vs_pred(pred_df, actual_pred_path)
    save_residual_plot(pred_df, residual_plot_path)
    save_feature_importance(feature_importance_df, feature_importance_path)

    bundle = {
        "model": model,
        "scaler": data["scaler"],
        "feature_columns": feature_columns,
        "sensor_columns": data["sensor_columns"],
        "removed_sensors": data["removed_sensors"],
    }
    joblib.dump(bundle, model_output_path, compress=3)

    return {
        "rmse": model_metrics["rmse"],
        "mae": model_metrics["mae"],
        "correlation_coefficient": model_metrics["correlation_coefficient"],
        "baseline_metrics": baseline_metrics,
        "baseline_name": "Linear Baseline",
        "comparison_df": comparison_df,
        "model_path": str(Path(model_output_path)),
        "plot_paths": {
            "actual_vs_pred": str(actual_pred_path),
            "residual_plot": str(residual_plot_path),
            "feature_importance": str(feature_importance_path),
            "rul_distribution": str(rul_plot_path),
        },
        "train_preview": raw_train.head(10).copy(),
        "test_preview": raw_test.head(10).copy(),
        "predictions_df": pred_df,
        "feature_importance_df": feature_importance_df,
        "removed_sensors": data["removed_sensors"],
        "sensor_columns": data["sensor_columns"],
        "feature_columns": feature_columns,
        "train_df": raw_train,
        "test_df": raw_test,
        "processed_train_df": train_df,
        "processed_test_df": test_df,
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    result = train_and_evaluate(
        base / "data" / "train_FD001.txt",
        base / "data" / "test_FD001.txt",
        base / "data" / "RUL_FD001.txt",
    )
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"MAE: {result['mae']:.4f}")
