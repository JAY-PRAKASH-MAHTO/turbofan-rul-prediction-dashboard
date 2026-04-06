"""Inference utilities for turbofan Remaining Useful Life prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from utils.preprocess import ROLLING_WINDOW, add_temporal_features, load_demo_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_CANDIDATES = [PROJECT_ROOT / "model" / "model.pkl", PROJECT_ROOT / "model.pkl"]
DEFAULT_SCALER_CANDIDATES = [PROJECT_ROOT / "model" / "scaler.pkl"]


@dataclass(frozen=True)
class ModelAssets:
    """Container for cached inference artifacts."""

    model: Any
    scaler: StandardScaler
    feature_columns: list[str]
    sensor_columns: list[str]
    removed_sensors: list[str]
    model_path: str
    scaler_path: str


def _find_existing_path(candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _infer_sensor_columns(feature_columns: Sequence[str]) -> list[str]:
    return [column for column in feature_columns if re.fullmatch(r"sensor_\d+", column)]


def _infer_removed_sensors(sensor_columns: Sequence[str]) -> list[str]:
    available = {f"sensor_{index}" for index in range(1, 22)}
    return sorted(available.difference(sensor_columns), key=lambda value: int(value.split("_")[1]))


@st.cache_resource(show_spinner=False)
def load_model(
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> ModelAssets:
    """Load the saved model and scaler with backwards-compatible artifact support."""

    resolved_model_path = Path(model_path) if model_path else _find_existing_path(DEFAULT_MODEL_CANDIDATES)
    if resolved_model_path is None:
        raise FileNotFoundError("No saved model artifact was found in model/model.pkl or model.pkl.")

    raw_artifact = joblib.load(resolved_model_path)
    resolved_scaler_path = Path(scaler_path) if scaler_path else _find_existing_path(DEFAULT_SCALER_CANDIDATES)

    if isinstance(raw_artifact, dict) and "model" in raw_artifact:
        model = raw_artifact["model"]
        scaler = raw_artifact.get("scaler")
        feature_columns = list(raw_artifact.get("feature_columns", []))
        sensor_columns = list(raw_artifact.get("sensor_columns", []))
        removed_sensors = list(raw_artifact.get("removed_sensors", []))
        if scaler is None and resolved_scaler_path is not None:
            scaler = joblib.load(resolved_scaler_path)
    else:
        model = raw_artifact
        if resolved_scaler_path is None:
            raise FileNotFoundError("The model artifact was found, but model/scaler.pkl is missing.")
        scaler = joblib.load(resolved_scaler_path)
        feature_columns = list(getattr(scaler, "feature_names_in_", []))
        sensor_columns = _infer_sensor_columns(feature_columns)
        removed_sensors = _infer_removed_sensors(sensor_columns)

    if scaler is None:
        raise FileNotFoundError("Unable to locate a saved scaler artifact.")

    if not feature_columns:
        feature_columns = list(getattr(scaler, "feature_names_in_", []))
    if not sensor_columns:
        sensor_columns = _infer_sensor_columns(feature_columns)
    if not removed_sensors:
        removed_sensors = _infer_removed_sensors(sensor_columns)

    return ModelAssets(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        sensor_columns=sensor_columns,
        removed_sensors=removed_sensors,
        model_path=str(resolved_model_path),
        scaler_path=str(resolved_scaler_path or ""),
    )


def predict_rul(model: Any, sequence: np.ndarray) -> dict[str, np.ndarray]:
    """Predict RUL from the latest row in each sliding window and estimate uncertainty."""

    window_array = np.asarray(sequence, dtype=float)
    if window_array.ndim == 2:
        window_array = window_array[np.newaxis, ...]
    if window_array.ndim != 3:
        raise ValueError("Expected sequence to have shape [windows, sequence_length, features].")

    latest_steps = window_array[:, -1, :]
    if hasattr(model, "feature_names_in_"):
        prediction_input = pd.DataFrame(latest_steps, columns=list(model.feature_names_in_))
    else:
        prediction_input = latest_steps

    predictions = np.clip(model.predict(prediction_input), a_min=0.0, a_max=None)
    tree_input = latest_steps

    if hasattr(model, "estimators_") and model.estimators_:
        estimator_predictions = np.vstack(
            [np.clip(estimator.predict(tree_input), a_min=0.0, a_max=None) for estimator in model.estimators_]
        )
        uncertainty = estimator_predictions.std(axis=0)
    else:
        uncertainty = pd.Series(predictions).rolling(window=5, min_periods=1).std().fillna(0.0).to_numpy()

    return {"prediction": predictions, "uncertainty": uncertainty}


def smooth_predictions(predictions: Sequence[float], window: int = 5) -> np.ndarray:
    """Apply light smoothing so the UI trend line is stable and readable."""

    if window < 1:
        raise ValueError("window must be at least 1.")

    prediction_series = pd.Series(predictions, dtype=float)
    return prediction_series.rolling(window=window, min_periods=1).mean().to_numpy()


def build_prediction_frame(
    window_index: pd.DataFrame,
    predictions: Sequence[float],
    uncertainty: Sequence[float],
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Combine prediction outputs with cycle metadata for plotting and summaries."""

    frame = window_index.copy()
    frame["predicted_rul_raw"] = np.asarray(predictions, dtype=float)
    frame["predicted_rul"] = smooth_predictions(frame["predicted_rul_raw"], window=smoothing_window)
    frame["prediction_uncertainty"] = smooth_predictions(uncertainty, window=min(3, smoothing_window))
    frame["lower_bound"] = np.clip(frame["predicted_rul"] - frame["prediction_uncertainty"], a_min=0.0, a_max=None)
    frame["upper_bound"] = frame["predicted_rul"] + frame["prediction_uncertainty"]
    return frame


def classify_risk(rul_value: float) -> dict[str, str | float]:
    """Map predicted RUL to a dashboard-friendly risk category."""

    if rul_value > 50:
        return {
            "level": "Green",
            "label": "Green",
            "color": "#12715b",
            "message": "Healthy runway",
        }
    if rul_value >= 20:
        return {
            "level": "Yellow",
            "label": "Yellow",
            "color": "#b7791f",
            "message": "Maintenance planning advised",
        }
    return {
        "level": "Red",
        "label": "Red",
        "color": "#b42318",
        "message": "Critical attention required",
    }


@st.cache_data(show_spinner=False)
def load_training_feature_bounds(
    train_data_path: str | Path,
    sensor_columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Build feature-wise training ranges for confidence warnings."""

    path = Path(train_data_path)
    if not path.exists():
        return {}

    reference_frame = load_demo_dataset(path)
    engineered_reference = add_temporal_features(reference_frame, list(sensor_columns), window=ROLLING_WINDOW)
    feature_frame = engineered_reference.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")

    return {
        "min": feature_frame.min(numeric_only=True).to_dict(),
        "max": feature_frame.max(numeric_only=True).to_dict(),
    }


def assess_training_range_deviation(
    feature_frame: pd.DataFrame,
    reference_bounds: dict[str, dict[str, float]] | None = None,
    scaled_feature_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Flag inputs that drift outside the operating envelope seen during training."""

    if reference_bounds and reference_bounds.get("min") and reference_bounds.get("max"):
        lower_bounds = pd.Series(reference_bounds["min"])
        upper_bounds = pd.Series(reference_bounds["max"])
        common_columns = [column for column in feature_frame.columns if column in lower_bounds.index]
        aligned_frame = feature_frame.loc[:, common_columns]
        outside_mask = aligned_frame.lt(lower_bounds[common_columns], axis=1) | aligned_frame.gt(
            upper_bounds[common_columns],
            axis=1,
        )
    elif scaled_feature_frame is not None:
        outside_mask = scaled_feature_frame.abs() > 3.0
    else:
        return {"warning": False, "outside_feature_pct": 0.0, "flagged_features": []}

    outside_feature_pct = float(outside_mask.to_numpy().mean() * 100)
    latest_row = outside_mask.iloc[-1] if not outside_mask.empty else pd.Series(dtype=bool)
    flagged_features = latest_row[latest_row].index.tolist()[:5]
    warning = outside_feature_pct > 1.0 or bool(latest_row.any())

    return {
        "warning": warning,
        "outside_feature_pct": outside_feature_pct,
        "flagged_features": flagged_features,
    }
