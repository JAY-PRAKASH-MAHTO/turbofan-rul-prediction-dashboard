"""Preprocessing utilities for production-style turbofan RUL inference."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


INPUT_SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
UPLOAD_REQUIRED_COLUMNS = [
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *INPUT_SENSOR_COLUMNS,
]
MODEL_SCHEMA_COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *INPUT_SENSOR_COLUMNS,
]
UPLOAD_TO_MODEL_COLUMN_MAP = {
    "op_setting_1": "setting_1",
    "op_setting_2": "setting_2",
    "op_setting_3": "setting_3",
}
DEFAULT_SEQUENCE_LENGTH = 30
ROLLING_WINDOW = 5


class InvalidSchemaError(ValueError):
    """Raised when uploaded data does not match the expected CMAPSS schema."""


class InsufficientCyclesError(ValueError):
    """Raised when an engine trajectory is shorter than the required window size."""


@st.cache_data(show_spinner=False)
def load_data(file: bytes | str | Path, file_name: str = "uploaded.csv") -> pd.DataFrame:
    """Load uploaded CSV engine telemetry."""

    if isinstance(file, (str, Path)):
        return pd.read_csv(file)

    try:
        return pd.read_csv(BytesIO(file))
    except Exception as exc:
        raise ValueError(f"Unable to read {file_name} as CSV.") from exc


@st.cache_data(show_spinner=False)
def load_demo_dataset(path: str | Path) -> pd.DataFrame:
    """Load the local FD001 demo dataset used by the engine selector."""

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] > len(MODEL_SCHEMA_COLUMNS):
        df = df.iloc[:, : len(MODEL_SCHEMA_COLUMNS)]
    df.columns = MODEL_SCHEMA_COLUMNS[: df.shape[1]]
    return df


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the incoming schema and align it with the trained model schema."""

    if df.empty:
        raise InvalidSchemaError("The provided file is empty. Upload a CSV with engine cycles and sensor values.")

    aligned = df.copy()
    aligned.columns = [str(column).strip().lower() for column in aligned.columns]

    missing_columns = []
    for expected_column in UPLOAD_REQUIRED_COLUMNS:
        aliases = [expected_column]
        if expected_column.startswith("op_setting_"):
            aliases.append(expected_column.replace("op_", "", 1))
        if not any(alias in aligned.columns for alias in aliases):
            missing_columns.append(expected_column)

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise InvalidSchemaError(
            "Missing required columns: "
            f"{missing_text}. Expected CSV columns are cycle, op_setting_1, op_setting_2, "
            "op_setting_3, and sensor_1 to sensor_21."
        )

    aligned = aligned.rename(columns=UPLOAD_TO_MODEL_COLUMN_MAP)
    if "engine_id" not in aligned.columns:
        aligned["engine_id"] = 1

    selected_columns = [column for column in MODEL_SCHEMA_COLUMNS if column in aligned.columns]
    return aligned.loc[:, selected_columns].copy()


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Coerce numeric fields and repair missing values with local trajectory context."""

    cleaned = df.copy()
    numeric_columns = [column for column in cleaned.columns if column in MODEL_SCHEMA_COLUMNS]
    cleaned.loc[:, numeric_columns] = cleaned.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")

    missing_value_count = int(cleaned.loc[:, numeric_columns].isna().sum().sum())
    cleaned = cleaned.sort_values(["engine_id", "cycle"]).drop_duplicates(
        subset=["engine_id", "cycle"],
        keep="last",
    )
    cleaned = cleaned.reset_index(drop=True)

    cleaned.loc[:, numeric_columns] = cleaned.groupby("engine_id", group_keys=False)[numeric_columns].transform(
        lambda column: column.ffill().bfill()
    )
    cleaned.loc[:, numeric_columns] = cleaned.loc[:, numeric_columns].fillna(
        cleaned.loc[:, numeric_columns].median(numeric_only=True)
    )
    cleaned.loc[:, numeric_columns] = cleaned.loc[:, numeric_columns].fillna(0.0)

    cleaned["engine_id"] = cleaned["engine_id"].round().astype(int)
    cleaned["cycle"] = cleaned["cycle"].round().astype(int)
    return cleaned, missing_value_count


def add_temporal_features(
    df: pd.DataFrame,
    sensor_columns: Sequence[str],
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Generate the same temporal features used during training."""

    engineered = df.sort_values(["engine_id", "cycle"]).copy()
    grouped = engineered.groupby("engine_id", group_keys=False)

    for sensor_name in sensor_columns:
        rolling_mean_name = f"{sensor_name}_roll_mean_{window}"
        rolling_std_name = f"{sensor_name}_roll_std_{window}"
        trend_name = f"{sensor_name}_trend"

        engineered[rolling_mean_name] = grouped[sensor_name].transform(
            lambda values: values.rolling(window=window, min_periods=1).mean()
        )
        engineered[rolling_std_name] = grouped[sensor_name].transform(
            lambda values: values.rolling(window=window, min_periods=1).std().fillna(0.0)
        )
        engineered[trend_name] = grouped[sensor_name].transform(lambda values: values.diff().fillna(0.0))

    return engineered


def normalize_using_saved_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Scale features using the exact scaler artifact saved during training."""

    missing_features = [feature for feature in feature_columns if feature not in df.columns]
    if missing_features:
        missing_text = ", ".join(missing_features[:8])
        raise ValueError(f"Unable to normalize data because feature columns are missing: {missing_text}")

    normalized = df.copy()
    normalized = normalized.astype({column: float for column in feature_columns})
    normalized.loc[:, feature_columns] = scaler.transform(normalized.loc[:, feature_columns])
    return normalized


def create_sliding_window(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    metadata_frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create trailing windows across the engine sequence for inference-time prediction."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    window_frames: list[np.ndarray] = []
    metadata_rows: list[dict[str, int]] = []

    metadata_source = metadata_frame if metadata_frame is not None else df

    for engine_id, engine_df in df.groupby("engine_id", sort=True):
        ordered = engine_df.sort_values("cycle").reset_index(drop=True)
        ordered_metadata = (
            metadata_source.loc[metadata_source["engine_id"] == engine_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )
        if len(ordered) < sequence_length:
            message = (
                f"Engine {engine_id} has only {len(ordered)} cycles. "
                f"At least {sequence_length} cycles are required to generate a prediction window."
            )
            warnings.warn(message, stacklevel=2)
            raise InsufficientCyclesError(message)
        if len(ordered_metadata) != len(ordered):
            raise ValueError("Sliding-window metadata does not align with the normalized feature frame.")

        feature_matrix = ordered.loc[:, feature_columns].to_numpy(dtype=float)
        cycles = ordered_metadata["cycle"].to_numpy(dtype=int)
        for end_index in range(sequence_length, len(ordered) + 1):
            start_index = end_index - sequence_length
            window_frames.append(feature_matrix[start_index:end_index])
            metadata_rows.append(
                {
                    "engine_id": int(engine_id),
                    "cycle": int(cycles[end_index - 1]),
                    "window_start_cycle": int(cycles[start_index]),
                    "window_end_cycle": int(cycles[end_index - 1]),
                }
            )

    return np.stack(window_frames), pd.DataFrame(metadata_rows)


def prepare_inference_data(
    source: pd.DataFrame | bytes | str | Path,
    scaler: StandardScaler,
    feature_columns: Sequence[str],
    sensor_columns: Sequence[str],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    rolling_window: int = ROLLING_WINDOW,
) -> dict[str, pd.DataFrame | np.ndarray | int]:
    """Run the end-to-end preprocessing pipeline for uploaded or demo engine data."""

    if isinstance(source, pd.DataFrame):
        raw_frame = source.copy()
    else:
        raw_frame = load_data(source)

    validated_frame = validate_columns(raw_frame)
    clean_frame, missing_value_count = handle_missing_values(validated_frame)
    engineered_frame = add_temporal_features(clean_frame, sensor_columns=sensor_columns, window=rolling_window)
    normalized_frame = normalize_using_saved_scaler(engineered_frame, scaler, feature_columns)
    windows, window_index = create_sliding_window(
        normalized_frame,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        metadata_frame=clean_frame,
    )

    return {
        "raw_frame": raw_frame,
        "validated_frame": validated_frame,
        "clean_frame": clean_frame,
        "engineered_frame": engineered_frame,
        "normalized_frame": normalized_frame,
        "windows": windows,
        "window_index": window_index,
        "missing_value_count": missing_value_count,
    }
