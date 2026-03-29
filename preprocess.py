"""Preprocessing helpers for the NASA CMAPSS RUL project.

The module keeps the data flow small and explicit so the Streamlit app can
reuse the same functions for uploaded data and for the default FD001 files.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

# These sensors are known to be constant in FD001 and are commonly removed.
KNOWN_IRRELEVANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


PathLikeOrBuffer = str | Path | IO[str] | IO[bytes]


def load_cmapss_data(source: PathLikeOrBuffer) -> pd.DataFrame:
    """Load a CMAPSS text file and assign the standard column names."""

    df = pd.read_csv(source, sep=r"\s+", header=None, engine="python")
    if df.shape[1] > len(BASE_COLUMNS):
        df = df.iloc[:, : len(BASE_COLUMNS)]
    df.columns = BASE_COLUMNS[: df.shape[1]]
    return df


def load_rul_targets(source: PathLikeOrBuffer) -> pd.DataFrame:
    """Load the FD001 RUL file and align it with engine ordering."""

    rul = pd.read_csv(source, sep=r"\s+", header=None, engine="python")
    rul = rul.iloc[:, 0].astype(float).reset_index(drop=True)
    return pd.DataFrame({"engine_id": np.arange(1, len(rul) + 1), "rul": rul})


def compute_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the remaining useful life for each training row."""

    result = train_df.copy()
    max_cycle = result.groupby("engine_id")["cycle"].transform("max")
    result["rul"] = max_cycle - result["cycle"]
    return result


def compute_test_rul(test_df: pd.DataFrame, rul_targets: pd.DataFrame) -> pd.DataFrame:
    """Compute the full-row RUL for the CMAPSS test split.

    The test file is truncated before failure. The provided RUL file contains the
    final remaining cycles for each engine at the last observed test cycle, so
    we add that offset to every row of the corresponding engine.
    """

    result = test_df.copy()
    max_cycle = result.groupby("engine_id")["cycle"].transform("max")
    rul_map = rul_targets.set_index("engine_id")["rul"].to_dict()
    result["rul"] = max_cycle - result["cycle"] + result["engine_id"].map(rul_map)
    return result


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def identify_constant_sensors(
    train_df: pd.DataFrame,
    sensor_columns: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return the kept sensors and the removed sensors.

    We combine the known FD001 constant sensors with a data-driven check so the
    function remains safe if a user uploads a slightly different split.
    """

    if sensor_columns is None:
        sensor_columns = get_sensor_columns(train_df)

    removable = set(KNOWN_IRRELEVANT_SENSORS)
    for col in sensor_columns:
        if col in train_df.columns and train_df[col].nunique(dropna=False) <= 1:
            removable.add(col)

    kept = [col for col in sensor_columns if col not in removable]
    removed = [col for col in sensor_columns if col in removable]
    return kept, removed


def add_rolling_features(df: pd.DataFrame, sensor_columns: Sequence[str], window: int = 5) -> pd.DataFrame:
    """Add rolling mean, rolling std, and first-difference trend features."""

    result = df.sort_values(["engine_id", "cycle"]).copy()
    grouped = result.groupby("engine_id", group_keys=False)

    for col in sensor_columns:
        roll_mean_name = f"{col}_roll_mean_{window}"
        roll_std_name = f"{col}_roll_std_{window}"
        trend_name = f"{col}_trend"

        result[roll_mean_name] = grouped[col].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        result[roll_std_name] = grouped[col].transform(
            lambda s: s.rolling(window=window, min_periods=1).std().fillna(0.0)
        )
        result[trend_name] = grouped[col].transform(lambda s: s.diff().fillna(0.0))

    return result


def select_feature_columns(sensor_columns: Sequence[str], window: int = 5) -> list[str]:
    """Build the final model feature list."""

    base_features = ["cycle", "setting_1", "setting_2", "setting_3"]
    engineered = []
    for sensor in sensor_columns:
        engineered.extend(
            [
                sensor,
                f"{sensor}_roll_mean_{window}",
                f"{sensor}_roll_std_{window}",
                f"{sensor}_trend",
            ]
        )
    return base_features + engineered


def fit_standard_scaler(train_df: pd.DataFrame, feature_columns: Sequence[str]) -> StandardScaler:
    """Fit a StandardScaler on the model input columns."""

    scaler = StandardScaler()
    scaler.fit(train_df.loc[:, feature_columns])
    return scaler


def transform_with_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy with the selected feature columns scaled."""

    result = df.copy()
    # Pandas 3.x is strict about writing scaled floats back into integer columns.
    # Cast the model inputs first so the assignment stays lossless and stable.
    result = result.astype({col: float for col in feature_columns})
    scaled = scaler.transform(result.loc[:, feature_columns])
    result.loc[:, feature_columns] = scaled
    return result


def prepare_datasets(
    train_source: PathLikeOrBuffer,
    test_source: PathLikeOrBuffer,
    rul_source: PathLikeOrBuffer,
    window: int = 5,
) -> dict:
    """Load, clean, engineer, and scale the CMAPSS train/test splits."""

    raw_train = load_cmapss_data(train_source)
    raw_test = load_cmapss_data(test_source)
    rul_targets = load_rul_targets(rul_source)

    train_df = compute_train_rul(raw_train)
    test_df = compute_test_rul(raw_test, rul_targets)

    sensor_columns = get_sensor_columns(train_df)
    kept_sensors, removed_sensors = identify_constant_sensors(train_df, sensor_columns)

    train_engineered = add_rolling_features(train_df, kept_sensors, window=window)
    test_engineered = add_rolling_features(test_df, kept_sensors, window=window)

    feature_columns = select_feature_columns(kept_sensors, window=window)

    scaler = fit_standard_scaler(train_engineered, feature_columns)
    train_processed = transform_with_scaler(train_engineered, scaler, feature_columns)
    test_processed = transform_with_scaler(test_engineered, scaler, feature_columns)

    return {
        "raw_train": train_df,
        "raw_test": test_df,
        "train_processed": train_processed,
        "test_processed": test_processed,
        "scaler": scaler,
        "sensor_columns": kept_sensors,
        "removed_sensors": removed_sensors,
        "feature_columns": feature_columns,
    }


def get_sensor_curve(
    df: pd.DataFrame,
    engine_id: int,
    sensor_name: str,
) -> pd.DataFrame:
    """Return one engine's cycle/sensor curve for dashboard plotting."""

    curve = df.loc[df["engine_id"] == engine_id, ["cycle", sensor_name]].sort_values("cycle")
    return curve.reset_index(drop=True)
