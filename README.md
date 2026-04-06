# Turbofan Engine RUL Prediction Dashboard

Production-style Streamlit dashboard for turbofan Remaining Useful Life prediction using the NASA CMAPSS FD001 dataset and a saved machine learning model.

## Overview

This project has been refactored from a static analysis demo into a live inference dashboard that can:

- accept new engine telemetry through CSV upload
- run the same preprocessing logic used by the trained model
- normalize features with the saved scaler artifact
- generate RUL predictions from rolling engine windows
- estimate simplified confidence as `RUL = x +/- y`
- classify risk as Green, Yellow, or Red
- warn when incoming data drifts outside the training range

The UI is built in Streamlit with a cleaner operations-style layout, sidebar controls, summary cards, and interactive Plotly charts.

## Dashboard Features

### 1. Two Input Modes

- `Upload CSV`
  Upload fresh engine telemetry with the required schema.
- `Demo Engine Selector`
  Choose an existing engine trajectory from the local NASA FD001 test split.

### 2. Production-Style Preprocessing

The inference pipeline in `utils/preprocess.py` includes:

- `load_data`
- `validate_columns`
- `handle_missing_values`
- `normalize_using_saved_scaler`
- `create_sliding_window`

It blocks prediction when there are fewer than `30` cycles available for the selected window size.

### 3. Inference Module

The model layer in `utils/predict.py` includes:

- `load_model`
- `predict_rul`
- `smooth_predictions`

It also handles:

- simplified confidence estimation from ensemble variance
- risk classification
- training-range drift checks for model confidence warnings

### 4. UI Sections

- Header with project overview
- Sidebar with input mode, upload control, demo engine selection, and window size
- Prediction summary cards
- RUL trend graph
- Sensor visualization panel
- Data preview table

## Accepted CSV Schema

Upload CSV files must include these columns:

```text
cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_16, sensor_17, sensor_18, sensor_19, sensor_20, sensor_21
```

Notes:

- `engine_id` is optional for uploads. If omitted, the dashboard assumes a single engine.
- Missing numeric values are repaired before inference using forward fill, backward fill, and median fallback.
- Invalid schema or unreadable files are rejected with explicit Streamlit warnings.

## How Prediction Works

The saved model is a `RandomForestRegressor` trained on engineered per-cycle features. For live inference:

1. The dashboard validates and cleans the incoming engine data.
2. It recreates the engineered features used during training:
   - raw retained sensors
   - rolling mean
   - rolling standard deviation
   - first-difference trend
3. It applies the saved `StandardScaler`.
4. It creates trailing windows with a default length of `30` cycles.
5. It predicts RUL from each trailing window and smooths the resulting trend for visualization.

Because the saved model is feature-based rather than a deep sequence model, the rolling windows are used to create stable trajectory-level inference and trend visualization around the latest available cycle.

## Risk Logic

- `Green`: `RUL > 50`
- `Yellow`: `20 <= RUL <= 50`
- `Red`: `RUL < 20`

The dashboard also shows:

- a confidence interval
- a critical warning when the engine is in the red zone
- a model confidence warning when engineered features fall outside the training range

## Repository Structure

```text
turbofan-rul-prediction-dashboard/
|
|-- app.py
|-- requirements.txt
|-- run.txt
|-- README.md
|
|-- utils/
|   |-- __init__.py
|   |-- preprocess.py
|   |-- predict.py
|
|-- model/
|   |-- model.pkl
|   |-- scaler.pkl
|
|-- data/
|   |-- train_FD001.txt
|   |-- test_FD001.txt
|   |-- RUL_FD001.txt
|
|-- assets/
|   |-- .gitkeep
|
|-- reports/
|   |-- legacy training report outputs
|
|-- train.py
|-- preprocess.py
|-- model.pkl
```

Notes:

- `utils/` contains the active modular inference pipeline used by the dashboard.
- `model/` contains the production-style model and scaler artifacts used at runtime.
- Top-level `train.py`, `preprocess.py`, and `model.pkl` are retained from the earlier training-oriented version of the project.

## Quick Start

### 1. Install dependencies

```powershell
py -m pip install -r requirements.txt
```

### 2. Run the dashboard

```powershell
py -m streamlit run app.py
```

If port `8501` is busy:

```powershell
py -m streamlit run app.py --server.port 8502
```

## Example Workflow

1. Launch the Streamlit app.
2. Choose `Upload CSV` or `Demo Engine Selector`.
3. Set the window size in the sidebar.
4. Review the prediction summary card for the latest RUL estimate.
5. Inspect the RUL trend graph and confidence band.
6. Explore sensor behavior in the sensor visualization panel.
7. Review the cleaned input data in the preview table.

## Tech Stack

- `Streamlit`
- `Pandas`
- `NumPy`
- `scikit-learn`
- `Plotly`
- `joblib`

## Local Data Source

Demo mode uses local NASA CMAPSS FD001 data from:

- `data/train_FD001.txt`
- `data/test_FD001.txt`
- `data/RUL_FD001.txt`

## Reference

Dataset context is documented in:

- `data/readme.txt`
- `data/Damage Propagation Modeling.pdf`
