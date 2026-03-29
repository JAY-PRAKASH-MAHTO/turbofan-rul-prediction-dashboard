"""Streamlit dashboard for NASA CMAPSS Remaining Useful Life prediction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from preprocess import get_sensor_curve
from train import train_and_evaluate


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = BASE_DIR / "data" / "train_FD001.txt"
DEFAULT_TEST = BASE_DIR / "data" / "test_FD001.txt"
DEFAULT_RUL = BASE_DIR / "data" / "RUL_FD001.txt"
MODEL_PATH = BASE_DIR / "model.pkl"
REPORTS_DIR = BASE_DIR / "reports"


def run_default_training() -> dict:
    return train_and_evaluate(
        DEFAULT_TRAIN,
        DEFAULT_TEST,
        DEFAULT_RUL,
        model_output_path=MODEL_PATH,
        reports_dir=REPORTS_DIR,
    )


def describe_feature(feature_name: str) -> tuple[str, str]:
    """Return a short explanation and an interpretation hint for a feature."""

    if feature_name == "cycle":
        return (
            "Elapsed operating cycles for the engine.",
            "This often acts as a direct aging signal because wear typically grows over time.",
        )

    if feature_name.startswith("setting_"):
        return (
            "Operational setting that changes the engine regime.",
            "The model uses settings to separate true degradation from normal operating-condition shifts.",
        )

    if "_roll_mean_" in feature_name:
        sensor_name = feature_name.split("_roll_mean_")[0]
        return (
            f"Five-cycle rolling mean for {sensor_name}.",
            "A smoothed sensor level helps the model focus on persistent degradation instead of noise.",
        )

    if "_roll_std_" in feature_name:
        sensor_name = feature_name.split("_roll_std_")[0]
        return (
            f"Five-cycle rolling standard deviation for {sensor_name}.",
            "Rising variability can signal instability before a clearer fault trend appears.",
        )

    if feature_name.endswith("_trend"):
        sensor_name = feature_name.removesuffix("_trend")
        return (
            f"Cycle-to-cycle change for {sensor_name}.",
            "Trend features capture whether a sensor is currently drifting toward a faulty state.",
        )

    return (
        f"Raw measurement from {feature_name}.",
        "Raw sensors preserve the original physical signal that the engineered features build on top of.",
    )


def build_feature_explanation_table(feature_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_features = feature_df.head(top_n).copy()
    explanations = [describe_feature(name) for name in top_features["feature"]]
    top_features["explanation"] = [item[0] for item in explanations]
    top_features["insight"] = [item[1] for item in explanations]
    return top_features


def render_graph_explainer(what_it_shows: str, how_it_works: str, why_it_matters: str) -> None:
    st.markdown("**What it shows**")
    st.write(what_it_shows)
    st.markdown("**How it works**")
    st.write(how_it_works)
    st.markdown("**Why it matters**")
    st.write(why_it_matters)


def render_technical_context() -> None:
    with st.expander("Technical Problem Statement", expanded=False):
        st.markdown(
            """
            This project solves a prognostics regression problem on the NASA CMAPSS FD001 subset.

            Each engine is represented as a multivariate time-series trajectory:
            - `x_(i,t) = [settings_(i,t), sensors_(i,t)]`
            - `i` identifies the engine
            - `t` identifies the operating cycle

            In the training set, each engine is observed until failure, so row-level Remaining Useful Life is defined as:
            - `RUL_(i,t) = T_fail,i - t`

            In the test set, each trajectory is right-censored before failure. The file `RUL_FD001.txt` provides the
            remaining cycles after the final observed cycle for each engine, which allows reconstruction of row-level
            targets across the full truncated trajectory.

            The learning objective is to estimate a function:
            - `f(x_(i,1:t)) -> RUL_(i,t)`

            This is operationally important because an accurate RUL estimate supports condition-based maintenance,
            reduces unplanned downtime, and lowers the risk of letting a degrading engine operate too close to failure.
            """
        )

    with st.expander("Methodology", expanded=False):
        st.markdown(
            """
            The modeling pipeline follows a standard prognostics workflow:

            1. Load the FD001 train, test, and RUL target files from the local `data/` directory.
            2. Assign the CMAPSS schema:
               `engine_id`, `cycle`, `setting_1..3`, `sensor_1..21`.
            3. Compute row-level RUL:
               - training from each engine's observed failure cycle
               - test from the provided terminal RUL offsets
            4. Remove constant or non-informative sensors that do not contribute meaningful degradation signal.
            5. Engineer temporal features per engine:
               - rolling mean with window size 5
               - rolling standard deviation with window size 5
               - first-order difference trend
            6. Standardize model inputs with `StandardScaler` fitted on the training split only.
            7. Train a `RandomForestRegressor` as the main nonlinear model.
            8. Train a `LinearRegression` baseline for comparison.
            9. Evaluate with:
               - RMSE for large-error sensitivity
               - MAE for average absolute error
               - Pearson correlation coefficient for trend alignment

            The Random Forest is suitable here because it can learn nonlinear interactions between operating settings,
            raw sensors, smoothed signals, and local trends without requiring deep sequence models or heavy training cost.
            """
        )


def render_overview_metrics(result: dict) -> None:
    baseline_rmse = result["baseline_metrics"]["rmse"]
    rmse_gain = baseline_rmse - result["rmse"]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Random Forest RMSE", f"{result['rmse']:.3f}", delta=f"{rmse_gain:.3f} vs baseline")
    metric_col2.metric("Random Forest MAE", f"{result['mae']:.3f}")
    metric_col3.metric("Correlation Coefficient", f"{result['correlation_coefficient']:.3f}")
    metric_col4.metric("Removed Sensors", str(len(result["removed_sensors"])))


def render_baseline_comparison(result: dict) -> None:
    st.subheader("Baseline Comparison")
    comparison_df = result["comparison_df"].copy()
    numeric_cols = ["rmse", "mae", "correlation_coefficient"]
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(3)
    st.dataframe(comparison_df, hide_index=True)
    render_graph_explainer(
        "A side-by-side comparison of the Random Forest and a simpler linear baseline using RMSE, MAE, and correlation.",
        "Both models are trained on the same engineered feature set, then evaluated on the same test trajectories so the comparison stays fair.",
        "It validates whether the main model is learning real nonlinear degradation behavior instead of only reproducing a simple global trend.",
    )


def render_rul_distribution(train_df: pd.DataFrame, bins: int) -> None:
    st.subheader("1. RUL Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_df["rul"], bins=bins, color="#2c7be5", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Remaining Useful Life")
    ax.set_ylabel("Count")
    ax.set_title("Training RUL Distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    render_graph_explainer(
        "The distribution of row-level RUL values in the training data.",
        "Each bar counts how many training observations fall into a given RUL interval after row-level targets are computed from engine failure cycles.",
        "It helps diagnose target imbalance. If the distribution is heavily skewed, the model may become better at common life stages and weaker at rare ones.",
    )


def render_degradation_curve(
    train_df: pd.DataFrame,
    sensor_columns: list[str],
    selected_engine: int,
    selected_sensor: str,
) -> None:
    st.subheader("2. Sensor Degradation Curve")
    curve_df = get_sensor_curve(train_df, selected_engine, selected_sensor)
    smooth_curve = curve_df[selected_sensor].rolling(window=5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve_df["cycle"], curve_df[selected_sensor], color="#9d4edd", alpha=0.45, label="Raw sensor")
    ax.plot(curve_df["cycle"], smooth_curve, color="#5a189a", linewidth=2.2, label="5-cycle rolling mean")
    ax.set_xlabel("Cycle")
    ax.set_ylabel(selected_sensor)
    ax.set_title(f"{selected_sensor} over Time for Engine {selected_engine}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    render_graph_explainer(
        f"The trajectory of {selected_sensor} across cycles for engine {selected_engine}, including raw values and a smoothed rolling mean.",
        "The chart isolates one engine, orders rows by cycle, and overlays the original sensor with its 5-cycle rolling average to suppress short-term noise.",
        "It reveals whether the sensor carries a monotonic drift, volatility increase, or regime change that could act as a degradation signature.",
    )


def render_feature_importance(feature_df: pd.DataFrame, top_n: int) -> None:
    st.subheader("3. Feature Importance")
    top_features = feature_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    ax.barh(top_features["feature"], top_features["importance"], color="#00a896")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    render_graph_explainer(
        "The highest-impact features used by the Random Forest during RUL estimation.",
        "Random Forest feature importance is computed from the average impurity reduction contributed by each feature across all decision trees.",
        "It identifies which sensors and engineered temporal descriptors carry the strongest predictive signal, which is useful for model interpretation and sensor prioritization.",
    )

    st.markdown("**Feature Explanation + Insights**")
    explanation_df = build_feature_explanation_table(feature_df, top_n)
    explanation_df["importance"] = explanation_df["importance"].round(4)
    st.dataframe(explanation_df[["feature", "importance", "explanation", "insight"]], hide_index=True)


def render_prediction_line_plot(
    prediction_df: pd.DataFrame,
    sample_window: tuple[int, int],
    show_baseline: bool,
) -> None:
    st.subheader("4. Actual vs Predicted Line Plot")
    start_idx, end_idx = sample_window
    window_df = prediction_df.iloc[start_idx : end_idx + 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window_df["sample_index"], window_df["actual_rul"], label="Actual RUL", linewidth=2)
    ax.plot(
        window_df["sample_index"],
        window_df["random_forest_predicted_rul"],
        label="Random Forest",
        linewidth=2,
    )
    if show_baseline:
        ax.plot(
            window_df["sample_index"],
            window_df["baseline_predicted_rul"],
            label="Linear Baseline",
            linewidth=1.8,
            linestyle="--",
        )
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("RUL")
    ax.set_title("Predicted RUL vs Actual RUL")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    render_graph_explainer(
        "A sequential comparison between true RUL and model-predicted RUL over a selected slice of the test set.",
        "The plot draws actual RUL and predicted RUL against sample index, with an optional baseline overlay for direct visual comparison.",
        "It shows where the model follows the degradation path well and where it systematically lags, overshoots, or smooths sharp transitions.",
    )


def render_scatter_plot(prediction_df: pd.DataFrame, model_choice: str, correlation_value: float) -> None:
    st.subheader("5. Scatter Plot")
    pred_column = (
        "random_forest_predicted_rul"
        if model_choice == "Random Forest"
        else "baseline_predicted_rul"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        prediction_df["actual_rul"],
        prediction_df[pred_column],
        alpha=0.55,
        color="#007f5f",
    )
    bounds = [
        min(prediction_df["actual_rul"].min(), prediction_df[pred_column].min()),
        max(prediction_df["actual_rul"].max(), prediction_df[pred_column].max()),
    ]
    ax.plot(bounds, bounds, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel(f"{model_choice} Predicted RUL")
    ax.set_title(f"{model_choice}: Actual vs Predicted")
    ax.grid(alpha=0.2)
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {correlation_value:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    render_graph_explainer(
        f"The relationship between actual RUL and {model_choice} predicted RUL for all test samples.",
        "Each point represents one test observation. The dashed diagonal is the ideal line where predicted RUL equals actual RUL, and Pearson correlation summarizes linear alignment.",
        "It helps distinguish calibration quality from pure trend following. Tight concentration around the diagonal indicates stronger predictive consistency.",
    )


def render_residual_distribution(prediction_df: pd.DataFrame, model_choice: str, bins: int) -> None:
    st.subheader("6. Residual Error Distribution")
    residual_column = (
        "random_forest_residual"
        if model_choice == "Random Forest"
        else "baseline_residual"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prediction_df[residual_column], bins=bins, color="#e55353", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_choice} Residual Distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    render_graph_explainer(
        f"The distribution of prediction errors for the {model_choice} model.",
        "Residuals are computed as `actual RUL - predicted RUL`. The histogram shows how often the model underpredicts or overpredicts by different magnitudes.",
        "A narrow distribution centered near zero indicates lower bias and tighter error spread, while wide tails indicate unstable performance on some operating conditions or engines.",
    )


def main() -> None:
    st.set_page_config(page_title="NASA CMAPSS RUL Dashboard", layout="wide")
    st.title("Remaining Useful Life Prediction Dashboard")
    st.caption("Default source: NASA CMAPSS FD001 turbofan dataset from local project files")

    if "training_result" not in st.session_state:
        st.session_state.training_result = None

    with st.sidebar:
        st.header("Default Dataset")
        st.caption("Using local files from the `data/` folder:")
        st.code("train_FD001.txt\ntest_FD001.txt\nRUL_FD001.txt")
        retrain_model = st.button("Retrain Model")

    if st.session_state.training_result is None or retrain_model:
        with st.spinner("Training Random Forest on default FD001 data..."):
            st.session_state.training_result = run_default_training()

    result = st.session_state.training_result
    prediction_df = result["predictions_df"]
    engine_ids = result["train_df"]["engine_id"].drop_duplicates().sort_values().tolist()
    sensor_columns = result["sensor_columns"]

    with st.sidebar:
        st.divider()
        st.header("Dynamic Controls")
        hist_bins = st.slider("RUL histogram bins", min_value=10, max_value=60, value=30, step=5)
        default_sensor_index = sensor_columns.index("sensor_11") if "sensor_11" in sensor_columns else 0
        selected_engine = st.selectbox("Engine for degradation curve", engine_ids, index=0)
        selected_sensor = st.selectbox("Sensor for degradation curve", sensor_columns, index=default_sensor_index)
        top_n_features = st.slider("Top features to display", min_value=5, max_value=20, value=10, step=1)
        line_plot_end = min(250, len(prediction_df) - 1)
        sample_window = st.slider(
            "Sample window for line plot",
            min_value=0,
            max_value=len(prediction_df) - 1,
            value=(0, line_plot_end),
        )
        show_baseline = st.checkbox("Overlay baseline on line plot", value=True)
        comparison_model = st.radio("Model for scatter and residual charts", ["Random Forest", "Linear Baseline"])
        residual_bins = st.slider("Residual histogram bins", min_value=10, max_value=60, value=30, step=5)

    render_technical_context()
    render_overview_metrics(result)
    render_baseline_comparison(result)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_rul_distribution(result["train_df"], hist_bins)
    with chart_col2:
        render_degradation_curve(result["train_df"], sensor_columns, selected_engine, selected_sensor)

    render_feature_importance(result["feature_importance_df"], top_n_features)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        render_prediction_line_plot(prediction_df, sample_window, show_baseline)
    with chart_col4:
        correlation_value = (
            result["correlation_coefficient"]
            if comparison_model == "Random Forest"
            else result["baseline_metrics"]["correlation_coefficient"]
        )
        render_scatter_plot(prediction_df, comparison_model, correlation_value)

    render_residual_distribution(prediction_df, comparison_model, residual_bins)

    st.subheader("Prediction Sample")
    prediction_preview = prediction_df[
        [
            "sample_index",
            "engine_id",
            "cycle",
            "actual_rul",
            "random_forest_predicted_rul",
            "baseline_predicted_rul",
            "random_forest_residual",
            "baseline_residual",
        ]
    ].head(20)
    st.dataframe(prediction_preview, hide_index=True)


if __name__ == "__main__":
    main()
