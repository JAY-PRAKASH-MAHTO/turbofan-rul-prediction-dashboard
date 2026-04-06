"""Industry-grade Streamlit dashboard for turbofan Remaining Useful Life prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from preprocess import (
    add_rolling_features as legacy_add_rolling_features,
    compute_test_rul as legacy_compute_test_rul,
    compute_train_rul as legacy_compute_train_rul,
    load_cmapss_data as legacy_load_cmapss_data,
    load_rul_targets as legacy_load_rul_targets,
    transform_with_scaler as legacy_transform_with_scaler,
)
from utils.preprocess import (
    DEFAULT_SEQUENCE_LENGTH,
    InvalidSchemaError,
    InsufficientCyclesError,
    UPLOAD_REQUIRED_COLUMNS,
    load_data,
    load_demo_dataset,
    prepare_inference_data,
)
from utils.predict import (
    assess_training_range_deviation,
    build_prediction_frame,
    classify_risk,
    load_model,
    load_training_feature_bounds,
    predict_rul,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEMO_DATA_PATH = DATA_DIR / "test_FD001.txt"
TRAIN_REFERENCE_PATH = DATA_DIR / "train_FD001.txt"
RUL_TARGET_PATH = DATA_DIR / "RUL_FD001.txt"
DEFAULT_SENSOR = "sensor_11"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #0f172a;
                --muted: #334155;
                --surface: #ffffff;
                --surface-soft: #eef4ff;
                --border: #c7d7ea;
                --shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
                --success: #0f8a5f;
                --warning: #b86c00;
                --danger: #c0392b;
                --accent: #1459d9;
                --accent-soft: #dbeafe;
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(20, 89, 217, 0.10), transparent 28%),
                    linear-gradient(180deg, #f8fbff 0%, #edf4fb 100%);
                color: var(--ink);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }

            .stApp [data-testid="stMarkdownContainer"] p,
            .stApp [data-testid="stMetricLabel"],
            .stApp [data-testid="stMetricValue"],
            .stApp [data-testid="stCaptionContainer"] {
                color: var(--ink) !important;
            }

            .stApp h1,
            .stApp h2,
            .stApp h3 {
                color: var(--ink) !important;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #eef4ff 100%);
                border-right: 1px solid #d8e5f5;
                color: #0f172a;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
                padding-top: 1rem;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div,
            [data-testid="stSidebar"] small {
                color: #0f172a !important;
            }

            [data-testid="stSidebar"] .stCaption,
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
                color: #475569 !important;
            }

            [data-testid="stSidebar"] .stRadio > label,
            [data-testid="stSidebar"] .stSlider > label,
            [data-testid="stSidebar"] .stSelectbox > label,
            [data-testid="stSidebar"] .stFileUploader > label,
            [data-testid="stSidebar"] .stExpander > label {
                color: #0f172a !important;
                font-weight: 700 !important;
            }

            [data-testid="stSidebar"] [data-baseweb="radio"] label {
                color: #0f172a !important;
                font-weight: 500 !important;
            }

            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #c7d7ea !important;
                box-shadow: 0 8px 18px rgba(20, 89, 217, 0.06);
            }

            [data-testid="stSidebar"] .stTextInput input,
            [data-testid="stSidebar"] .stNumberInput input,
            [data-testid="stSidebar"] textarea {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #c7d7ea !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                background: #ffffff !important;
                border: 2px dashed #bfd3ef !important;
                border-radius: 18px !important;
                padding: 1rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
                color: #475569 !important;
            }

            [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
                background: #ffffff !important;
                color: #1459d9 !important;
                border: 1px solid #bfd3ef !important;
            }

            [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
                background: #1459d9 !important;
                border-color: #1459d9 !important;
            }

            [data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
                background: #1459d9 !important;
            }

            [data-testid="stSidebar"] input[type="radio"] {
                accent-color: #1459d9;
            }

            [data-testid="stSidebar"] details {
                background: rgba(255, 255, 255, 0.75);
                border: 1px solid #d8e5f5;
                border-radius: 14px;
                padding: 0.2rem 0.4rem;
            }

            .hero-card {
                background:
                    linear-gradient(135deg, rgba(16, 37, 66, 0.98), rgba(29, 78, 216, 0.9)),
                    linear-gradient(180deg, #102542 0%, #1d4ed8 100%);
                border-radius: 26px;
                padding: 2rem 2.2rem;
                border: 1px solid rgba(255, 255, 255, 0.14);
                box-shadow: var(--shadow);
                color: #f8fafc;
                margin-bottom: 1.5rem;
            }

            .hero-kicker {
                display: inline-block;
                padding: 0.3rem 0.75rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.16);
                font-size: 0.78rem;
                letter-spacing: 0.03em;
                text-transform: uppercase;
                margin-bottom: 0.8rem;
            }

            .hero-title {
                font-size: 2.2rem;
                line-height: 1.15;
                font-weight: 700;
                margin: 0 0 0.4rem 0;
            }

            .hero-subtitle {
                font-size: 1rem;
                line-height: 1.6;
                color: rgba(248, 250, 252, 0.88);
                margin: 0;
                max-width: 860px;
            }

            .summary-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1.2rem 1.25rem;
                box-shadow: var(--shadow);
                height: 100%;
            }

            .summary-label {
                color: var(--muted);
                font-size: 0.86rem;
                text-transform: uppercase;
                letter-spacing: 0.03em;
                margin: 0 0 0.55rem 0;
            }

            .summary-value {
                color: var(--ink);
                font-size: 2rem;
                line-height: 1.1;
                font-weight: 700;
                margin: 0;
            }

            .summary-subtitle {
                color: var(--muted);
                margin: 0.45rem 0 0 0;
                font-size: 0.95rem;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                border-radius: 999px;
                padding: 0.45rem 0.8rem;
                font-weight: 600;
                font-size: 0.92rem;
                margin-top: 0.4rem;
            }

            .status-green {
                background: var(--success);
                color: #ffffff;
            }

            .status-yellow {
                background: var(--warning);
                color: #ffffff;
            }

            .status-red {
                background: var(--danger);
                color: #ffffff;
            }

            .section-note {
                color: var(--muted);
                font-size: 0.98rem;
                margin-top: 0.25rem;
                margin-bottom: 0.75rem;
            }

            .explain-card {
                background: linear-gradient(180deg, #f8fbff 0%, var(--surface-soft) 100%);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                margin-top: 0.85rem;
            }

            .explain-title {
                color: var(--accent);
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin: 0 0 0.35rem 0;
            }

            .explain-text {
                color: var(--ink);
                font-size: 0.95rem;
                line-height: 1.6;
                margin: 0;
            }

            .panel-shell {
                background: rgba(255, 255, 255, 0.78);
                border-radius: 22px;
                padding: 0.2rem;
            }

            .stDataFrame, div[data-testid="stDataFrame"] {
                border-radius: 16px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="hero-card">
            <div class="hero-kicker">Live inference dashboard</div>
            <h1 class="hero-title">Turbofan Engine RUL Prediction Dashboard</h1>
            <p class="hero-subtitle">
                Upload fresh engine telemetry or compare against the NASA FD001 demo fleet.
                The dashboard validates incoming data, applies the saved training scaler, predicts
                Remaining Useful Life, and surfaces risk with a clean operations-facing view.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    with st.container(border=True):
        st.subheader("Awaiting Engine Data")
        st.write(
            "Upload a CSV in the sidebar or switch to the demo engine mode to run the inference pipeline."
        )
        st.caption("Expected CSV schema")
        st.code(", ".join(UPLOAD_REQUIRED_COLUMNS), language="text")


def render_summary_card(title: str, value: str, subtitle: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="summary-card" style="border-top: 5px solid {accent};">
            <p class="summary-label">{title}</p>
            <p class="summary-value">{value}</p>
            <p class="summary-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(risk_summary: dict[str, str | float]) -> None:
    risk_class = f"status-{risk_summary['level'].lower()}"
    st.markdown(
        f"""
        <div class="summary-card" style="border-top: 5px solid {risk_summary['color']};">
            <p class="summary-label">Risk Status</p>
            <p class="summary-value">{risk_summary['label']}</p>
            <div class="status-pill {risk_class}">{risk_summary['message']}</div>
            <p class="summary-subtitle">Thresholds: Green &gt; 50, Yellow 20-50, Red &lt; 20</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explainer(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="explain-card">
            <p class="explain-title">{title}</p>
            <p class="explain-text">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def describe_feature(feature_name: str) -> tuple[str, str]:
    if feature_name == "cycle":
        return (
            "Elapsed operating cycles for the engine.",
            "Cycle acts as a direct aging signal because overall wear usually increases with operating time.",
        )
    if feature_name.startswith("setting_"):
        return (
            "Operational setting that shifts the engine regime.",
            "Settings help the model separate normal operating-condition changes from real degradation.",
        )
    if "_roll_mean_" in feature_name:
        sensor_name = feature_name.split("_roll_mean_")[0]
        return (
            f"Rolling mean for {sensor_name}.",
            "A smoothed value helps the model focus on persistent degradation rather than noisy fluctuations.",
        )
    if "_roll_std_" in feature_name:
        sensor_name = feature_name.split("_roll_std_")[0]
        return (
            f"Rolling standard deviation for {sensor_name}.",
            "Rising variability can indicate instability before failure becomes obvious in the raw reading.",
        )
    if feature_name.endswith("_trend"):
        sensor_name = feature_name.removesuffix("_trend")
        return (
            f"Cycle-to-cycle trend for {sensor_name}.",
            "Trend shows whether a sensor is drifting toward a more degraded state right now.",
        )
    return (
        f"Raw measurement from {feature_name}.",
        "Raw sensor values preserve the original physical signal used by the model.",
    )


def build_feature_explanation_table(feature_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_features = feature_df.head(top_n).copy()
    explanations = [describe_feature(name) for name in top_features["feature"]]
    top_features["description"] = [item[0] for item in explanations]
    top_features["why_it_matters"] = [item[1] for item in explanations]
    return top_features


@st.cache_data(show_spinner=False)
def load_model_diagnostics(
    train_path: str | Path,
    test_path: str | Path,
    rul_path: str | Path,
) -> dict[str, pd.DataFrame | float]:
    model_assets = load_model()

    train_df = legacy_compute_train_rul(legacy_load_cmapss_data(train_path))
    test_df = legacy_compute_test_rul(
        legacy_load_cmapss_data(test_path),
        legacy_load_rul_targets(rul_path),
    )

    train_engineered = legacy_add_rolling_features(train_df, model_assets.sensor_columns, window=5)
    test_engineered = legacy_add_rolling_features(test_df, model_assets.sensor_columns, window=5)
    test_processed = legacy_transform_with_scaler(
        test_engineered,
        model_assets.scaler,
        model_assets.feature_columns,
    )

    feature_matrix = test_processed.loc[:, model_assets.feature_columns]
    if hasattr(model_assets.model, "feature_names_in_"):
        feature_matrix = feature_matrix.loc[:, list(model_assets.model.feature_names_in_)]

    predictions = np.clip(model_assets.model.predict(feature_matrix), a_min=0.0, a_max=None)
    actual = test_df["rul"].astype(float).to_numpy()
    residuals = actual - predictions
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    correlation = 0.0
    if len(actual) > 1 and np.std(actual) > 0 and np.std(predictions) > 0:
        correlation = float(np.corrcoef(actual, predictions)[0, 1])

    prediction_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(test_df)),
            "engine_id": test_df["engine_id"].to_numpy(),
            "cycle": test_df["cycle"].to_numpy(),
            "actual_rul": actual,
            "predicted_rul": predictions,
            "residual": residuals,
        }
    )

    feature_importance_df = pd.DataFrame(
        {
            "feature": model_assets.feature_columns,
            "importance": model_assets.model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    return {
        "train_df": train_df,
        "prediction_df": prediction_df,
        "feature_importance_df": feature_importance_df,
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
    }


def build_rul_trend_figure(prediction_frame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prediction_frame["cycle"],
            y=prediction_frame["upper_bound"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_frame["cycle"],
            y=prediction_frame["lower_bound"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(29, 78, 216, 0.14)",
            name="Confidence band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_frame["cycle"],
            y=prediction_frame["predicted_rul_raw"],
            mode="lines",
            line={"color": "#f97316", "width": 2, "dash": "dot"},
            name="Raw prediction",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_frame["cycle"],
            y=prediction_frame["predicted_rul"],
            mode="lines+markers",
            line={"color": "#1459d9", "width": 4},
            marker={"size": 7, "color": "#0f172a"},
            name="Smoothed prediction",
        )
    )
    fig.add_hline(y=50, line_dash="dot", line_color="#0f8a5f", opacity=0.75)
    fig.add_hline(y=20, line_dash="dot", line_color="#c0392b", opacity=0.75)
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=380,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a", "size": 14},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis_title="Cycle",
        yaxis_title="Predicted RUL",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#94a3b8", tickfont={"color": "#0f172a"})
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.28)", linecolor="#94a3b8", tickfont={"color": "#0f172a"})
    return fig


def build_sensor_figure(clean_frame, sensor_name: str) -> go.Figure:
    sensor_view = clean_frame.sort_values("cycle").copy()
    sensor_view["rolling_mean"] = sensor_view[sensor_name].rolling(window=5, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sensor_view["cycle"],
            y=sensor_view[sensor_name],
            mode="lines",
            line={"color": "#0ea5e9", "width": 2.6},
            name="Raw sensor",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sensor_view["cycle"],
            y=sensor_view["rolling_mean"],
            mode="lines",
            line={"color": "#0f8a5f", "width": 4},
            name="5-cycle rolling mean",
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=380,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a", "size": 14},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis_title="Cycle",
        yaxis_title=sensor_name,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#94a3b8", tickfont={"color": "#0f172a"})
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.28)", linecolor="#94a3b8", tickfont={"color": "#0f172a"})
    return fig


def build_rul_distribution_figure(train_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Histogram(
                x=train_df["rul"],
                nbinsx=30,
                marker={"color": "#1459d9", "line": {"color": "#ffffff", "width": 1}},
                opacity=0.9,
                name="Training RUL",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=360,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a", "size": 14},
        xaxis_title="Remaining Useful Life",
        yaxis_title="Count",
        bargap=0.06,
    )
    fig.update_xaxes(showgrid=False, linecolor="#94a3b8")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.28)", linecolor="#94a3b8")
    return fig


def build_feature_importance_figure(feature_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    top_features = feature_df.head(top_n).sort_values("importance", ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=top_features["importance"],
                y=top_features["feature"],
                orientation="h",
                marker={
                    "color": top_features["importance"],
                    "colorscale": [[0.0, "#93c5fd"], [1.0, "#1459d9"]],
                },
                name="Importance",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=max(360, top_n * 30),
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a", "size": 14},
        xaxis_title="Feature Importance",
        yaxis_title="Feature",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.28)", linecolor="#94a3b8")
    fig.update_yaxes(showgrid=False, linecolor="#94a3b8")
    return fig


def build_actual_vs_predicted_figure(prediction_df: pd.DataFrame, sample_window: tuple[int, int]) -> go.Figure:
    start_index, end_index = sample_window
    view = prediction_df.iloc[start_index : end_index + 1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["sample_index"],
            y=view["actual_rul"],
            mode="lines",
            line={"color": "#0f8a5f", "width": 3},
            name="Actual RUL",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["sample_index"],
            y=view["predicted_rul"],
            mode="lines",
            line={"color": "#1459d9", "width": 3},
            name="Predicted RUL",
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=360,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a", "size": 14},
        xaxis_title="Test Sample Index",
        yaxis_title="RUL",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_xaxes(showgrid=False, linecolor="#94a3b8")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.28)", linecolor="#94a3b8")
    return fig


def build_scatter_figure(prediction_df: pd.DataFrame, correlation_value: float) -> go.Figure:
    bounds = [
        float(min(prediction_df["actual_rul"].min(), prediction_df["predicted_rul"].min())),
        float(max(prediction_df["actual_rul"].max(), prediction_df["predicted_rul"].max())),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prediction_df["actual_rul"],
            y=prediction_df["predicted_rul"],
            mode="markers",
            marker={"color": "#1459d9", "size": 7, "opacity": 0.55},
            name="Samples",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bounds,
            y=bounds,
            mode="lines",
            line={"color": "#f97316", "width": 2, "dash": "dash"},
            name="Ideal fit",
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=360,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a", "size": 14},
        xaxis_title="Actual RUL",
        yaxis_title="Predicted RUL",
        showlegend=False,
        annotations=[
            {
                "x": 0.02,
                "y": 0.98,
                "xref": "paper",
                "yref": "paper",
                "text": f"Pearson r = {correlation_value:.3f}",
                "showarrow": False,
                "font": {"color": "#0f172a", "size": 13},
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": "#c7d7ea",
                "borderwidth": 1,
            }
        ],
    )
    fig.update_xaxes(showgrid=False, linecolor="#94a3b8")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.28)", linecolor="#94a3b8")
    return fig


def build_residual_distribution_figure(prediction_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Histogram(
                x=prediction_df["residual"],
                nbinsx=30,
                marker={"color": "#c0392b", "line": {"color": "#ffffff", "width": 1}},
                opacity=0.9,
                name="Residual",
            )
        ]
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#0f172a", opacity=0.8)
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=360,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a", "size": 14},
        xaxis_title="Actual RUL - Predicted RUL",
        yaxis_title="Count",
        bargap=0.06,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor="#94a3b8")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.28)", linecolor="#94a3b8")
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Turbofan Engine RUL Prediction Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    render_header()

    try:
        model_assets = load_model()
    except Exception as exc:
        st.error(f"Unable to load the saved model assets: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Controls")
        input_mode = st.radio("Input mode", ["Upload CSV", "Demo Engine Selector"])
        sequence_length = st.slider(
            "Window size",
            min_value=30,
            max_value=60,
            value=DEFAULT_SEQUENCE_LENGTH,
            step=5,
            help="Minimum cycles required before the dashboard can issue a prediction.",
        )

        with st.expander("Expected CSV schema", expanded=False):
            st.code(", ".join(UPLOAD_REQUIRED_COLUMNS), language="text")

        source_frame = None
        source_label = ""
        if input_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload engine CSV", type=["csv"])
            if uploaded_file is not None:
                source_label = uploaded_file.name
                try:
                    source_frame = load_data(uploaded_file.getvalue(), file_name=uploaded_file.name)
                except Exception as exc:
                    st.error(f"Invalid file. The CSV could not be parsed: {exc}")
        else:
            demo_data = load_demo_dataset(DEMO_DATA_PATH)
            engine_ids = sorted(demo_data["engine_id"].unique().tolist())
            selected_engine = st.selectbox("Demo engine", engine_ids, index=0)
            source_frame = demo_data.loc[demo_data["engine_id"] == selected_engine].copy()
            source_label = f"NASA FD001 demo engine {selected_engine}"
            st.caption(f"{len(source_frame)} cycles available")

    if source_frame is None:
        render_empty_state()
        st.stop()

    try:
        prepared = prepare_inference_data(
            source=source_frame,
            scaler=model_assets.scaler,
            feature_columns=model_assets.feature_columns,
            sensor_columns=model_assets.sensor_columns,
            sequence_length=sequence_length,
        )
    except InvalidSchemaError as exc:
        st.warning(str(exc))
        st.stop()
    except InsufficientCyclesError as exc:
        st.warning(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected preprocessing error: {exc}")
        st.stop()

    prediction_output = predict_rul(model_assets.model, prepared["windows"])
    prediction_frame = build_prediction_frame(
        window_index=prepared["window_index"],
        predictions=prediction_output["prediction"],
        uncertainty=prediction_output["uncertainty"],
        smoothing_window=5,
    )

    reference_bounds = load_training_feature_bounds(
        TRAIN_REFERENCE_PATH,
        tuple(model_assets.sensor_columns),
        tuple(model_assets.feature_columns),
    )
    drift_summary = assess_training_range_deviation(
        feature_frame=prepared["engineered_frame"].loc[:, model_assets.feature_columns],
        reference_bounds=reference_bounds,
        scaled_feature_frame=prepared["normalized_frame"].loc[:, model_assets.feature_columns],
    )

    current_rul = float(prediction_frame["predicted_rul"].iloc[-1])
    current_uncertainty = float(prediction_frame["prediction_uncertainty"].iloc[-1])
    risk_summary = classify_risk(current_rul)
    sensor_columns = [col for col in prepared["clean_frame"].columns if col.startswith("sensor_")]
    default_sensor_index = sensor_columns.index(DEFAULT_SENSOR) if DEFAULT_SENSOR in sensor_columns else 0

    info_col1, info_col2, info_col3 = st.columns([2.1, 1, 1])
    info_col1.caption(f"Data source: {source_label}")
    info_col2.caption(f"Cycles ingested: {len(prepared['clean_frame'])}")
    info_col3.caption(f"Prediction windows: {len(prediction_frame)}")

    if drift_summary["warning"]:
        st.warning(
            "Model Confidence Warning: Prediction may be unreliable. "
            f"{drift_summary['outside_feature_pct']:.1f}% of engineered features are outside the training range."
        )

    if risk_summary["level"] == "Red":
        st.error("Critical risk detected. Predicted RUL is below 20 cycles and should be treated as urgent.")

    diagnostic_data = None
    diagnostic_error = None
    try:
        diagnostic_data = load_model_diagnostics(
            TRAIN_REFERENCE_PATH,
            DEMO_DATA_PATH,
            RUL_TARGET_PATH,
        )
    except Exception as exc:
        diagnostic_error = exc

    with st.container(border=True):
        st.subheader("Prediction Summary")
        st.markdown(
            '<p class="section-note">Latest estimate based on the most recent validated prediction window.</p>',
            unsafe_allow_html=True,
        )
        summary_col1, summary_col2, summary_col3 = st.columns([1.25, 1, 1])
        with summary_col1:
            render_summary_card(
                title="Predicted RUL",
                value=f"{current_rul:.0f} cycles",
                subtitle="Smoothed forecast for the latest available cycle",
                accent="#1d4ed8",
            )
        with summary_col2:
            render_summary_card(
                title="Confidence Interval",
                value=f"{current_rul:.0f} +/- {current_uncertainty:.0f}",
                subtitle="Prediction variance estimated from the ensemble spread",
                accent="#0f766e",
            )
        with summary_col3:
            render_status_card(risk_summary)

        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        detail_col1.metric("Latest cycle", f"{int(prediction_frame['cycle'].iloc[-1])}")
        detail_col2.metric("Missing values repaired", f"{prepared['missing_value_count']}")
        detail_col3.metric("Out-of-range features", f"{drift_summary['outside_feature_pct']:.1f}%")
        detail_col4.metric("Model features used", f"{len(model_assets.feature_columns)}")

    chart_col1, chart_col2 = st.columns([1.2, 1.0])

    with chart_col1:
        with st.container(border=True):
            st.subheader("RUL Trend Graph")
            st.markdown(
                '<p class="section-note">Predicted Remaining Useful Life across the engine trajectory with clearer threshold colors and confidence context.</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_rul_trend_figure(prediction_frame), use_container_width=True)
            render_explainer(
                "Why this graph matters",
                "This chart shows how the engine's predicted life changes cycle by cycle. It helps you see whether the asset is staying healthy, degrading steadily, or entering a critical zone. The confidence band is useful because it shows how stable the prediction is, which supports better maintenance planning instead of relying on a single number alone.",
            )

    with chart_col2:
        with st.container(border=True):
            st.subheader("Sensor Visualization")
            st.markdown(
                '<p class="section-note">Inspect raw sensor behavior against a short rolling mean to understand which signals are changing most clearly.</p>',
                unsafe_allow_html=True,
            )
            selected_sensor = st.selectbox("Select sensor", sensor_columns, index=default_sensor_index)
            st.plotly_chart(
                build_sensor_figure(prepared["clean_frame"], selected_sensor),
                use_container_width=True,
            )
            render_explainer(
                "Why this graph matters",
                "This graph helps explain the prediction by showing the underlying sensor pattern over time. The raw line captures actual operating behavior, while the rolling mean removes short-term noise so you can spot sustained drift, instability, or degradation trends that may be influencing the RUL estimate.",
            )

    with st.container(border=True):
        st.subheader("Data Preview")
        st.markdown(
            '<p class="section-note">Validated and cleaned engine records used for inference, shown for quick audit and trust checks before acting on the prediction.</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(prepared["clean_frame"], use_container_width=True, height=340)
        render_explainer(
            "Why this table matters",
            "This table is important because it lets you verify the actual data that reached the model after validation and missing-value handling. It is useful for checking whether the upload was parsed correctly, whether the cycle order looks right, and whether any suspicious values might explain an unexpected prediction.",
        )

    st.markdown("### Model Diagnostics")
    st.markdown(
        '<p class="section-note">These diagnostics bring back the earlier analysis views using the saved model against the local NASA FD001 reference split.</p>',
        unsafe_allow_html=True,
    )

    if diagnostic_error is not None:
        st.info(f"Model diagnostic charts are temporarily unavailable: {diagnostic_error}")
    elif diagnostic_data is not None:
        diagnostic_prediction_df = diagnostic_data["prediction_df"]
        diagnostic_feature_df = diagnostic_data["feature_importance_df"]
        top_n_features = min(10, len(diagnostic_feature_df))
        sample_end_index = min(250, len(diagnostic_prediction_df) - 1)

        diag_metric1, diag_metric2, diag_metric3 = st.columns(3)
        diag_metric1.metric("Reference RMSE", f"{diagnostic_data['rmse']:.2f}")
        diag_metric2.metric("Reference MAE", f"{diagnostic_data['mae']:.2f}")
        diag_metric3.metric("Reference Correlation", f"{diagnostic_data['correlation']:.3f}")

        diag_row1_col1, diag_row1_col2 = st.columns(2)
        with diag_row1_col1:
            with st.container(border=True):
                st.subheader("Training RUL Distribution")
                st.plotly_chart(
                    build_rul_distribution_figure(diagnostic_data["train_df"]),
                    use_container_width=True,
                )
                render_explainer(
                    "Why this graph matters",
                    "This distribution shows where the model saw the most examples during training. It is useful because predictions are usually more reliable in regions where the model has seen enough similar RUL values, and less reliable in rare parts of the lifecycle.",
                )

        with diag_row1_col2:
            with st.container(border=True):
                st.subheader("Feature Importance")
                st.plotly_chart(
                    build_feature_importance_figure(diagnostic_feature_df, top_n=top_n_features),
                    use_container_width=True,
                )
                render_explainer(
                    "Why this graph matters",
                    "This chart highlights which engineered signals contribute most to the model's decision making. It is important because it helps explain whether the model is relying on meaningful operational patterns such as sensor drift, variability, or cycle progression.",
                )
                feature_table = build_feature_explanation_table(diagnostic_feature_df, top_n_features)
                st.dataframe(
                    feature_table[["feature", "importance", "description", "why_it_matters"]],
                    use_container_width=True,
                    hide_index=True,
                    height=290,
                )
                render_explainer(
                    "Why this table matters",
                    "This table explains the top features in plain language so the chart becomes actionable. It helps connect model importance scores to real engineering meaning instead of leaving the feature names abstract.",
                )

        diag_row2_col1, diag_row2_col2 = st.columns(2)
        with diag_row2_col1:
            with st.container(border=True):
                st.subheader("Actual vs Predicted RUL")
                st.plotly_chart(
                    build_actual_vs_predicted_figure(
                        diagnostic_prediction_df,
                        sample_window=(0, sample_end_index),
                    ),
                    use_container_width=True,
                )
                render_explainer(
                    "Why this graph matters",
                    "This view compares the true RUL curve with the model prediction across a slice of the reference test set. It is useful for seeing whether the model follows the degradation trend smoothly or misses sharp changes in engine behavior.",
                )

        with diag_row2_col2:
            with st.container(border=True):
                st.subheader("Actual vs Predicted Scatter")
                st.plotly_chart(
                    build_scatter_figure(
                        diagnostic_prediction_df,
                        correlation_value=diagnostic_data["correlation"],
                    ),
                    use_container_width=True,
                )
                render_explainer(
                    "Why this graph matters",
                    "This scatter plot shows overall agreement between real and predicted RUL values. Points close to the diagonal indicate stronger calibration, while a wide spread highlights where the model may overpredict or underpredict asset life.",
                )

        with st.container(border=True):
            st.subheader("Residual Error Distribution")
            st.plotly_chart(
                build_residual_distribution_figure(diagnostic_prediction_df),
                use_container_width=True,
            )
            render_explainer(
                "Why this graph matters",
                "Residuals show the size and direction of prediction error. This graph is important because it reveals whether the model has a systematic bias, such as repeatedly predicting too high or too low, and whether a few cases are causing large misses.",
            )


if __name__ == "__main__":
    main()
