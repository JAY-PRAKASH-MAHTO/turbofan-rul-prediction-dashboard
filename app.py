"""Industry-grade Streamlit dashboard for turbofan Remaining Useful Life prediction."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

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
DEFAULT_SENSOR = "sensor_11"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #102542;
                --muted: #5b6b7f;
                --surface: #ffffff;
                --border: #d9e3ef;
                --shadow: 0 18px 45px rgba(16, 37, 66, 0.08);
                --success: #12715b;
                --warning: #b7791f;
                --danger: #b42318;
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(29, 78, 216, 0.08), transparent 28%),
                    linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #102542 100%);
                color: #f8fafc;
            }

            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stCaption {
                color: #e2e8f0 !important;
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
                background: rgba(18, 113, 91, 0.12);
                color: var(--success);
            }

            .status-yellow {
                background: rgba(183, 121, 31, 0.14);
                color: var(--warning);
            }

            .status-red {
                background: rgba(180, 35, 24, 0.12);
                color: var(--danger);
            }

            .section-note {
                color: var(--muted);
                font-size: 0.95rem;
                margin-top: 0.25rem;
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
            line={"color": "rgba(100, 116, 139, 0.55)", "width": 1.6, "dash": "dot"},
            name="Raw prediction",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_frame["cycle"],
            y=prediction_frame["predicted_rul"],
            mode="lines+markers",
            line={"color": "#1d4ed8", "width": 3},
            marker={"size": 6, "color": "#0f172a"},
            name="Smoothed prediction",
        )
    )
    fig.add_hline(y=50, line_dash="dot", line_color="#12715b", opacity=0.65)
    fig.add_hline(y=20, line_dash="dot", line_color="#b7791f", opacity=0.65)
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=380,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis_title="Cycle",
        yaxis_title="Predicted RUL",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.25)")
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
            line={"color": "rgba(2, 132, 199, 0.45)", "width": 2},
            name="Raw sensor",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sensor_view["cycle"],
            y=sensor_view["rolling_mean"],
            mode="lines",
            line={"color": "#0f766e", "width": 3},
            name="5-cycle rolling mean",
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=380,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis_title="Cycle",
        yaxis_title=sensor_name,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.25)")
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
                '<p class="section-note">Predicted Remaining Useful Life across the engine trajectory.</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_rul_trend_figure(prediction_frame), use_container_width=True)

    with chart_col2:
        with st.container(border=True):
            st.subheader("Sensor Visualization")
            st.markdown(
                '<p class="section-note">Inspect raw sensor behavior against a short rolling mean for context.</p>',
                unsafe_allow_html=True,
            )
            selected_sensor = st.selectbox("Select sensor", sensor_columns, index=default_sensor_index)
            st.plotly_chart(
                build_sensor_figure(prepared["clean_frame"], selected_sensor),
                use_container_width=True,
            )

    with st.container(border=True):
        st.subheader("Data Preview")
        st.markdown(
            '<p class="section-note">Validated and cleaned engine records used for inference.</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(prepared["clean_frame"], use_container_width=True, height=340)


if __name__ == "__main__":
    main()
