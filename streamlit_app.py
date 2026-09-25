"""Streamlit demo for DriftSentrix.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, then on share.streamlit.io point at this
                file (main file path: streamlit_app.py).

This is a DEMO / VISUALIZATION layer, not the production deliverable --
the actual served system is the FastAPI service in src/anomaly/serve/
(see README.md "Running the service"). Streamlit Cloud hosts Streamlit
apps, not arbitrary FastAPI services, so this app calls the detection
code directly (in-process) rather than proxying to a running API.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

from anomaly.data.loaders import make_synthetic
from anomaly.eval.metrics import event_metrics, pa_f1, point_f1, pr_auc
from anomaly.eval.thresholds import best_f1_threshold, percentile_threshold
from anomaly.models.baselines import (
    EWMADetector,
    IsolationForestDetector,
    PCADetector,
    RandomDetector,
    ZScoreDetector,
)
from anomaly.viz import plot_timeline

st.set_page_config(page_title="DriftSentrix", layout="wide")

BASELINES = {
    "Z-score": ZScoreDetector,
    "EWMA": EWMADetector,
    "Isolation Forest": IsolationForestDetector,
    "PCA": PCADetector,
    "Random (sanity check)": RandomDetector,
}

NO_SEED_MODELS = {"Z-score", "EWMA"}  # these two take no seed argument


def build_baseline(name: str):
    cls = BASELINES[name]
    return cls() if name in NO_SEED_MODELS else cls(seed=0)


st.title("DriftSentrix")
st.caption(
    "Multivariate time-series anomaly detection with an honest evaluation protocol. "
    "[Full code, tests, FastAPI service, and the LSTM/Transformer autoencoders on GitHub]"
    "(https://github.com/) -- this page is a live demo of the baseline detectors and the "
    "project's headline finding, not the full pipeline."
)

tab1, tab2, tab3 = st.tabs(["Live detection demo", "PA-inflation study", "Try your own data"])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Run a detector on synthetic data")
    col1, col2 = st.columns([1, 2])

    with col1:
        model_name = st.selectbox("Detector", list(BASELINES))
        threshold_mode = st.radio("Threshold", ["Oracle (uses labels -- upper bound)", "99th percentile (deployable)"])
        seed = st.number_input("Data seed", value=0, step=1)
        n_test = st.slider("Test length", 200, 2000, 1000, step=100)

    train, test, labels = make_synthetic(n_train=2000, n_test=n_test, d=3, seed=int(seed))
    det = build_baseline(model_name)
    det.fit(train)
    scores = det.score(test)

    if threshold_mode.startswith("Oracle"):
        threshold = best_f1_threshold(labels, scores)
    else:
        threshold = percentile_threshold(det.score(train[-400:]), 99.0)

    y_pred = (scores >= threshold).astype(int)
    p_p, p_r, p_f1 = point_f1(labels, y_pred)
    _, _, pa_f1_val = pa_f1(labels, y_pred)
    ev = event_metrics(labels, y_pred)
    auc = pr_auc(labels, scores)

    with col2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PR-AUC", f"{auc:.3f}")
        m2.metric("Point-F1", f"{p_f1:.3f}")
        m3.metric("PA-F1", f"{pa_f1_val:.3f}", delta=f"{pa_f1_val - p_f1:+.3f} vs point-F1")
        m4.metric("Event-F1", f"{ev['event_f1']:.3f}")
        st.caption(
            f"{ev['n_false_alarms']:.0f} false alarms · mean detection delay "
            f"{ev['mean_detection_delay']:.1f} steps"
        )

    fig_path = "/tmp/streamlit_timeline.png"
    plot_timeline(test, labels, scores, threshold, fig_path)
    st.image(fig_path, use_container_width=True)

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Why point-adjusted F1 is misleading")
    st.markdown(
        "For each detector above, take its predicted **rate** of flagged points, then build a "
        "**pure-random** detector that flags points at that exact same rate. If the random "
        "detector's PA-F1 is anywhere close to the real detector's, PA-F1 was mostly rewarding "
        "*flagging enough points somewhere*, not finding the right ones."
    )
    try:
        pa_df = pd.read_csv("reports/pa_inflation.csv")
        real = pa_df[pa_df["model"] != "random"]
        mean_point = real["random_point_f1"].mean()
        mean_pa = real["random_pa_f1"].mean()
        st.metric(
            "Mean PA-F1 of a rate-matched RANDOM detector",
            f"{mean_pa:.3f}",
            delta=f"vs. mean point-F1 of only {mean_point:.3f} ({mean_pa / mean_point:.1f}x inflation)",
            delta_color="inverse",
        )
        st.dataframe(pa_df, use_container_width=True)
        st.image("docs/img/pa_inflation.png", use_container_width=True)
    except FileNotFoundError:
        st.info(
            "Run `python -m anomaly.scripts.pa_inflation_study --config configs/full_benchmark.yaml` "
            "to generate `reports/pa_inflation.csv` before this tab has data to show."
        )

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Upload your own multivariate CSV")
    st.caption("Numeric columns only, one row per timestep. The first 80% of rows are used to fit the detector.")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded).select_dtypes(include=[np.number])
        if df.shape[1] == 0:
            st.error("No numeric columns found.")
        else:
            data = df.to_numpy(dtype=np.float32)
            split = int(0.8 * len(data))
            train_u, test_u = data[:split], data[split:]
            model_name_u = st.selectbox("Detector", list(BASELINES), key="upload_model")
            det_u = build_baseline(model_name_u)
            det_u.fit(train_u)
            scores_u = det_u.score(test_u)
            threshold_u = percentile_threshold(det_u.score(train_u[-max(1, len(train_u) // 5):]), 99.0)
            alerts_u = scores_u >= threshold_u
            st.write(f"{alerts_u.sum()} of {len(test_u)} test points flagged at the 99th-percentile threshold.")
            chart_df = pd.DataFrame(test_u, columns=df.columns)
            chart_df["anomaly_score"] = scores_u
            st.line_chart(chart_df["anomaly_score"])
            st.dataframe(chart_df[alerts_u], use_container_width=True)

st.divider()
st.caption(
    "Point-F1, PA-F1, and event-F1 are always reported together in this project -- see "
    "docs/EVALUATION.md on GitHub for why PA-F1 alone is misleading."
)
