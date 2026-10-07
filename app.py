"""Credit Card Fraud Detection dashboard.

Launch with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import dataset_summary, load_data
from model import (
    ARTIFACT_PATH,
    CV_FOLDS,
    evaluate,
    load_artifacts,
    predict_proba,
    save_artifacts,
    train_and_evaluate,
)

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

MERCHANTS = ["Clothing", "Electronics", "Food", "Grocery", "Travel"]
INPUT_DEFAULTS = {
    "amount": 120.0,
    "transaction_hour": 12,
    "merchant_category": "Grocery",
    "foreign_transaction": False,
    "location_mismatch": False,
    "device_trust_score": 65,
    "velocity_last_24h": 2,
    "cardholder_age": 44,
}


@st.cache_resource
def get_artifacts() -> dict:
    """Load the trained model; train and save it if no compatible artifact exists."""
    if ARTIFACT_PATH.exists():
        try:
            return load_artifacts()
        except Exception:
            pass  # missing/incompatible artifact (e.g. other sklearn version on the cloud) -> retrain
    artifacts = train_and_evaluate()
    save_artifacts(artifacts)
    return artifacts


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


artifacts = get_artifacts()
df = get_data()
summary = dataset_summary(df)
pipeline = artifacts["pipeline"]
tuned_threshold = artifacts["tuning"]["threshold"]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("⚙️ Decision threshold")
threshold = st.sidebar.slider(
    "Fraud threshold",
    min_value=0.05,
    max_value=0.99,
    value=float(round(tuned_threshold, 3)),
    step=0.001,
    format="%.3f",
    help="Transactions with predicted probability ≥ threshold are flagged as fraud.",
)
st.sidebar.caption(
    f"CV-tuned optimum (max F1): **{tuned_threshold:.3f}**. "
    "Move the slider to explore the precision/recall trade-off; all metrics and "
    "predictions update live."
)
if st.sidebar.button("Retrain model"):
    get_artifacts.clear()
    if ARTIFACT_PATH.exists():
        ARTIFACT_PATH.unlink()
    st.rerun()

metrics = evaluate(artifacts["y_test"], artifacts["test_proba"], threshold)

st.title("💳 Credit Card Fraud Detection")
st.caption("Logistic Regression · class_weight='balanced' · CV-tuned decision threshold")

tab_perf, tab_predict = st.tabs(["📊 Model Performance & Analysis", "🔍 Interactive Fraud Prediction"])

# ---------------------------------------------------------------- tab 1
with tab_perf:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{metrics['precision']:.3f}")
    c2.metric("Recall", f"{metrics['recall']:.3f}")
    c3.metric("F1-Score", f"{metrics['f1']:.3f}")
    c4.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    c5.metric("PR AUC", f"{metrics['pr_auc']:.3f}")
    st.caption(
        f"Test set: {artifacts['n_test']:,} transactions, only "
        f"**{artifacts['n_test_fraud']} fraud cases** — one misclassified fraud moves "
        f"recall by ~{100 / artifacts['n_test_fraud']:.1f} pts. Threshold = {threshold:.2f}."
    )

    col_cm, col_roc, col_pr = st.columns(3)

    with col_cm:
        st.subheader("Confusion Matrix")
        normalize = st.toggle("Normalize by actual class", value=False)
        cm = metrics["confusion_matrix_norm"] if normalize else metrics["confusion_matrix"]
        labels = ["Legitimate", "Fraud"]
        fig = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=".2%" if normalize else "d",
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "Share" if normalize else "Count"},
        )
        fig.update_layout(coloraxis_showscale=False, height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with col_roc:
        st.subheader("ROC Curve")
        fpr, tpr = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {metrics['roc_auc']:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(dash="dash", color="gray")))
        tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
        fig.add_trace(go.Scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers",
                                 name="Current threshold", marker=dict(size=12, color="red")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          height=420, margin=dict(t=10, b=10), legend=dict(x=0.45, y=0.1))
        st.plotly_chart(fig, width="stretch")

    with col_pr:
        st.subheader("Precision-Recall Curve")
        prec, rec = metrics["pr_curve"]
        baseline = artifacts["n_test_fraud"] / artifacts["n_test"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP = {metrics['pr_auc']:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines",
                                 name=f"Baseline ({baseline:.1%})", line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(x=[metrics["recall"]], y=[metrics["precision"]], mode="markers",
                                 name="Current threshold", marker=dict(size=12, color="red")))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=420,
                          margin=dict(t=10, b=10), legend=dict(x=0.02, y=0.1))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    col_data, col_method = st.columns(2)

    with col_data:
        st.subheader("Dataset Statistics")
        d1, d2, d3 = st.columns(3)
        d1.metric("Transactions", f"{summary['n_rows']:,}")
        d2.metric("Fraud cases", f"{summary['n_fraud']:,}", f"{summary['fraud_rate']:.2%}", delta_color="off")
        d3.metric("Imbalance", f"1 : {summary['imbalance_ratio']:.0f}")
        means = summary["class_means"].T.rename(columns={0: "Legitimate (mean)", 1: "Fraud (mean)"})
        st.dataframe(means.style.format("{:.2f}"), width="stretch")

        coefs = artifacts["coefficients"]
        fig = px.bar(coefs.iloc[::-1], x="coefficient", y="feature", orientation="h",
                     color=np.where(coefs.iloc[::-1]["coefficient"] > 0, "↑ fraud risk", "↓ fraud risk"),
                     color_discrete_map={"↑ fraud risk": "#d62728", "↓ fraud risk": "#2ca02c"},
                     title="Model coefficients (standardized numeric features)")
        fig.update_layout(height=420, legend_title_text="", yaxis_title="", margin=dict(b=10))
        st.plotly_chart(fig, width="stretch")

    with col_method:
        st.subheader("Imbalance Handling Approach")
        t = artifacts["tuning"]
        d = artifacts["metrics_default"]
        st.markdown(
            f"""
**1. Class weighting** — `LogisticRegression(class_weight='balanced')` makes each
fraud error cost ~{summary['imbalance_ratio']:.0f}× more than a legitimate one, so the model
learns from all {artifacts['n_train']:,} training rows without synthetic data.

**2. Threshold tuning** — Balanced weighting inflates fraud probabilities, so the
default 0.5 cut-off yields many false alarms. Using **{CV_FOLDS}-fold stratified CV**
out-of-fold probabilities on the training set, the threshold maximizing F1 was
selected: **{t['threshold']:.3f}** (CV F1 = {t['cv_f1']:.3f} vs {t['cv_f1_at_05']:.3f} at 0.5).
The test set was never used for tuning.

**Preprocessing:** StandardScaler (numeric), passthrough (binary), One-Hot (merchant
category); `transaction_id` dropped. Stratified 80/20 train/test split.
"""
        )
        comparison = pd.DataFrame(
            {
                "Default (0.5)": [d["precision"], d["recall"], d["f1"]],
                f"Tuned ({tuned_threshold:.3f})": [
                    artifacts["metrics"]["precision"],
                    artifacts["metrics"]["recall"],
                    artifacts["metrics"]["f1"],
                ],
            },
            index=["Precision", "Recall", "F1"],
        )
        st.markdown("**Test-set impact of threshold tuning**")
        st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
        st.info(
            "Note: because of class weighting, the predicted probabilities are *risk scores* "
            "that overstate the true fraud likelihood (base rate ≈1.5%). Compare them to the "
            "threshold rather than reading them as calibrated probabilities."
        )

# ---------------------------------------------------------------- tab 2
def _set_inputs(values: dict) -> None:
    for k, v in values.items():
        st.session_state[k] = v


def _sample_from_test(label: int) -> dict:
    """Pick a random test transaction of the given class that the model classifies correctly."""
    X_test, y_test, proba = artifacts["X_test"], artifacts["y_test"], artifacts["test_proba"]
    correct = (proba >= tuned_threshold) == bool(label)
    idx = np.flatnonzero((y_test == label) & correct)
    row = X_test.iloc[np.random.choice(idx)]
    return {
        "amount": float(row["amount"]),
        "transaction_hour": int(row["transaction_hour"]),
        "merchant_category": row["merchant_category"],
        "foreign_transaction": bool(row["foreign_transaction"]),
        "location_mismatch": bool(row["location_mismatch"]),
        "device_trust_score": int(row["device_trust_score"]),
        "velocity_last_24h": int(row["velocity_last_24h"]),
        "cardholder_age": int(row["cardholder_age"]),
    }


for key, value in INPUT_DEFAULTS.items():
    st.session_state.setdefault(key, value)

with tab_predict:
    st.subheader("Transaction details")
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    b1.button("🚨 Load Fraud Sample", on_click=lambda: _set_inputs(_sample_from_test(1)), width="stretch")
    b2.button("✅ Load Legitimate Sample", on_click=lambda: _set_inputs(_sample_from_test(0)), width="stretch")
    b3.button("↺ Reset", on_click=lambda: _set_inputs(INPUT_DEFAULTS), width="stretch")
    st.caption("Sample buttons load a real transaction from the held-out test set.")

    with st.form("predict_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            st.number_input("Amount ($)", min_value=0.0, max_value=5000.0, step=10.0, key="amount")
            st.slider("Transaction hour", 0, 23, key="transaction_hour")
            st.selectbox("Merchant category", MERCHANTS, key="merchant_category")
        with f2:
            st.slider("Device trust score", 0, 99, key="device_trust_score",
                      help="Low = unfamiliar / suspicious device")
            st.slider("Transactions in last 24h", 0, 15, key="velocity_last_24h")
            st.slider("Cardholder age", 18, 90, key="cardholder_age")
        with f3:
            st.checkbox("Foreign transaction", key="foreign_transaction")
            st.checkbox("Location mismatch", key="location_mismatch")
        submitted = st.form_submit_button("🔮 Predict", type="primary", width="stretch")

    if submitted:
        features = {k: st.session_state[k] for k in INPUT_DEFAULTS}
        features["foreign_transaction"] = int(features["foreign_transaction"])
        features["location_mismatch"] = int(features["location_mismatch"])
        proba = predict_proba(pipeline, features)
        is_fraud = proba >= threshold

        color, label, risk = (
            ("#d62728", "FRAUD", "HIGH RISK") if is_fraud else ("#2ca02c", "LEGITIMATE", "LOW RISK")
        )
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Fraud risk score", f"{proba:.1%}")
            st.markdown(
                f"""<div style="background:{color};color:white;padding:18px;border-radius:10px;
                text-align:center;font-size:1.4rem;font-weight:700;">
                {'🚨' if is_fraud else '✅'} {risk}<br>
                <span style="font-size:1rem;font-weight:500;">Classified as {label}</span></div>""",
                unsafe_allow_html=True,
            )
            st.caption(f"Decision threshold: {threshold:.2f}")
        with r2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%", "valueformat": ".1f"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, threshold * 100], "color": "rgba(44,160,44,0.2)"},
                            {"range": [threshold * 100, 100], "color": "rgba(214,39,40,0.2)"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 4}, "value": threshold * 100},
                    },
                )
            )
            fig.update_layout(height=260, margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch")
