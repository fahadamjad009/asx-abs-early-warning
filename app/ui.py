"""ASX/ABS Early Warning — Self-contained Streamlit Dashboard."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from joblib import load

# ── Paths ──
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")
DATA_PATH = Path("data/processed/market_firm_features.csv")

# ── Load artifacts ──
@st.cache_resource
def load_model():
    return load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

FEATURE_COLS = ["ret_12m", "vol_12m", "drawdown_12m", "mom_3m", "liq_proxy"]
PREDICT_COLS = ["ticker", "gics_industry_group"] + FEATURE_COLS

def predict_df(model, df, threshold):
    pred_df = df[PREDICT_COLS].copy()
    probas = model.predict_proba(pred_df)[:, 1]
    preds = (probas >= threshold).astype(int)
    result = df.copy()
    result["probability"] = probas
    result["prediction"] = preds
    result["risk_level"] = pd.cut(probas, bins=[0, 0.3, 0.6, 0.85, 1.0],
                                   labels=["Low", "Medium", "High", "Critical"])
    return result.sort_values("probability", ascending=False)

# ── Page config ──
st.set_page_config(page_title="ASX/ABS Early Warning", page_icon="📊", layout="wide")

# ── Load everything ──
model = load_model()
metrics = load_metrics()
data = load_data()
threshold = float(metrics.get("best_threshold", 0.5))

# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
st.markdown("# 📊 ASX/ABS Early Warning — Risk Scoring Platform")
st.caption("Production-style financial risk analytics combining ASX market signals with macroeconomic context.")

# ── Hero metrics ──
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Companies", len(data) if data is not None else 0)
m2.metric("Calibrated Threshold", f"{threshold:.4f}")
m3.metric("ROC AUC", f"{metrics.get('roc_auc', 'n/a')}")
m4.metric("Avg Precision", f"{metrics.get('avg_precision', 'n/a')}")
m5.metric("Model", "Stacked Ensemble")

st.divider()

if data is None:
    st.error("No dataset found at data/processed/market_firm_features.csv")
    st.stop()

# ══════════════════════════════════════════
# SCORE ALL COMPANIES
# ══════════════════════════════════════════
scored = predict_df(model, data, threshold)

# ══════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Risk Overview", "🔍 Company Lookup", "📈 Market Analysis",
    "🧪 Model Evaluation", "📋 Full Dataset"
])

# ── TAB 1: RISK OVERVIEW ──
with tab1:
    st.header("Risk Overview")

    # Top risk companies
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Top 15 Highest-Risk Companies")
        top = scored.head(15)[["ticker", "gics_industry_group", "probability", "risk_level",
                                "ret_12m", "drawdown_12m", "vol_12m"]].reset_index(drop=True)
        st.dataframe(top, use_container_width=True, height=400)

    with col_b:
        st.subheader("Risk Distribution")
        risk_counts = scored["risk_level"].value_counts()
        fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                         color=risk_counts.index,
                         color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12",
                                             "High": "#e74c3c", "Critical": "#8e44ad"},
                         hole=0.4)
        fig_pie.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Sector risk heatmap
    st.subheader("Sector Risk Heatmap")
    sector_stats = scored.groupby("gics_industry_group").agg(
        avg_risk=("probability", "mean"),
        max_risk=("probability", "max"),
        count=("ticker", "count"),
        avg_drawdown=("drawdown_12m", "mean"),
    ).sort_values("avg_risk", ascending=False).reset_index()

    fig_hm = px.bar(sector_stats, x="gics_industry_group", y="avg_risk",
                    color="avg_risk", color_continuous_scale="RdYlGn_r",
                    hover_data=["max_risk", "count", "avg_drawdown"],
                    title="Average Risk Score by GICS Sector")
    fig_hm.update_layout(xaxis_title="Sector", yaxis_title="Avg Risk Probability",
                         margin=dict(l=20, r=20, t=60, b=80), height=400)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Probability distribution
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(scored, x="probability", nbins=30, marginal="box",
                            color_discrete_sequence=["#3498db"],
                            title="Distribution of Risk Probabilities Across All Companies")
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold: {threshold:.3f}")
    fig_hist.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

# ── TAB 2: COMPANY LOOKUP ──
with tab2:
    st.header("Single Company Risk Scoring")
    tickers = sorted(scored["ticker"].unique().tolist())
    selected = st.selectbox("Select a company", tickers)

    row = scored[scored["ticker"] == selected].iloc[0]

    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Risk Probability", f"{row['probability']:.4f}")
    lc2.metric("Risk Level", row["risk_level"])
    lc3.metric("Prediction", "⚠️ DISTRESS" if row["prediction"] == 1 else "✅ HEALTHY")
    lc4.metric("Sector", row["gics_industry_group"])

    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    fc1.metric("Return 12m", f"{row['ret_12m']:.2%}")
    fc2.metric("Volatility 12m", f"{row['vol_12m']:.4f}")
    fc3.metric("Drawdown 12m", f"{row['drawdown_12m']:.2%}")
    fc4.metric("Momentum 3m", f"{row['mom_3m']:.2%}")
    fc5.metric("Liquidity", f"{row['liq_proxy']:.4f}")

    # Peer comparison
    st.subheader("Peer Comparison (Same Sector)")
    peers = scored[scored["gics_industry_group"] == row["gics_industry_group"]]
    fig_peer = px.bar(peers.sort_values("probability", ascending=False),
                      x="ticker", y="probability",
                      color="probability", color_continuous_scale="RdYlGn_r",
                      title=f"Risk Scores: {row['gics_industry_group']} Sector")
    fig_peer.add_hline(y=threshold, line_dash="dash", line_color="red")
    fig_peer.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=380)
    st.plotly_chart(fig_peer, use_container_width=True)

    # Manual scoring
    st.subheader("Custom Input Scoring")
    with st.expander("Enter custom values"):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            c_ticker = st.text_input("Ticker", value="CUSTOM")
            c_gics = st.text_input("GICS Group", value="Other")
        with cc2:
            c_ret = st.number_input("Return 12m", value=0.0, format="%.4f")
            c_vol = st.number_input("Volatility 12m", value=0.25, format="%.4f")
            c_dd = st.number_input("Drawdown 12m", value=-0.20, format="%.4f")
        with cc3:
            c_mom = st.number_input("Momentum 3m", value=0.0, format="%.4f")
            c_liq = st.number_input("Liquidity proxy", value=1.0, format="%.4f")

        if st.button("Score Custom Input"):
            custom = pd.DataFrame([{
                "ticker": c_ticker, "gics_industry_group": c_gics,
                "ret_12m": c_ret, "vol_12m": c_vol, "drawdown_12m": c_dd,
                "mom_3m": c_mom, "liq_proxy": c_liq,
            }])
            proba = float(model.predict_proba(custom[PREDICT_COLS])[:, 1][0])
            pred = int(proba >= threshold)
            st.metric("Risk Probability", f"{proba:.4f}")
            st.metric("Prediction", "⚠️ DISTRESS" if pred == 1 else "✅ HEALTHY")

# ── TAB 3: MARKET ANALYSIS ──
with tab3:
    st.header("Market-Wide Analysis")

    # Scatter: Return vs Drawdown
    st.subheader("Return vs Drawdown (bubble = risk probability)")
    fig_sc = px.scatter(scored, x="drawdown_12m", y="ret_12m",
                        size=np.clip(scored["probability"], 0.05, 1.0),
                        color="risk_level",
                        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12",
                                            "High": "#e74c3c", "Critical": "#8e44ad"},
                        hover_data=["ticker", "probability", "gics_industry_group", "vol_12m"],
                        title="Return vs Drawdown — Risk-Sized Scatter")
    fig_sc.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=500)
    st.plotly_chart(fig_sc, use_container_width=True)

    # Volatility vs Momentum
    st.subheader("Volatility vs Momentum")
    fig_vm = px.scatter(scored, x="vol_12m", y="mom_3m",
                        color="probability", color_continuous_scale="RdYlGn_r",
                        hover_data=["ticker", "gics_industry_group", "drawdown_12m"],
                        title="Volatility vs Momentum — Color = Risk Score")
    fig_vm.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=450)
    st.plotly_chart(fig_vm, use_container_width=True)

    # Feature correlation
    st.subheader("Feature Correlation Matrix")
    corr = scored[FEATURE_COLS + ["probability"]].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         title="Feature Correlation Heatmap")
    fig_corr.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

# ── TAB 4: MODEL EVALUATION ──
with tab4:
    st.header("Model Evaluation")

    if "target_distress_12m" not in scored.columns:
        st.warning("No target column (target_distress_12m) in dataset — evaluation metrics unavailable.")
        st.info("Training metrics from models/metrics.json are shown in the header.")
    else:
        from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                                     precision_recall_curve, average_precision_score,
                                     classification_report)

        y_true = scored["target_distress_12m"].astype(int).values
        y_score = scored["probability"].values
        y_pred = scored["prediction"].values

        ev1, ev2, ev3, ev4 = st.columns(4)
        ev1.metric("Total Samples", len(y_true))
        ev2.metric("Positive Rate", f"{y_true.mean():.3f}")

        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ev3.metric("ROC AUC (live)", f"{auc:.4f}")
            ev4.metric("Avg Precision (live)", f"{ap:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, x=["Pred Healthy", "Pred Distress"],
                               y=["True Healthy", "True Distress"],
                               color_continuous_scale="Blues", title="Confusion Matrix")
            fig_cm.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dash"), name="Chance"))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                                  margin=dict(l=20, r=20, t=60, b=20), height=400)
            st.plotly_chart(fig_roc, use_container_width=True)

            # PR curve
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, name=f"PR (AP={ap:.3f})"))
            fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                                 yaxis_title="Precision",
                                 margin=dict(l=20, r=20, t=60, b=20), height=400)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            ev3.metric("ROC AUC", "n/a (single class)")
            ev4.metric("Avg Precision", "n/a")
            st.info("Only one class present — ROC/PR curves unavailable.")

# ── TAB 5: FULL DATASET ──
with tab5:
    st.header("Full Scored Dataset")
    st.dataframe(scored, use_container_width=True, height=600)
    st.download_button("Download scored CSV", scored.to_csv(index=False),
                       "asx_scored_results.csv", "text/csv")

# ── Footer ──
st.divider()
st.caption("ASX/ABS Early Warning Platform | Fahad Amjad | github.com/fahadamjad009/asx-abs-early-warning")