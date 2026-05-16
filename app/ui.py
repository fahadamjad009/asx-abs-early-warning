from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Plotly (interactive charts)
import plotly.express as px
import plotly.graph_objects as go

API_URL_DEFAULT = "http://127.0.0.1:8000"

# Prefer real market dataset output; fall back to baseline if present
MARKET_CSV = Path("data/processed/market_firm_features.csv")
BASELINE_CSV = Path("data/processed/baseline_firm_features.csv")


# -----------------------------
# Helpers (HTTP + payload shaping)
# -----------------------------
def _safe_get_json(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if not r.ok:
            return None
        return r.json()
    except Exception:
        return None


def _build_rows_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r0 in df.iterrows():
        rows.append(
            {
                "ticker": str(r0.get("ticker", "")),
                "gics_industry_group": None
                if ("gics_industry_group" not in df.columns) or pd.isna(r0.get("gics_industry_group"))
                else str(r0.get("gics_industry_group")),
                "ret_12m": float(r0.get("ret_12m", 0.0)),
                "vol_12m": float(r0.get("vol_12m", 0.0)),
                "drawdown_12m": float(r0.get("drawdown_12m", 0.0)),
                "mom_3m": float(r0.get("mom_3m", 0.0)),
                "liq_proxy": float(r0.get("liq_proxy", 0.0)),
            }
        )
    return rows


def _post_predict_batch(api_url: str, rows: List[Dict[str, Any]], timeout: int = 120) -> pd.DataFrame:
    rr = requests.post(f"{api_url}/predict_batch", json={"rows": rows}, timeout=timeout)
    rr.raise_for_status()
    data = rr.json()  # API returns a list of dicts
    return pd.DataFrame(data)


def _predict_batch_chunked(
    api_url: str,
    rows: List[Dict[str, Any]],
    chunk_size: int = 50,
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Chunk /predict_batch calls to avoid connection resets on reruns or large payloads.
    """
    out_frames: List[pd.DataFrame] = []
    total = len(rows)
    for start in range(0, total, chunk_size):
        chunk = rows[start : start + chunk_size]
        df_chunk = _post_predict_batch(api_url, chunk, timeout=timeout)
        out_frames.append(df_chunk)
    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)


# -----------------------------
# Plotly chart helpers (interactive, responsive)
# -----------------------------
def _plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
    z = np.array(cm, dtype=int)
    fig = px.imshow(
        z,
        text_auto=True,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        aspect="auto",
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=320)
    st.plotly_chart(fig, use_container_width=True)


def _plot_probability_histogram(prob: np.ndarray, thr: float, title: str) -> None:
    dfp = pd.DataFrame({"probability": prob})
    fig = px.histogram(
        dfp,
        x="probability",
        nbins=40,
        title=title,
        marginal="box",
    )
    fig.add_vline(
        x=float(thr),
        line_dash="dash",
        annotation_text=f"thr={float(thr):.3f}",
        annotation_position="top right",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=380)
    st.plotly_chart(fig, use_container_width=True)


def _plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, title: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=20, r=20, t=60, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_pr_curve(recall: np.ndarray, precision: np.ndarray, ap: float, title: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AP={ap:.3f})"))
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(l=20, r=20, t=60, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_feature_scatter(sample: pd.DataFrame, y_true: np.ndarray, y_score: np.ndarray) -> None:
    """
    Quick EDA-style chart: drawdown vs vol, colored by true label, sized by probability.
    """
    dfp = sample.copy()
    dfp["_y_true"] = y_true
    dfp["_prob"] = y_score
    # Guard in case columns missing
    if ("drawdown_12m" not in dfp.columns) or ("vol_12m" not in dfp.columns):
        return

    fig = px.scatter(
        dfp,
        x="drawdown_12m",
        y="vol_12m",
        color=dfp["_y_true"].astype(str),
        size=np.clip(dfp["_prob"], 0.01, 1.0),
        hover_data=["ticker", "_prob", "ret_12m", "mom_3m", "liq_proxy"],
        title="Data explorer: drawdown_12m vs vol_12m (color=true label, size=pred prob)",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=420)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="ASX/ABS Early Warning", layout="wide")
st.title("ASX/ABS Early Warning — Risk Scoring Dashboard")

api_url = st.sidebar.text_input("API URL", value=API_URL_DEFAULT)

# -----------------------------
# Show active threshold from API (single source of truth)
# -----------------------------
root = _safe_get_json(f"{api_url}/", timeout=10)
active_threshold = None
if isinstance(root, dict):
    active_threshold = root.get("threshold")

if active_threshold is not None:
    st.caption(f"Using calibrated threshold (from API): **{float(active_threshold):.4f}**")
else:
    st.caption("Threshold: unavailable (API not reachable).")

# -----------------------------
# Load dataset for autofill + top-risk scoring
# -----------------------------
data_df: pd.DataFrame | None = None
data_source = None

if MARKET_CSV.exists():
    data_df = pd.read_csv(MARKET_CSV)
    data_source = str(MARKET_CSV)
elif BASELINE_CSV.exists():
    data_df = pd.read_csv(BASELINE_CSV)
    data_source = str(BASELINE_CSV)

st.sidebar.markdown("### Data source")
if data_source:
    st.sidebar.success(f"Loaded:\n{data_source}")
else:
    st.sidebar.warning("No local dataset found. Train/build dataset to enable autofill + top-risk table.")

# -----------------------------
# Single company scoring
# -----------------------------
st.header("Single company scoring")

default_ticker = "CBA"
default_gics = "Banks"
default_ret = 0.10
default_vol = 0.25
default_dd = -0.20
default_mom = 0.02
default_liq = 1.0

if data_df is not None and "ticker" in data_df.columns:
    st.caption("Auto-fill enabled (loaded local dataset).")

    ticker_list = data_df["ticker"].dropna().astype(str).unique().tolist()
    ticker_selected = st.selectbox("Pick a ticker", options=sorted(ticker_list), index=0)

    row = data_df.loc[data_df["ticker"].astype(str) == str(ticker_selected)].iloc[0]

    default_ticker = str(row.get("ticker", default_ticker))

    if "gics_industry_group" in data_df.columns:
        gics_val = row.get("gics_industry_group", "")
        default_gics = "" if pd.isna(gics_val) else str(gics_val)
    else:
        default_gics = ""

    default_ret = float(row.get("ret_12m", default_ret))
    default_vol = float(row.get("vol_12m", default_vol))
    default_dd = float(row.get("drawdown_12m", default_dd))
    default_mom = float(row.get("mom_3m", default_mom))
    default_liq = float(row.get("liq_proxy", default_liq))
else:
    st.caption("Auto-fill disabled (no dataset found).")

c1, c2, c3 = st.columns(3)
with c1:
    ticker = st.text_input("Ticker", value=default_ticker)
    gics = st.text_input("GICS industry group (optional)", value=default_gics)
with c2:
    ret_12m = st.number_input("Return 12m", value=float(default_ret))
    vol_12m = st.number_input("Volatility 12m", value=float(default_vol))
with c3:
    drawdown_12m = st.number_input("Drawdown 12m", value=float(default_dd))
    mom_3m = st.number_input("Momentum 3m", value=float(default_mom))
    liq_proxy = st.number_input("Liquidity proxy", value=float(default_liq))

if st.button("Score"):
    payload = {
        "ticker": ticker,
        "gics_industry_group": gics if gics.strip() else None,
        "ret_12m": float(ret_12m),
        "vol_12m": float(vol_12m),
        "drawdown_12m": float(drawdown_12m),
        "mom_3m": float(mom_3m),
        "liq_proxy": float(liq_proxy),
    }
    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=30)
        r.raise_for_status()
        out = r.json()

        st.metric("Risk probability", f"{out['probability']:.3f}")

        thr = out.get("threshold", active_threshold if active_threshold is not None else 0.5)
        st.metric(f"Prediction (>= {float(thr):.3f})", out["prediction"])

        st.caption(f"API version: {out.get('api_version', 'n/a')}")
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")

st.divider()

# -----------------------------
# Top risk list (FAST via /predict_batch)
# -----------------------------
st.header("Top risk companies (scored from local dataset)")
if data_df is None:
    st.info(
        "No local dataset found. Generate one:\n"
        "- Real market: python -c \"from src.build_dataset import build_market_dataset; build_market_dataset(limit=200, force=False)\"\n"
        "- Then train: python -m src.train --data .\\data\\processed\\market_firm_features.csv --out .\\models"
    )
else:
    n = st.slider("How many rows to score", min_value=20, max_value=300, value=80, step=20)
    sample = data_df.head(n).copy()
    batch_rows = _build_rows_from_df(sample)

    try:
        with st.spinner("Batch scoring..."):
            scored = _predict_batch_chunked(api_url, batch_rows, chunk_size=50, timeout=120)

        if "probability" in scored.columns:
            scored = scored.sort_values("probability", ascending=False)

        st.dataframe(scored.head(25), use_container_width=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Batch scoring failed: {e}")

st.divider()

# -----------------------------
# Batch scoring via CSV upload (FAST via /predict_batch)
# -----------------------------
st.header("Batch scoring (CSV upload)")
st.caption(
    "CSV must contain: ticker, ret_12m, vol_12m, drawdown_12m, mom_3m, liq_proxy "
    "(and optional gics_industry_group)."
)

up = st.file_uploader("Upload CSV", type=["csv"])
if up is not None:
    df_up = pd.read_csv(up)
    st.write("Preview", df_up.head(10))

    if st.button("Score batch upload"):
        rows = _build_rows_from_df(df_up)

        try:
            with st.spinner("Batch scoring upload..."):
                scored = _predict_batch_chunked(api_url, rows, chunk_size=50, timeout=120)

            st.write("Scored", scored)
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False),
                "scored.csv",
                "text/csv",
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Batch upload scoring failed: {e}")

# -----------------------------
# Model evaluation (interactive, button-triggered, chunked calls)
# -----------------------------
st.divider()
st.header("Model evaluation (on local dataset)")

# Show training metrics (if available)
metrics_path = Path("models/metrics.json")
if metrics_path.exists():
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        st.caption(
            f"Training metrics — ROC AUC: {m.get('roc_auc')}, "
            f"Avg Precision: {m.get('avg_precision')}, "
            f"Best threshold: {float(m.get('best_threshold', 0.5)):.4f}"
        )
    except Exception:
        st.caption("Training metrics found but could not be read.")

EVAL_CSV = MARKET_CSV if MARKET_CSV.exists() else BASELINE_CSV

if not EVAL_CSV.exists():
    st.info("No evaluation dataset found.")
else:
    eval_df = pd.read_csv(EVAL_CSV)

    if "target_distress_12m" not in eval_df.columns:
        st.warning("Evaluation needs 'target_distress_12m' column in the dataset.")
    else:
        # Prefer API threshold; fall back to 0.5
        thr = float(active_threshold) if active_threshold is not None else 0.5

        n_eval = st.slider(
            "Rows to evaluate",
            min_value=50,
            max_value=min(500, len(eval_df)),
            value=min(170, len(eval_df)),
            step=10,
        )

        chunk_size = st.sidebar.slider("Eval batch chunk size", 10, 100, 50, 10)

        # Don’t auto-fire evaluation on every rerun.
        run_eval = st.button("Run evaluation")

        # Cache in session state so slider movement doesn't refetch unless you press button
        if "eval_cache" not in st.session_state:
            st.session_state.eval_cache = None

        if run_eval:
            sample = eval_df.head(n_eval).copy()
            rows = _build_rows_from_df(sample)

            try:
                with st.spinner("Scoring evaluation sample via API..."):
                    scored = _predict_batch_chunked(api_url, rows, chunk_size=int(chunk_size), timeout=180)

                st.session_state.eval_cache = {
                    "n_eval": int(n_eval),
                    "threshold": float(thr),
                    "sample": sample,
                    "scored": scored,
                }
            except requests.exceptions.RequestException as e:
                st.error(
                    "Evaluation call failed. Make sure FastAPI is still running on 127.0.0.1:8000 "
                    "(ideally in a separate terminal). If you run uvicorn with --reload, "
                    "it can restart mid-request and reset connections.\n\n"
                    f"Error: {e}"
                )
                st.session_state.eval_cache = None

        cache = st.session_state.eval_cache
        if cache is None:
            st.info("Click **Run evaluation** to score and render metrics.")
        else:
            # Lazy import sklearn only when needed (keeps UI boot fast)
            try:
                from sklearn.metrics import (
                    average_precision_score,
                    confusion_matrix,
                    precision_recall_curve,
                    roc_auc_score,
                    roc_curve,
                )
            except Exception as e:
                st.error(
                    "scikit-learn is required for evaluation plots/metrics.\n\n"
                    "Install it with: `pip install scikit-learn`\n\n"
                    f"Import error: {e}"
                )
                st.stop()

            sample = cache["sample"]
            scored = cache["scored"]
            thr = float(cache["threshold"])

            # Align y_true / y_score
            y_true = sample["target_distress_12m"].astype(int).to_numpy()

            if "probability" not in scored.columns:
                st.error("API /predict_batch response missing 'probability' column.")
                st.stop()

            y_score = scored["probability"].astype(float).to_numpy()
            y_pred = (y_score >= float(thr)).astype(int)

            # Summary metrics
            pos_rate = float(y_true.mean())
            c1m, c2m, c3m, c4m = st.columns(4)
            with c1m:
                st.metric("Rows evaluated", int(len(sample)))
            with c2m:
                st.metric("Positive rate", f"{pos_rate:.3f}")
            with c3m:
                st.metric("Threshold", f"{thr:.3f}")
            with c4m:
                st.metric("Mean prob", f"{float(np.mean(y_score)):.3f}")

            if len(set(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_score))
                ap = float(average_precision_score(y_true, y_score))
            else:
                auc = float("nan")
                ap = float("nan")

            k1, k2 = st.columns(2)
            with k1:
                st.metric("ROC AUC", f"{auc:.3f}" if np.isfinite(auc) else "n/a")
            with k2:
                st.metric("Avg Precision", f"{ap:.3f}" if np.isfinite(ap) else "n/a")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            st.subheader("Confusion matrix (using calibrated threshold)")
            _plot_confusion_matrix(cm, title="Confusion matrix (calibrated threshold)")

            # Probability distribution
            st.subheader("Probability distribution")
            _plot_probability_histogram(y_score, thr, title="Predicted probability distribution")

            # ROC / PR curves (only if both classes exist)
            if len(set(y_true)) > 1:
                st.subheader("ROC curve")
                fpr, tpr, _ = roc_curve(y_true, y_score)
                _plot_roc_curve(fpr, tpr, auc, title="ROC curve")

                st.subheader("Precision-Recall curve")
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                _plot_pr_curve(recall, precision, ap, title="Precision-Recall curve")
            else:
                st.info("ROC/PR curves unavailable (only one class present in the evaluated sample).")

            # Small EDA explorer chart
            st.subheader("Quick data explorer (EDA-style)")
            _plot_feature_scatter(sample, y_true, y_score)

            # Error analysis tables
            st.subheader("Error analysis")

            scored_view = scored.copy()
            scored_view["y_true"] = y_true
            scored_view["y_pred"] = y_pred

            fp = scored_view[(scored_view.y_true == 0) & (scored_view.y_pred == 1)]
            fn = scored_view[(scored_view.y_true == 1) & (scored_view.y_pred == 0)]

            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("**False Positives (Pred=1, True=0)**")
                st.dataframe(fp.head(20), use_container_width=True)
            with ec2:
                st.markdown("**False Negatives (Pred=0, True=1)**")
                st.dataframe(fn.head(20), use_container_width=True)

            st.subheader("Top scored (inspection)")
            st.dataframe(scored_view.sort_values("probability", ascending=False).head(30), use_container_width=True)