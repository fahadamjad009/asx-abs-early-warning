from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL_DEFAULT = "http://127.0.0.1:8000"
BASELINE_CSV = Path("data/processed/baseline_firm_features.csv")

st.set_page_config(page_title="ASX/ABS Early Warning", layout="wide")
st.title("ASX/ABS Early Warning — Risk Scoring Dashboard (Baseline)")

api_url = st.sidebar.text_input("API URL", value=API_URL_DEFAULT)

# -----------------------------
# Load baseline dataset (if available)
# -----------------------------
baseline_df: pd.DataFrame | None = None
if BASELINE_CSV.exists():
    baseline_df = pd.read_csv(BASELINE_CSV)

st.header("Single company scoring")

# Defaults (overridden by autofill if baseline CSV exists)
default_ticker = "CBA"
default_gics = "Banks"
default_ret = 0.10
default_vol = 0.25
default_dd = -0.20
default_mom = 0.02
default_liq = 1.0

if baseline_df is not None:
    st.caption("Auto-fill enabled (loaded baseline dataset from training).")

    ticker_list = baseline_df["ticker"].dropna().astype(str).unique().tolist()
    ticker_selected = st.selectbox("Pick a ticker", options=sorted(ticker_list), index=0)

    row = baseline_df.loc[baseline_df["ticker"].astype(str) == str(ticker_selected)].iloc[0]

    default_ticker = str(row.get("ticker", default_ticker))
    gics_val = row.get("gics_industry_group", "")
    default_gics = "" if pd.isna(gics_val) else str(gics_val)

    default_ret = float(row.get("ret_12m", default_ret))
    default_vol = float(row.get("vol_12m", default_vol))
    default_dd = float(row.get("drawdown_12m", default_dd))
    default_mom = float(row.get("mom_3m", default_mom))
    default_liq = float(row.get("liq_proxy", default_liq))
else:
    st.caption("Auto-fill disabled (run: python -m src.train to generate baseline CSV).")

c1, c2, c3 = st.columns(3)
with c1:
    ticker = st.text_input("Ticker", value=default_ticker)
    gics = st.text_input("GICS industry group (optional)", value=default_gics)
with c2:
    ret_12m = st.number_input("Return 12m", value=default_ret)
    vol_12m = st.number_input("Volatility 12m", value=default_vol)
with c3:
    drawdown_12m = st.number_input("Drawdown 12m", value=default_dd)
    mom_3m = st.number_input("Momentum 3m", value=default_mom)
    liq_proxy = st.number_input("Liquidity proxy", value=default_liq)

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
    r = requests.post(f"{api_url}/predict", json=payload, timeout=30)
    r.raise_for_status()
    out = r.json()
    st.metric("Risk probability", f"{out['probability']:.3f}")
    st.metric("Prediction (>=0.5)", out["prediction"])
    st.caption(f"API version: {out.get('api_version', 'n/a')}")

st.divider()

# -----------------------------
# Top risk list (FAST via /predict_batch)
# -----------------------------
st.header("Top risk companies (scored from baseline dataset)")
if baseline_df is None:
    st.info("Run training to generate data/processed/baseline_firm_features.csv, then refresh this page.")
else:
    n = st.slider("How many rows to score", min_value=20, max_value=300, value=80, step=20)

    sample = baseline_df.head(n).copy()

    batch_rows = []
    for _, r0 in sample.iterrows():
        batch_rows.append(
            {
                "ticker": str(r0.get("ticker", "")),
                "gics_industry_group": None
                if pd.isna(r0.get("gics_industry_group"))
                else str(r0.get("gics_industry_group")),
                "ret_12m": float(r0.get("ret_12m", 0.0)),
                "vol_12m": float(r0.get("vol_12m", 0.0)),
                "drawdown_12m": float(r0.get("drawdown_12m", 0.0)),
                "mom_3m": float(r0.get("mom_3m", 0.0)),
                "liq_proxy": float(r0.get("liq_proxy", 0.0)),
            }
        )

    with st.spinner("Batch scoring..."):
        rr = requests.post(
            f"{api_url}/predict_batch",
            json={"rows": batch_rows},
            timeout=60,
        )
        rr.raise_for_status()
        scored = pd.DataFrame(rr.json()).sort_values("probability", ascending=False)

    st.dataframe(scored.head(25), use_container_width=True)

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
    df = pd.read_csv(up)
    st.write("Preview", df.head(10))

    if st.button("Score batch"):
        # Build batch rows
        rows = []
        for _, r0 in df.iterrows():
            rows.append(
                {
                    "ticker": str(r0.get("ticker", "")),
                    "gics_industry_group": None
                    if pd.isna(r0.get("gics_industry_group"))
                    else str(r0.get("gics_industry_group")),
                    "ret_12m": float(r0.get("ret_12m", 0.0)),
                    "vol_12m": float(r0.get("vol_12m", 0.0)),
                    "drawdown_12m": float(r0.get("drawdown_12m", 0.0)),
                    "mom_3m": float(r0.get("mom_3m", 0.0)),
                    "liq_proxy": float(r0.get("liq_proxy", 0.0)),
                }
            )

        with st.spinner("Batch scoring upload..."):
            rr = requests.post(
                f"{api_url}/predict_batch",
                json={"rows": rows},
                timeout=120,
            )
            rr.raise_for_status()
            scored = pd.DataFrame(rr.json())

        st.write("Scored", scored)
        st.download_button(
            "Download scored CSV",
            scored.to_csv(index=False),
            "scored.csv",
            "text/csv",
        )