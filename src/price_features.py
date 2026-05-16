from __future__ import annotations

import numpy as np
import pandas as pd


# -------------------------------------------------
# helpers
# -------------------------------------------------

def _max_drawdown(series: pd.Series) -> float:
    """Compute max drawdown of a price series."""
    if len(series) == 0:
        return np.nan
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return dd.min()


def _forward_max_drawdown(series: pd.Series, horizon: int) -> pd.Series:
    """
    Forward-looking max drawdown over horizon days.
    Used for event labeling.
    """
    out = np.full(len(series), np.nan)

    prices = series.values
    n = len(prices)

    for i in range(n):
        j_end = min(n, i + horizon)
        if j_end <= i + 1:
            continue

        window = prices[i:j_end]
        peak = window[0]
        mdd = 0.0

        for p in window:
            if p > peak:
                peak = p
            dd = p / peak - 1.0
            if dd < mdd:
                mdd = dd

        out[i] = mdd

    return pd.Series(out, index=series.index)


# -------------------------------------------------
# MAIN FEATURE BUILDER
# -------------------------------------------------

def build_price_features(
    df: pd.DataFrame,
    horizon_days: int = 252,  # ~12 months
) -> pd.DataFrame:
    """
    Build firm-level features from daily OHLCV.

    Returns one row per ticker (latest snapshot).
    """

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["ticker", "date"]).copy()

    out_rows = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()

        if len(g) < 260:
            # not enough history
            continue

        close = g["adj_close"].astype(float)
        vol = g["volume"].astype(float)

        # ----------------------------
        # FEATURES
        # ----------------------------

        ret_12m = close.iloc[-1] / close.iloc[-252] - 1.0
        mom_3m = close.iloc[-1] / close.iloc[-63] - 1.0

        daily_ret = close.pct_change()
        vol_12m = daily_ret.rolling(252).std().iloc[-1] * np.sqrt(252)

        drawdown_12m = _max_drawdown(close.iloc[-252:])

        liq_proxy = vol.iloc[-63:].mean() / 1e6  # scaled volume proxy

        # ----------------------------
        # FORWARD EVENT LABEL
        # ----------------------------

        fwd_dd = _forward_max_drawdown(close, horizon_days)
        target = int(fwd_dd.iloc[-horizon_days] < -0.40) if len(fwd_dd.dropna()) else 0

        out_rows.append(
            {
                "ticker": ticker,
                "ret_12m": ret_12m,
                "vol_12m": vol_12m,
                "drawdown_12m": drawdown_12m,
                "mom_3m": mom_3m,
                "liq_proxy": liq_proxy,
                "target_distress_12m": target,
            }
        )

    return pd.DataFrame(out_rows)