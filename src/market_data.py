from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# yfinance can be slow / rate-limited; we cache to disk
import yfinance as yf


CACHE_DIR_DEFAULT = Path("data/cache/yahoo")
CACHE_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PriceFetchConfig:
    cache_dir: Path = CACHE_DIR_DEFAULT
    interval: str = "1d"
    period: str = "2y"
    sleep_s: float = 0.2
    cache_ttl_hours: float = 24.0


def _asx_yahoo_symbol(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t.endswith(".AX"):
        return t
    return f"{t}.AX"


def _cache_path(cfg: PriceFetchConfig, ticker: str) -> Path:
    t = str(ticker).strip().upper().replace(".", "_")
    return cfg.cache_dir / f"{t}.parquet"


def _cache_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    age_s = time.time() - path.stat().st_mtime
    return age_s <= ttl_hours * 3600.0


def fetch_prices_one(
    ticker: str,
    cfg: Optional[PriceFetchConfig] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for one ASX ticker using Yahoo Finance (yfinance),
    using local parquet cache to avoid refetching.

    Returns columns:
      date, open, high, low, close, adj_close, volume, ticker
    """
    cfg = cfg or PriceFetchConfig()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    p = _cache_path(cfg, ticker)
    if (not force) and _cache_fresh(p, cfg.cache_ttl_hours):
        return pd.read_parquet(p)

    sym = _asx_yahoo_symbol(ticker)

    df = yf.download(
        sym,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )

    if df is None or df.empty:
        out = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]
        )
        out.to_parquet(p, index=False)
        return out

    # yfinance sometimes returns MultiIndex columns; flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if x not in ("", None)]).strip()
            for tup in df.columns.values
        ]

    df = df.reset_index()

    # -------------------------------------------------
    # Normalize column names
    # -------------------------------------------------
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # -------------------------------------------------
    # Strip ticker suffixes that yfinance sometimes appends:
    # e.g. open_cba.ax, close_cba_ax, adj_close_cba.ax
    # -------------------------------------------------
    tkr = str(ticker).strip().lower()
    cleaned_cols = {}
    for c in df.columns:
        new_c = c

        # remove common suffix patterns
        if f"_{tkr}.ax" in new_c:
            new_c = new_c.replace(f"_{tkr}.ax", "")
        if f"_{tkr}_ax" in new_c:
            new_c = new_c.replace(f"_{tkr}_ax", "")

        # if any lingering ".ax" inside name
        new_c = new_c.replace(".ax", "")

        cleaned_cols[c] = new_c

    df = df.rename(columns=cleaned_cols)

    # -------------------------------------------------
    # Map known variants to canonical names
    # -------------------------------------------------
    rename_map = {}

    # date column variants
    if "date" not in df.columns:
        if "datetime" in df.columns:
            rename_map["datetime"] = "date"
        elif "index" in df.columns:
            rename_map["index"] = "date"

    # adj close variants
    if "adjclose" in df.columns:
        rename_map["adjclose"] = "adj_close"
    if "adj_close" in df.columns:
        rename_map["adj_close"] = "adj_close"

    # sometimes flattened multiindex creates suffixes
    for c in list(df.columns):
        if c.endswith("_adj_close") and "adj_close" not in df.columns:
            rename_map[c] = "adj_close"
        if c.endswith("_close") and "close" not in df.columns:
            rename_map[c] = "close"
        if c.endswith("_open") and "open" not in df.columns:
            rename_map[c] = "open"
        if c.endswith("_high") and "high" not in df.columns:
            rename_map[c] = "high"
        if c.endswith("_low") and "low" not in df.columns:
            rename_map[c] = "low"
        if c.endswith("_volume") and "volume" not in df.columns:
            rename_map[c] = "volume"

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    # Convert date; drop invalid
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Fill adj_close if missing but close exists
    if df["adj_close"].isna().all() and (not df["close"].isna().all()):
        df["adj_close"] = df["close"]

    df["ticker"] = str(ticker).strip().upper()

    # Keep only required cols
    df = df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Cache
    df.to_parquet(p, index=False)

    if cfg.sleep_s and cfg.sleep_s > 0:
        time.sleep(cfg.sleep_s)

    return df


def fetch_prices_many(
    tickers: list[str],
    cfg: Optional[PriceFetchConfig] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch many tickers and concatenate into one dataframe.
    """
    cfg = cfg or PriceFetchConfig()
    frames = [fetch_prices_one(t, cfg=cfg, force=force) for t in tickers]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()