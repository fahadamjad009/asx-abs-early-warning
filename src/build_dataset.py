from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from src.preprocessing import load_asx_universe
from src.market_data import fetch_prices_one, PriceFetchConfig
from src.price_features import build_price_features


OUT_PATH = Path("data/processed/market_firm_features.csv")
FAIL_PATH = Path("data/processed/market_dataset_failures.csv")


@dataclass
class BuildResult:
    ok_rows: int
    ok_tickers: int
    failed: int
    out_path: Path
    fail_path: Path


def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_market_dataset(
    limit: int = 120,          # start small; scale later
    force: bool = False,
    min_history_rows: int = 260,

    # ---- Step 2.1: sanity thresholds (tune later) ----
    max_vol_12m: float = 5.0,          # drop extreme volatility outliers
    min_liq_proxy: float = 1e-9,       # drop zero/near-zero liquidity proxy
    clip_ret_12m: Tuple[float, float] = (-0.95, 3.0),
    clip_mom_3m: Tuple[float, float] = (-1.0, 1.0),

    # Optional: allow a test “whitelist”
    allow_tickers: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, BuildResult]:
    asx = load_asx_universe()

    tickers = (
        asx["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )
    tickers = sorted(tickers)

    if allow_tickers:
        allow = {str(t).strip().upper() for t in allow_tickers}
        tickers = [t for t in tickers if t in allow]

    tickers = tickers[: max(1, int(limit))]

    cfg = PriceFetchConfig()

    feat_frames: List[pd.DataFrame] = []
    failures = []

    for t in tickers:
        try:
            prices = fetch_prices_one(t, cfg=cfg, force=force)

            if prices is None or prices.empty:
                failures.append({"ticker": t, "reason": "empty_prices"})
                continue

            if len(prices) < min_history_rows:
                failures.append(
                    {
                        "ticker": t,
                        "reason": f"insufficient_history_rows<{min_history_rows}",
                        "rows": int(len(prices)),
                    }
                )
                continue

            feats = build_price_features(prices)
            if feats is None or feats.empty:
                failures.append({"ticker": t, "reason": "empty_features"})
                continue

            # -----------------------------
            # Step 2.1 — Sanity filters + clipping (senior-grade hygiene)
            # -----------------------------
            try:
                v = float(feats.loc[0, "vol_12m"])
                liq = float(feats.loc[0, "liq_proxy"])
                r12 = float(feats.loc[0, "ret_12m"])
                m3 = float(feats.loc[0, "mom_3m"])
            except Exception as e:
                failures.append({"ticker": t, "reason": "feature_parse_error", "error": _safe_str(e)})
                continue

            if pd.isna(v) or pd.isna(liq) or pd.isna(r12) or pd.isna(m3):
                failures.append({"ticker": t, "reason": "nan_in_features"})
                continue

            # Skip insane volatility (often garbage data / extreme microcaps)
            if v > max_vol_12m:
                failures.append({"ticker": t, "reason": "volatility_too_high", "vol_12m": v})
                continue

            # Skip zero/near-zero liquidity
            if liq <= min_liq_proxy:
                failures.append({"ticker": t, "reason": "liq_proxy_too_low", "liq_proxy": liq})
                continue

            # Clip extreme returns/momentum (winsorization-lite)
            feats.loc[0, "ret_12m"] = _clip(r12, clip_ret_12m[0], clip_ret_12m[1])
            feats.loc[0, "mom_3m"] = _clip(m3, clip_mom_3m[0], clip_mom_3m[1])

            feat_frames.append(feats)

        except KeyboardInterrupt:
            # You pressed Ctrl+C; write what we have so far, then stop gracefully.
            failures.append({"ticker": t, "reason": "keyboard_interrupt"})
            break
        except Exception as e:
            failures.append({"ticker": t, "reason": "exception", "error": _safe_str(e)})
            continue

    out = pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()
    fail_df = pd.DataFrame(failures)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    FAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fail_df.to_csv(FAIL_PATH, index=False)

    res = BuildResult(
        ok_rows=int(len(out)),
        ok_tickers=int(out["ticker"].nunique()) if (not out.empty and "ticker" in out.columns) else 0,
        failed=int(len(fail_df)),
        out_path=OUT_PATH,
        fail_path=FAIL_PATH,
    )

    return out, fail_df, res


if __name__ == "__main__":
    df, fail, res = build_market_dataset(limit=120, force=False)
    print(f"Wrote: {res.out_path.as_posix()} rows={res.ok_rows} tickers={res.ok_tickers}")
    print(f"Wrote: {res.fail_path.as_posix()} failed={res.failed}")
    if not df.empty:
        print(df.head(5).to_string(index=False))
    if not fail.empty:
        print("\nFailures (first 10):")
        print(fail.head(10).to_string(index=False))