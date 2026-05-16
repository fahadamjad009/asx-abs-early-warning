"""Download real ASX data for demo dashboard."""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# Known active ASX200 blue-chips (won't be delisted)
TICKERS = [
    "CBA.AX","BHP.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","MQG.AX",
    "WES.AX","WOW.AX","TLS.AX","RIO.AX","FMG.AX","ALL.AX","COL.AX",
    "GMG.AX","TCL.AX","STO.AX","WDS.AX","AMC.AX","QBE.AX","SUN.AX",
    "IAG.AX","REA.AX","XRO.AX","CPU.AX","ORG.AX","MIN.AX","JHX.AX",
    "ASX.AX","MPL.AX","RMD.AX","SEK.AX","SHL.AX","NCM.AX","NST.AX",
    "EVN.AX","NEM.AX","APA.AX","GPT.AX","MGR.AX","SCG.AX","SGP.AX",
    "DXS.AX","LLC.AX","QAN.AX","TAH.AX","JBH.AX","HVN.AX","SUL.AX",
    "TWE.AX","TPG.AX","CAR.AX","IEL.AX","PME.AX","WHC.AX","NHC.AX",
    "S32.AX","OZL.AX","IGO.AX","LYC.AX","PLS.AX","SFR.AX","AWC.AX",
    "BSL.AX","BEN.AX","BOQ.AX","HUB.AX","NWL.AX","CGF.AX","IFL.AX",
    "AMP.AX","PDN.AX","DEG.AX","GOR.AX","RRL.AX","WAF.AX","CMM.AX",
    "EDV.AX","ALX.AX","AZJ.AX","AGL.AX","MEZ.AX","VCX.AX","CLW.AX",
]

GICS_MAP = {
    "CBA.AX":"Banks","BHP.AX":"Materials","CSL.AX":"Pharma","NAB.AX":"Banks",
    "WBC.AX":"Banks","ANZ.AX":"Banks","MQG.AX":"Diversified Financials",
    "WES.AX":"Retailing","WOW.AX":"Food Retail","TLS.AX":"Telecom",
    "RIO.AX":"Materials","FMG.AX":"Materials","QAN.AX":"Transport",
    "JBH.AX":"Retailing","XRO.AX":"Software","REA.AX":"Software",
    "STO.AX":"Energy","WDS.AX":"Energy","ORG.AX":"Energy","AGL.AX":"Energy",
    "QBE.AX":"Insurance","SUN.AX":"Insurance","IAG.AX":"Insurance",
    "NCM.AX":"Gold","NST.AX":"Gold","EVN.AX":"Gold","RRL.AX":"Gold",
}

def build():
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in TICKERS:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="2y")
            if hist.empty or len(hist) < 60:
                print(f"  SKIP {t} (not enough data)")
                continue
            c = hist["Close"]
            ret_12m = float((c.iloc[-1] / c.iloc[-252] - 1)) if len(c) >= 252 else float((c.iloc[-1] / c.iloc[0] - 1))
            vol_12m = float(c.pct_change().rolling(252).std().iloc[-1]) if len(c) >= 252 else float(c.pct_change().std())
            peak = c.cummax()
            dd = float(((c - peak) / peak).min())
            mom_3m = float((c.iloc[-1] / c.iloc[-63] - 1)) if len(c) >= 63 else 0.0
            liq_proxy = float(hist["Volume"].tail(20).mean() / max(hist["Volume"].mean(), 1))

            # Synthetic distress target: companies with >30% drawdown and negative 12m return
            target = 1 if (dd < -0.30 and ret_12m < 0) else 0

            rows.append({
                "ticker": t.replace(".AX",""),
                "gics_industry_group": GICS_MAP.get(t, "Other"),
                "ret_12m": round(ret_12m, 4),
                "vol_12m": round(vol_12m, 4),
                "drawdown_12m": round(dd, 4),
                "mom_3m": round(mom_3m, 4),
                "liq_proxy": round(liq_proxy, 4),
                "target_distress_12m": target,
            })
            print(f"  OK {t} ret={ret_12m:.2%} dd={dd:.2%}")
        except Exception as e:
            print(f"  FAIL {t}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "market_firm_features.csv", index=False)
    print(f"\nSaved {len(df)} companies to {out / 'market_firm_features.csv'}")

if __name__ == "__main__":
    build()