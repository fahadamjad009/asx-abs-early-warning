# 📊 ASX/ABS Early Warning Platform

> Production-style financial risk analytics combining ASX market signals with macroeconomic context to flag early distress indicators for Australian-listed companies.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://asx-early-warning.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-8%2F8_passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**🔗 Live Dashboard:** [asx-early-warning.streamlit.app](https://asx-early-warning.streamlit.app)

---

## 🎯 What this is

An end-to-end machine learning system that scores 79 ASX-listed companies for short-horizon distress risk using market-derived features (returns, volatility, drawdown, momentum, liquidity proxies) and a stacked ensemble model. Built as a portfolio-grade prototype — not a notebook — with a deployed dashboard, FastAPI scoring service, SHAP explainability, automated tests, and Docker-based reproducibility.

**Who it's for:** financial analysts, risk teams, and ML practitioners interested in market-signal-driven early warning systems for emerging-markets equity portfolios.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧮 **Stacked Ensemble Model** | Combines tree-based and linear learners for calibrated risk probabilities |
| 🔍 **SHAP Explainability** | Per-company waterfall plots + global feature importance — every score is auditable |
| 📊 **Interactive Dashboard** | 5-tab Streamlit app: risk overview, company lookup, market analysis, model evaluation, full dataset |
| 🌐 **FastAPI Scoring Service** | `/predict`, `/predict_batch`, `/health` endpoints with OpenAPI docs |
| 🐳 **Docker Compose** | One-command local stack: API + dashboard |
| ✅ **Automated Tests** | 8/8 pytest passing, covering API contract and feature pipelines |
| 📈 **Live Data Pipeline** | yfinance integration pulling real ASX market data |

---

## 🖼️ Dashboard Preview

> *Screenshots coming — visit the [live demo](https://asx-early-warning.streamlit.app) in the meantime.*

The dashboard surfaces:
- **Top 15 highest-risk companies** with sector context and ranked probabilities
- **Per-company SHAP waterfall** explaining each risk score in feature contributions
- **Sector risk heatmap** aggregating average risk by GICS industry group
- **Volatility vs momentum scatter** for market regime exploration
- **Model evaluation panel** with confusion matrix, ROC, and PR curves

---

## 🏗️ Architecture

```
ASX / Yahoo Finance data
        │
        ▼
data/cache/yahoo/*.parquet
        │
        ▼
src/market_data.py
src/price_features.py
src/build_dataset.py
        │
        ▼
data/processed/market_firm_features.csv
        │
        ▼
src/train.py  ──►  models/model.joblib + models/metrics.json
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
FastAPI scoring service          Streamlit dashboard
app/api.py                       app/ui.py
        │                                  │
        ▼                                  ▼
/predict, /predict_batch          Risk Overview · Company Lookup
/health, /docs                    Market Analysis · Model Eval
                                  Full Dataset
```

---

## 🧰 Tech Stack

**ML & Data:** Python 3.11 · scikit-learn 1.5.2 · imbalanced-learn · pandas · numpy · joblib
**Explainability:** SHAP (KernelExplainer)
**API:** FastAPI · Uvicorn · Pydantic
**Dashboard:** Streamlit · Plotly
**Data Sources:** yfinance (ASX market) · ABS macroeconomic indicators
**DevOps:** Docker · docker-compose · GitHub Actions · pytest

---

## 📐 Model & Methodology

- **Target:** binary distress indicator over a 12-month forward window
- **Features:** `ret_12m`, `vol_12m`, `drawdown_12m`, `mom_3m`, `liq_proxy` + GICS sector
- **Model:** stacked ensemble with calibrated probability outputs
- **Threshold:** selected via precision-recall optimisation on the validation fold

### ⚠️ Honest Limitations

This is a **prototype**, not production-deployed risk infrastructure:

- The current dataset is small (79 firms) — model evaluation metrics in `models/metrics.json` reflect training-set performance and should not be interpreted as out-of-sample generalisation
- **Roadmap (next):** rebuild with proper train/test split, point-in-time feature engineering to eliminate look-ahead bias, and walk-forward backtesting
- The target labelling currently includes some leakage between features and outcome — being refactored
- No live broker integration; scoring is batch-mode only

These limitations are flagged explicitly because portfolio honesty matters more than inflated metrics.

---

## 🚀 Quick Start

### Run locally with Docker

```bash
git clone https://github.com/fahadamjad009/asx-abs-early-warning.git
cd asx-abs-early-warning
docker compose up --build
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### Run locally with Python

```bash
git clone https://github.com/fahadamjad009/asx-abs-early-warning.git
cd asx-abs-early-warning
pip install -r requirements.txt

# Start the dashboard
streamlit run app/ui.py

# Or start the API
uvicorn app.api:app --reload
```

### Run tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
asx-abs-early-warning/
├── app/
│   ├── api.py              # FastAPI scoring endpoints
│   └── ui.py               # Streamlit dashboard
├── src/
│   ├── market_data.py      # yfinance ingestion
│   ├── price_features.py   # feature engineering
│   ├── build_dataset.py    # dataset assembly
│   └── train.py            # model training pipeline
├── data/processed/         # feature CSVs
├── models/                 # model.joblib + metrics.json
├── tests/                  # pytest test suite
├── docs/                   # model cards, methodology notes
├── scripts/                # utility CLIs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🛣️ Roadmap

- [x] FastAPI scoring service with batch prediction
- [x] Streamlit dashboard with 5 analytical views
- [x] Docker Compose for local reproducibility
- [x] SHAP explainability (global + per-company)
- [x] Streamlit Cloud deployment
- [ ] GitHub Actions daily data refresh workflow
- [ ] Walk-forward backtesting with proper train/test isolation
- [ ] Point-in-time feature engineering to eliminate look-ahead bias
- [ ] Expand coverage to ASX 200 + cross-listed ADRs
- [ ] Macroeconomic feature integration (RBA cash rate, AUD/USD, commodity prices)
- [ ] PDF risk report auto-generation per company

---

## 👤 Author

**Fahad Amjad** — Master of Data Science & Innovation (UTS) · Master of Professional Accounting

🌐 [GitHub Portfolio](https://github.com/fahadamjad009) 

---

