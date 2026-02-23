from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field

from src.utils import ARTIFACTS, read_json

app = FastAPI(title="ASX/ABS Early Warning API", version="0.1.0")

# Load artifacts (created by: python -m src.train)
SCHEMA = read_json(ARTIFACTS / "schema.json")
MODEL = load(ARTIFACTS / "model.joblib")

# Load recommended threshold (created by: python -m src.train)
THRESH = read_json(ARTIFACTS / "threshold.json") if (ARTIFACTS / "threshold.json").exists() else None
DEFAULT_THRESHOLD = float(THRESH["threshold"]) if THRESH and "threshold" in THRESH else 0.5


class PredictRequest(BaseModel):
    ticker: str = Field(..., examples=["CBA"])
    gics_industry_group: Optional[str] = Field(None, examples=["Banks"])
    ret_12m: float
    vol_12m: float
    drawdown_12m: float
    mom_3m: float
    liq_proxy: float


class PredictResponse(BaseModel):
    ticker: str
    probability: float
    prediction: int
    threshold: float
    api_version: str = "0.1.0"


class PredictBatchRequest(BaseModel):
    rows: List[PredictRequest]


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "ASX/ABS Early Warning API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "predict": "/predict",
        "predict_batch": "/predict_batch",
        "schema_loaded": bool(SCHEMA),
        "threshold": DEFAULT_THRESHOLD,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    row = pd.DataFrame([req.model_dump()])
    proba = float(MODEL.predict_proba(row)[:, 1][0])
    pred = int(proba >= DEFAULT_THRESHOLD)
    return PredictResponse(
        ticker=req.ticker,
        probability=proba,
        prediction=pred,
        threshold=DEFAULT_THRESHOLD,
    )


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest) -> List[Dict[str, Any]]:
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    probas = MODEL.predict_proba(df)[:, 1]
    preds = (probas >= DEFAULT_THRESHOLD).astype(int)

    out: List[Dict[str, Any]] = []
    for i in range(len(df)):
        out.append(
            {
                "ticker": str(df.iloc[i]["ticker"]),
                "probability": float(probas[i]),
                "prediction": int(preds[i]),
                "threshold": float(DEFAULT_THRESHOLD),
                "api_version": "0.1.0",
            }
        )
    return out