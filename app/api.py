from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from joblib import load

from src.utils import ARTIFACTS, read_json

app = FastAPI(title="ASX/ABS Early Warning API", version="0.1.0")

SCHEMA = read_json(ARTIFACTS / "schema.json")
MODEL = load(ARTIFACTS / "model.joblib")


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
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    row = pd.DataFrame([req.model_dump()])
    proba = float(MODEL.predict_proba(row)[:, 1][0])
    pred = int(proba >= 0.5)
    return PredictResponse(ticker=req.ticker, probability=proba, prediction=pred)


@app.post("/predict_batch", response_model=List[PredictResponse])
def predict_batch(req: PredictBatchRequest) -> List[PredictResponse]:
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    probas = MODEL.predict_proba(df)[:, 1]
    preds = (probas >= 0.5).astype(int)

    out: List[PredictResponse] = []
    for i in range(len(df)):
        out.append(
            PredictResponse(
                ticker=str(df.loc[i, "ticker"]),
                probability=float(probas[i]),
                prediction=int(preds[i]),
            )
        )
    return out