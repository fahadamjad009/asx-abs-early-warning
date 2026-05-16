"""Tests for ASX/ABS Early Warning API."""
import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

# ── Root ──
def test_root_returns_200():
    r = client.get("/")
    assert r.status_code == 200

def test_root_has_required_keys():
    data = client.get("/").json()
    for key in ("name", "docs", "health", "predict", "threshold"):
        assert key in data

def test_root_threshold_is_float():
    data = client.get("/").json()
    assert isinstance(data["threshold"], float)
    assert 0.0 <= data["threshold"] <= 1.0

# ── Health ──
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

# ── Predict ──
SAMPLE = {
    "ticker": "CBA",
    "gics_industry_group": "Banks",
    "ret_12m": 0.12,
    "vol_12m": 0.25,
    "drawdown_12m": -0.08,
    "mom_3m": 0.03,
    "liq_proxy": 1.2,
}

def test_predict_returns_200():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200

def test_predict_response_shape():
    data = client.post("/predict", json=SAMPLE).json()
    assert "ticker" in data
    assert "probability" in data
    assert "prediction" in data
    assert data["ticker"] == "CBA"
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in (0, 1)

def test_predict_missing_field_returns_422():
    r = client.post("/predict", json={"ticker": "CBA"})
    assert r.status_code == 422

# ── Predict Batch ──
def test_predict_batch_returns_list():
    r = client.post("/predict_batch", json={"rows": [SAMPLE, SAMPLE]})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 2

