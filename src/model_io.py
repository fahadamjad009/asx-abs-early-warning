from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Tuple

import joblib


DEFAULT_MODELS_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "model.joblib")
DEFAULT_METRICS_PATH = os.path.join(DEFAULT_MODELS_DIR, "metrics.json")


@lru_cache(maxsize=1)
def load_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    metrics_path: str = DEFAULT_METRICS_PATH,
) -> Tuple[Any, Dict[str, Any]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    model = joblib.load(model_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return model, metrics


def get_threshold(metrics: Dict[str, Any], default: float = 0.5) -> float:
    try:
        t = float(metrics.get("best_threshold", default))
        # guard rails
        if not (0.0 < t < 1.0):
            return default
        return t
    except Exception:
        return default