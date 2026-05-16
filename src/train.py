from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Config
# -----------------------------
DEFAULT_DATA_PATH = os.path.join("data", "processed", "market_firm_features.csv")
DEFAULT_OUT_DIR = "models"

# label column auto-detection candidates
# FIX APPLIED: add your real label column name used by build_dataset.py
LABEL_CANDIDATES = [
    "label",
    "distress_label",
    "y",
    "target",
    "forward_distress",
    "event",
    "bankrupt",
    "failure",
    "target_distress_12m",  # <-- ADDED (your dataset label)
]

# columns that are identifiers (kept for debugging but excluded from training features)
ID_COL_CANDIDATES = [
    "ticker",
    "asx_ticker",
    "symbol",
    "company",
    "name",
    "date",
    "asof_date",
]


@dataclass
class TrainArtifacts:
    model_path: str
    metrics_path: str


def _now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _detect_label_column(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    for c in LABEL_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(
        "Could not auto-detect label column.\n"
        f"Tried: {LABEL_CANDIDATES}\n"
        f"Found columns: {list(df.columns)[:60]}{'...' if len(df.columns) > 60 else ''}\n\n"
        "Fix: either rename your label column to one of the candidates above, "
        "or add your exact label column name into LABEL_CANDIDATES in src/train.py."
    )


def _pick_id_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            cols.append(c)
    return cols


def _find_best_threshold(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, dict]:
    """
    Choose threshold that maximizes F1 on the PR curve.
    Returns: (best_threshold, info)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, p)

    eps = 1e-12
    f1 = (2 * precision * recall) / (precision + recall + eps)

    # precision_recall_curve returns thresholds of length n-1;
    # the last precision/recall point has no threshold.
    f1 = f1[:-1]
    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        return 0.5, {"note": "No thresholds produced; defaulting to 0.5"}

    best_i = int(np.nanargmax(f1))
    return float(thresholds[best_i]), {
        "best_index": best_i,
        "best_f1": float(f1[best_i]),
        "best_precision": float(precision[best_i]),
        "best_recall": float(recall[best_i]),
    }


def build_model(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
    )

    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("lr", LogisticRegression(max_iter=3000, n_jobs=-1)),
    ]

    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=4000),
        passthrough=False,
        n_jobs=-1,
    )

    calibrated = CalibratedClassifierCV(stack, method="isotonic", cv=3)

    return Pipeline(steps=[("pre", pre), ("clf", calibrated)])


def train(data_path: str, out_dir: str, test_size: float, random_state: int) -> TrainArtifacts:
    df = pd.read_csv(data_path)

    label_col = _detect_label_column(df)
    id_cols = _pick_id_columns(df)

    y = df[label_col].astype(int).to_numpy()

    X = df.drop(columns=[label_col], errors="ignore").copy()
    if id_cols:
        X_ids = X[id_cols].copy()
        X = X.drop(columns=id_cols, errors="ignore")
    else:
        X_ids = None

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = build_model(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else float("nan")
    ap = average_precision_score(y_test, p_test) if len(np.unique(y_test)) > 1 else float("nan")

    best_thr, thr_info = _find_best_threshold(y_test, p_test)
    y_hat = (p_test >= best_thr).astype(int)
    cm = confusion_matrix(y_test, y_hat).tolist()

    metrics = {
        "trained_at_utc": _now_utc(),
        "data_path": data_path,
        "rows": int(df.shape[0]),
        "features": int(X.shape[1]),
        "label_col": label_col,
        "id_cols_dropped": id_cols,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "roc_auc": float(auc) if auc == auc else None,
        "avg_precision": float(ap) if ap == ap else None,
        "best_threshold": float(best_thr),
        "threshold_details": thr_info,
        "confusion_matrix": cm,
        "pos_rate_overall": float(np.mean(y)),
        "pos_rate_test": float(np.mean(y_test)),
    }

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.joblib")
    metrics_path = os.path.join(out_dir, "metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if X_ids is not None:
        sample_path = os.path.join(out_dir, "debug_ids_sample.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump({"id_cols": id_cols, "head": X_ids.head(20).to_dict(orient="records")}, f, indent=2)

    return TrainArtifacts(model_path=model_path, metrics_path=metrics_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to processed CSV")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory for model + metrics")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    artifacts = train(
        data_path=args.data,
        out_dir=args.out,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(f"[OK] saved model -> {artifacts.model_path}")
    print(f"[OK] saved metrics -> {artifacts.metrics_path}")


if __name__ == "__main__":
    main()