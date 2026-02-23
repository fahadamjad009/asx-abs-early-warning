from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt

from .preprocessing import (
    make_training_dataset,
    fit_and_save_preprocessor,
    build_full_feature_frame,
)
from .utils import ARTIFACTS, FIGURES, write_json, ensure_dirs

# Optional polish: silence loky "physical cores" warning on some Windows setups.
# Set to your logical core count if you want (doesn't affect results).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """
    Pick a probability threshold that maximizes F1 on the provided labels/probabilities.
    Returns dict: threshold, precision, recall, f1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    # precision_recall_curve returns precision/recall arrays length n_thresholds+1
    # thresholds length n_thresholds; align by trimming last p/r entry
    precision = precision[:-1]
    recall = recall[:-1]

    denom = precision + recall
    f1 = np.where(denom > 0, 2 * (precision * recall) / denom, 0.0)

    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    idx = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
    }


def train(
    abs_csv_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    ensure_dirs()

    # Build training dataset (X/y) + schema
    X, y, schema = make_training_dataset(abs_csv_path=abs_csv_path)

    # Export full baseline dataset for UI exploration (ids + features + target)
    full_df = build_full_feature_frame(abs_csv_path=abs_csv_path)
    full_out = Path("data/processed/baseline_firm_features.csv")
    full_out.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(full_out, index=False)

    # Fit and persist preprocessing pipeline
    pre = fit_and_save_preprocessor(X, schema)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    base_estimators = [
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
        ),
    ]

    final_est = LogisticRegression(max_iter=2000, class_weight="balanced")

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_est,
        passthrough=False,
        n_jobs=-1,
    )

    model = ImbPipeline(
        steps=[
            ("pre", pre),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", stack),
        ]
    )

    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    proba = calibrated.predict_proba(X_test)[:, 1]

    # ---- key upgrade: choose threshold instead of hard 0.5 ----
    thr_info = _best_f1_threshold(y_test.to_numpy(), proba)
    threshold = float(thr_info["threshold"])
    pred = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
        "threshold": thr_info,
        "ui_export_csv": str(full_out.as_posix()),
    }

    dump(calibrated, ARTIFACTS / "model.joblib")
    write_json(ARTIFACTS / "metrics.json", metrics)
    write_json(ARTIFACTS / "threshold.json", thr_info)

    # Probability histogram
    plt.figure()
    plt.hist(proba, bins=30)
    plt.title("Predicted risk probability (test set)")
    plt.xlabel("p(distress)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIGURES / "probability_hist.png", dpi=160)
    plt.close()

    # PR curve plot (nice for recruiter-grade reports)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(FIGURES / "pr_curve.png", dpi=160)
    plt.close()

    model_card = {
        "problem": "ASX/ABS early warning (baseline demo scaffold)",
        "label": "target_distress_12m (noisy synthetic baseline label)",
        "notes": [
            "Baseline uses placeholder firm features + noisy synthetic label (non-perfect AUC by design).",
            "Next: replace with real ASX price features + event-based labeling.",
            "ABS features can be merged by GICS industry group (or mapped industry codes).",
            f"Recommended threshold (F1-opt): {threshold:.4f}",
            f"UI baseline export: {str(full_out.as_posix())}",
        ],
        "metrics": {k: metrics[k] for k in ["roc_auc", "pr_auc", "n_train", "n_test"]},
        "threshold": thr_info,
    }
    write_json(ARTIFACTS / "model_card.json", model_card)

    return metrics


if __name__ == "__main__":
    m = train()
    print("Training complete. Metrics:")
    print({k: m[k] for k in ["roc_auc", "pr_auc"]})
    print("Best threshold:", m["threshold"])
    print("Baseline CSV written to: data/processed/baseline_firm_features.csv")