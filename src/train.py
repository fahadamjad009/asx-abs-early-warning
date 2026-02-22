from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
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

    # Handle imbalance after preprocessing via SMOTE
    model = ImbPipeline(
        steps=[
            ("pre", pre),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", stack),
        ]
    )

    # Calibrate for probability quality
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    # Evaluate
    proba = calibrated.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, output_dict=True
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
        "ui_export_csv": str(full_out.as_posix()),
    }

    # Save artifacts
    dump(calibrated, ARTIFACTS / "model.joblib")
    write_json(ARTIFACTS / "metrics.json", metrics)

    # Plot
    plt.figure()
    plt.hist(proba, bins=30)
    plt.title("Predicted risk probability (test set)")
    plt.xlabel("p(distress)")
    plt.ylabel("count")
    plt.tight_layout()
    fig_path = FIGURES / "probability_hist.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    model_card = {
        "problem": "ASX/ABS early warning (baseline demo scaffold)",
        "label": "target_distress_12m (placeholder rule: drawdown_12m < -0.35)",
        "notes": [
            "This baseline uses placeholder firm features + a placeholder label rule.",
            "Next: replace with real ASX price features and event-based labeling.",
            "ABS features can be merged by GICS industry group (or mapped industry codes).",
            f"UI baseline export: {str(full_out.as_posix())}",
        ],
        "metrics": {k: metrics[k] for k in ["roc_auc", "pr_auc", "n_train", "n_test"]},
    }
    write_json(ARTIFACTS / "model_card.json", model_card)

    return metrics


if __name__ == "__main__":
    m = train()
    print("Training complete. Metrics:")
    print({k: m[k] for k in ["roc_auc", "pr_auc"]})
    print("Baseline CSV written to: data/processed/baseline_firm_features.csv")