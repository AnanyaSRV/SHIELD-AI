
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "lightgbm_tuning"
HOLDOUT_DIR = ROOT / "reports" / "holdout_evaluation"

THRESHOLD = 0.371294741468585


def main():
    print("Loading saved tuned model and test split...")

    model = joblib.load(OUT / "best_validation_model.pkl")
    preprocessing = joblib.load(OUT / "preprocessing.pkl")

    test_data = np.load(HOLDOUT_DIR / "split_indices.npz")
    test_indices = test_data["test_indices"]

    # Read only the target labels and rows belonging to the saved test split.
    # No training, threshold selection, or model selection occurs here.
    df = pd.read_csv(
        ROOT / "data" / "train_transaction.csv",
        low_memory=False,
    )

    X_test = df.iloc[test_indices].drop(
        columns=["isFraud", "TransactionID"],
        errors="ignore",
    ).copy()

    y_test = (
        df.iloc[test_indices]["isFraud"]
        .astype(np.int8)
        .to_numpy()
    )

    cat_cols = preprocessing["categorical_columns"]
    num_cols = preprocessing["numeric_columns"]
    encoder = preprocessing["encoder"]
    medians = preprocessing["numeric_medians"]
    feature_cols = preprocessing["feature_columns"]

    for col in cat_cols:
        X_test[col] = (
            X_test[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    if cat_cols:
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    for col in num_cols:
        X_test[col] = pd.to_numeric(
            X_test[col], errors="coerce"
        )
        X_test[col] = X_test[col].fillna(medians[col])

    X_test = X_test[feature_cols].astype(np.float32)

    print(f"Test rows: {len(y_test):,}")
    print(f"Fraud rate: {y_test.mean():.4%}")
    print(f"Threshold fixed from validation: {THRESHOLD:.8f}")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    results = {
        "model": "tuned_baseline_lightgbm",
        "threshold": THRESHOLD,
        "test_rows": int(len(y_test)),
        "test_fraud_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print("\nTUNED MODEL — UNTOUCHED TEST RESULTS")
    print("=" * 50)

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            digits=5,
            zero_division=0,
        )
    )

    with open(
        OUT / "final_holdout_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=4)

    np.savez_compressed(
        OUT / "final_holdout_predictions.npz",
        y_true=y_test,
        y_probability=probabilities,
        y_prediction=predictions,
        threshold=np.array(THRESHOLD),
    )

    print(f"\nResults saved to: {OUT}")
    print("Evaluation complete. No test-set tuning was performed.")


if __name__ == "__main__":
    main()
