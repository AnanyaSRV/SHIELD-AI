
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train_transaction.csv"
OUTPUT_DIR = ROOT / "reports" / "holdout_evaluation"

TARGET = "isFraud"
ID_COLUMN = "TransactionID"
RANDOM_STATE = 42
MIN_RECALL = 0.80


def main():
    print("Loading dataset and saved experiment artifacts...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    y = df[TARGET].astype(np.int8).reset_index(drop=True)
    X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")

    # Reconstruct exactly the same stratified split.
    from sklearn.model_selection import train_test_split

    row_indices = np.arange(len(df))

    (
        _X_dev,
        X_remaining,
        _y_dev,
        y_remaining,
        _idx_dev,
        idx_remaining,
    ) = train_test_split(
        X,
        y,
        row_indices,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    (
        X_validation,
        _X_test,
        y_validation,
        _y_test,
        _idx_validation,
        _idx_test,
    ) = train_test_split(
        X_remaining,
        y_remaining,
        idx_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    # Load preprocessing and trained model.
    model = joblib.load(OUTPUT_DIR / "holdout_model.pkl")
    preprocessing = joblib.load(OUTPUT_DIR / "preprocessing.pkl")

    X_validation = X_validation.copy()

    categorical_columns = preprocessing["categorical_columns"]
    numeric_columns = preprocessing["numeric_columns"]
    encoder = preprocessing["encoder"]
    medians = preprocessing["numeric_medians"]
    feature_columns = preprocessing["feature_columns"]

    # Apply the same transformations used during training.
    for column in categorical_columns:
        X_validation[column] = (
            X_validation[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    if categorical_columns:
        X_validation[categorical_columns] = encoder.transform(
            X_validation[categorical_columns]
        )

    for column in numeric_columns:
        X_validation[column] = pd.to_numeric(
            X_validation[column], errors="coerce"
        )
        X_validation[column] = X_validation[column].fillna(
            medians[column]
        )

    X_validation = X_validation[feature_columns].astype(np.float32)
    y_validation = y_validation.reset_index(drop=True).astype(np.int8)

    print(f"Validation rows: {len(y_validation):,}")
    print(f"Validation fraud rate: {y_validation.mean():.4%}")

    # Threshold selection uses validation data only.
    probabilities = model.predict_proba(X_validation)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        probabilities,
    )

    # The final precision/recall point has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = np.flatnonzero(recall >= MIN_RECALL)

    if len(eligible) == 0:
        raise RuntimeError(
            f"No threshold achieves recall >= {MIN_RECALL:.0%}."
        )

    # Choose the highest threshold that meets the recall requirement.
    best_index = eligible[np.argmax(thresholds[eligible])]
    threshold = float(thresholds[best_index])

    predictions = (probabilities >= threshold).astype(np.int8)

    final_precision = precision_score(
        y_validation, predictions, zero_division=0
    )
    final_recall = recall_score(
        y_validation, predictions, zero_division=0
    )
    final_f1 = f1_score(
        y_validation, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_validation, predictions, labels=[0, 1]
    ).ravel()

    results = {
        "threshold": threshold,
        "minimum_required_recall": MIN_RECALL,
        "validation_precision": float(final_precision),
        "validation_recall": float(final_recall),
        "validation_f1": float(final_f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "validation_rows": int(len(y_validation)),
        "selection_data": "early-stopping validation set only",
    }

    print("\nVALIDATION THRESHOLD RESULTS")
    print("=" * 45)

    for key, value in results.items():
        print(f"{key}: {value}")

    output_path = OUTPUT_DIR / "selected_threshold.json"

    import json
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nSaved threshold selection to: {output_path}")
    print(
        "\nThe untouched test set was not used to choose this threshold."
    )


if __name__ == "__main__":
    main()
