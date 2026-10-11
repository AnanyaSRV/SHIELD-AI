
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from lightgbm import LGBMClassifier, early_stopping, log_evaluation


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train_transaction.csv"
OUT = ROOT / "reports" / "lightgbm_tuning"

RANDOM_STATE = 42
MIN_RECALL = 0.80


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    y = df["isFraud"].astype(np.int8)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    # Recreate the existing 60/20/20 stratified split.
    X_dev, X_remaining, y_dev, y_remaining = train_test_split(
        X, y, test_size=0.40,
        random_state=RANDOM_STATE, stratify=y
    )

    X_val, _, y_val, _ = train_test_split(
        X_remaining, y_remaining, test_size=0.50,
        random_state=RANDOM_STATE, stratify=y_remaining
    )

    y_dev = y_dev.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    X_dev = X_dev.copy()
    X_val = X_val.copy()

    cat_cols = X_dev.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    num_cols = [
        c for c in X_dev.columns if c not in cat_cols
    ]

    # Fit preprocessing on development data only.
    for c in cat_cols:
        for frame in (X_dev, X_val):
            frame[c] = (
                frame[c].astype("string")
                .fillna("__MISSING__").astype(str)
            )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32,
    )

    if cat_cols:
        X_dev[cat_cols] = encoder.fit_transform(X_dev[cat_cols])
        X_val[cat_cols] = encoder.transform(X_val[cat_cols])

    medians = {}

    for c in num_cols:
        X_dev[c] = pd.to_numeric(X_dev[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")

        median = X_dev[c].median()
        if pd.isna(median):
            median = 0.0

        medians[c] = float(median)
        X_dev[c] = X_dev[c].fillna(median)
        X_val[c] = X_val[c].fillna(median)

    X_dev = X_dev.astype(np.float32)
    X_val = X_val[X_dev.columns].astype(np.float32)

    # Three configurations; same split and same validation metric.
    configurations = [
        {
            "name": "baseline",
            "num_leaves": 64,
            "min_child_samples": 20,
            "scale_pos_weight": 10,
            "colsample_bytree": 0.8,
            "reg_lambda": 0.0,
        },
        {
            "name": "regularized",
            "num_leaves": 31,
            "min_child_samples": 50,
            "scale_pos_weight": 10,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        {
            "name": "higher_class_weight",
            "num_leaves": 31,
            "min_child_samples": 50,
            "scale_pos_weight": 15,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        },
    ]

    results = []
    best_ap = -1.0
    best_model = None
    best_config = None
    best_probabilities = None

    for config in configurations:
        name = config["name"]
        params = {k: v for k, v in config.items() if k != "name"}

        print(f"\nTraining configuration: {name}")

        model = LGBMClassifier(
            objective="binary",
            n_estimators=1500,
            learning_rate=0.03,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
            subsample=0.8,
            subsample_freq=1,
            **params,
        )

        model.fit(
            X_dev,
            y_dev,
            eval_set=[(X_val, y_val)],
            eval_metric=["average_precision", "auc"],
            callbacks=[
                early_stopping(
                    stopping_rounds=100,
                    first_metric_only=True,
                    verbose=False,
                ),
                log_evaluation(period=0),
            ],
        )

        probabilities = model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, probabilities)
        auc = roc_auc_score(y_val, probabilities)

        # Select the highest threshold meeting the recall target.
        precision, recall, thresholds = precision_recall_curve(
            y_val, probabilities
        )
        precision = precision[:-1]
        recall = recall[:-1]

        eligible = np.flatnonzero(recall >= MIN_RECALL)

        if len(eligible):
            idx = eligible[np.argmax(thresholds[eligible])]
            threshold = float(thresholds[idx])
            predictions = (probabilities >= threshold).astype(int)

            p = precision_score(
                y_val, predictions, zero_division=0
            )
            r = recall_score(
                y_val, predictions, zero_division=0
            )
            f1 = f1_score(
                y_val, predictions, zero_division=0
            )
        else:
            threshold = None
            p = r = f1 = None

        result = {
            "name": name,
            "parameters": params,
            "best_iteration": int(model.best_iteration_),
            "validation_average_precision": float(ap),
            "validation_roc_auc": float(auc),
            "threshold_for_target_recall": threshold,
            "validation_precision_at_threshold": p,
            "validation_recall_at_threshold": r,
            "validation_f1_at_threshold": f1,
        }
        results.append(result)

        print(f"Best iteration: {model.best_iteration_}")
        print(f"Validation average precision: {ap:.6f}")
        print(f"Validation ROC-AUC: {auc:.6f}")
        print(f"Threshold for >=80% recall: {threshold}")
        print(f"Precision at threshold: {p}")
        print(f"Recall at threshold: {r}")
        print(f"F1 at threshold: {f1}")

        # Primary selection criterion: validation average precision.
        if ap > best_ap:
            best_ap = ap
            best_model = model
            best_config = result
            best_probabilities = probabilities.copy()

    # Save tuning outputs separately from the existing holdout experiment.
    with open(OUT / "tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    joblib.dump(best_model, OUT / "best_validation_model.pkl")

    joblib.dump(
        {
            "encoder": encoder,
            "categorical_columns": cat_cols,
            "numeric_columns": num_cols,
            "numeric_medians": medians,
            "feature_columns": X_dev.columns.tolist(),
        },
        OUT / "preprocessing.pkl",
    )

    np.savez_compressed(
        OUT / "validation_predictions.npz",
        y_true=y_val.to_numpy(),
        y_probability=best_probabilities,
    )

    with open(OUT / "best_candidate.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4)

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Selected configuration: {best_config['name']}")
    print(f"Validation average precision: {best_ap:.6f}")
    print(f"Outputs saved to: {OUT}")
    print(
        "The untouched test set was not loaded or evaluated."
    )


if __name__ == "__main__":
    main()
