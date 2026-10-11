
from pathlib import Path
from datetime import datetime
import json
import shutil

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "train_transaction.csv"
OUTPUT_DIR = ROOT / "reports" / "holdout_evaluation"

TARGET = "isFraud"
ID_COLUMN = "TransactionID"

RANDOM_STATE = 42
PRESELECTED_THRESHOLD = 0.425

N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 100


# ============================================================
# HELPERS
# ============================================================

def backup_existing_outputs():
    """
    Preserve files from a previous holdout run before replacing them.
    The original project model and validation reports are not modified.
    """
    if not OUTPUT_DIR.exists():
        return

    existing_files = list(OUTPUT_DIR.iterdir())

    if not existing_files:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = OUTPUT_DIR.parent / (
        f"holdout_evaluation_previous_run_{timestamp}"
    )

    backup_dir.mkdir(parents=True, exist_ok=False)

    for item in existing_files:
        destination = backup_dir / item.name

        if item.is_file():
            shutil.copy2(item, destination)
        elif item.is_dir():
            shutil.copytree(item, destination)

    print(f"Previous holdout outputs backed up to: {backup_dir}")


def preprocess_data(
    X_dev,
    X_validation,
    X_test,
):
    """
    Fit preprocessing on development data only.
    Apply the same transformations to validation and test data.
    """

    X_dev = X_dev.copy()
    X_validation = X_validation.copy()
    X_test = X_test.copy()

    # Identify categorical columns using development data.
    categorical_columns = X_dev.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_columns = [
        column
        for column in X_dev.columns
        if column not in categorical_columns
    ]

    # Convert categorical values to strings consistently.
    for column in categorical_columns:
        for frame in (X_dev, X_validation, X_test):
            frame[column] = (
                frame[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

    # Fit the categorical encoder on development data only.
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32,
    )

    if categorical_columns:
        X_dev[categorical_columns] = encoder.fit_transform(
            X_dev[categorical_columns]
        )

        X_validation[categorical_columns] = encoder.transform(
            X_validation[categorical_columns]
        )

        X_test[categorical_columns] = encoder.transform(
            X_test[categorical_columns]
        )

    # Convert numeric columns and learn medians from development data.
    numeric_medians = {}

    for column in numeric_columns:
        X_dev[column] = pd.to_numeric(
            X_dev[column], errors="coerce"
        )
        X_validation[column] = pd.to_numeric(
            X_validation[column], errors="coerce"
        )
        X_test[column] = pd.to_numeric(
            X_test[column], errors="coerce"
        )

        median_value = X_dev[column].median()

        # Handle a column that is entirely missing in development data.
        if pd.isna(median_value):
            median_value = 0.0

        numeric_medians[column] = float(median_value)

        X_dev[column] = X_dev[column].fillna(median_value)
        X_validation[column] = X_validation[column].fillna(
            median_value
        )
        X_test[column] = X_test[column].fillna(median_value)

    # Ensure compatible numeric matrices.
    X_dev = X_dev.astype(np.float32)
    X_validation = X_validation.astype(np.float32)
    X_test = X_test.astype(np.float32)

    preprocessing = {
        "encoder": encoder,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "numeric_medians": numeric_medians,
        "feature_columns": X_dev.columns.tolist(),
    }

    return X_dev, X_validation, X_test, preprocessing


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("SHIELD-AI: THREE-WAY HOLDOUT EVALUATION")
    print("=" * 70)

    # Preserve artifacts from the previous run.
    backup_existing_outputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1/7] Loading transaction dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing from the dataset."
        )

    y = pd.to_numeric(df[TARGET], errors="raise").astype(np.int8)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "The target must contain only 0 (legitimate) and 1 (fraud)."
        )

    X = df.drop(
        columns=[TARGET, ID_COLUMN],
        errors="ignore",
    )

    # Retain the source row positions so split membership is reproducible.
    row_indices = np.arange(len(df))

    print(f"Dataset rows: {len(df):,}")
    print(f"Feature columns before preprocessing: {X.shape[1]:,}")
    print(f"Legitimate transactions: {(y == 0).sum():,}")
    print(f"Fraudulent transactions: {(y == 1).sum():,}")
    print(f"Overall fraud rate: {y.mean():.4%}")

    # --------------------------------------------------------
    # 2. STRATIFIED THREE-WAY SPLIT
    #    60% development, 20% validation, 20% untouched test
    # --------------------------------------------------------

    print("\n[2/7] Creating stratified data splits...")

    (
        X_dev_raw,
        X_remaining_raw,
        y_dev,
        y_remaining,
        idx_dev,
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
        X_validation_raw,
        X_test_raw,
        y_validation,
        y_test,
        idx_validation,
        idx_test,
    ) = train_test_split(
        X_remaining_raw,
        y_remaining,
        idx_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    y_dev = y_dev.reset_index(drop=True)
    y_validation = y_validation.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"Development rows: {len(X_dev_raw):,}")
    print(f"Early-stopping validation rows: {len(X_validation_raw):,}")
    print(f"Untouched test rows: {len(X_test_raw):,}")

    print(f"Development fraud rate: {y_dev.mean():.4%}")
    print(f"Validation fraud rate: {y_validation.mean():.4%}")
    print(f"Test fraud rate: {y_test.mean():.4%}")

    # Save split positions so this experiment can be reproduced.
    np.savez_compressed(
        OUTPUT_DIR / "split_indices.npz",
        development_indices=idx_dev,
        validation_indices=idx_validation,
        test_indices=idx_test,
    )

    # --------------------------------------------------------
    # 3. PREPROCESSING
    # --------------------------------------------------------

    print("\n[3/7] Fitting preprocessing on development data only...")

    X_dev, X_validation, X_test, preprocessing = preprocess_data(
        X_dev_raw,
        X_validation_raw,
        X_test_raw,
    )

    print(f"Final feature count: {X_dev.shape[1]:,}")
    print(
        "Categorical features encoded: "
        f"{len(preprocessing['categorical_columns']):,}"
    )
    print(
        "Numeric features: "
        f"{len(preprocessing['numeric_columns']):,}"
    )

    # --------------------------------------------------------
    # 4. TRAIN MODEL
    # --------------------------------------------------------

    print("\n[4/7] Training LightGBM...")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        num_leaves=64,
        scale_pos_weight=10,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_dev,
        y_dev,
        eval_set=[
            (X_validation, y_validation),
        ],
        eval_metric=["auc", "binary_logloss"],
        callbacks=[
            # IMPORTANT: monitor AUC only.
            early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=True,
            ),
            log_evaluation(period=100),
        ],
    )

    print(f"\nBest iteration: {model.best_iteration_}")

    # Check that the recorded best iteration agrees with AUC history.
    auc_history = model.evals_result_["valid_0"]["auc"]
    best_auc_iteration = int(np.argmax(auc_history)) + 1
    best_auc_value = float(np.max(auc_history))

    print(f"Recorded AUC iterations: {len(auc_history)}")
    print(f"Highest validation AUC: {best_auc_value:.6f}")
    print(f"Iteration with highest recorded AUC: {best_auc_iteration}")

    if model.best_iteration_ != best_auc_iteration:
        print(
            "WARNING: best_iteration_ does not match the maximum "
            "recorded AUC. Inspect the LightGBM evaluation history."
        )
    else:
        print("AUC early-stopping consistency check: PASSED")

    # Save the trained model and preprocessing separately.
    joblib.dump(
        model,
        OUTPUT_DIR / "holdout_model.pkl",
    )

    joblib.dump(
        preprocessing,
        OUTPUT_DIR / "preprocessing.pkl",
    )

    # --------------------------------------------------------
    # 5. EVALUATE ON THE UNTOUCHED TEST SET
    # --------------------------------------------------------

    print("\n[5/7] Evaluating on the untouched test set...")

    # Do not tune the threshold using the test labels.
    y_probability = model.predict_proba(X_test)[:, 1]

    y_prediction = (
        y_probability >= PRESELECTED_THRESHOLD
    ).astype(np.int8)

    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(
        y_test, y_prediction, zero_division=0
    )
    recall = recall_score(
        y_test, y_prediction, zero_division=0
    )
    f1 = f1_score(
        y_test, y_prediction, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_probability)
    average_precision = average_precision_score(
        y_test, y_probability
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_prediction,
        labels=[0, 1],
    ).ravel()

    report = classification_report(
        y_test,
        y_prediction,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        digits=6,
        zero_division=0,
    )

    metrics = {
        "experiment": "three_way_stratified_holdout",
        "random_state": RANDOM_STATE,
        "development_rows": int(len(X_dev)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_dev.shape[1]),
        "best_iteration": int(model.best_iteration_),
        "best_validation_auc": best_auc_value,
        "best_auc_iteration": best_auc_iteration,
        "threshold": PRESELECTED_THRESHOLD,
        "threshold_source": (
            "Preselected threshold from the separate original model; "
            "not tuned on this holdout test set."
        ),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "test_fraud_rate": float(y_test.mean()),
        "probability_min": float(np.min(y_probability)),
        "probability_median": float(np.median(y_probability)),
        "probability_95th_percentile": float(
            np.percentile(y_probability, 95)
        ),
        "probability_max": float(np.max(y_probability)),
    }

    print("\n" + "=" * 70)
    print("UNTOUCHED HOLDOUT TEST RESULTS")
    print("=" * 70)

    for name in [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]:
        print(f"{name}: {metrics[name]}")

    print("\nClassification report:")
    print(report)

    print("Prediction probability distribution:")
    print(f"Minimum: {metrics['probability_min']:.6f}")
    print(f"Median: {metrics['probability_median']:.6f}")
    print(
        "95th percentile: "
        f"{metrics['probability_95th_percentile']:.6f}"
    )
    print(f"Maximum: {metrics['probability_max']:.6f}")

    # --------------------------------------------------------
    # 6. SAVE RESULTS
    # --------------------------------------------------------

    print("\n[6/7] Saving evaluation results...")

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("SHIELD-AI Holdout Evaluation\n")
        file.write("=" * 50 + "\n\n")

        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

        file.write("\nClassification report:\n")
        file.write(report)

    np.savez_compressed(
        OUTPUT_DIR / "holdout_scores.npz",
        y_true=y_test.to_numpy(),
        y_probability=y_probability,
        y_prediction=y_prediction,
        threshold=np.array(PRESELECTED_THRESHOLD),
    )

    # --------------------------------------------------------
    # 7. FINISH
    # --------------------------------------------------------

    print("\n[7/7] Evaluation complete.")

    print(f"\nResults directory: {OUTPUT_DIR}")
    print(f"Model: {OUTPUT_DIR / 'holdout_model.pkl'}")
    print(f"Preprocessing: {OUTPUT_DIR / 'preprocessing.pkl'}")
    print(f"Metrics: {OUTPUT_DIR / 'metrics.json'}")
    print(
        f"Classification report: "
        f"{OUTPUT_DIR / 'classification_report.txt'}"
    )
    print(f"Predictions: {OUTPUT_DIR / 'holdout_scores.npz'}")
    print(f"Split indices: {OUTPUT_DIR / 'split_indices.npz'}")

    print(
        "\nNote: This is a separate experiment using "
        "train_transaction.csv only. Its results must not be "
        "presented as the performance of the original merged-data model."
    )


if __name__ == "__main__":
    main()
