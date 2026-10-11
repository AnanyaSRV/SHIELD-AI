
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

INPUT_PATH = ROOT / "data" / "train_transaction.csv"
MODEL_PATH = (
    ROOT / "reports" / "lightgbm_tuning" / "best_validation_model.pkl"
)
PREPROCESSING_PATH = (
    ROOT / "reports" / "lightgbm_tuning" / "preprocessing.pkl"
)
OUTPUT_DIR = ROOT / "reports" / "lightgbm_tuning"

CHUNK_SIZE = 25_000
MONITOR_THRESHOLD = 0.60
BLOCK_THRESHOLD = 0.80


def preprocess_chunk(chunk, preprocessing):
    feature_columns = preprocessing["feature_columns"]
    categorical_columns = preprocessing["categorical_columns"]
    numeric_columns = preprocessing["numeric_columns"]
    numeric_medians = preprocessing["numeric_medians"]
    encoder = preprocessing["encoder"]

    features = chunk.drop(
        columns=["TransactionID", "isFraud"],
        errors="ignore",
    ).copy()

    missing = set(feature_columns) - set(features.columns)
    extra = set(features.columns) - set(feature_columns)

    if missing or extra:
        raise ValueError(
            "Feature schema mismatch.\n"
            f"Missing features: {sorted(missing)[:10]}\n"
            f"Unexpected features: {sorted(extra)[:10]}"
        )

    features = features[feature_columns].copy()

    for column in categorical_columns:
        features[column] = (
            features[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    if categorical_columns:
        features[categorical_columns] = encoder.transform(
            features[categorical_columns]
        )

    for column in numeric_columns:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )
        features[column] = features[column].fillna(
            numeric_medians[column]
        )

    features = features[feature_columns].astype(np.float32)

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(
            "Preprocessed chunk contains infinite or invalid values."
        )

    return features


def main():
    for path in (INPUT_PATH, MODEL_PATH, PREPROCESSING_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_output = (
        OUTPUT_DIR / "full_dataset_transaction_decisions.csv"
    )
    block_output = (
        OUTPUT_DIR / "full_dataset_block_transactions.csv"
    )

    # Remove old output files so reruns do not append duplicate results.
    for path in (all_output, block_output):
        if path.exists():
            path.unlink()

    print("Loading saved LightGBM model...")
    model = joblib.load(MODEL_PATH)
    preprocessing = joblib.load(PREPROCESSING_PATH)

    expected_features = preprocessing["feature_columns"]
    if len(expected_features) != 392:
        raise ValueError(
            f"Expected 392 features, found {len(expected_features)}."
        )

    print(f"Input dataset: {INPUT_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print(f"BLOCK threshold: {BLOCK_THRESHOLD:.2f}")
    print("Processing the full dataset...\n")

    total_rows = 0
    total_block = 0
    total_monitor = 0
    total_allow = 0
    total_actual_fraud = 0
    actual_fraud_available = True
    first_chunk = True

    for chunk in pd.read_csv(
        INPUT_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        rows_in_chunk = len(chunk)

        # Keep the CSV record position, excluding the header.
        data_row_numbers = (
            np.arange(rows_in_chunk, dtype=np.int64) + total_rows + 1
        )
        csv_line_numbers = data_row_numbers + 1

        transaction_ids = (
            chunk["TransactionID"].to_numpy()
            if "TransactionID" in chunk.columns
            else np.full(rows_in_chunk, None, dtype=object)
        )

        actual_labels = (
            chunk["isFraud"].to_numpy()
            if "isFraud" in chunk.columns
            else None
        )

        if actual_labels is None:
            actual_fraud_available = False
        else:
            total_actual_fraud += int(
                pd.to_numeric(
                    pd.Series(actual_labels), errors="coerce"
                ).fillna(0).sum()
            )

        features = preprocess_chunk(chunk, preprocessing)
        probabilities = model.predict_proba(features)[:, 1]

        decisions = np.where(
            probabilities >= BLOCK_THRESHOLD,
            "BLOCK",
            np.where(
                probabilities >= MONITOR_THRESHOLD,
                "MONITOR",
                "ALLOW",
            ),
        )

        output = pd.DataFrame(
            {
                "data_row_number": data_row_numbers,
                "csv_line_number": csv_line_numbers,
                "TransactionID": transaction_ids,
                "fraud_probability": probabilities,
                "decision": decisions,
            }
        )

        # Preserve known labels for later analysis, but never use them
        # as model input.
        if actual_labels is not None:
            output["actual_isFraud"] = actual_labels

        output.to_csv(
            all_output,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        blocked = output[output["decision"] == "BLOCK"]
        if not blocked.empty:
            blocked.to_csv(
                block_output,
                mode="w" if total_block == 0 else "a",
                header=(total_block == 0),
                index=False,
            )
            total_block += len(blocked)

        total_monitor += int((decisions == "MONITOR").sum())
        total_allow += int((decisions == "ALLOW").sum())
        total_rows += rows_in_chunk
        first_chunk = False

        print(
            f"Processed {total_rows:,} rows | "
            f"BLOCK: {total_block:,} | "
            f"MONITOR: {total_monitor:,} | "
            f"ALLOW: {total_allow:,}"
        )

    print("\n" + "=" * 65)
    print("FULL DATASET SCORING COMPLETE")
    print("=" * 65)
    print(f"Total transactions scored: {total_rows:,}")
    print(f"BLOCK:   {total_block:,}")
    print(f"MONITOR: {total_monitor:,}")
    print(f"ALLOW:   {total_allow:,}")

    if actual_fraud_available:
        print(f"Known fraud labels in source: {total_actual_fraud:,}")
    else:
        print("Actual fraud labels were not available in the input.")

    if total_block == 0:
        print("\nNo transactions met the BLOCK threshold.")
        print("No block-only CSV was created.")
    else:
        print(f"\nBLOCK transactions saved to:\n{block_output}")

    print(f"\nAll decisions saved to:\n{all_output}")
    print(
        "\nReminder: BLOCK is a model-policy decision, not proof of fraud."
    )


if __name__ == "__main__":
    main()
