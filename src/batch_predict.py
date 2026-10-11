
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from risk_agent import RiskAssessmentAgent


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data" / "train_transaction.csv"

DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "full_dataset_predictions.csv"
)

DEFAULT_BATCH_SIZE = 10_000


def predict_csv(
    input_path=DEFAULT_INPUT,
    output_path=DEFAULT_OUTPUT,
    batch_size=DEFAULT_BATCH_SIZE,
):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if input_path == output_path:
        raise ValueError(
            "Output path cannot overwrite the input dataset."
        )

    print("=" * 65)
    print("SHIELD-AI: FULL DATASET BATCH PREDICTION")
    print("=" * 65)
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Batch size: {batch_size}")

    agent = RiskAssessmentAgent()

    feature_columns = agent.feature_columns
    threshold = agent.threshold

    # Inspect the CSV header without loading its rows.
    available_columns = pd.read_csv(
        input_path,
        nrows=0,
    ).columns.tolist()

    missing_columns = (
        set(feature_columns) - set(available_columns)
    )

    if missing_columns:
        raise ValueError(
            "Input dataset is missing model features: "
            f"{sorted(missing_columns)[:20]}"
        )

    usecols = list(
        dict.fromkeys(
            ["TransactionID", *feature_columns]
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_rows = 0
    fraud_predictions = 0
    legitimate_predictions = 0
    first_write = True

    # Start a fresh output file for this run.
    if output_path.exists():
        output_path.unlink()

    try:
        reader = pd.read_csv(
            input_path,
            usecols=usecols,
            chunksize=batch_size,
            low_memory=False,
        )

        for batch_number, chunk in enumerate(
            reader,
            start=1,
        ):
            if "TransactionID" in chunk.columns:
                transaction_ids = (
                    chunk["TransactionID"].reset_index(drop=True)
                )
            else:
                transaction_ids = pd.Series(
                    range(
                        total_rows,
                        total_rows + len(chunk),
                    ),
                    name="TransactionID",
                )

            features = chunk.drop(
                columns=["TransactionID"],
                errors="ignore",
            )

            processed = agent._preprocess_batch(features)

            probabilities = agent.model.predict_proba(
                processed
            )[:, 1]

            if not np.isfinite(probabilities).all():
                raise ValueError(
                    f"Invalid probabilities in batch {batch_number}."
                )

            predictions = probabilities >= threshold

            output = pd.DataFrame(
                {
                    "TransactionID": transaction_ids,
                    "fraud_probability": probabilities,
                    "model_prediction": np.where(
                        predictions,
                        "Fraud",
                        "Legitimate",
                    ),
                    "classification_threshold": threshold,
                }
            )

            output.to_csv(
                output_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False

            rows_in_batch = len(chunk)
            total_rows += rows_in_batch

            fraud_predictions += int(predictions.sum())
            legitimate_predictions += int(
                (~predictions).sum()
            )

            print(
                f"Batch {batch_number}: "
                f"{rows_in_batch:,} rows processed | "
                f"Total: {total_rows:,} | "
                f"Fraud predictions: {fraud_predictions:,}"
            )

        if total_rows == 0:
            raise ValueError("The input dataset contains no rows.")

    except Exception:
        print(
            "\nPrediction failed. The output file may be incomplete: "
            f"{output_path}"
        )
        raise

    print("\n" + "=" * 65)
    print("BATCH PREDICTION COMPLETED")
    print("=" * 65)
    print(f"Total transactions:     {total_rows:,}")
    print(f"Fraud predictions:      {fraud_predictions:,}")
    print(f"Legitimate predictions: {legitimate_predictions:,}")
    print(f"Threshold:              {threshold:.15f}")
    print(f"Output file:            {output_path}")
    print("=" * 65)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run SHIELD-AI predictions on a CSV dataset in batches."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV path.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output predictions CSV path.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of transactions processed per batch.",
    )

    args = parser.parse_args()

    predict_csv(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
