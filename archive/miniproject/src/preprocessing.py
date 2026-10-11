# src/preprocessing.py

import gc
import pandas as pd
import numpy as np


def preprocess_data(
    transaction_path: str,
    identity_path: str,
):
    """
    IEEE-CIS preprocessing pipeline.

    Steps:
    1. Load train_transaction.csv
    2. Load train_identity.csv
    3. Merge using TransactionID
    4. Separate target (isFraud)
    5. Remove leakage-prone identifiers
    6. Basic missing-value handling
    7. Return clean master dataframe

    NOTE:
    - NO train/validation split here
    - NO SMOTE here
    - NO feature engineering here
    """

    print("=" * 60)
    print("LOADING IEEE-CIS DATASET")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD TRANSACTION DATA
    # ---------------------------------------------------------
    print("Reading transaction dataset...")

    transaction_df = pd.read_csv(
        transaction_path
    )

    print(
        f"Transaction shape: {transaction_df.shape}"
    )

    # ---------------------------------------------------------
    # LOAD IDENTITY DATA
    # ---------------------------------------------------------
    print("\nReading identity dataset...")

    identity_df = pd.read_csv(
        identity_path
    )

    print(
        f"Identity shape: {identity_df.shape}"
    )

    # ---------------------------------------------------------
    # MERGE
    # ---------------------------------------------------------
    print("\nMerging datasets...")

    df = pd.merge(
        transaction_df,
        identity_df,
        on="TransactionID",
        how="left"
    )

    del transaction_df
    del identity_df
    gc.collect()

    print(
        f"Master dataset shape: {df.shape}"
    )

    # ---------------------------------------------------------
    # CHECK TARGET
    # ---------------------------------------------------------
    if "isFraud" not in df.columns:
        raise ValueError(
            "Target column 'isFraud' not found."
        )

    print("\nClass Distribution:")

    counts = (
        df["isFraud"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df["isFraud"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        f"Legitimate (0): "
        f"{counts.get(0, 0)} "
        f"({percentages.get(0, 0):.2f}%)"
    )

    print(
        f"Fraud (1): "
        f"{counts.get(1, 0)} "
        f"({percentages.get(1, 0):.2f}%)"
    )

    # ---------------------------------------------------------
    # DROP HIGH-RISK IDENTIFIERS
    # ---------------------------------------------------------
    # These identifiers can lead to memorization.
    # TransactionID should never be used.

    drop_cols = [
        "TransactionID"
    ]

    existing_drop_cols = [
        col
        for col in drop_cols
        if col in df.columns
    ]

    df.drop(
        columns=existing_drop_cols,
        inplace=True
    )

    print(
        f"\nDropped columns: "
        f"{existing_drop_cols}"
    )

    # ---------------------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------------------
    print("\nHandling missing values...")

    for col in df.columns:

        if col == "isFraud":
            continue

        if (
            df[col].dtype == "object"
            or str(df[col].dtype)
            == "category"
        ):

            df[col] = (
                df[col]
                .fillna("Unknown")
            )

        else:

            median_value = (
                df[col]
                .median()
            )

            df[col] = (
                df[col]
                .fillna(median_value)
            )

    # ---------------------------------------------------------
    # REMOVE CONSTANT COLUMNS
    # ---------------------------------------------------------
    constant_cols = [
        col
        for col in df.columns
        if col != "isFraud"
        and df[col].nunique()
        <= 1
    ]

    if constant_cols:

        print(
            f"\nRemoving "
            f"{len(constant_cols)} "
            f"constant columns..."
        )

        df.drop(
            columns=constant_cols,
            inplace=True
        )

    # ---------------------------------------------------------
    # FINAL CLEANUP
    # ---------------------------------------------------------
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    for col in df.columns:

        if col == "isFraud":
            continue

        if (
            df[col].dtype != "object"
        ):

            df[col] = (
                df[col]
                .fillna(
                    df[col].median()
                )
            )

    print("\nPreprocessing complete.")

    print(
        f"Final dataset shape: "
        f"{df.shape}"
    )

    print("=" * 60)

    return df


if __name__ == "__main__":

    TRANSACTION_PATH = (
        "data/train_transaction.csv"
    )

    IDENTITY_PATH = (
        "data/train_identity.csv"
    )

    master_df = preprocess_data(
        TRANSACTION_PATH,
        IDENTITY_PATH
    )

    print("\nPreview:")

    print(master_df.head())

    print(
        "\nColumns:",
        len(master_df.columns)
    )