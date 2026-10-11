from pathlib import Path

import pandas as pd


def preprocess_data(
    transaction_path="data/train_transaction.csv",
    identity_path="data/train_identity.csv",
):
    """
    Load and merge IEEE-CIS training data.

    Returns
    -------
    X : pandas.DataFrame
        Features, excluding TransactionID and isFraud.
    y : pandas.Series
        Binary fraud target.

    This function does not fit encoders, scalers, or samplers.
    Those operations must be fitted on training data only.
    """
    transaction_path = Path(transaction_path)
    identity_path = Path(identity_path)

    if not transaction_path.is_file():
        raise FileNotFoundError(
            f"Transaction file not found: {transaction_path}"
        )

    if not identity_path.is_file():
        raise FileNotFoundError(
            f"Identity file not found: {identity_path}"
        )

    print("Loading IEEE-CIS transaction data...")
    transactions = pd.read_csv(transaction_path)

    print("Loading IEEE-CIS identity data...")
    identity = pd.read_csv(identity_path)

    if "TransactionID" not in transactions.columns:
        raise ValueError(
            "Transaction data must contain TransactionID."
        )

    if "TransactionID" not in identity.columns:
        raise ValueError(
            "Identity data must contain TransactionID."
        )

    if "isFraud" not in transactions.columns:
        raise ValueError(
            "Training transaction data must contain isFraud."
        )

    if transactions["TransactionID"].duplicated().any():
        raise ValueError(
            "Duplicate TransactionID values found in transaction data."
        )

    if identity["TransactionID"].duplicated().any():
        raise ValueError(
            "Duplicate TransactionID values found in identity data."
        )

    print("Merging transaction and identity data...")
    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    y = merged["isFraud"].astype("int8").copy()
    X = merged.drop(
        columns=["isFraud", "TransactionID"]
    ).copy()

    if not y.isin([0, 1]).all():
        raise ValueError("isFraud must contain only 0 and 1.")

    print(f"Merged data shape: {merged.shape}")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("Target distribution:")
    print(y.value_counts().sort_index())

    return X, y