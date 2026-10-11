# src/encoder.py

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def encode_features(X_train, X_val):
    """
    Encode categorical features using OrdinalEncoder.

    IMPORTANT:
    - Fit ONLY on training data.
    - Transform validation data.
    - Unknown validation categories become -1.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_val : pd.DataFrame

    Returns
    -------
    X_train_encoded
    X_val_encoded
    encoder
    """

    print("=" * 60)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 60)

    
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Validate feature alignment before encoding.
    if not X_train.columns.is_unique or not X_val.columns.is_unique:
        raise ValueError("Duplicate feature column names detected.")

    if not X_train.columns.equals(X_val.columns):
        raise ValueError(
            "Training and validation feature columns must match "
            "in both names and order."
        )


    # ---------------------------------------------------------
    # FIND CATEGORICAL COLUMNS
    # ---------------------------------------------------------
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(
        f"\nCategorical columns found: "
        f"{len(categorical_cols)}"
    )

    if len(categorical_cols) > 0:

        print("\nExamples:")

        print(
            categorical_cols[:10]
        )

    # ---------------------------------------------------------
    # FIT ENCODER ONLY ON TRAINING DATA
    # ---------------------------------------------------------
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype="float32"
    )

    if len(categorical_cols) > 0:

        X_train[categorical_cols] = (
            encoder.fit_transform(
                X_train[categorical_cols]
            )
        )

        X_val[categorical_cols] = (
            encoder.transform(
                X_val[categorical_cols]
            )
        )

    
    # CONVERT EVERYTHING TO NUMERIC
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")

    # Replace infinite values with NaN, then apply the existing
    # missing-value strategy.
    import numpy as np

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    X_train = X_train.fillna(0).astype("float32")
    X_val = X_val.fillna(0).astype("float32")


    
    print("=" * 60)

    return X_train, X_val, encoder


if __name__ == "__main__":
    from src.preprocessing import preprocess_data
    from sklearn.model_selection import train_test_split

    TRANSACTION_PATH = "data/train_transaction.csv"
    IDENTITY_PATH = "data/train_identity.csv"

    # Load and split the dataset.
    X, y = preprocess_data(
        TRANSACTION_PATH,
        IDENTITY_PATH
    )

    print("\nSplitting Dataset...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.20,
        random_state=42
    )

    print(f"\nTraining Samples: {len(X_train)}")
    print(f"Validation Samples: {len(X_val)}")

    # Encode features.
    X_train_encoded, X_val_encoded, encoder = encode_features(
        X_train,
        X_val
    )

    print("\nPreview:")
    print(X_train_encoded.head())

    print("\nFinal Shapes")
    print("X_train:", X_train_encoded.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val_encoded.shape)
    print("y_val:", y_val.shape)
