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

    # ----------------------------------------
    # Find categorical columns
    # ----------------------------------------
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(
        f"\nCategorical columns found: "
        f"{len(categorical_cols)}"
    )

    if len(categorical_cols) > 0:
        print(
            "\nExamples:"
        )

        print(
            categorical_cols[:10]
        )

    # ----------------------------------------
    # Fit encoder ONLY on training set
    # ----------------------------------------
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    if len(categorical_cols) > 0:

        X_train[categorical_cols] = encoder.fit_transform(
            X_train[categorical_cols]
        )

        X_val[categorical_cols] = encoder.transform(
            X_val[categorical_cols]
        )

    # ----------------------------------------
    # Convert all columns to numeric
    # ----------------------------------------
    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_val = X_val.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ----------------------------------------
    # Fill any remaining NaNs
    # ----------------------------------------
    X_train = X_train.fillna(0)

    X_val = X_val.fillna(0)

    print("\nEncoding complete.")

    print(
        f"\nX_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"X_val shape: "
        f"{X_val.shape}"
    )

    print("=" * 60)

    return (
        X_train,
        X_val,
        encoder
    )


if __name__ == "__main__":

    from preprocessing import preprocess_data
    from sklearn.model_selection import train_test_split

    master_df = preprocess_data(
        "data/train_transaction.csv",
        "data/train_identity.csv"
    )

    X = master_df.drop(
        columns=["isFraud"]
    )

    y = master_df["isFraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    (
        X_train_encoded,
        X_val_encoded,
        encoder
    ) = encode_features(
        X_train,
        X_val
    )

    print("\nPreview:")

    print(
        X_train_encoded.head()
    )