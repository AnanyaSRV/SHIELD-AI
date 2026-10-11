# src/balancer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTEENN

from src.preprocessing import preprocess_data
from src.encoder import encode_features


def split_encode_and_oversample(
    master_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    sampling_strategy: float = 0.3
):
    """
    Leakage-free pipeline:

    Master Data
        ↓
    Stratified Split
        ↓
    Encode Train/Validation
        ↓
    SMOTE on TRAIN ONLY
        ↓
    Return balanced train + untouched validation
    """

    print("=" * 60)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 60)

    # -------------------------------------------------
    # FEATURES AND TARGET
    # -------------------------------------------------
    X = master_df.drop(columns=["isFraud"])
    y = master_df["isFraud"]

    print("\nOriginal Dataset")
    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")

    print("\nOriginal Class Distribution:")
    print(y.value_counts())

    # -------------------------------------------------
    # SPLIT
    # -------------------------------------------------
    print("\nPerforming Stratified Split...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print("\nTraining Set:")
    print(X_train.shape)
    print(y_train.value_counts())

    print("\nValidation Set:")
    print(X_val.shape)
    print(y_val.value_counts())

    # -------------------------------------------------
    # ENCODING
    # -------------------------------------------------
    print("\nEncoding Features...")

    X_train_encoded, X_val_encoded, encoder = encode_features(
        X_train,
        X_val
    )

    # -------------------------------------------------
    # SMOTE
    # -------------------------------------------------
    print("\nApplying SMOTE (TRAIN ONLY)...")

    smote = SMOTEENN(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5
    )

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_encoded,
        y_train
    )

    print("\nBalanced Training Set:")

    print(X_train_balanced.shape)

    print(
        pd.Series(y_train_balanced).value_counts()
    )

    print("\nValidation Set Remains Untouched:")

    print(X_val_encoded.shape)

    print(y_val.value_counts())

    print("=" * 60)

    return (
        X_train_balanced,
        X_val_encoded,
        y_train_balanced,
        y_val,
        encoder
    )


if __name__ == "__main__":

    TRANSACTION_PATH = "data/train_transaction.csv"
    IDENTITY_PATH = "data/train_identity.csv"

    master_df = preprocess_data(
        TRANSACTION_PATH,
        IDENTITY_PATH
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        encoder
    ) = split_encode_and_oversample(
        master_df,
        sampling_strategy=0.3
    )

    print("\nFinal Shapes")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)