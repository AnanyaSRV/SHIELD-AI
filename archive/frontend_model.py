# src/frontend_model.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from preprocessing import preprocess_data


def train_frontend_model():
    print("=" * 60)
    print("TRAINING JAVA FRONTEND MODEL")
    print("=" * 60)

    # Load IEEE-CIS data
    df = preprocess_data(
        transaction_path="data/train_transaction.csv",
        identity_path="data/train_identity.csv"
    )

    # Use only fields collected by Java GUI
    required_cols = [
        "card1",           # sender card
        "addr1",           # sender region
        "TransactionAmt",  # amount
        "isFraud"
    ]

    # Check columns exist
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise Exception(f"Missing columns: {missing}")

    df = df[required_cols].copy()

    # Fill missing values
    df["card1"] = df["card1"].fillna(df["card1"].median())
    df["addr1"] = df["addr1"].fillna(df["addr1"].median())
    df["TransactionAmt"] = df["TransactionAmt"].fillna(
        df["TransactionAmt"].median()
    )

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    print("\nDataset Shape:", X.shape)

    print("\nClass Distribution:")
    print(y.value_counts())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    # Train Random Forest
    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)

    print("\n" + "=" * 40)
    print("FRONTEND MODEL RESULTS")
    print("=" * 40)

    print(
        f"Accuracy : {accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"Precision: {precision_score(y_test, y_pred):.4f}"
    )

    print(
        f"Recall   : {recall_score(y_test, y_pred):.4f}"
    )

    print(
        f"F1 Score : {f1_score(y_test, y_pred):.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(y_test, y_pred)
    )

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        rf,
        "models/rf_frontend.pkl"
    )

    print("\nSaved frontend model:")
    print("models/rf_frontend.pkl")

    return rf


if __name__ == "__main__":
    train_frontend_model()