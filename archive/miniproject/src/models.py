# src/models.py

import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from xgboost import XGBClassifier


def train_and_evaluate_models(X_train, X_val, y_train, y_val):

    print("=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    # Logistic Regression
    print("\nTraining Logistic Regression...")

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    lr.fit(X_train, y_train)

    # Random Forest
    print("Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/rf.pkl")
    print("Random Forest saved.")

    # XGBoost
    print("Training XGBoost...")

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    xgb.fit(X_train, y_train)

    # Voting Ensemble
    print("Training Voting Ensemble...")

    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("xgb", xgb),
        ],
        voting="soft",
        n_jobs=-1,
    )

    ensemble.fit(X_train, y_train)

    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb,
        "Voting Ensemble": ensemble,
    }

    print("\nUsing Random Forest threshold = 0.20")

    # Evaluation
    for name, model in models.items():

        print("\n" + "=" * 40)
        print(name)
        print("=" * 40)

        # Threshold tuning ONLY for Random Forest
        if name == "Random Forest":

            y_prob = model.predict_proba(X_val)[:, 1]

            threshold = 0.30

            y_pred = (y_prob >= threshold).astype(int)

        else:

            y_pred = model.predict(X_val)

        print(
            f"Accuracy : {accuracy_score(y_val, y_pred):.4f}"
        )

        print(
            f"Precision: {precision_score(y_val, y_pred):.4f}"
        )

        print(
            f"Recall   : {recall_score(y_val, y_pred):.4f}"
        )

        print(
            f"F1 Score : {f1_score(y_val, y_pred):.4f}"
        )

        print("\nClassification Report:")

        print(
            classification_report(y_val, y_pred)
        )

    # Save ensemble
    joblib.dump(
        ensemble,
        "models/fraud_ensemble.pkl",
    )

    print(
        "\nSaved ensemble to models/fraud_ensemble.pkl"
    )

    return ensemble