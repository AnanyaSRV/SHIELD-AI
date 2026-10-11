import os
import joblib
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data
from encoder import encode_features


def train_lightgbm():

    print("=" * 60)
    print("LIGHTGBM TRAINING")
    print("=" * 60)

    # ---------------------------------------------------
    # LOAD + PREPROCESS
    # ---------------------------------------------------
    X, y = preprocess_data(
        "data/train_transaction.csv",
        "data/train_identity.csv"
    )

    # ---------------------------------------------------
    # SPLIT
    # ---------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\nTrain:", X_train.shape)
    print("Validation:", X_val.shape)

    # ---------------------------------------------------
    # ENCODE
    # ---------------------------------------------------
    X_train, X_val, encoder = encode_features(
        X_train,
        X_val
    )

    # ---------------------------------------------------
    # LIGHTGBM
    # ---------------------------------------------------
    print("\nTraining model...")

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",

        n_estimators=1000,
        learning_rate=0.03,

        num_leaves=128,
        max_depth=-1,

        subsample=0.8,
        colsample_bytree=0.8,

        min_child_samples=50,

        # Reduced from 27.58
        scale_pos_weight=10,

        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[(X_val, y_val)],
        eval_metric="auc",

        callbacks=[
        lgb.early_stopping(
        stopping_rounds=100,
        first_metric_only=True,
        verbose=True
    ),
    lgb.log_evaluation(50)
    ]
    )

    # ---------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------
    y_prob = model.predict_proba(
        X_val,
        num_iteration=model.best_iteration_
    )[:, 1]

    # ---------------------------------------------------
    # FIND BEST THRESHOLD
    # ---------------------------------------------------
    best_threshold = 0.50
    best_f1 = 0

    print("\nThreshold Search:")

    for threshold in [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50
    ]:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            y_pred
        )

        print(
            f"Threshold {threshold:.2f}"
            f" → F1 = {score:.5f}"
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print(
        f"\nBest Threshold: "
        f"{best_threshold:.2f}"
    )

    y_pred = (
        y_prob >= best_threshold
    ).astype(int)

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------
    roc_auc = roc_auc_score(
        y_val,
        y_prob
    )

    pr_auc = average_precision_score(
        y_val,
        y_prob
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Best Iteration : "
        f"{model.best_iteration_}"
    )

    print(
        f"ROC-AUC        : "
        f"{roc_auc:.5f}"
    )

    print(
        f"PR-AUC         : "
        f"{pr_auc:.5f}"
    )

    print(
        f"Best F1        : "
        f"{best_f1:.5f}"
    )

    print(
        f"Threshold      : "
        f"{best_threshold:.2f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_val,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    # ---------------------------------------------------
    # SAVE
    # ---------------------------------------------------
    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/lightgbm_model.pkl"
    )

    joblib.dump(
        encoder,
        "models/encoder.pkl"
    )

    print("\nModel saved.")
    joblib.dump(X_val, "models/X_val.pkl")
    joblib.dump(y_val, "models/y_val.pkl")

    return model


if __name__ == "__main__":

    train_lightgbm()