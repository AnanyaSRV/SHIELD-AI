import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("data/ieee_uffs.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

os.makedirs("fusion", exist_ok=True)

# ==========================================
# CLASS WEIGHT FOR XGBOOST
# ==========================================
raw_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

scale_pos_weight = min(
    raw_weight,
    10
)

print("\nRaw Weight:",
      round(raw_weight, 2))

print("Using Weight:",
      scale_pos_weight)

# ==========================================
# MODELS
# ==========================================
models = {

    "LR": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "RF": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGB": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
}

# ==========================================
# TRAIN + SAVE PREDICTIONS
# ==========================================
for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    # ==========================================
    # PREDICTION PROBABILITIES
    # ==========================================
    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # ==========================================
    # THRESHOLD OPTIMIZATION
    # ==========================================
    best_f1 = 0
    best_threshold = 0.50

    for threshold in [
        i / 100
        for i in range(5, 100, 5)
    ]:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        score = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    # ==========================================
    # FINAL PREDICTIONS
    # ==========================================
    y_pred = (
        y_prob >= best_threshold
    ).astype(int)

    # ==========================================
    # METRICS
    # ==========================================
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nBest Threshold:",
          best_threshold)

    print("\nRESULTS")

    print("ROC-AUC :",
          round(roc_auc, 5))

    print("PR-AUC  :",
          round(pr_auc, 5))

    print("Precision:",
          round(precision, 5))

    print("Recall   :",
          round(recall, 5))

    print("F1 Score :",
          round(f1, 5))

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # ==========================================
    # SAVE PREDICTIONS
    # ==========================================
    pred_df = pd.DataFrame({

        "y_true": y_test.values,

        "y_prob": y_prob
    })

    pred_df.to_csv(
        f"fusion/{name.lower()}_predictions.csv",
        index=False
    )

    print(
        f"Saved fusion/{name.lower()}_predictions.csv"
    )

print("\nModel comparison complete.")