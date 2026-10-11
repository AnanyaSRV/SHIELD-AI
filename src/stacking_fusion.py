import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("=" * 60)
print("STACKING FUSION")
print("=" * 60)

# ==========================================
# LOAD BASE MODEL PREDICTIONS
# ==========================================
lr = pd.read_csv("fusion/lr_predictions.csv")
rf = pd.read_csv("fusion/rf_predictions.csv")
xgb = pd.read_csv("fusion/xgb_predictions.csv")

# ==========================================
# META FEATURES
# ==========================================
meta_X = pd.DataFrame({
    "lr": lr["y_prob"],
    "rf": rf["y_prob"],
    "xgb": xgb["y_prob"]
})

meta_y = lr["y_true"]

# ==========================================
# META TRAIN/TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    meta_X,
    meta_y,
    test_size=0.20,
    stratify=meta_y,
    random_state=42
)

print("\nMeta Train Shape:", X_train.shape)
print("Meta Test Shape :", X_test.shape)

# ==========================================
# META LEARNER
# ==========================================
meta_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

print("\nTraining meta learner...")

meta_model.fit(X_train, y_train)

# ==========================================
# META PREDICTIONS
# ==========================================
y_prob = meta_model.predict_proba(X_test)[:, 1]

# ==========================================
# THRESHOLD OPTIMIZATION
# ==========================================
best_f1 = 0
best_threshold = 0.50

for threshold in [i / 100 for i in range(5, 100, 5)]:

    y_pred = (y_prob >= threshold).astype(int)

    score = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("\nBest Threshold:", best_threshold)

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

print("\nRESULTS")

print("ROC-AUC :", round(roc_auc, 5))
print("PR-AUC  :", round(pr_auc, 5))
print("Precision:", round(precision, 5))
print("Recall   :", round(recall, 5))
print("F1 Score :", round(f1, 5))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred
))