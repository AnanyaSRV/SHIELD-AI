import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("=" * 60)
print("SOFT VOTING FUSION")
print("=" * 60)

# ==========================================
# LOAD PREDICTIONS
# ==========================================
lr = pd.read_csv("fusion/lr_predictions.csv")
rf = pd.read_csv("fusion/rf_predictions.csv")
xgb = pd.read_csv("fusion/xgb_predictions.csv")

# Ground Truth
y_true = lr["y_true"]

# ==========================================
# SOFT VOTING
# ==========================================
fusion_prob = (
    lr["y_prob"]
    + rf["y_prob"]
    + xgb["y_prob"]
) / 3

# ==========================================
# THRESHOLD OPTIMIZATION
# ==========================================
best_f1 = 0
best_threshold = 0.50

for threshold in [i / 100 for i in range(5, 100, 5)]:

    y_pred = (
        fusion_prob >= threshold
    ).astype(int)

    score = f1_score(
        y_true,
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
    fusion_prob >= best_threshold
).astype(int)

# ==========================================
# METRICS
# ==========================================
roc_auc = roc_auc_score(
    y_true,
    fusion_prob
)

pr_auc = average_precision_score(
    y_true,
    fusion_prob
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
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
    y_true,
    y_pred
))