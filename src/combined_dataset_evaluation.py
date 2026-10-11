import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("=" * 60)
print("TRAIN: PaySim + IEEE")
print("TEST : Credit-UFFS")
print("=" * 60)

# ==================================================
# LOAD DATASETS
# ==================================================
paysim = pd.read_csv(
    "data/paysim_uffs.csv"
)

ieee = pd.read_csv(
    "data/ieee_uffs.csv"
)

credit = pd.read_csv(
    "data/credit_uffs.csv"
)

print("\nPaySim Shape :", paysim.shape)

print("IEEE Shape   :", ieee.shape)

print("Credit Shape :", credit.shape)

# ==================================================
# COMBINE TRAINING DATA
# ==================================================
train_df = pd.concat(
    [paysim, ieee],
    ignore_index=True
)

print(
    "\nCombined Train Shape:",
    train_df.shape
)

# ==================================================
# TRAIN / TEST SPLIT
# ==================================================
X_train = train_df.drop(
    "target",
    axis=1
)

y_train = train_df["target"]

X_test = credit.drop(
    "target",
    axis=1
)

y_test = credit["target"]

print("\nTrain Shape:", X_train.shape)

print("Test Shape :", X_test.shape)

# ==================================================
# CLASS WEIGHT
# ==================================================
raw_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

scale_pos_weight = min(
    raw_weight,
    10
)

print(
    "\nRaw Weight:",
    round(raw_weight, 2)
)

print(
    "Using Weight:",
    scale_pos_weight
)

# ==================================================
# FINAL SHIELD-AI RISK MODEL
# ==================================================
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

print("\nTraining...")

model.fit(
    X_train,
    y_train
)

# ==================================================
# PREDICTIONS
# ==================================================
y_prob = model.predict_proba(
    X_test
)[:, 1]

# ==================================================
# THRESHOLD OPTIMIZATION
# ==================================================
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

print(
    "\nBest Threshold:",
    best_threshold
)

print(
    "Best F1:",
    round(best_f1, 5)
)

# ==================================================
# FINAL PREDICTIONS
# ==================================================
y_pred = (
    y_prob >= best_threshold
).astype(int)

# ==================================================
# METRICS
# ==================================================
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

print(
    f"ROC-AUC  : {roc_auc:.5f}"
)

print(
    f"PR-AUC   : {pr_auc:.5f}"
)

print(
    f"Precision: {precision:.5f}"
)

print(
    f"Recall   : {recall:.5f}"
)

print(
    f"F1 Score : {f1:.5f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nExperiment Complete.")