import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc
)

# ==========================================
# LOAD PREDICTIONS
# ==========================================
models = {

    "LR":
        pd.read_csv(
            "fusion/lr_predictions.csv"
        ),

    "RF":
        pd.read_csv(
            "fusion/rf_predictions.csv"
        ),

    "XGB":
        pd.read_csv(
            "fusion/xgb_predictions.csv"
        )
}

# ==========================================
# SOFT VOTING
# ==========================================
soft_prob = (
    models["LR"]["y_prob"]
    + models["RF"]["y_prob"]
    + models["XGB"]["y_prob"]
) / 3

models["Soft Voting"] = pd.DataFrame({

    "y_true":
        models["LR"]["y_true"],

    "y_prob":
        soft_prob

})

# ==========================================
# WEIGHTED FUSION
# ==========================================
weighted_prob = (

    0.10 * models["LR"]["y_prob"]

    + 0.70 * models["RF"]["y_prob"]

    + 0.20 * models["XGB"]["y_prob"]

)

models["Weighted Fusion"] = pd.DataFrame({

    "y_true":
        models["LR"]["y_true"],

    "y_prob":
        weighted_prob

})

# ==========================================
# ROC CURVES
# ==========================================
plt.figure(figsize=(8, 6))

for name, df in models.items():

    fpr, tpr, _ = roc_curve(
        df["y_true"],
        df["y_prob"]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()

plt.savefig(
    "roc_curves.png"
)

plt.close()

print("Saved: roc_curves.png")

# ==========================================
# PR CURVES
# ==========================================
plt.figure(figsize=(8, 6))

for name, df in models.items():

    precision, recall, _ = (
        precision_recall_curve(
            df["y_true"],
            df["y_prob"]
        )
    )

    pr_auc = auc(
        recall,
        precision
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AUC={pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(
    "Precision-Recall Curves"
)

plt.legend()

plt.savefig(
    "pr_curves.png"
)

plt.close()

print("Saved: pr_curves.png")

# ==========================================
# INFERENCE TIME
# ==========================================
print("\nInference Time")

for name, df in models.items():

    start = time.time()

    _ = df["y_prob"] > 0.5

    end = time.time()

    latency = (
        (end - start)
        * 1000
    )

    print(
        f"{name}: "
        f"{latency:.3f} ms"
    )

print("\nPhase 8 Complete.")