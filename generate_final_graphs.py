
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "lightgbm_tuning"
OUT = ROOT / "reports" / "final_graphs"
OUT.mkdir(parents=True, exist_ok=True)

# Load the saved final holdout predictions.
data = np.load(REPORT / "final_holdout_predictions.npz")
y_true = data["y_true"].astype(int)
prob = data["y_probability"].astype(float)
threshold = float(data["threshold"])

if len(y_true) != len(prob):
    raise ValueError("Labels and probabilities have different lengths.")

pred = (prob >= threshold).astype(int)

# 1. ROC curve
fpr, tpr, _ = roc_curve(y_true, prob)
auc = roc_auc_score(y_true, prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"LightGBM (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SHIELD-AI: ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=300)
plt.close()

# 2. Precision–Recall curve
precision, recall, _ = precision_recall_curve(y_true, prob)
ap = average_precision_score(y_true, prob)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision, label=f"LightGBM (AP = {ap:.4f})")
plt.axhline(y_true.mean(), linestyle="--", label="Fraud prevalence")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("SHIELD-AI: Precision–Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "precision_recall_curve.png", dpi=300)
plt.close()

# 3. Confusion matrix at the saved validation-selected threshold
cm = confusion_matrix(y_true, pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"],
)
disp.plot(values_format=",", colorbar=False)
plt.title(f"SHIELD-AI: Confusion Matrix (threshold = {threshold:.4f})")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=300)
plt.close()

# 4. Classification metrics at candidate thresholds
thresholds = [0.30, 0.35, threshold, 0.40, 0.45, 0.50, 0.60, 0.65, 0.70, 0.80]
rows = []

for t in sorted(set(thresholds)):
    p = (prob >= t).astype(int)
    rows.append({
        "threshold": t,
        "precision": precision_score(y_true, p, zero_division=0),
        "recall": recall_score(y_true, p, zero_division=0),
        "f1": f1_score(y_true, p, zero_division=0),
    })

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "threshold_metrics.csv", index=False)

plt.figure(figsize=(9, 6))
plt.plot(metrics["threshold"], metrics["precision"], marker="o", label="Precision")
plt.plot(metrics["threshold"], metrics["recall"], marker="o", label="Recall")
plt.plot(metrics["threshold"], metrics["f1"], marker="o", label="F1-score")
plt.axvline(threshold, linestyle="--", label=f"Selected threshold = {threshold:.3f}")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("SHIELD-AI: Threshold Performance Trade-off")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "threshold_tradeoff.png", dpi=300)
plt.close()

# 5. Decision-policy trade-off on the held-out data.
# Diagnostic only: do not use this holdout chart to select a new policy.
policy_path = REPORT / "policy_combination_results.csv"


if policy_path.exists():
    policy = pd.read_csv(policy_path)

    # Normalize column names to handle CSV naming differences.
    policy.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in policy.columns
    ]

    if {"block_threshold", "block_precision", "fraud_recall_at_block"}.issubset(policy.columns):
        policy = (
            policy.drop_duplicates("block_threshold")
            .sort_values("block_threshold")
        )

        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.plot(
            policy["block_threshold"],
            policy["block_precision"],
            marker="o",
            label="Block precision",
        )
        ax1.plot(
            policy["block_threshold"],
            policy["fraud_recall_at_block"],
            marker="o",
            label="Fraud recall at BLOCK",
        )
        ax1.set_xlabel("BLOCK Threshold")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1)
        ax1.set_title("SHIELD-AI: Holdout Decision-Policy Trade-off")
        ax1.legend()
        fig.tight_layout()
        fig.savefig(OUT / "holdout_policy_tradeoff.png", dpi=300)
        plt.close(fig)
    else:
        print("Skipping policy chart: expected columns not found.")
        print("Available columns:", list(policy.columns))
# Summary of saved evaluation metrics
metrics_path = REPORT / "final_holdout_metrics.json"
if metrics_path.exists():
    with open(metrics_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(OUT / "graph_summary.txt", "w", encoding="utf-8") as f:
        f.write("SHIELD-AI Final Holdout Graph Summary\n")
        f.write("=" * 40 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

print("Graph generation completed.")
print(f"Output folder: {OUT}")
print("Files:")
for file in sorted(OUT.iterdir()):
    print(f" - {file.name}")
print("Note: Holdout policy chart is diagnostic, not for threshold selection.")