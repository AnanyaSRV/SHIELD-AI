
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports" / "lightgbm_tuning"
INPUT = REPORT_DIR / "validation_predictions.npz"
OUTPUT = REPORT_DIR / "validation_policy_results.csv"

data = np.load(INPUT)
y_true = data["y_true"].astype(int)
y_probability = data["y_probability"].astype(float)

if len(y_true) != len(y_probability):
    raise ValueError("Labels and probabilities have different lengths.")
if not np.isfinite(y_probability).all():
    raise ValueError("Probabilities contain non-finite values.")
if not np.isin(y_true, [0, 1]).all():
    raise ValueError("Expected binary labels 0 and 1.")
if not np.isin(y_true, [1]).any():
    raise ValueError("No fraud examples found.")

monitor_thresholds = [0.40, 0.50, 0.60, 0.70]
block_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

rows = []
total_fraud = int(y_true.sum())

for monitor_t in monitor_thresholds:
    for block_t in block_thresholds:
        if monitor_t >= block_t:
            continue

        allowed = y_probability < monitor_t
        monitored = (y_probability >= monitor_t) & (y_probability < block_t)
        blocked = y_probability >= block_t

        # Three-way operational outcomes
        action = np.zeros(len(y_true), dtype=int)  # ALLOW
        action[monitored] = 1                     # MONITOR
        action[blocked] = 2                       # BLOCK

        fraud_blocked = int(y_true[blocked].sum())
        legitimate_blocked = int(blocked.sum()) - fraud_blocked

        # Standard binary fraud metrics across ALL validation rows.
        # For this diagnostic, only BLOCK is considered a positive prediction.
        y_pred_block = blocked.astype(int)
        precision = precision_score(y_true, y_pred_block, zero_division=0)
        recall = recall_score(y_true, y_pred_block, zero_division=0)
        f1 = f1_score(y_true, y_pred_block, zero_division=0)

        rows.append({
            "monitor_threshold": monitor_t,
            "block_threshold": block_t,
            "allowed_count": int(allowed.sum()),
            "fraud_allowed": int(y_true[allowed].sum()),
            "monitored_count": int(monitored.sum()),
            "fraud_monitored": int(y_true[monitored].sum()),
            "blocked_count": int(blocked.sum()),
            "fraud_blocked": fraud_blocked,
            "legitimate_blocked": legitimate_blocked,
            "block_precision": precision,
            "block_recall": recall,
            "block_f1": f1,
            "total_fraud": total_fraud,
        })

results = pd.DataFrame(rows)
results = results.sort_values(
    ["block_f1", "block_precision"],
    ascending=False
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 30)

print("\nCORRECTED VALIDATION POLICY COMPARISON")
print("=" * 150)
print(results.to_string(
    index=False,
    formatters={
        "block_precision": "{:.2%}".format,
        "block_recall": "{:.2%}".format,
        "block_f1": "{:.2%}".format,
    }
))
print("=" * 150)
print(f"\nValidation rows: {len(y_true):,}")
print(f"Actual fraud cases: {total_fraud:,}")
print(f"Results saved to: {OUTPUT}")
print("\nNo model, holdout predictions, or live decision rules were modified.")
