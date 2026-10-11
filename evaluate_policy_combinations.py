
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

prediction_file = Path(
    "reports/lightgbm_tuning/final_holdout_predictions.npz"
)

data = np.load(prediction_file)
y_true = data["y_true"]
probabilities = data["y_probability"]

monitor_thresholds = [0.40, 0.50, 0.60, 0.70]
block_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

results = []

for monitor_threshold in monitor_thresholds:
    for block_threshold in block_thresholds:
        if monitor_threshold >= block_threshold:
            continue

        allow = probabilities < monitor_threshold
        monitor = (
            (probabilities >= monitor_threshold)
            & (probabilities < block_threshold)
        )
        block = probabilities >= block_threshold

        # Count outcomes in each action group.
        allow_rows = int(allow.sum())
        monitor_rows = int(monitor.sum())
        block_rows = int(block.sum())

        fraud_allowed = int(y_true[allow].sum())
        fraud_monitored = int(y_true[monitor].sum())
        fraud_blocked = int(y_true[block].sum())

        legitimate_blocked = block_rows - fraud_blocked

        # Treat BLOCK as the positive class for these metrics.
        block_predictions = block.astype(int)

        results.append({
            "Monitor threshold": monitor_threshold,
            "Block threshold": block_threshold,
            "Allowed": allow_rows,
            "Fraud allowed": fraud_allowed,
            "Monitored": monitor_rows,
            "Fraud monitored": fraud_monitored,
            "Blocked": block_rows,
            "Fraud blocked": fraud_blocked,
            "Legitimate blocked": legitimate_blocked,
            "Block precision": precision_score(
                y_true, block_predictions, zero_division=0
            ),
            "Fraud recall at BLOCK": recall_score(
                y_true, block_predictions, zero_division=0
            ),
            "Block F1": f1_score(
                y_true, block_predictions, zero_division=0
            ),
        })

df = pd.DataFrame(results)

# Show the strongest block precision candidates first.
df = df.sort_values(
    ["Block F1", "Block precision"],
    ascending=False,
)

print("\nMONITOR + BLOCK THRESHOLD COMPARISON")
print("=" * 140)

print(df.to_string(
    index=False,
    formatters={
        "Monitor threshold": "{:.2f}".format,
        "Block threshold": "{:.2f}".format,
        "Block precision": "{:.2%}".format,
        "Fraud recall at BLOCK": "{:.2%}".format,
        "Block F1": "{:.2%}".format,
    },
))

print("=" * 140)

# Save the analysis results, not model artifacts.
output_file = Path(
    "reports/lightgbm_tuning/policy_combination_results.csv"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print("No model or live decision rules were modified.")
