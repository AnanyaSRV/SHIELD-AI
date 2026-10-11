
import numpy as np
from pathlib import Path

prediction_file = Path(
    "reports/lightgbm_tuning/final_holdout_predictions.npz"
)

data = np.load(prediction_file)
y_true = data["y_true"]
probabilities = data["y_probability"]

actions = [
    ("ALLOW", probabilities < 0.60),
    ("MONITOR", (probabilities >= 0.60) & (probabilities < 0.80)),
    ("BLOCK", probabilities >= 0.80),
]

print("\nHELD-OUT ACTION POLICY EVALUATION")
print("=" * 72)
print(
    f"{'Action':<12}{'Rows':>10}{'Fraud':>10}"
    f"{'Legitimate':>14}{'Fraud rate':>14}"
)
print("-" * 72)

for action, mask in actions:
    rows = int(mask.sum())
    fraud = int(y_true[mask].sum())
    legitimate = rows - fraud
    fraud_rate = fraud / rows if rows else None

    rate_text = f"{fraud_rate:.2%}" if rows else "N/A"

    print(
        f"{action:<12}{rows:>10,}{fraud:>10,}"
        f"{legitimate:>14,}{rate_text:>14}"
    )

print("-" * 72)
print(f"Total transactions: {len(y_true):,}")
print(
    "\nOffline evaluation only. These results do not establish "
    "optimal production thresholds."
)
