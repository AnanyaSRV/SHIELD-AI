
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

# Candidate BLOCK thresholds.
# The MONITOR threshold remains fixed at 0.60.
block_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

results = []

for block_threshold in block_thresholds:
    allow_mask = probabilities < 0.60
    monitor_mask = (
        (probabilities >= 0.60)
        & (probabilities < block_threshold)
    )
    block_mask = probabilities >= block_threshold

    # BLOCK is treated as a positive fraud decision for this analysis.
    block_predictions = block_mask.astype(int)

    results.append({
        "Block threshold": block_threshold,
        "Allowed": int(allow_mask.sum()),
        "Fraud allowed": int(y_true[allow_mask].sum()),
        "Monitored": int(monitor_mask.sum()),
        "Fraud monitored": int(y_true[monitor_mask].sum()),
        "Blocked": int(block_mask.sum()),
        "Fraud blocked": int(y_true[block_mask].sum()),
        "Legitimate blocked": int(block_mask.sum() - y_true[block_mask].sum()),
        "Block precision": precision_score(
            y_true, block_predictions, zero_division=0
        ),
        "Fraud recall at BLOCK threshold": recall_score(
            y_true, block_predictions, zero_division=0
        ),
        "Block F1": f1_score(
            y_true, block_predictions, zero_division=0
        ),
    })

df = pd.DataFrame(results)

print("\nBLOCK THRESHOLD COMPARISON")
print("=" * 115)
print(df.to_string(
    index=False,
    formatters={
        "Block precision": "{:.2%}".format,
        "Fraud recall at BLOCK threshold": "{:.2%}".format,
        "Block F1": "{:.2%}".format,
        "Block threshold": "{:.2f}".format,
    },
))
print("=" * 115)
print(
    "\nDiagnostic comparison only. The 0.60 MONITOR threshold is fixed; "
    "the BLOCK threshold varies. No project files or models are modified."
)
