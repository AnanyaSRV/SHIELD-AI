
import joblib
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

model = joblib.load("models/lightgbm_model.pkl")
X_val = joblib.load("models/X_val.pkl")
y_val = np.asarray(joblib.load("models/y_val.pkl")).ravel()
probabilities = model.predict_proba(X_val)[:, 1]

print("Threshold | Precision | Recall | F1")
print("-" * 43)

for threshold in np.arange(0.420, 0.431, 0.001):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:.3f} | "
        f"{precision_score(y_val, predictions, zero_division=0):.4f} | "
        f"{recall_score(y_val, predictions, zero_division=0):.4f} | "
        f"{f1_score(y_val, predictions, zero_division=0):.4f} | "
        f"TP={tp:,} FP={fp:,} TN={tn:,} FN={fn:,}"
    )
