
import joblib
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

print("Loading saved model and validation data...")

model = joblib.load("models/lightgbm_model.pkl")
X_val = joblib.load("models/X_val.pkl")
y_val = np.asarray(joblib.load("models/y_val.pkl")).ravel()

probabilities = model.predict_proba(X_val)[:, 1]

for threshold in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    print(f"\nThreshold: {threshold:.2f}")
    print(f"TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
    print(f"Precision: {precision_score(y_val, predictions):.4f}")
    print(f"Recall:    {recall_score(y_val, predictions):.4f}")
    print(f"F1-score:  {f1_score(y_val, predictions):.4f}")
