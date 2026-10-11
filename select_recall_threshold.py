
import joblib
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

TARGET_RECALL = 0.80
GRID_STEP = 0.001

print("Loading saved model and validation data...")

model = joblib.load("models/lightgbm_model.pkl")
X_val = joblib.load("models/X_val.pkl")
y_val = np.asarray(joblib.load("models/y_val.pkl")).ravel()

print("Calculating fraud probabilities...")
probabilities = model.predict_proba(X_val)[:, 1]

# Dense grid of thresholds; no model training or artifact saving.
thresholds = np.arange(
    GRID_STEP, 1.0, GRID_STEP
)

eligible = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    if recall >= TARGET_RECALL:
        eligible.append((threshold, recall))

if not eligible:
    raise RuntimeError(
        f"No tested threshold achieved recall >= {TARGET_RECALL:.0%}."
    )

# Highest tested threshold that meets the recall target.
best_threshold = eligible[-1][0]
predictions = (probabilities >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val, predictions, labels=[0, 1]
).ravel()

precision = precision_score(
    y_val, predictions, zero_division=0
)
recall = recall_score(
    y_val, predictions, zero_division=0
)
f1 = f1_score(
    y_val, predictions, zero_division=0
)

print("\n" + "=" * 55)
print("RECALL-CONSTRAINED THRESHOLD RESULTS")
print("=" * 55)
print(f"Target recall:       {TARGET_RECALL:.2%}")
print(f"Grid step:           {GRID_STEP:.3f}")
print(f"Selected threshold:  {best_threshold:.3f}")
print(f"Precision:           {precision:.4f} ({precision:.2%})")
print(f"Recall:              {recall:.4f} ({recall:.2%})")
print(f"F1-score:            {f1:.4f} ({f1:.2%})")
print(f"TP (fraud detected): {tp:,}")
print(f"FP (false alarms):   {fp:,}")
print(f"TN (legit correct):  {tn:,}")
print(f"FN (fraud missed):   {fn:,}")
print("=" * 55)
print("Validation-set result only; not independent test performance.")
print("No model artifacts were changed.")
