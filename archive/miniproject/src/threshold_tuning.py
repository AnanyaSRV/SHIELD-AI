import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

print("=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

# Load saved model and validation data
rf = joblib.load("models/rf.pkl")
X_val = joblib.load("models/X_val.pkl")
y_val = joblib.load("models/y_val.pkl")

# Fraud probabilities
y_prob = rf.predict_proba(X_val)[:, 1]

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    results.append(
        [
            threshold,
            accuracy,
            precision,
            recall,
            f1,
        ]
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ],
)

print("\nThreshold Tuning Results:")
print(results_df.round(4))

best_f1 = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n" + "=" * 60)
print("BEST THRESHOLD (Highest F1)")
print("=" * 60)

print(best_f1)

results_df.to_csv(
    "models/threshold_results.csv",
    index=False,
)

print("\nSaved results to:")
print("models/threshold_results.csv")