import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

# Load saved files
model = joblib.load("models/lightgbm_model.pkl")
X_val = joblib.load("models/X_val.pkl")
y_val = joblib.load("models/y_val.pkl")

# Predict probabilities
y_prob = model.predict_proba(X_val)[:, 1]

# Change this whenever you want
threshold = 0.50

y_pred = (y_prob >= threshold).astype(int)

print("Threshold:", threshold)
print("F1 Score:", f1_score(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        digits=4,
        zero_division=0
    )
)