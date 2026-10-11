
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

root = Path("reports/holdout_evaluation")
model = joblib.load(root / "holdout_model.pkl")
X = joblib.load("models/X_val.pkl")

# This is a diagnostic only. The existing validation features may not
# represent the new model's preprocessing, so do not use this as evaluation.
meta = joblib.load(root / "preprocessing.pkl")

print("Best iteration:", model.best_iteration_)
print("Model expects features:", model.n_features_in_)
print("Existing validation feature count:", X.shape[1])
print("\nModel score diagnostics need the new holdout features.")
print("The script deliberately does not score mismatched data.")
print("Next, we will make the evaluation script save the holdout scores")
print("and their labels, then inspect the real probability distribution.")
