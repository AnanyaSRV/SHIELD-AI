import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

PREDICTIONS_PATH = (
    "reports/lightgbm_tuning/validation_predictions.npz"
)

def main():
    data = np.load(PREDICTIONS_PATH)
    y_true = data["y_true"]
    probabilities = data["y_probability"]

    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities have different lengths.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain invalid values.")

    print(f"Validation rows: {len(y_true):,}")
    print(f"Validation fraud rate: {y_true.mean():.4%}")
    print()
    print("Threshold | Precision | Recall | F1")
    print("-" * 43)

    thresholds = [
        0.30, 0.35, 0.371294741468585,
        0.40, 0.45, 0.50, 0.55, 0.60,
    ]

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int8)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        print(
            f"{threshold:9.4f} | {precision:9.4f} | "
            f"{recall:6.4f} | {f1:6.4f}"
        )


if __name__ == "__main__":
    main()