
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

THRESHOLD = 0.425
MODEL_PATH = Path("models/lightgbm_model.pkl")
X_PATH = Path("models/X_val.pkl")
Y_PATH = Path("models/y_val.pkl")
OUTPUT_DIR = Path("reports/validation_evaluation")


def main():
    for path in (MODEL_PATH, X_PATH, Y_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved artifact not found: {path}")

    model = joblib.load(MODEL_PATH)
    X = joblib.load(X_PATH)
    y = np.asarray(joblib.load(Y_PATH)).ravel()

    if len(X) != len(y):
        raise ValueError("Validation features and labels have different lengths.")
    if len(np.unique(y)) != 2:
        raise ValueError("Expected both legitimate and fraud classes.")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "evaluation_type": "Validation; not an independent test",
        "threshold": THRESHOLD,
        "transactions": len(y),
        "accuracy": accuracy_score(y, predictions),
        "fraud_precision": precision_score(y, predictions, zero_division=0),
        "fraud_recall": recall_score(y, predictions, zero_division=0),
        "fraud_f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / "metrics.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write("SHIELD-AI — LIGHTGBM VALIDATION EVALUATION\n")
        file.write("=" * 48 + "\n")
        file.write("Warning: validation results, not independent test results.\n\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                file.write(f"{key}: {value:.6f}\n")
            else:
                file.write(f"{key}: {value}\n")

        file.write("\nClassification report:\n")
        file.write(
            classification_report(
                y,
                predictions,
                labels=[0, 1],
                target_names=["Legitimate", "Fraud"],
                digits=4,
                zero_division=0,
            )
        )

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay = None
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        labels=[0, 1],
        display_labels=["Legitimate", "Fraud"],
        cmap="Blues",
        ax=ax,
        values_format=",",
    )
    ax.set_title(f"SHIELD-AI Confusion Matrix (threshold={THRESHOLD})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y, probabilities, ax=ax)
    ax.set_title("SHIELD-AI Validation ROC Curve")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=ax)
    ax.set_title("SHIELD-AI Validation Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=200)
    plt.close(fig)

    print("Evaluation report created successfully.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Fraud recall: {metrics['fraud_recall']:.4%}")
    print(f"Fraud precision: {metrics['fraud_precision']:.4%}")
    print(f"Fraud F1: {metrics['fraud_f1']:.4%}")
    print("Saved: metrics.txt, confusion_matrix.png, roc_curve.png,")
    print("       precision_recall_curve.png")
    print("Saved model and validation artifacts were not modified.")


if __name__ == "__main__":
    main()
