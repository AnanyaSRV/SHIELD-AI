from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "lightgbm_tuning"
VALIDATION_FILE = REPORT_DIR / "validation_predictions.npz"

TARGET_RECALL = 0.80
CURRENT_THRESHOLD = 0.371294741468585


def metrics_at_threshold(y_true, probabilities, threshold):
    pred = probabilities >= threshold
    positive = y_true == 1
    negative = ~positive

    tp = int(np.count_nonzero(pred & positive))
    fp = int(np.count_nonzero(pred & negative))
    fn = int(np.count_nonzero(~pred & positive))
    tn = int(np.count_nonzero(~pred & negative))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "predicted_fraud": tp + fp,
    }


def main():
    if not VALIDATION_FILE.is_file():
        raise FileNotFoundError(
            f"Validation predictions not found: {VALIDATION_FILE}"
        )

    data = np.load(VALIDATION_FILE)
    y = np.asarray(data["y_true"]).reshape(-1)
    scores = np.asarray(
        data["y_probability"], dtype=np.float64
    ).reshape(-1)

    if len(y) == 0 or len(y) != len(scores):
        raise ValueError("Validation arrays are empty or mismatched.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Probabilities contain invalid values.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")
    if not np.any(y == 1):
        raise ValueError("Validation set contains no fraud examples.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # At each distinct score, predict fraud for scores >= threshold.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y[order].astype(np.int8)

    # Cumulative positive and negative counts from highest scores down.
    reverse_y = sorted_y[::-1]
    reverse_scores = sorted_scores[::-1]
    cumulative_tp = np.cumsum(reverse_y == 1)
    cumulative_fp = np.cumsum(reverse_y == 0)

    # Keep only the final row for each distinct score.
    distinct_end = np.r_[
        np.flatnonzero(reverse_scores[:-1] != reverse_scores[1:]),
        len(reverse_scores) - 1,
    ]

    tp = cumulative_tp[distinct_end]
    fp = cumulative_fp[distinct_end]
    thresholds = reverse_scores[distinct_end]

    total_positive = int(np.count_nonzero(y == 1))
    fn = total_positive - tp
    tn = len(y) - total_positive - fp

    precision = np.divide(
        tp, tp + fp,
        out=np.zeros(len(tp), dtype=float),
        where=(tp + fp) > 0,
    )
    recall = tp / total_positive
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(tp), dtype=float),
        where=(precision + recall) > 0,
    )
    accuracy = (tp + tn) / len(y)

    results = pd.DataFrame({
        "threshold": thresholds,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "predicted_fraud": tp + fp,
    })

    results.to_csv(
        REPORT_DIR / "validation_threshold_results.csv",
        index=False,
    )

    current = metrics_at_threshold(y, scores, CURRENT_THRESHOLD)
    eligible = results[results["recall"] >= TARGET_RECALL]

    print("=" * 60)
    print("SHIELD-AI LIGHTGBM VALIDATION THRESHOLD TUNING")
    print("=" * 60)
    print(f"Validation rows: {len(y):,}")
    print(f"Fraud rate: {y.mean():.4%}")
    print(f"Thresholds evaluated: {len(results):,}")
    print(f"Recall target: {TARGET_RECALL:.0%}")

    print("\nCurrent threshold:")
    print(json.dumps(current, indent=2))

    if eligible.empty:
        summary = {
            "status": "target_not_met",
            "target_recall": TARGET_RECALL,
            "current_threshold": CURRENT_THRESHOLD,
            "current_validation_metrics": current,
            "selected_threshold": None,
        }
        print("\nNo threshold meets the recall target.")
    else:
        best = eligible.sort_values(
            ["precision", "recall", "f1_score", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]

        selected = {
            key: (
                float(value)
                if isinstance(value, (np.floating, float))
                else int(value)
                if isinstance(value, (np.integer, int))
                else value
            )
            for key, value in best.to_dict().items()
        }

        summary = {
            "status": "selected",
            "target_recall": TARGET_RECALL,
            "current_threshold": CURRENT_THRESHOLD,
            "current_validation_metrics": current,
            "selected_threshold": selected["threshold"],
            "selected_validation_metrics": selected,
            "selection_rule": (
                "Highest precision with validation recall >= target; "
                "ties prefer recall, then F1, then higher threshold."
            ),
            "warning": (
                "Selection uses validation data only. Assess the chosen "
                "threshold on the held-out test set once; do not tune "
                "the threshold using test results."
            ),
        }

        print("\nSelected threshold:")
        print(json.dumps(selected, indent=2))

    summary_path = REPORT_DIR / "selected_validation_threshold.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(REPORT_DIR / "validation_threshold_results.csv")
    print(summary_path)
    print("\nNo model training or test-set evaluation was performed.")


if __name__ == "__main__":
    main()