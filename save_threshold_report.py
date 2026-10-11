
import csv
from pathlib import Path

output_path = Path("reports") / "lightgbm_threshold_selection.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

report = {
    "model": "LightGBM",
    "threshold": 0.425,
    "target_recall": 0.80,
    "validation_precision": 0.6931,
    "validation_recall": 0.8001,
    "validation_f1": 0.7428,
    "true_positives": 3307,
    "false_positives": 1464,
    "true_negatives": 112511,
    "false_negatives": 826,
    "status": "Validation-derived candidate; not production-locked",
}

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=report.keys())
    writer.writeheader()
    writer.writerow(report)

print(f"Threshold report saved to: {output_path}")
print("Existing model artifacts were not modified.")
