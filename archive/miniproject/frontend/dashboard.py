
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
METRICS_PATH = (
    ROOT / "reports" / "lightgbm_tuning" / "final_holdout_metrics.json"
)


def render_dashboard():
    st.header("Fraud Monitoring Dashboard")

    st.caption(
        "SHIELD-AI · IEEE-CIS evaluation results. "
        "These are offline test metrics, not live transaction statistics."
    )

    if not METRICS_PATH.is_file():
        st.error(f"Evaluation metrics not found: {METRICS_PATH}")
        return

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not load evaluation metrics: {exc}")
        return

    preferred_metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
        ("Average precision", "average_precision"),
    ]

    available = [
        (label, key)
        for label, key in preferred_metrics
        if key in metrics
    ]

    if not available:
        st.warning(
            "The metrics file does not contain the expected metric keys."
        )
        st.json(metrics)
        return

    columns = st.columns(min(3, len(available)))

    for index, (label, key) in enumerate(available):
        value = metrics[key]

        with columns[index % len(columns)]:
            st.metric(label, f"{float(value):.2%}")

    st.divider()
    st.subheader("Evaluation details")

    displayed_keys = {key for _, key in available}
    details = {
        key: value
        for key, value in metrics.items()
        if key not in displayed_keys
    }

    if details:
        st.json(details)

    st.info(
        "The reported scores come from a held-out dataset. "
        "They do not represent live transaction statistics."
    )
