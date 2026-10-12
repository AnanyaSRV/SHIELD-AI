
import math


class ReportAgent:
    """Generate a transparent, readable investigator-style report."""

    def generate_report(
        self,
        risk_result,
        graph_result=None,
        evidence=None,
        action="VERIFY",
    ):
        """Build a Markdown report from model, graph, and evidence results."""

        risk_result = risk_result or {}
        graph_result = graph_result or {}
        evidence = evidence or []

        probability = risk_result.get("fraud_probability")
        prediction = risk_result.get("prediction", "Unknown")
        threshold = risk_result.get("threshold")
        threshold_status = risk_result.get(
            "threshold_status", "not specified"
        )

        transaction_found = graph_result.get("transaction_found")
        in_graph_sample = graph_result.get("in_graph_sample")

        raw_status = str(
            graph_result.get(
                "graph_investigation_status",
                graph_result.get("status", ""),
            )
            or ""
        )

        if transaction_found is True and in_graph_sample is True:
            graph_status = (
                "Completed — transaction found in the graph; "
                "shared-card proxy relationships were checked."
            )
        elif transaction_found is True and in_graph_sample is False:
            graph_status = (
                "Partial — transaction exists in the dataset but is "
                "outside the constructed graph; neighbors could not "
                "be computed."
            )
        elif transaction_found is False:
            graph_status = (
                "Unavailable — transaction was not found in the graph, "
                "or graph lookup could not be completed."
            )
            if raw_status:
                graph_status += f" Details: {raw_status}"
        else:
            graph_status = (
                raw_status
                if raw_status
                else "Unavailable — graph lookup status was not provided."
            )

        proxy_neighbors = graph_result.get(
            "proxy_neighbor_count",
            graph_result.get("shared_card_proxy_neighbors", 0),
        )

        try:
            proxy_neighbors = max(0, int(proxy_neighbors or 0))
        except (TypeError, ValueError, OverflowError):
            proxy_neighbors = 0

        # Shared-card links are not validated fraud findings.
        graph_validated = graph_result.get("finding_validated") is True

        hub_detected = (
            graph_result.get("hub_detected") is True
            if graph_validated
            else False
        )
        fraud_ring_detected = (
            graph_result.get("fraud_ring_detected") is True
            if graph_validated
            else False
        )

        lines = [
            "# SHIELD-AI Investigation Report",
            "",
            "## 1. Model Assessment",
            "",
            f"- **Model prediction:** {prediction}",
        ]

        if probability is not None:
            try:
                probability_value = float(probability)
                if math.isfinite(probability_value):
                    lines.append(
                        f"- **Fraud probability:** {probability_value:.2%}"
                    )
                else:
                    lines.append(
                        f"- **Fraud probability:** {probability}"
                    )
            except (TypeError, ValueError, OverflowError):
                lines.append(f"- **Fraud probability:** {probability}")

        if threshold is not None:
            try:
                threshold_value = float(threshold)
                if math.isfinite(threshold_value):
                    lines.append(
                        f"- **Classification threshold:** "
                        f"{threshold_value:.6f}"
                    )
                else:
                    lines.append(
                        f"- **Classification threshold:** {threshold}"
                    )
            except (TypeError, ValueError, OverflowError):
                lines.append(
                    f"- **Classification threshold:** {threshold}"
                )

            lines.append(f"- **Threshold status:** {threshold_status}")

        lines.extend([
            "",
            "## 2. Graph Investigation",
            "",
            f"- **Investigation status:** {graph_status}",
            (
                "- **Transaction found:** "
                + (
                    "Yes" if transaction_found is True
                    else "No" if transaction_found is False
                    else "Unknown"
                )
            ),
            (
                "- **Included in constructed graph:** "
                + (
                    "Yes" if in_graph_sample is True
                    else "No" if in_graph_sample is False
                    else "Unknown"
                )
            ),
            f"- **Shared-card proxy neighbors:** {proxy_neighbors}",
            (
                "- **Validated hub detected:** "
                + (
                    "Yes" if hub_detected
                    else "No validated finding"
                )
            ),
            (
                "- **Validated fraud ring detected:** "
                + (
                    "Yes" if fraud_ring_detected
                    else "No validated finding"
                )
            ),
            "",
            (
                "> **Graph limitation:** Shared-card links are exploratory "
                "proxy relationships. They do not independently establish "
                "account ownership, coordinated fraud, or a confirmed fraud "
                "ring. No validated finding does not prove that a transaction "
                "is legitimate."
            ),
            "",
            "## 3. Evidence",
            "",
        ])

        if evidence:
            for item in evidence:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
        else:
            lines.append("- No evidence items were provided.")

        lines.extend([
            "",
            "## 4. Recommended Action",
            "",
            f"**{action}**",
            "",
            "## 5. Interpretation and Limitations",
            "",
            (
                "This report presents a model-assisted assessment, not a "
                "confirmed fraud finding. The predicted probability and "
                "classification depend on the trained model and selected "
                "classification threshold. Graph relationships are "
                "exploratory unless independently validated. The recommended "
                "action follows the application's configured decision policy "
                "and is not a guarantee of transaction safety."
            ),
            "",
        ])

        return "\n".join(lines)
