
import math


class ReportAgent:
    """Generate a transparent, readable investigator-style report."""

    def generate_report(
        self, risk_result, graph_result=None, evidence=None, action="VERIFY"
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

        graph_status = graph_result.get(
            "graph_investigation_status",
            "Graph investigation status not provided.",
        )

        # The current graph uses exploratory shared-card proxy links.
        # Do not describe them as suspicious or confirmed fraud.
        proxy_neighbors = graph_result.get(
            "proxy_neighbor_count",
            graph_result.get("shared_card_proxy_neighbors", 0),
        )

        try:
            proxy_neighbors = int(proxy_neighbors or 0)
        except (TypeError, ValueError):
            proxy_neighbors = 0

        # Graph findings are only described as validated if the graph
        # agent explicitly marks them as validated.
        graph_validated = (
            graph_result.get("finding_validated") is True
            and graph_status == "validated"
        )

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
            except (TypeError, ValueError):
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
            except (TypeError, ValueError):
                lines.append(
                    f"- **Classification threshold:** {threshold}"
                )

            lines.append(f"- **Threshold status:** {threshold_status}")

        lines.extend([
            "",
            "## 2. Graph Investigation",
            "",
            f"- **Investigation status:** {graph_status}",
            f"- **Validated hub detected:** {hub_detected}",
            f"- **Validated fraud ring detected:** {fraud_ring_detected}",
            f"- **Shared-card proxy neighbors:** {proxy_neighbors}",
            "",
            (
                "> **Graph limitation:** Shared-card links are exploratory "
                "proxy relationships. They do not independently establish "
                "account ownership, coordinated fraud, or a confirmed "
                "fraud ring. An absent graph finding does not prove that "
                "a transaction is legitimate."
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
                "This report presents a model-assisted assessment, not "
                "a confirmed fraud finding. The predicted probability "
                "and classification depend on the trained model and "
                "selected classification threshold. Graph relationships "
                "are exploratory unless independently validated. The "
                "recommended action follows the application's configured "
                "decision policy and is not a guarantee of transaction safety."
            ),
            "",
        ])

        return "\n".join(lines)
