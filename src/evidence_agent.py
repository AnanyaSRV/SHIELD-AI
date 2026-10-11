
import math
import pandas as pd


class EvidenceAgent:
    """Collect model, feature, transaction, and graph evidence."""

    def collect_evidence(
        self,
        risk_result,
        graph_result,
        transaction_info=None,
    ):
        evidence = []

        risk_result = risk_result or {}
        graph_result = graph_result or {}

        # 1. Overall model risk
        probability = risk_result.get("fraud_probability")
        threshold = risk_result.get("threshold")

        if (
            isinstance(probability, (int, float))
            and math.isfinite(probability)
        ):
            if isinstance(threshold, (int, float)) and math.isfinite(threshold):
                comparison = (
                    "meets or exceeds"
                    if probability >= threshold
                    else "is below"
                )
                evidence.append(
                    f"Model risk score ({probability:.2%}) "
                    f"{comparison} the classification threshold "
                    f"({threshold:.2%})."
                )
            else:
                evidence.append(
                    f"Model-assigned fraud probability: {probability:.2%}."
                )

        prediction = risk_result.get("prediction")
        if prediction:
            evidence.append(f"Model classification: {prediction}.")

        # 2. SHAP feature attribution
        explanation_status = risk_result.get("explanation_status")

        if explanation_status == "available":
            positive = risk_result.get("top_positive_features", []) or []
            negative = risk_result.get("top_negative_features", []) or []

            if positive:
                evidence.append(
                    "Features pushing the model output toward fraud "
                    "(positive SHAP contributions):"
                )
                for item in positive:
                    if not isinstance(item, dict):
                        continue

                    feature = item.get("feature")
                    value = item.get("shap_value")

                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue

                    if feature is not None and math.isfinite(value):
                        evidence.append(f"{feature}: {value:+.4f}")

            if negative:
                evidence.append(
                    "Features pushing the model output away from fraud "
                    "(negative SHAP contributions):"
                )
                for item in negative:
                    if not isinstance(item, dict):
                        continue

                    feature = item.get("feature")
                    value = item.get("shap_value")

                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue

                    if feature is not None and math.isfinite(value):
                        evidence.append(f"{feature}: {value:+.4f}")

            evidence.append(
                "SHAP values explain contributions to the model's raw "
                "output, not probability-point changes or causation."
            )

        elif explanation_status:
            evidence.append(
                "SHAP feature attribution is unavailable for this prediction."
            )

        # 3. Observable transaction characteristics
        row = None

        if transaction_info is not None:
            if isinstance(transaction_info, pd.DataFrame):
                if not transaction_info.empty:
                    row = transaction_info.iloc[0]
            elif isinstance(transaction_info, dict):
                row = transaction_info

        if row is not None:
            amount = row.get("TransactionAmt")

            try:
                amount = float(amount)
                if math.isfinite(amount):
                    evidence.append(
                        f"Observed transaction amount: {amount:,.2f}; "
                        "currency/unit has not been independently verified."
                    )
            except (TypeError, ValueError):
                pass

        # 4. Graph evidence
        # Shared-card proxy relationships are exploratory, not proof of fraud.
        proxy_count = graph_result.get("proxy_neighbor_count", 0)

        try:
            proxy_count = int(proxy_count)
        except (TypeError, ValueError):
            proxy_count = 0

        if proxy_count > 0:
            evidence.append(
                f"Graph analysis identified {proxy_count} shared-card "
                "proxy neighbor(s). These links are exploratory and do not "
                "confirm fraud, account ownership, or a fraud ring."
            )

        # Only report a validated graph finding when explicitly marked.
        graph_validated = (
            graph_result.get("finding_validated") is True
            and graph_result.get("graph_investigation_status") == "validated"
        )

        if graph_validated and graph_result.get("hub_detected") is True:
            evidence.append(
                "Validated graph analysis identified a hub pattern."
            )

        if (
            graph_validated
            and graph_result.get("fraud_ring_detected") is True
        ):
            evidence.append(
                "Validated graph analysis identified a potential fraud ring "
                "requiring further investigation."
            )

        graph_status = graph_result.get("graph_investigation_status")
        if graph_status:
            evidence.append(f"Graph investigation status: {graph_status}.")

        if not evidence:
            evidence.append(
                "No evidence could be generated from the available inputs."
            )

        return evidence
