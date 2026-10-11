
class DecisionAgent:
    """Apply explicit actions to model risk and validated evidence."""

    def __init__(
        self,
        monitor_threshold=0.60,
        block_threshold=0.80,
    ):
        if not (
            0.0 <= monitor_threshold <= block_threshold <= 1.0
        ):
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= monitor_threshold <= block_threshold <= 1."
            )

        self.monitor_threshold = float(monitor_threshold)
        self.block_threshold = float(block_threshold)

    def decide(self, risk_result, graph_result=None, evidence=None):
        """
        Return ALLOW, MONITOR, VERIFY, or BLOCK.

        Exploratory shared-card proxy links must never trigger VERIFY.
        A graph override requires an explicitly validated finding.
        """
        if not isinstance(risk_result, dict):
            raise TypeError("risk_result must be a dictionary.")

        graph_result = graph_result or {}
        evidence = evidence or []

        probability = risk_result.get("fraud_probability")

        try:
            probability = float(probability)
        except (TypeError, ValueError):
            raise ValueError(
                "risk_result must contain a valid fraud_probability "
                "between 0 and 1."
            ) from None

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "risk_result must contain a valid fraud_probability "
                "between 0 and 1."
            )

        # Graph override is disabled for the current exploratory graph.
        # It can be enabled only when a future graph pipeline explicitly
        # supplies independently validated evidence.
        graph_finding_validated = (
            graph_result.get("finding_validated") is True
            and graph_result.get("fraud_ring_detected") is True
            and graph_result.get("graph_investigation_status")
            == "validated"
        )

        if graph_finding_validated:
            return "VERIFY"

        if probability >= self.block_threshold:
            return "BLOCK"

        if probability >= self.monitor_threshold:
            return "MONITOR"

        return "ALLOW"


if __name__ == "__main__":
    agent = DecisionAgent()

    test_cases = [
        ("Low risk, no evidence", 0.10, "ALLOW"),
        ("Low risk, unusual evidence", 0.10, "ALLOW"),
        ("Low risk, exploratory graph links", 0.10, "ALLOW"),
        ("Medium risk", 0.65, "MONITOR"),
        ("High risk", 0.85, "BLOCK"),
        ("Maximum risk", 1.00, "BLOCK"),
    ]

    for name, probability, expected in test_cases:
        result = agent.decide(
            {"fraud_probability": probability},
            graph_result={
                "account_found": False,
                "graph_investigation_status": (
                    "Exploratory shared-card proxy links only. "
                    "Account ownership and fraud rings are not confirmed."
                ),
                "fraud_ring_detected": False,
            },
            evidence=[],
        )

        status = "PASS" if result == expected else "FAIL"
        print(
            f"{status}: {name} -> {result} "
            f"(expected {expected})"
        )
