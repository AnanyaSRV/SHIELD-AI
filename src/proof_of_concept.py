class ReportAgent:

    def generate_report(
        self,
        transaction_id,
        risk_score,
        graph_result,
        evidence,
        decision
    ):

        report = []

        report.append(
            f"Transaction ID: {transaction_id}"
        )

        report.append(
            f"Risk Score: {risk_score:.2f}"
        )

        report.append(
            f"Decision: {decision}"
        )

        report.append("\nEvidence:")

        if evidence:

            for item in evidence:

                report.append(
                    f"• {item}"
                )

        else:

            report.append(
                "• No evidence detected"
            )

        report.append("\nGraph Findings:")

        if graph_result:

            report.append(
                f"• Fraud Ring: "
                f"{graph_result.get('fraud_ring_detected', False)}"
            )

            report.append(
                f"• Hub Account: "
                f"{graph_result.get('hub_detected', False)}"
            )

            report.append(
                f"• Suspicious Neighbors: "
                f"{graph_result.get('suspicious_neighbors', 0)}"
            )

        return "\n".join(report)


class DecisionAgent:

    def decide(
        self,
        risk_score,
        graph_result,
        evidence
    ):

        fraud_ring = graph_result.get(
            "fraud_ring_detected",
            False
        )

        evidence_count = len(
            evidence
        )

        if fraud_ring:

            return "BLOCK"

        if (
            risk_score >= 0.80
            and evidence_count >= 3
        ):

            return "BLOCK"

        if risk_score >= 0.60:

            return "MONITOR"

        if (
            risk_score >= 0.30
            or evidence_count >= 2
        ):

            return "VERIFY"

        return "ALLOW"


if __name__ == "__main__":

    report_agent = ReportAgent()

    decision_agent = DecisionAgent()

    scenarios = [

        {
            "transaction_id": "TXN001",
            "risk_score": 0.10,
            "graph_result": {
                "fraud_ring_detected": False,
                "hub_detected": False,
                "suspicious_neighbors": 0
            },
            "evidence": [
                "Low transaction amount"
            ]
        },

        {
            "transaction_id": "TXN002",
            "risk_score": 0.68,
            "graph_result": {
                "fraud_ring_detected": False,
                "hub_detected": False,
                "suspicious_neighbors": 1
            },
            "evidence": [
                "Night-time transaction",
                "Rapid transaction behaviour"
            ]
        },

        {
            "transaction_id": "TXN003",
            "risk_score": 0.42,
            "graph_result": {
                "fraud_ring_detected": False,
                "hub_detected": True,
                "suspicious_neighbors": 4
            },
            "evidence": [
                "High transaction amount",
                "Connected to suspicious accounts"
            ]
        },

        {
            "transaction_id": "TXN004",
            "risk_score": 0.91,
            "graph_result": {
                "fraud_ring_detected": True,
                "hub_detected": True,
                "suspicious_neighbors": 8
            },
            "evidence": [
                "High transaction amount",
                "Night-time transaction",
                "Fraud ring detected"
            ]
        }

    ]

    print("=" * 60)
    print("SHIELD-AI PROOF OF CONCEPT")
    print("=" * 60)

    for scenario in scenarios:

        decision = decision_agent.decide(
            scenario["risk_score"],
            scenario["graph_result"],
            scenario["evidence"]
        )

        report = report_agent.generate_report(
            transaction_id=scenario["transaction_id"],
            risk_score=scenario["risk_score"],
            graph_result=scenario["graph_result"],
            evidence=scenario["evidence"],
            decision=decision
        )

        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)