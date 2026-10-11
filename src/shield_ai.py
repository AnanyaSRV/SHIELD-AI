
from pathlib import Path

import pandas as pd

from risk_agent import RiskAssessmentAgent
from evidence_agent import EvidenceAgent
from decision_agent import DecisionAgent
from report_agent import ReportAgent
from graph_agent import GraphInvestigationAgent


DEFAULT_THRESHOLD = 0.371294741468585
TOP_EXPLANATION_FEATURES = 5


class SHIELDAI:
    """
    SHIELD-AI risk assessment, exploratory graph investigation,
    evidence generation, decision support, and reporting.

    The deployed risk scorer is the tuned LightGBM model.
    Shared-card graph links are exploratory proxies and do not establish
    verified account ownership or confirm fraud rings.
    """

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        print("=" * 60)
        print("INITIALIZING SHIELD-AI")
        print("=" * 60)

        self.risk_agent = RiskAssessmentAgent(threshold=threshold)
        self.evidence_agent = EvidenceAgent()
        self.decision_agent = DecisionAgent()
        self.report_agent = ReportAgent()

        self.graph_agent = None
        self.graph_initialization_error = None

        print("Tuned LightGBM risk workflow ready.")
        print("Exploratory graph investigation initializes on demand.")

    def _get_graph_agent(self):
        """Initialize the graph agent only when needed."""
        if self.graph_agent is not None:
            return self.graph_agent

        if self.graph_initialization_error is not None:
            return None

        try:
            self.graph_agent = GraphInvestigationAgent()
            return self.graph_agent
        except Exception as exc:
            self.graph_initialization_error = str(exc)
            print(f"Graph initialization failed: {exc}")
            return None

    @staticmethod
    def _unavailable_graph_result(status):
        """Return a consistent graph result when lookup is unavailable."""
        return {
            "account_found": False,
            "transaction_found": False,
            "in_graph_sample": False,
            "proxy_relationships_found": False,
            "graph_investigation_status": status,
            "hub_detected": False,
            "fraud_ring_detected": False,
            "suspicious_neighbors": 0,
            "proxy_neighbor_count": 0,
            "proxy_neighbors": [],
            "proxy_relationships": [],
            "relationship_type": (
                "shared card1 proxy; not verified account identity"
            ),
        }

    def _investigate_graph(self, transaction_id):
        """Investigate a transaction ID without changing model predictions."""
        if transaction_id is None or pd.isna(transaction_id):
            return self._unavailable_graph_result(
                "Graph lookup not performed: no TransactionID was supplied."
            )

        agent = self._get_graph_agent()

        if agent is None:
            return self._unavailable_graph_result(
                "Graph initialization failed: "
                + str(self.graph_initialization_error)
            )

        try:
            normalized_id = transaction_id

            if isinstance(normalized_id, str):
                normalized_id = normalized_id.strip()

                try:
                    numeric_id = float(normalized_id)
                    if numeric_id.is_integer():
                        normalized_id = int(numeric_id)
                except ValueError:
                    pass

            if (
                isinstance(normalized_id, float)
                and normalized_id.is_integer()
            ):
                normalized_id = int(normalized_id)

            raw = agent.investigate_transaction(normalized_id)

            if not raw.get("transaction_found", False):
                result = self._unavailable_graph_result(
                    raw.get(
                        "status",
                        "Transaction was not found in the graph dataset.",
                    )
                )
                result["transaction_id"] = normalized_id
                return result

            neighbor_count = int(
                raw.get("proxy_neighbor_count", 0) or 0
            )
            neighbors = raw.get("proxy_neighbors", []) or []
            relationships = raw.get("proxy_relationships", []) or []

            return {
                "account_found": False,
                "transaction_found": True,
                "transaction_id": raw.get(
                    "transaction_id", normalized_id
                ),
                "in_graph_sample": bool(
                    raw.get("in_graph_sample", False)
                ),
                "proxy_relationships_found": neighbor_count > 0,
                "graph_investigation_status": raw.get(
                    "status",
                    "Exploratory proxy graph lookup completed.",
                ),
                "hub_detected": False,
                "fraud_ring_detected": False,

                # Shared-card proxy links are not confirmed suspicious activity.
                "suspicious_neighbors": 0,
                "proxy_neighbor_count": neighbor_count,
                "proxy_neighbors": neighbors,
                "proxy_relationships": relationships,
                "relationship_type": raw.get(
                    "relationship_type",
                    "shared card1 proxy; not verified account identity",
                ),
                "graph_sample_rows": len(agent.df),
                "TransactionAmt": raw.get("TransactionAmt"),
                "TransactionDT": raw.get("TransactionDT"),
            }

        except Exception as exc:
            return self._unavailable_graph_result(
                f"Graph lookup failed safely: {exc}"
            )

    @staticmethod
    def _prepare_explanations(risk_result):
        """
        Convert SHAP output into frontend-friendly records.

        SHAP values describe contributions to the model's raw output.
        They are not probability percentages or proof of causation.
        """
        if not isinstance(risk_result, dict):
            return []

        explanations = []

        groups = [
            (
                risk_result.get("top_positive_features", []) or [],
                "increases model output",
                "This feature pushes the model output toward the fraud class.",
            ),
            (
                risk_result.get("top_negative_features", []) or [],
                "decreases model output",
                "This feature pushes the model output away from the fraud class.",
            ),
        ]

        for items, direction, interpretation in groups:
            for item in items[:TOP_EXPLANATION_FEATURES]:
                if not isinstance(item, dict):
                    continue

                value = item.get("shap_value")
                if value is None:
                    continue

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                explanations.append({
                    "feature": str(item.get("feature", "Unknown feature")),
                    "contribution": value,
                    "shap_value": value,
                    "direction": direction,
                    "interpretation": interpretation,
                })

        return explanations

    @staticmethod
    def _build_fusion_summary(risk_result):
        """
        Describe the deployed scoring result honestly.

        This workflow currently uses one tuned LightGBM scorer.
        Do not claim that LR, RF, and XGBoost were fused per transaction.
        """
        probability = risk_result.get("fraud_probability")
        threshold = risk_result.get("threshold")

        return {
            "status": "Risk score available",
            "scoring_method": "Tuned LightGBM",
            "fusion_performed": False,
            "message": (
                "The current investigation uses the tuned LightGBM model. "
                "A multi-model soft-voting fusion was not performed for "
                "this transaction."
            ),
            "fraud_probability": probability,
            "classification_threshold": threshold,
            "prediction": risk_result.get("prediction", "Unknown"),
            "threshold_status": risk_result.get(
                "threshold_status", "not specified"
            ),
        }

    def _build_decision_summary(
        self, risk_result, graph_result, evidence, action
    ):
        """Explain the action returned by the DecisionAgent."""
        probability = risk_result.get("fraud_probability")
        monitor_threshold = self.decision_agent.monitor_threshold
        block_threshold = self.decision_agent.block_threshold

        if probability is None:
            reason = "The risk probability was unavailable."
        elif action == "VERIFY":
            reason = (
                "The decision policy selected VERIFY based on a "
                "validated graph finding."
            )
        elif action == "BLOCK":
            reason = (
                f"The fraud probability ({probability:.2%}) meets or "
                f"exceeds the block threshold ({block_threshold:.0%})."
            )
        elif action == "MONITOR":
            reason = (
                f"The fraud probability ({probability:.2%}) meets or "
                f"exceeds the monitor threshold ({monitor_threshold:.0%}) "
                "but is below the block threshold."
            )
        else:
            reason = (
                f"The fraud probability ({probability:.2%}) is below "
                f"the monitor threshold ({monitor_threshold:.0%})."
            )

        return {
            "recommended_action": action,
            "decision_policy": "Probability threshold policy",
            "fraud_probability": probability,
            "monitor_threshold": monitor_threshold,
            "block_threshold": block_threshold,
            "reason": reason,
            "evidence_item_count": len(evidence),
            "graph_status": graph_result.get(
                "graph_investigation_status",
                "Graph status not provided.",
            ),
            "graph_override_applied": action == "VERIFY",
            "note": (
                "These action thresholds are application policy settings; "
                "they are not independently calibrated production rules."
            ),
        }
    def investigate_transaction(
        self,
        transaction_features,
        transaction_id=None,
    ):
        """
        Assess one transaction.

        transaction_features must be a one-row DataFrame containing
        the expected model features, without TransactionID or isFraud.
        """
        if not isinstance(transaction_features, pd.DataFrame):
            raise TypeError(
                "transaction_features must be a pandas DataFrame."
            )

        if len(transaction_features) != 1:
            raise ValueError("Expected exactly one transaction row.")

        forbidden = {"isFraud", "TransactionID"} & set(
            transaction_features.columns
        )
        if forbidden:
            raise ValueError(
                "Remove these columns from model features: "
                + ", ".join(sorted(forbidden))
            )

        # 1. Risk assessment.
        risk_result = self.risk_agent.assess_risk(transaction_features)

        if not isinstance(risk_result, dict):
            raise TypeError(
                "RiskAssessmentAgent.assess_risk() must return a dictionary."
            )

        # 2. Explainability.
        explanations = self._prepare_explanations(risk_result)

        # 3. Exploratory graph lookup.
        graph_result = self._investigate_graph(transaction_id)

        # 4. Evidence collection.
        evidence = self.evidence_agent.collect_evidence(
            risk_result,
            graph_result,
            transaction_info=transaction_features,
        )

        # Correct any legacy evidence wording that labels proxy links
        # as suspicious neighbors without independent confirmation.
        corrected_evidence = []
        for item in evidence:
            text = str(item)
            if "suspicious neighbor" in text.lower():
                count = graph_result.get("proxy_neighbor_count", 0)
                text = (
                    f"Graph analysis identified {count} shared-card proxy "
                    "neighbor(s). These links are exploratory and do not "
                    "confirm fraud or account ownership."
                )
            corrected_evidence.append(text)
        evidence = corrected_evidence

        # 5. Decision.
        action = self.decision_agent.decide(
            risk_result,
            graph_result,
            evidence,
        )

        # 6. Build explicit summaries for the frontend.
        fusion_result = self._build_fusion_summary(risk_result)
        decision_result = self._build_decision_summary(
            risk_result,
            graph_result,
            evidence,
            action,
        )

        # 7. Report.
        report = self.report_agent.generate_report(
            risk_result,
            graph_result,
            evidence,
            action,
        )

        # Correct legacy report wording, if present.
        if isinstance(report, str) and "suspicious neighbor" in report.lower():
            count = graph_result.get("proxy_neighbor_count", 0)
            report = report.replace(
                f"Graph analysis reported {count} suspicious neighbor(s).",
                (
                    f"Graph analysis identified {count} shared-card proxy "
                    "neighbor(s). These links are exploratory and do not "
                    "confirm fraud or account ownership."
                ),
            )

        # Keep original fields and add the missing frontend fields.
        return {
            "risk_result": risk_result,
            "graph_result": graph_result,
            "evidence": evidence,
            "action": action,
            "report": report,
            "fusion_result": fusion_result,
            "decision_result": decision_result,
            "decision": decision_result,
            "explanations": explanations,
            "feature_attributions": risk_result.get(
                "feature_attributions", []
            ),
            "top_positive_features": risk_result.get(
                "top_positive_features", []
            ),
            "top_negative_features": risk_result.get(
                "top_negative_features", []
            ),
            "explanation_status": risk_result.get(
                "explanation_status", "unavailable"
            ),
            "explanation_method": risk_result.get(
                "explanation_method", "SHAP TreeExplainer"
            ),
            "explanation_scale": risk_result.get(
                "explanation_scale",
                "raw model output, not probability points",
            ),
            "explanation_error": risk_result.get("explanation_error"),
        }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "train_transaction.csv"

    print("Loading one transaction for an end-to-end smoke test...")

    sample = pd.read_csv(data_path, nrows=1, low_memory=False)

    transaction_id = (
        sample["TransactionID"].iloc[0]
        if "TransactionID" in sample.columns
        else None
    )

    sample = sample.drop(
        columns=["isFraud", "TransactionID"],
        errors="ignore",
    )

    shield_ai = SHIELDAI()
    result = shield_ai.investigate_transaction(
        sample,
        transaction_id=transaction_id,
    )

    print("\nRISK RESULT")
    print(result["risk_result"])

    print("\nFUSION / SCORING SUMMARY")
    print(result["fusion_result"])

    print("\nDECISION SUMMARY")
    print(result["decision_result"])

    print("\nGRAPH RESULT")
    print(result["graph_result"])

    print("\nEVIDENCE")
    for item in result["evidence"]:
        print(f"- {item}")

    print("\nFINAL REPORT")
    print(result["report"])
