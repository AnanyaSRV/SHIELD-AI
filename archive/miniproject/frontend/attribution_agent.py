import streamlit as st


def render_attribution():
    st.header("Fraud Attribution")
    st.caption(
        "Review the model's risk assessment and the evidence returned "
        "by SHIELD-AI."
    )

    result = st.session_state.get("shield_ai_last_result")

    if result is None:
        st.info(
            "No investigation is available yet. Open Fraud Detection, "
            "upload a compatible IEEE-CIS CSV, and investigate a transaction."
        )
        st.button(
            "Go to Fraud Detection",
            on_click=lambda: st.session_state.update(
                {"Choose Module": "Fraud Detection"}
            ),
        )
        return

    risk = result.get("risk_result", {})
    evidence = result.get("evidence", [])
    graph = result.get("graph_result", {})
    action = result.get("action", "Not available")

    probability = risk.get("fraud_probability")
    prediction = risk.get("prediction", "Not available")
    threshold = risk.get("threshold")

    st.subheader("Risk assessment")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Fraud probability",
        f"{probability:.2%}" if isinstance(probability, (int, float)) else "N/A",
    )
    col2.metric("Prediction", str(prediction))
    col3.metric("Recommended action", str(action))

    if isinstance(threshold, (int, float)):
        st.caption(f"Classification threshold: {threshold:.6f}")

    st.divider()
    st.subheader("Evidence returned by the agent")

    if isinstance(evidence, (list, tuple)) and evidence:
        for item in evidence:
            st.write(f"• {item}")
    elif isinstance(evidence, dict) and evidence:
        st.json(evidence)
    else:
        st.info(
            "The evidence agent returned no significant evidence for "
            "this investigation."
        )

    st.divider()
    st.subheader("Graph investigation status")

    graph_status = graph.get(
        "graph_investigation_status",
        "No graph investigation status was returned.",
    )
    st.info(graph_status)

    st.write(
        "**Fraud ring detected:**",
        "Yes" if graph.get("fraud_ring_detected") else "No reported finding",
    )
    st.write(
        "**Hub detected:**",
        "Yes" if graph.get("hub_detected") else "No reported finding",
    )
    st.write(
        "**Suspicious neighbors:**",
        graph.get("suspicious_neighbors", "Not available"),
    )

    st.info(
    "Graph investigation uses exploratory shared-card proxy relationships. "
    "Account ownership and fraud rings have not been independently "
    "validated. A lack of reported findings does not prove that a "
    "transaction is safe."
    )  

    st.divider()
    st.subheader("Risk result details")
    st.json(risk)


render_attribution()