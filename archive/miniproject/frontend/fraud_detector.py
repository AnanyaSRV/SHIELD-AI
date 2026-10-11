
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PATH CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

for directory in (PROJECT_ROOT, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


# =========================================================
# SHIELD-AI IMPORT
# =========================================================

try:
    from shield_ai import SHIELDAI
except ImportError as exc:
    SHIELDAI = None
    SHIELD_IMPORT_ERROR = str(exc)
else:
    SHIELD_IMPORT_ERROR = None


# =========================================================
# PAGE CONTENT
# app.py manages st.set_page_config().
# Do not call it again in this imported module.
# =========================================================

st.title("🛡️ SHIELD-AI Fraud Detection")
st.caption(
    "Multi-agent digital payment fraud investigation "
    "with model explanations and exploratory graph analysis."
)


# =========================================================
# INITIALIZE SHIELD-AI
# =========================================================

@st.cache_resource
def load_shield_ai():
    if SHIELDAI is None:
        raise ImportError(
            SHIELD_IMPORT_ERROR or "Unable to import SHIELDAI."
        )

    return SHIELDAI()


try:
    shield_ai = load_shield_ai()
except Exception as exc:
    st.error(f"Could not initialize SHIELD-AI: {exc}")
    st.info(
        "Check your model artifacts, preprocessing files, "
        "and graph-agent configuration."
    )
    st.stop()


# =========================================================
# DISPLAY HELPERS
# =========================================================

def format_integer_like(value):
    """Format integer-like numeric values without trailing .0."""
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass

    try:
        numeric = float(value)

        if numeric.is_integer():
            return str(int(numeric))

        return f"{numeric:g}"

    except (TypeError, ValueError):
        return str(value)


def format_amount(value):
    """Format a transaction amount."""
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"

        return f"{float(value):,.2f}"

    except (TypeError, ValueError):
        return str(value)


def format_time_offset(value):
    """Format TransactionDT as a relative seconds offset."""
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"

        return f"{float(value):,.0f}"

    except (TypeError, ValueError):
        return str(value)


def get_first_present(data, keys, default=None):
    """Get the first non-None value from a dictionary."""
    if not isinstance(data, dict):
        return default

    for key in keys:
        value = data.get(key)

        if value is not None:
            return value

    return default


def display_labeled_value(label, value):
    """Display a labeled value."""
    if value is None:
        value = "N/A"

    st.markdown(f"**{label}:** {value}")


def find_column(dataframe, candidates):
    """Find a column using case-insensitive matching."""
    if dataframe.empty:
        return None

    lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        column = lookup.get(candidate.strip().lower())

        if column is not None:
            return column

    return None


# =========================================================
# GRAPH RELATIONSHIP DISPLAY
# =========================================================

def normalize_relationships(graph_result):
    """Convert graph relationships into a DataFrame."""
    relationships = get_first_present(
        graph_result,
        [
            "proxy_relationships",
            "relationships",
            "linked_transactions",
        ],
        default=[],
    )

    if isinstance(relationships, pd.DataFrame):
        return relationships.copy()

    if isinstance(relationships, dict):
        relationships = [relationships]

    if not isinstance(relationships, (list, tuple)):
        return pd.DataFrame()

    if not relationships:
        return pd.DataFrame()

    return pd.DataFrame(relationships)


def display_linked_transactions(graph_result):
    """Display proxy-linked transaction records."""
    st.markdown("#### Linked transaction details")

    relationship_df = normalize_relationships(graph_result)

    if relationship_df.empty:
        st.info(
            "No linked transaction relationship records "
            "were returned for this transaction."
        )
        return

    transaction_id_col = find_column(
        relationship_df,
        [
            "Transaction ID",
            "TransactionID",
            "transaction_id",
            "neighbor_transaction_id",
            "proxy_neighbor_id",
            "neighbor_id",
        ],
    )

    shared_card_col = find_column(
        relationship_df,
        [
            "Shared card1 proxy",
            "shared_card1",
            "shared_card1_proxy",
            "card1",
        ],
    )

    amount_col = find_column(
        relationship_df,
        [
            "Transaction Amount",
            "TransactionAmt",
            "transaction_amount",
            "amount",
        ],
    )

    time_col = find_column(
        relationship_df,
        [
            "Transaction Time Offset (seconds)",
            "TransactionDT",
            "transaction_time",
            "time_offset",
        ],
    )

    if shared_card_col is not None:
        shared_card_value = relationship_df[
            shared_card_col
        ].iloc[0]

        display_labeled_value(
            "Shared card1 proxy value",
            format_integer_like(shared_card_value),
        )

    if transaction_id_col is None:
        st.warning(
            "No recognized transaction ID column was found "
            "in the graph relationship records."
        )

        st.write(relationship_df.to_dict(orient="records"))
        return

    for _, row in relationship_df.iterrows():
        transaction_id = format_integer_like(
            row[transaction_id_col]
        )

        shared_card = (
            format_integer_like(row[shared_card_col])
            if shared_card_col is not None
            else "N/A"
        )

        amount = (
            format_amount(row[amount_col])
            if amount_col is not None
            else "N/A"
        )

        time_offset = (
            format_time_offset(row[time_col])
            if time_col is not None
            else "N/A"
        )

        st.markdown(
            f"- **Transaction ID:** `{transaction_id}`"
            f" | **Shared card1:** `{shared_card}`"
            f" | **Amount:** `{amount}`"
            f" | **Time offset:** `{time_offset}` seconds"
        )


def display_graph_investigation(graph_result):
    """Display graph investigation information."""
    st.markdown("---")
    st.subheader("Graph Investigation")

    st.caption(
        "Exploratory shared-card proxy links only. "
        "Shared attributes do not establish verified account ownership "
        "or confirm a fraud ring."
    )

    if not isinstance(graph_result, dict):
        st.info("No graph investigation result is available.")
        return

    transaction_found = get_first_present(
        graph_result,
        ["transaction_found", "found"],
        default=None,
    )

    in_graph_sample = get_first_present(
        graph_result,
        ["in_graph_sample"],
        default=None,
    )

    neighbor_count = get_first_present(
        graph_result,
        [
            "proxy_neighbor_count",
            "neighbor_count",
            "num_neighbors",
        ],
        default=0,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        value = (
            "Yes" if transaction_found else "No"
        ) if transaction_found is not None else "Unknown"

        st.metric("Transaction found in graph", value)

    with col2:
        value = (
            "Yes" if in_graph_sample else "No"
        ) if in_graph_sample is not None else "Unknown"

        st.metric("In graph sample", value)

    with col3:
        st.metric(
            "Shared-card proxy neighbors",
            format_integer_like(neighbor_count),
        )

    relationship_type = get_first_present(
        graph_result,
        ["relationship_types", "relationship_type"],
        default=None,
    )

    if relationship_type:
        if isinstance(relationship_type, (list, tuple)):
            relationship_type = ", ".join(
                str(item) for item in relationship_type
            )

        display_labeled_value(
            "Relationship type",
            relationship_type,
        )
    else:
        display_labeled_value(
            "Relationship type",
            "Shared card1 proxy; not verified account identity",
        )

    selected_amount = get_first_present(
        graph_result,
        ["TransactionAmt", "transaction_amount", "amount"],
        default=None,
    )

    if selected_amount is not None:
        display_labeled_value(
            "Selected transaction amount",
            format_amount(selected_amount),
        )

    selected_time = get_first_present(
        graph_result,
        ["TransactionDT", "transaction_time", "time_offset"],
        default=None,
    )

    if selected_time is not None:
        display_labeled_value(
            "Selected transaction time offset",
            f"{format_time_offset(selected_time)} seconds",
        )

    offline_label = get_first_present(
        graph_result,
        ["offline_dataset_label"],
        default=None,
    )

    if offline_label is not None:
        st.caption(
            "Offline dataset label (research reference only): "
            f"{format_integer_like(offline_label)}"
        )

    display_linked_transactions(graph_result)


# =========================================================
# UPLOAD TRANSACTION DATA
# =========================================================

st.markdown("### Upload Transaction Data")

st.write(
    "Upload a CSV containing the transaction features expected "
    "by the trained model. TransactionID is retained for investigation "
    "but excluded from model features."
)

uploaded_file = st.file_uploader(
    "Choose a transaction CSV file",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "Upload a CSV to begin fraud detection and investigation."
    )
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if uploaded_df.empty:
    st.error("The uploaded CSV contains no transactions.")
    st.stop()

st.success(
    f"Loaded {len(uploaded_df):,} transactions "
    f"with {len(uploaded_df.columns):,} columns."
)

with st.expander("Preview uploaded data"):
    st.write(f"Rows: {len(uploaded_df):,}")
    st.write(f"Columns: {len(uploaded_df.columns):,}")

    st.dataframe(
        uploaded_df.head(10),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# SEPARATE TRANSACTION IDs FROM MODEL FEATURES
# =========================================================

transaction_id_column = next(
    (
        column
        for column in uploaded_df.columns
        if str(column).strip().lower() == "transactionid"
    ),
    None,
)

if transaction_id_column is not None:
    transaction_ids = uploaded_df[
        transaction_id_column
    ].reset_index(drop=True)

    model_input_df = uploaded_df.drop(
        columns=[transaction_id_column]
    ).copy()
else:
    transaction_ids = pd.Series(
        [None] * len(uploaded_df),
        dtype=object,
    )

    model_input_df = uploaded_df.copy()


# =========================================================
# EXCLUDE TARGET LABELS FROM MODEL INPUT
# =========================================================

label_names = {
    "isfraud",
    "target",
    "fraud_label",
    "label",
}

label_columns = [
    column
    for column in model_input_df.columns
    if str(column).strip().lower() in label_names
]

if label_columns:
    st.warning(
        "Target/label columns were excluded from model input: "
        + ", ".join(map(str, label_columns))
    )

    model_input_df = model_input_df.drop(
        columns=label_columns
    )


# =========================================================
# VALIDATE MODEL FEATURES
# =========================================================

EXPECTED_FEATURE_COUNT = 392

if len(model_input_df.columns) != EXPECTED_FEATURE_COUNT:
    st.error(
        f"The model expects {EXPECTED_FEATURE_COUNT} features, "
        f"but the uploaded file contains "
        f"{len(model_input_df.columns)} model-input columns "
        "after excluding transaction IDs and labels."
    )

    st.write(
        "Available model-input columns:",
        list(model_input_df.columns),
    )

    st.info(
        "Use the feature schema expected by the trained model."
    )
    st.stop()


# =========================================================
# SELECT TRANSACTION
# =========================================================

st.markdown("### Select Transaction")

selected_index = st.selectbox(
    "Choose a transaction row",
    options=list(range(len(model_input_df))),
    format_func=lambda index: (
        f"Row {index + 1} | Transaction ID: "
        f"{format_integer_like(transaction_ids.iloc[index])}"
        if transaction_ids.iloc[index] is not None
        else f"Row {index + 1}"
    ),
)

# IMPORTANT:
# SHIELDAI.investigate_transaction() requires a pandas DataFrame.
# Use double brackets to retain a one-row DataFrame instead of
# converting the selected row to a dictionary.
selected_features = model_input_df.iloc[
    [selected_index]
].copy()

selected_transaction_id = transaction_ids.iloc[
    selected_index
]

col1, col2 = st.columns(2)

with col1:
    display_labeled_value(
        "Selected row",
        format_integer_like(selected_index + 1),
    )

with col2:
    display_labeled_value(
        "Transaction ID",
        format_integer_like(selected_transaction_id),
    )

with st.expander("Selected model features"):
    st.dataframe(
        selected_features,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# RUN INVESTIGATION
# =========================================================

if st.button(
    "Investigate Transaction",
    type="primary",
    use_container_width=True,
):
    with st.spinner(
        "Running risk assessment and graph investigation..."
    ):
        try:
            # Pass a DataFrame, not a dictionary.
            result = shield_ai.investigate_transaction(
                selected_features,
                transaction_id=selected_transaction_id,
            )

            if not isinstance(result, dict):
                raise TypeError(
                    "investigate_transaction() did not return "
                    "a dictionary."
                )

            st.session_state["shield_ai_last_result"] = result

            st.session_state[
                "shield_ai_last_transaction_id"
            ] = selected_transaction_id

            st.session_state[
                "shield_ai_last_row_index"
            ] = selected_index

        except Exception as exc:
            st.error(
                f"Transaction investigation failed: {exc}"
            )
            st.exception(exc)


# =========================================================
# DISPLAY RESULTS
# =========================================================

result = st.session_state.get("shield_ai_last_result")

if result is None:
    st.info(
        "Select a transaction and click Investigate Transaction "
        "to view the investigation results."
    )
    st.stop()

st.markdown("---")
st.header("Investigation Results")


# =========================================================
# RISK ASSESSMENT
# =========================================================

st.subheader("Risk Assessment")

risk_result = get_first_present(
    result,
    ["risk_assessment", "risk_result", "risk"],
    default={},
)

if not isinstance(risk_result, dict):
    risk_result = {"result": risk_result}

fusion_result = get_first_present(
    result,
    ["fusion_result", "fusion", "risk_fusion"],
    default={},
)

if not isinstance(fusion_result, dict):
    fusion_result = {"result": fusion_result}

decision_result = get_first_present(
    result,
    ["decision", "decision_result", "decision_output"],
    default={},
)

if not isinstance(decision_result, dict):
    decision_result = {"decision": decision_result}

risk_score = get_first_present(
    fusion_result,
    [
        "risk_score",
        "final_risk_score",
        "confidence",
        "score",
    ],
    default=get_first_present(
        risk_result,
        [
            "risk_score",
            "fraud_probability",
            "probability",
            "score",
        ],
        default=None,
    ),
)

decision = get_first_present(
    decision_result,
    [
        "action",
        "decision",
        "recommended_action",
        "status",
    ],
    default=get_first_present(
        result,
        ["decision_label", "action"],
        default="N/A",
    ),
)

risk_col, decision_col = st.columns(2)

with risk_col:
    if risk_score is not None:
        try:
            score = float(risk_score)

            if 0 <= score <= 1:
                st.metric("Risk Score", f"{score:.4f}")
            else:
                st.metric("Risk Score", f"{score:.2f}")

        except (TypeError, ValueError):
            st.metric("Risk Score", str(risk_score))
    else:
        st.metric("Risk Score", "N/A")

with decision_col:
    st.metric(
        "Recommended Action",
        str(decision).upper(),
    )

with st.expander("Risk Agent Output"):
    st.json(risk_result)

with st.expander("Fusion Agent Output"):
    st.json(fusion_result)

with st.expander("Decision Agent Output"):
    st.json(decision_result)


# =========================================================
# GRAPH INVESTIGATION
# =========================================================

graph_result = get_first_present(
    result,
    [
        "graph_investigation",
        "graph_result",
        "graph",
        "graph_evidence",
    ],
    default={},
)

display_graph_investigation(graph_result)


# =========================================================
# FRAUD ATTRIBUTION AND EXPLANATIONS
# =========================================================

st.markdown("---")
st.subheader("Fraud Attribution and Explanations")

explanations = get_first_present(
    result,
    [
        "explanations",
        "feature_importance",
        "shap_values",
        "attributions",
        "explanation",
    ],
    default=None,
)

if explanations is None:
    explanations = get_first_present(
        risk_result,
        [
            "explanations",
            "feature_importance",
            "shap_values",
            "attributions",
            "explanation",
        ],
        default=None,
    )

if explanations is None:
    st.info(
        "No separate feature-attribution output was returned "
        "for this transaction."
    )

elif isinstance(explanations, pd.DataFrame):
    st.dataframe(
        explanations,
        use_container_width=True,
        hide_index=True,
    )

elif isinstance(explanations, (list, tuple)):
    for item in explanations:
        if isinstance(item, dict):
            feature = item.get(
                "feature",
                item.get("name", "Feature"),
            )

            contribution = item.get(
                "contribution",
                item.get(
                    "importance",
                    item.get("value", "N/A"),
                ),
            )

            st.markdown(
                f"- **{feature}:** {contribution}"
            )
        else:
            st.write(f"- {item}")

elif isinstance(explanations, dict):
    st.json(explanations)

else:
    st.write(explanations)


# =========================================================
# EVIDENCE
# =========================================================

st.markdown("---")
st.subheader("Investigation Evidence")

evidence = get_first_present(
    result,
    [
        "evidence",
        "evidence_result",
        "evidence_items",
        "evidence_agent",
    ],
    default=None,
)

if evidence is None:
    st.info("No separate evidence records were returned.")

elif isinstance(evidence, (list, tuple)):
    for index, item in enumerate(evidence, start=1):
        with st.expander(f"Evidence item {index}"):
            if isinstance(item, dict):
                st.json(item)
            else:
                st.write(item)

elif isinstance(evidence, dict):
    st.json(evidence)

else:
    st.write(evidence)


# =========================================================
# INVESTIGATOR REPORT
# =========================================================

st.markdown("---")
st.subheader("Investigator Report")

report = get_first_present(
    result,
    [
        "report",
        "investigation_report",
        "report_result",
        "summary",
        "report_agent",
    ],
    default=None,
)

if report is None:
    st.info(
        "No separate investigator report was returned."
    )

elif isinstance(report, dict):
    st.json(report)

elif isinstance(report, (list, tuple)):
    for item in report:
        st.write(item)

else:
    st.markdown(str(report))


# =========================================================
# DEBUGGING OUTPUT
# =========================================================

with st.expander("Full SHIELD-AI Result (Debugging)"):
    st.json(result)
