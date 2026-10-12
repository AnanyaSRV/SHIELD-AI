
import sys
from pathlib import Path
import math

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
    except (TypeError, ValueError, OverflowError):
        return str(value)


def format_amount(value):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.2f}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def format_time_offset(value):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.0f}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def get_first_present(data, keys, default=None):
    if not isinstance(data, dict):
        return default

    for key in keys:
        value = data.get(key)
        if value is not None:
            return value

    return default


def display_labeled_value(label, value):
    if value is None:
        value = "N/A"
    st.markdown(f"**{label}:** {value}")


def find_column(dataframe, candidates):
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
        display_labeled_value(
            "Shared card1 proxy value",
            format_integer_like(
                relationship_df[shared_card_col].iloc[0]
            ),
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
# UPLOAD TRANSACTION DATA — MEMORY-SAFE ROW PROCESSING
# =========================================================

st.markdown("### Upload Transaction Data")

st.write(
    "Upload a CSV containing the transaction features expected by the "
    "trained model. The app reads the file in chunks and retains only "
    "the selected transaction for inference."
)

uploaded_file = st.file_uploader(
    "Choose a transaction CSV file",
    type=["csv"],
    key="shield_ai_transaction_upload",
)

if uploaded_file is None:
    st.info("Upload a CSV to begin fraud detection and investigation.")
    st.stop()


# =========================================================
# READ HEADER AND PREVIEW ONLY
# =========================================================

try:
    uploaded_file.seek(0)
    header_df = pd.read_csv(uploaded_file, nrows=0)
    uploaded_columns = list(header_df.columns)

    uploaded_file.seek(0)
    preview_df = pd.read_csv(uploaded_file, nrows=10)

    uploaded_file.seek(0)

except Exception as exc:
    st.error(f"Could not read the uploaded CSV header/preview: {exc}")
    st.stop()

if not uploaded_columns:
    st.error("The uploaded CSV has no columns.")
    st.stop()

st.success(
    f"CSV received: {uploaded_file.name} — "
    f"{len(uploaded_columns):,} columns. "
    "The full file has not been loaded into memory."
)

with st.expander("Preview uploaded data"):
    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True,
    )

del preview_df
del header_df


# =========================================================
# VALIDATE FEATURE SCHEMA
# =========================================================

transaction_id_column = next(
    (
        column
        for column in uploaded_columns
        if str(column).strip().lower() == "transactionid"
    ),
    None,
)

label_names = {
    "isfraud",
    "target",
    "fraud_label",
    "label",
}

label_columns = [
    column
    for column in uploaded_columns
    if str(column).strip().lower() in label_names
]

model_columns = [
    column
    for column in uploaded_columns
    if column != transaction_id_column
    and column not in label_columns
]

EXPECTED_FEATURE_COUNT = 392

if label_columns:
    st.warning(
        "Target/label columns will be excluded from model input: "
        + ", ".join(map(str, label_columns))
    )

if len(model_columns) != EXPECTED_FEATURE_COUNT:
    st.error(
        f"The trained model expects {EXPECTED_FEATURE_COUNT} features, "
        f"but this CSV has {len(model_columns)} model-input columns "
        "after excluding the transaction ID and target/label columns."
    )

    with st.expander("Inspect CSV columns"):
        st.write("CSV columns:", uploaded_columns)
        st.write("Model-input columns:", model_columns)

    st.info(
        "Please use a CSV with the feature schema expected by "
        "the trained model."
    )
    st.stop()


# =========================================================
# SELECT TRANSACTION
# =========================================================

st.markdown("### Select Transaction")

st.caption(
    "Enter the row number in the CSV, starting at 1. "
    "The header is not counted. The app searches in chunks."
)

selected_row_number = st.number_input(
    "Transaction row number",
    min_value=1,
    max_value=10_000_000,
    value=1,
    step=1,
    help="Use 1 for the first data row, 2 for the second, and so on.",
)

selected_index = int(selected_row_number) - 1

if st.button(
    "Load Selected Transaction",
    use_container_width=True,
):
    selected_row = None
    rows_seen = 0
    chunk_size = 5000

    try:
        uploaded_file.seek(0)

        for chunk in pd.read_csv(
            uploaded_file,
            chunksize=chunk_size,
            low_memory=True,
        ):
            chunk_end = rows_seen + len(chunk)

            if rows_seen <= selected_index < chunk_end:
                selected_row = chunk.iloc[
                    selected_index - rows_seen
                ].copy()
                break

            rows_seen = chunk_end

        if selected_row is None:
            st.error(
                f"Row {selected_row_number:,} was not found. "
                "The CSV may contain fewer rows than requested."
            )
            st.stop()

        selected_transaction_id = (
            selected_row[transaction_id_column]
            if transaction_id_column is not None
            else None
        )

        if (
            selected_transaction_id is not None
            and pd.isna(selected_transaction_id)
        ):
            selected_transaction_id = None

        selected_features = {
            column: selected_row[column]
            for column in model_columns
        }

        st.session_state[
            "shield_ai_selected_features"
        ] = selected_features

        st.session_state[
            "shield_ai_selected_transaction_id"
        ] = selected_transaction_id

        st.session_state[
            "shield_ai_selected_row_index"
        ] = selected_index

        st.session_state[
            "shield_ai_selected_upload_name"
        ] = uploaded_file.name

        st.session_state.pop("shield_ai_last_result", None)
        st.session_state.pop(
            "shield_ai_last_transaction_id",
            None,
        )
        st.session_state.pop(
            "shield_ai_last_row_index",
            None,
        )

        st.success(
            f"Loaded CSV data row {selected_row_number:,} "
            "for investigation."
        )

    except Exception as exc:
        st.error(f"Could not read the selected transaction: {exc}")
        st.exception(exc)


# =========================================================
# DISPLAY SELECTED TRANSACTION
# =========================================================

selected_features = st.session_state.get(
    "shield_ai_selected_features"
)

selected_transaction_id = st.session_state.get(
    "shield_ai_selected_transaction_id"
)

selected_index = st.session_state.get(
    "shield_ai_selected_row_index"
)

selected_upload_name = st.session_state.get(
    "shield_ai_selected_upload_name"
)

if (
    selected_features is None
    or selected_upload_name != uploaded_file.name
    or selected_index is None
):
    st.info(
        "Enter a row number and click **Load Selected Transaction** first."
    )
    st.stop()

col1, col2 = st.columns(2)

with col1:
    display_labeled_value(
        "Selected CSV row",
        format_integer_like(selected_index + 1),
    )

with col2:
    display_labeled_value(
        "Transaction ID",
        format_integer_like(selected_transaction_id),
    )

with st.expander("Selected model features"):
    st.json(selected_features)


# =========================================================
# RUN INVESTIGATION — PASS A DATAFRAME
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
            # IMPORTANT:
            # src/shield_ai.py requires transaction_features
            # to be a pandas DataFrame, not a dictionary.
            #
            # Preserve the selected feature names and their order.
            transaction_features_df = pd.DataFrame(
                [selected_features],
                columns=list(selected_features.keys()),
            )

            if not isinstance(
                transaction_features_df,
                pd.DataFrame,
            ):
                raise TypeError(
                    "Selected transaction could not be converted "
                    "to a pandas DataFrame."
                )

            if transaction_features_df.shape != (
                1,
                EXPECTED_FEATURE_COUNT,
            ):
                raise ValueError(
                    "Unexpected model input shape: "
                    f"{transaction_features_df.shape}. Expected "
                    f"(1, {EXPECTED_FEATURE_COUNT})."
                )

            result = shield_ai.investigate_transaction(
                transaction_features_df,
                transaction_id=selected_transaction_id,
            )

            if not isinstance(result, dict):
                raise TypeError(
                    "investigate_transaction() did not return "
                    "a dictionary."
                )

            st.session_state[
                "shield_ai_last_result"
            ] = result

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
        "Load a transaction and click **Investigate Transaction** "
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
    risk_result,
    ["fraud_probability", "probability"],
    default=None,
)

decision = get_first_present(
    decision_result,
    ["recommended_action", "action", "decision", "status"],
    default=get_first_present(
        result,
        ["action"],
        default="N/A",
    ),
)

risk_col, decision_col = st.columns(2)

with risk_col:
    try:
        score = float(risk_score)

        if math.isfinite(score) and 0 <= score <= 1:
            st.metric("Fraud Probability", f"{score:.2%}")
        else:
            st.metric("Fraud Probability", "N/A")

    except (TypeError, ValueError, OverflowError):
        st.metric("Fraud Probability", "N/A")

with decision_col:
    st.metric("Recommended Action", str(decision).upper())

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
    st.info("No separate investigator report was returned.")

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
