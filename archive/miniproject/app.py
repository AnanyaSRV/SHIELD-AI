import importlib
import sys

import streamlit as st

st.set_page_config(
    page_title="SHIELD-AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("SHIELD-AI")
st.subheader("Agentic Fraud Detection and Investigation System")

page = st.sidebar.selectbox(
    "Choose Module",
    [
        "Dashboard",
        "Fraud Detection",
        "Fraud Attribution",
        "Case Report",
    ],
)

page_modules = {
    "Dashboard": "frontend.dashboard",
    "Fraud Detection": "frontend.fraud_detector",
    "Fraud Attribution": "frontend.attribution_agent",
    "Case Report": "frontend.report_generator",
}

module_name = page_modules[page]

try:
    importlib.invalidate_caches()

    if module_name in sys.modules:
        module = sys.modules[module_name]
        try:
            module = importlib.reload(module)
        except ImportError:
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
    else:
        module = importlib.import_module(module_name)

except Exception as exc:
    st.error(f"Could not load the {page} page.")
    st.exception(exc)