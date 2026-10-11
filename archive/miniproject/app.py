
import importlib
import logging
import sys
import traceback

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shield_ai")

st.set_page_config(
    page_title="SHIELD-AI",
    page_icon="🛡️",
    layout="wide",
)

logger.info("SHIELD-AI application startup")

st.title("SHIELD-AI")
st.subheader("Agentic Fraud Detection and Investigation System")

page_modules = {
    "Dashboard": "frontend.dashboard",
    "Fraud Detection": "frontend.fraud_detector",
    "Fraud Attribution": "frontend.attribution_agent",
    "Case Report": "frontend.report_generator",
}

page = st.sidebar.selectbox(
    "Choose Module",
    list(page_modules.keys()),
)

module_name = page_modules[page]


try:
    logger.info("Loading page: %s (%s)", page, module_name)

    importlib.invalidate_caches()
    module = importlib.import_module(module_name)

    if page == "Dashboard":
        render_function = getattr(module, "render_dashboard", None)
    else:
        render_function = getattr(module, "render_page", None)

    if callable(render_function):
        render_function()
    elif page != "Dashboard":
        st.warning(f"The {page} module does not expose a render_page() function.")

    logger.info("Successfully loaded page: %s", page)

except Exception as exc:
    logger.exception("Failed to load page: %s", page)
    st.error(f"Could not load the {page} page.")
    st.exception(exc)