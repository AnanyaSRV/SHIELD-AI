
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

    # Add the archive/miniproject directory to the import path.
    app_dir = __file__.rsplit("/", 1)[0]
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    importlib.invalidate_caches()
    importlib.import_module(module_name)

    logger.info("Successfully loaded page: %s", page)

except Exception as exc:
    logger.exception("Failed to load page: %s", page)
    st.error(f"Could not load the {page} page.")
    st.exception(exc)
    st.code(traceback.format_exc())
