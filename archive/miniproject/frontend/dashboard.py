
import importlib
import logging
import sys
from pathlib import Path

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shield_ai")

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

    # Dashboard modules render their content during import.
    # Other modules may expose a render function instead.
    if page != "Dashboard":
        render_function = getattr(module, "render_page", None)

        if callable(render_function):
            render_function()

    logger.info("Successfully loaded page: %s", page)

except Exception as exc:
    logger.exception("Failed to load page: %s", page)
    st.error(f"Could not load the {page} page.")
    st.exception(exc)
