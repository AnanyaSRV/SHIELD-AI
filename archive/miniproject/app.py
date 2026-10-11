```python
import importlib
import logging
import sys

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
    key="Choose Module",
)

module_name = page_modules[page]

try:
    logger.info("Loading page: %s (%s)", page, module_name)

    importlib.invalidate_caches()

    # Reload modules on subsequent Streamlit reruns so that
    # pages which render at module level continue to work.
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)

    # The dashboard uses an explicit rendering function.
    if page == "Dashboard":
        render_function = getattr(module, "render_dashboard", None)

        if not callable(render_function):
            raise AttributeError(
                "frontend.dashboard must define render_dashboard()."
            )

        render_function()

    # These existing modules render their content when imported.
    # Do not call their rendering functions a second time.
    elif page == "Fraud Detection":
        pass

    elif page == "Fraud Attribution":
        pass

    elif page == "Case Report":
        pass

    logger.info("Successfully loaded page: %s", page)

except Exception as exc:
    logger.exception("Failed to load page: %s", page)
    st.error(f"Could not load the {page} page.")
    st.exception(exc)
```