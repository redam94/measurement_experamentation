"""
🔬 Measurement Design Agent — Home
"""
from __future__ import annotations

import streamlit as st

# ── Must be called before any other st.* call ──────────────────────────────────
st.set_page_config(
    page_title="Measurement Design Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared helpers live next to this file
from shared import init_session_state, render_sidebar, start_session

init_session_state()
render_sidebar()

# ── Landing page ───────────────────────────────────────────────────────────────
st.title("🔬 Measurement Design Agent")
st.markdown(
    "Design rigorous experiments to measure the effectiveness of your ad "
    "campaigns — no statistics background needed."
)

st.divider()

# ── Workflow overview ──────────────────────────────────────────────────────────
st.subheader("How It Works")

cols = st.columns(5)
steps = [
    ("1️⃣", "Elicitation Chat",     "Answer a few questions about your campaign goals, data, and constraints."),
    ("2️⃣", "Design Report",        "Receive a ranked comparison of measurement methods with detailed specs."),
    ("3️⃣", "Method Setup",         "Pick a method and provide baseline metrics, expected effects, and design choices."),
    ("4️⃣", "Simulation Results",   "View interactive power curves, MDE analysis, synthetic data, and validation."),
    ("5️⃣", "Data Templates",       "Download CSV templates and schema docs to prepare your real data."),
]

for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"### {icon}")
        st.markdown(f"**{title}**")
        st.caption(desc)

st.divider()

# ── Quick start ────────────────────────────────────────────────────────────────
if not st.session_state.session_id:
    st.info(
        "Click **Start New Session** in the sidebar to begin, then navigate "
        "to the **Elicitation Chat** page."
    )
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("🆕 Start New Session", type="primary", use_container_width=True):
            try:
                start_session()
                st.rerun()
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
else:
    # Show current session status
    st.success(f"Session active: `{st.session_state.session_id[:8]}…`")

    status_cols = st.columns(4)
    with status_cols[0]:
        n_covered = len(st.session_state.covered_topics or [])
        st.metric("Elicitation Topics", f"{n_covered} / 7")
    with status_cols[1]:
        st.metric("Elicitation Complete", "✅" if st.session_state.done else "⏳")
    with status_cols[2]:
        method = st.session_state.chosen_method
        from shared import METHOD_NAMES
        st.metric("Chosen Method", METHOD_NAMES.get(method, "—") if method else "—")
    with status_cols[3]:
        st.metric("Setup Complete", "✅" if st.session_state.setup_done else "⏳" if st.session_state.setup_active else "—")

    st.divider()

    # Navigation hints
    if not st.session_state.done:
        st.info("👉 Head to **Elicitation Chat** to continue the conversation.")
    elif not st.session_state.setup_active:
        st.info("👉 Head to **Design Report** to review results, then **Method Setup** to configure your experiment.")
    elif not st.session_state.setup_done:
        st.info("👉 Continue the conversation on the **Method Setup** page.")
    else:
        st.info("👉 Explore **Simulation Results** for interactive charts and **Data Templates** for CSV schemas.")
