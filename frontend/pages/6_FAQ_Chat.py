"""
Page 6 — FAQ Chat

Ask questions about method assumptions, statistical concepts,
and experiment design best practices.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="FAQ Chat · Measurement Design Agent",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from shared import (
    METHOD_NAMES,
    init_session_state,
    render_sidebar,
    send_faq_message,
)

init_session_state()
render_sidebar()

# ── Page header ─────────────────────────────────────────────────────────────────
st.header("❓ FAQ Chat — Understand the Methods")
st.markdown(
    "Ask me anything about the measurement methods, their assumptions, "
    "statistical concepts like power and MDE, or how to interpret your results. "
    "No question is too basic!"
)
st.divider()

# ── Method filter ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 3])
with col1:
    method_options = {"All Methods": None, **METHOD_NAMES}
    selected_label = st.selectbox(
        "Focus on a specific method (optional):",
        options=list(method_options.keys()),
        index=0,
        help="Select a method to get answers focused on its specific assumptions and requirements.",
    )
    selected_method_key = method_options[selected_label]

    # Update stored method key
    if selected_method_key != st.session_state.faq_method_key:
        st.session_state.faq_method_key = selected_method_key

with col2:
    st.markdown("**Example questions you can ask:**")
    st.caption(
        "• What does 'parallel trends' mean in plain English?\n"
        "• Why does statistical power matter for my test?\n"
        "• What happens if the SUTVA assumption is violated?\n"
        "• How many markets do I need for a geo lift test?\n"
        "• What's the difference between A/B test and DiD?\n"
        "• How do I check if my synthetic control is good enough?"
    )

st.divider()

# ── Quick-start buttons ────────────────────────────────────────────────────────
if not st.session_state.faq_messages:
    st.markdown("**Quick start — pick a topic:**")
    quick_cols = st.columns(3)
    quick_questions = [
        ("📊 Statistical Power", "What is statistical power and why does it matter for my test?"),
        ("🎯 MDE Explained", "What is the Minimum Detectable Effect and how do I think about it?"),
        ("🔀 Assumptions 101", "What are the main assumptions I should worry about for experiment design?"),
        ("🗺 Geo Methods", "When should I use a geo-based method instead of an A/B test?"),
        ("⚖️ Method Comparison", "What are the key differences between the available measurement methods?"),
        ("🚨 Red Flags", "What are common red flags that can make my test unreliable?"),
    ]
    for i, (label, question) in enumerate(quick_questions):
        with quick_cols[i % 3]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                try:
                    send_faq_message(question, st.session_state.faq_method_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Chat history ────────────────────────────────────────────────────────────────
for msg in st.session_state.faq_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ──────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a question about methods, assumptions, or statistics…"):
    try:
        send_faq_message(user_input, st.session_state.faq_method_key)
        st.rerun()
    except Exception as e:
        st.error(f"Error communicating with backend: {e}")

# ── Clear chat button ──────────────────────────────────────────────────────────
if st.session_state.faq_messages:
    st.divider()
    if st.button("🗑 Clear FAQ Chat", use_container_width=False):
        st.session_state.faq_messages = []
        st.rerun()
