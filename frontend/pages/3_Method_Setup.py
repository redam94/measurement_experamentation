"""
Page 3 — Method Setup

Pick one of the recommended methods from the elicitation results,
then complete the setup conversation (baseline metrics, effects, design choices).
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Method Setup · Measurement Design Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

from shared import (
    METHOD_NAMES,
    init_session_state,
    render_sidebar,
    require_report,
    require_session,
    send_setup_message,
    start_setup,
    stream_setup_message,
)

init_session_state()
render_sidebar()

# ── Page header ─────────────────────────────────────────────────────────────────
st.header("🚀 Method Setup")

if not require_session():
    st.stop()
if not require_report():
    st.stop()

# ── Method selection (if setup hasn't started yet) ─────────────────────────────
if not st.session_state.setup_active:
    st.markdown(
        "Select one of the recommended methods below to proceed with experiment "
        "setup. I'll help you calculate sample sizes, find the minimum detectable "
        "effect, and generate synthetic data to test everything before launch."
    )
    st.divider()

    ranked = st.session_state.report.get("ranked_methods", [])
    cols = st.columns(min(len(ranked), 3)) if ranked else []

    for i, item in enumerate(ranked):
        col_idx = i % 3
        with cols[col_idx]:
            method_key = item.get("key", "")
            method_name = METHOD_NAMES.get(method_key, method_key.replace("_", " ").title())
            score = item.get("score", 0)

            st.markdown(f"**#{item['rank']} {method_name}**")
            st.progress(score / 100)
            st.caption(f"Score: {score}/100")

            if st.button(
                f"Set up {method_name}",
                key=f"setup_{method_key}",
                use_container_width=True,
            ):
                try:
                    start_setup(method_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error starting setup: {e}")

    st.stop()

# ── Setup conversation ─────────────────────────────────────────────────────────
method_name = METHOD_NAMES.get(
    st.session_state.chosen_method, st.session_state.chosen_method or ""
)
st.caption(f"Configuring: **{method_name}**")
st.divider()

# Render chat history
for msg in st.session_state.setup_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Setup done banner
if st.session_state.setup_done:
    st.success(
        "🎉 Experiment setup complete!  "
        "Head to **Simulation Results** for interactive charts."
    )
    st.stop()

# Chat input
if user_input := st.chat_input("Your answer…"):
    # Show user message immediately
    st.session_state.setup_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response
    try:
        with st.chat_message("assistant"):
            response_text = st.write_stream(stream_setup_message(user_input))
        st.session_state.setup_messages.append({"role": "assistant", "content": response_text})
        st.rerun()
    except Exception as e:
        st.error(f"Error communicating with backend: {e}")
