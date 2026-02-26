"""
Page 1 — Elicitation Chat

Conversational interview covering 7 topics to learn about the user's
campaign and recommend measurement methods.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Elicitation Chat · Measurement Design Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from shared import (
    init_session_state,
    render_sidebar,
    require_session,
    send_message,
)

init_session_state()
render_sidebar()

# ── Page header ─────────────────────────────────────────────────────────────────
st.header("💬 Elicitation Chat")
st.caption(
    "I'll ask about your campaign goals, data availability, "
    "and constraints to recommend the best measurement approach."
)

if not require_session():
    st.stop()

# ── Guard: if setup is active, redirect ────────────────────────────────────────
if st.session_state.setup_active:
    st.info("Elicitation is complete. Head to **Method Setup** to continue.")
    st.stop()

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Elicitation done banner ────────────────────────────────────────────────────
if st.session_state.done:
    st.success(
        "✅ Elicitation complete! Navigate to **Design Report** to view "
        "your results and method rankings."
    )
    st.stop()

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Your answer…"):
    try:
        send_message(user_input)
        st.rerun()
    except Exception as e:
        st.error(f"Error communicating with backend: {e}")
