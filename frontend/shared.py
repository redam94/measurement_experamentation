"""
Shared constants, helpers, and session-state initialisation used by all pages.
"""
from __future__ import annotations

import json
import os
from typing import Any, Generator

import httpx
import streamlit as st

# Import domain knowledge from core library
from measurement_design.knowledge.schemas import (
    METHOD_NAMES,
    METHOD_SCHEMAS,
    PANEL_METHODS,
    ALL_TOPICS,
    ALL_SETUP_TOPICS,
)

# ── API base URL ────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ── Topic labels with emojis (UI presentation layer) ─────────────────────────
TOPIC_LABELS = {
    "objective":         "\U0001F3AF Objective",
    "randomization":     "\U0001F500 Randomisation",
    "data_history":      "\U0001F4C5 Data History",
    "geo_structure":     "\U0001F5FA  Geography",
    "treatment_control": "\U0001F39B  Treatment Control",
    "covariates":        "\U0001F4CA Covariates",
    "scale":             "\U0001F4CF Scale & Duration",
}

SETUP_TOPIC_LABELS = {
    "baseline_metrics":   "\U0001F4C8 Baseline Metrics",
    "expected_effect":    "\U0001F3AF Expected Effect",
    "statistical_design": "\u2699\uFE0F Statistical Design",
    "method_specific":    "\U0001F527 Method Details",
}

# ── Computation phases (setup pipeline) ─────────────────────────────────────────
COMPUTATION_PHASES = [
    ("power_analysis",  "\U0001F4CA Power Analysis"),
    ("mde_simulation",  "\U0001F3AF MDE Simulation"),
    ("synthetic_gen",   "\U0001F9EA Synthetic Data"),
    ("validation",      "\u2705 Code Validation"),
    ("review_results",  "\U0001F50D Results Review"),
    ("setup_output",    "\U0001F4CB Setup Report"),
]


# ── Session state initialisation ───────────────────────────────────────────────

def init_session_state():
    """Ensure all session-state keys exist (idempotent)."""
    defaults: dict[str, Any] = {
        "session_id": None,
        "messages": [],
        "phase": None,
        "done": False,
        "covered_topics": [],
        "report": None,
        # Setup workflow
        "setup_active": False,
        "setup_phase": None,
        "setup_done": False,
        "setup_topics_covered": [],
        "setup_results": None,
        "chosen_method": None,
        "red_flags": [],
        "setup_messages": [],
        # MDE detail (fetched separately)
        "mde_detail": None,
        # Sensitivity analysis
        "sensitivity_data": None,
        # FAQ chat
        "faq_messages": [],
        "faq_method_key": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── API helper ──────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs) -> Any:
    """Call the backend REST API and return parsed JSON."""
    url = f"{API_BASE}{path}"
    response = httpx.request(method, url, timeout=300, **kwargs)
    response.raise_for_status()
    return response.json()


# ── Session lifecycle helpers ───────────────────────────────────────────────────

def start_session():
    """Call POST /sessions and reset all session state."""
    data = api("POST", "/sessions")
    st.session_state.session_id = data["session_id"]
    st.session_state.phase = data["phase"]
    st.session_state.messages = [{"role": "assistant", "content": data["message"]}]
    st.session_state.done = False
    st.session_state.covered_topics = []
    st.session_state.report = None
    st.session_state.setup_active = False
    st.session_state.setup_phase = None
    st.session_state.setup_done = False
    st.session_state.setup_topics_covered = []
    st.session_state.setup_results = None
    st.session_state.chosen_method = None
    st.session_state.red_flags = []
    st.session_state.setup_messages = []
    st.session_state.mde_detail = None
    st.session_state.sensitivity_data = None
    st.session_state.faq_messages = []
    st.session_state.faq_method_key = None


def send_message(user_text: str):
    """Send an elicitation-phase chat message."""
    sid = st.session_state.session_id
    st.session_state.messages.append({"role": "user", "content": user_text})
    data = api("POST", f"/sessions/{sid}/turn", json={"message": user_text})
    st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
    st.session_state.phase = data["phase"]
    st.session_state.done = data["done"]
    st.session_state.covered_topics = data["covered_topics"]
    if data["done"]:
        fetch_report()


def fetch_report():
    """Fetch the elicitation report artefacts."""
    sid = st.session_state.session_id
    st.session_state.report = api("GET", f"/sessions/{sid}/report")


def start_setup(method_key: str):
    """Initiate the setup workflow for a chosen method."""
    sid = st.session_state.session_id
    data = api("POST", f"/sessions/{sid}/setup", json={"method_key": method_key})
    st.session_state.setup_active = True
    st.session_state.chosen_method = method_key
    st.session_state.setup_phase = data["setup_phase"]
    st.session_state.setup_done = data["done"]
    st.session_state.setup_topics_covered = data["setup_topics_covered"]
    st.session_state.red_flags = data.get("red_flags", [])
    # Start fresh message list for setup (don't carry over elicitation history)
    st.session_state.setup_messages = [{"role": "assistant", "content": data["reply"]}]


def send_setup_message(user_text: str):
    """Send a setup-phase chat message."""
    sid = st.session_state.session_id
    st.session_state.setup_messages.append({"role": "user", "content": user_text})
    data = api("POST", f"/sessions/{sid}/setup/turn", json={"message": user_text})
    st.session_state.setup_messages.append({"role": "assistant", "content": data["reply"]})
    st.session_state.setup_phase = data["setup_phase"]
    st.session_state.setup_done = data["done"]
    st.session_state.setup_topics_covered = data["setup_topics_covered"]
    st.session_state.red_flags = data.get("red_flags", [])
    if data["done"]:
        fetch_setup_results()


def fetch_setup_results():
    """Fetch all setup results (power, MDE, synthetic, validation)."""
    sid = st.session_state.session_id
    st.session_state.setup_results = api("GET", f"/sessions/{sid}/setup/results")
    # Also fetch MDE detail with power_by_effect curve
    try:
        st.session_state.mde_detail = api("GET", f"/sessions/{sid}/setup/mde-detail")
    except Exception:
        st.session_state.mde_detail = None
    # Fetch sensitivity analysis
    try:
        st.session_state.sensitivity_data = api("GET", f"/sessions/{sid}/setup/sensitivity")
    except Exception:
        st.session_state.sensitivity_data = None


def send_faq_message(user_text: str, method_key: str | None = None):
    """Send a message to the FAQ chat (stateless endpoint)."""
    st.session_state.faq_messages.append({"role": "user", "content": user_text})
    data = api(
        "POST",
        "/faq",
        json={
            "messages": st.session_state.faq_messages,
            "method_key": method_key,
        },
    )
    st.session_state.faq_messages.append({"role": "assistant", "content": data["reply"]})


# ── Streaming API helpers ───────────────────────────────────────────────────────

def _iter_sse(method: str, path: str, **kwargs) -> Generator[dict, None, None]:
    """Low-level SSE consumer — yields parsed JSON event dicts."""
    url = f"{API_BASE}{path}"
    with httpx.stream(method, url, timeout=300, **kwargs) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                continue


def stream_message(user_text: str) -> Generator[str, None, None]:
    """Generator yielding response tokens for elicitation streaming.

    Side-effect: updates session state from the final metadata event.
    Intended for use with ``st.write_stream()``.
    """
    sid = st.session_state.session_id
    for data in _iter_sse(
        "POST", f"/sessions/{sid}/turn/stream", json={"message": user_text}
    ):
        if "token" in data:
            yield data["token"]
        if data.get("done"):
            st.session_state.phase = data.get("phase", "")
            st.session_state.done = data.get("is_done", False)
            st.session_state.covered_topics = data.get("covered_topics", [])
            if data.get("is_done"):
                fetch_report()


def stream_setup_message(user_text: str) -> Generator[str, None, None]:
    """Generator yielding response tokens for setup streaming.

    Side-effect: updates setup session state from the final metadata event.
    """
    sid = st.session_state.session_id
    for data in _iter_sse(
        "POST", f"/sessions/{sid}/setup/turn/stream", json={"message": user_text}
    ):
        if "token" in data:
            yield data["token"]
        if data.get("done"):
            st.session_state.setup_phase = data.get("setup_phase", "")
            st.session_state.setup_done = data.get("setup_done", False)
            st.session_state.setup_topics_covered = data.get(
                "setup_topics_covered", []
            )
            st.session_state.red_flags = data.get("red_flags", [])
            if data.get("setup_done"):
                fetch_setup_results()


def stream_faq_message(
    user_text: str, method_key: str | None = None
) -> Generator[str, None, None]:
    """Generator yielding real LLM tokens for FAQ streaming."""
    for data in _iter_sse(
        "POST",
        "/faq/stream",
        json={
            "messages": st.session_state.faq_messages
            + [{"role": "user", "content": user_text}],
            "method_key": method_key,
        },
    ):
        if "token" in data:
            yield data["token"]


# ── Session listing / resumption ────────────────────────────────────────────────

def list_recent_sessions() -> list[dict]:
    """Fetch the list of persisted sessions from the backend."""
    try:
        return api("GET", "/sessions")
    except Exception:
        return []


def restore_session_state(session_id: str):
    """Load a previously saved session and populate session state."""
    data = api("GET", f"/sessions/{session_id}/restore")

    st.session_state.session_id = data["session_id"]
    st.session_state.phase = data.get("phase", "")
    st.session_state.done = data.get("done", False)
    st.session_state.covered_topics = data.get("covered_topics", [])
    st.session_state.messages = data.get("messages", [])

    # Report data
    if data.get("done") and data.get("report"):
        st.session_state.report = data["report"]
    elif data.get("done"):
        try:
            fetch_report()
        except Exception:
            st.session_state.report = None
    else:
        st.session_state.report = None

    # Setup session (if previously started)
    setup = data.get("setup")
    if setup and setup.get("active"):
        st.session_state.setup_active = True
        st.session_state.chosen_method = setup.get("chosen_method_key", "")
        st.session_state.setup_phase = setup.get("setup_phase", "")
        st.session_state.setup_done = setup.get("setup_done", False)
        st.session_state.setup_topics_covered = setup.get("setup_topics_covered", [])
        st.session_state.red_flags = setup.get("red_flags", [])
        st.session_state.setup_messages = setup.get("messages", [])
        if setup.get("setup_done"):
            try:
                fetch_setup_results()
            except Exception:
                st.session_state.setup_results = None
    else:
        st.session_state.setup_active = False
        st.session_state.setup_phase = None
        st.session_state.setup_done = False
        st.session_state.setup_topics_covered = []
        st.session_state.setup_results = None
        st.session_state.chosen_method = None
        st.session_state.red_flags = []
        st.session_state.setup_messages = []

    st.session_state.mde_detail = None
    st.session_state.sensitivity_data = None
    st.session_state.faq_messages = []
    st.session_state.faq_method_key = None


# ── Sidebar renderer ───────────────────────────────────────────────────────────

def render_sidebar():
    """Draw the shared sidebar used by all pages."""
    with st.sidebar:
        st.title("🔬 Measurement Design Agent")
        st.caption(
            "Design rigorous experiments to measure ad campaign effectiveness "
            "— no statistics background needed."
        )
        st.divider()

        if st.button("🆕 Start New Session", use_container_width=True, type="primary"):
            try:
                start_session()
                st.rerun()
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

        # ── Session info ────────────────────
        if st.session_state.session_id:
            st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

        # ── Recent sessions ─────────────────
        recent = list_recent_sessions()
        if recent:
            with st.expander("📂 Recent Sessions", expanded=False):
                for sess in recent[:10]:
                    sid = sess.get("session_id", "")
                    label = sess.get("label", sid[:8])
                    updated = sess.get("updated_at", "")[:16]
                    is_current = sid == st.session_state.session_id
                    btn_label = f"{'● ' if is_current else ''}{label}  ({updated})"
                    if st.button(
                        btn_label,
                        key=f"restore_{sid}",
                        use_container_width=True,
                        disabled=is_current,
                    ):
                        try:
                            restore_session_state(sid)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not restore session: {e}")

        st.divider()

        # ── Progress indicators ─────────────
        if not st.session_state.setup_active:
            st.subheader("Elicitation Progress")
            covered = st.session_state.covered_topics or []
            for topic in ALL_TOPICS:
                label = TOPIC_LABELS.get(topic, topic)
                icon = "✅" if topic in covered else "⬜"
                st.markdown(f"{icon} {label}")
        else:
            st.subheader("Setup Progress")
            method_name = METHOD_NAMES.get(
                st.session_state.chosen_method,
                st.session_state.chosen_method or "",
            )
            st.caption(f"Method: **{method_name}**")
            st.divider()

            setup_covered = st.session_state.setup_topics_covered or []
            for topic in ALL_SETUP_TOPICS:
                label = SETUP_TOPIC_LABELS.get(topic, topic)
                icon = "✅" if topic in setup_covered else "⬜"
                st.markdown(f"{icon} {label}")

            # Computation pipeline
            phase = st.session_state.setup_phase or ""
            phase_order = [p[0] for p in COMPUTATION_PHASES]
            current_idx = phase_order.index(phase) if phase in phase_order else -1
            done_idx = len(phase_order) if st.session_state.setup_done else current_idx

            st.divider()
            st.caption("Computation Pipeline")
            for i, (p_key, p_label) in enumerate(COMPUTATION_PHASES):
                if i < done_idx:
                    st.markdown(f"✅ {p_label}")
                elif i == current_idx and not st.session_state.setup_done:
                    st.markdown(f"⏳ {p_label}")
                else:
                    st.markdown(f"⬜ {p_label}")

            # Red flags summary
            rf = st.session_state.red_flags
            if rf:
                st.divider()
                n_crit = sum(1 for f in rf if f.get("severity") == "critical")
                n_warn = sum(1 for f in rf if f.get("severity") == "warning")
                parts = []
                if n_crit:
                    parts.append(f"🚨 {n_crit} critical")
                if n_warn:
                    parts.append(f"⚠️ {n_warn} warning(s)")
                st.caption(f"Flags: {' · '.join(parts)}")
                for f in rf:
                    icon = "🚨" if f.get("severity") == "critical" else "⚠️"
                    st.markdown(f"{icon} {f.get('title', 'Flag')}")

        # Phase display
        if st.session_state.phase or st.session_state.setup_phase:
            st.divider()
            phase_display = st.session_state.setup_phase or st.session_state.phase
            st.caption(f"Phase: `{phase_display}`")

        st.divider()
        st.caption("Powered by Claude · LangGraph · PyMC")


# ── Guard helpers ───────────────────────────────────────────────────────────────

def require_session() -> bool:
    """Show an info box and return False if no session is active."""
    if not st.session_state.session_id:
        st.info("👈 Start a **New Session** from the sidebar first.")
        return False
    return True


def require_report() -> bool:
    """Show an info box if the elicitation report isn't ready yet."""
    if not st.session_state.done or not st.session_state.report:
        st.info(
            "Complete the **Elicitation Chat** first to generate your "
            "measurement design report."
        )
        return False
    return True


def require_setup_done() -> bool:
    """Show an info box if setup isn't finished."""
    if not st.session_state.setup_done or not st.session_state.setup_results:
        st.info(
            "Complete the **Method Setup** workflow first to see simulation results."
        )
        return False
    return True
