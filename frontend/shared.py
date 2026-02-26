"""
Shared constants, helpers, and session-state initialisation used by all pages.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

# ── API base URL ────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ── Topic labels (elicitation) ──────────────────────────────────────────────────
TOPIC_LABELS = {
    "objective":         "🎯 Objective",
    "randomization":     "🔀 Randomisation",
    "data_history":      "📅 Data History",
    "geo_structure":     "🗺  Geography",
    "treatment_control": "🎛  Treatment Control",
    "covariates":        "📊 Covariates",
    "scale":             "📏 Scale & Duration",
}

ALL_TOPICS = list(TOPIC_LABELS.keys())

# ── Topic labels (setup) ───────────────────────────────────────────────────────
SETUP_TOPIC_LABELS = {
    "baseline_metrics":   "📈 Baseline Metrics",
    "expected_effect":    "🎯 Expected Effect",
    "statistical_design": "⚙️ Statistical Design",
    "method_specific":    "🔧 Method Details",
}

ALL_SETUP_TOPICS = list(SETUP_TOPIC_LABELS.keys())

# ── Method key → display name ──────────────────────────────────────────────────
METHOD_NAMES = {
    "ab_test":           "A/B Test",
    "did":               "Difference-in-Differences",
    "ddml":              "Double/Debiased ML",
    "geo_lift":          "Geo Lift Test",
    "synthetic_control": "Synthetic Control",
    "matched_market":    "Matched Market Test",
}

# Methods that have panel / time-series structure
PANEL_METHODS = {"did", "geo_lift", "synthetic_control", "matched_market"}

# ── Computation phases (setup pipeline) ─────────────────────────────────────────
COMPUTATION_PHASES = [
    ("power_analysis",  "📊 Power Analysis"),
    ("mde_simulation",  "🎯 MDE Simulation"),
    ("synthetic_gen",   "🧪 Synthetic Data"),
    ("validation",      "✅ Code Validation"),
    ("setup_output",    "📋 Setup Report"),
]

# ── Data-template schemas per method ────────────────────────────────────────────
METHOD_SCHEMAS: dict[str, dict[str, Any]] = {
    "ab_test": {
        "description": (
            "User-level data for a randomised A/B test. Each row represents one "
            "user assigned to either `control` or `treatment`."
        ),
        "columns": [
            {"name": "user_id",   "dtype": "int",    "description": "Unique user identifier",         "example": 1,          "required": True},
            {"name": "group",     "dtype": "string", "description": "Assignment group",                "example": "control",  "required": True},
            {"name": "converted", "dtype": "int",    "description": "Binary conversion flag (0/1)",    "example": 0,          "required": False},
            {"name": "outcome",   "dtype": "float",  "description": "Continuous outcome metric",       "example": 42.5,       "required": False},
        ],
        "notes": "Include *either* `converted` (proportions test) or `outcome` (continuous test), not both.",
        "form_fields": {
            "n_per_group":    {"label": "Users per group",    "type": "int",   "default": 1000, "min": 10},
            "baseline_rate":  {"label": "Baseline conv rate", "type": "float", "default": 0.05, "min": 0.001, "max": 0.999},
            "lift_abs":       {"label": "Expected lift (abs)","type": "float", "default": 0.005},
        },
    },
    "did": {
        "description": (
            "Panel data for Difference-in-Differences. Each row is one unit × one time period."
        ),
        "columns": [
            {"name": "unit_id",  "dtype": "string", "description": "Unit / market identifier",          "example": "T_1",     "required": True},
            {"name": "group",    "dtype": "string", "description": "treatment or control",              "example": "treatment","required": True},
            {"name": "period",   "dtype": "int",    "description": "Time period index (1-based)",       "example": 1,         "required": True},
            {"name": "is_post",  "dtype": "int",    "description": "1 if post-treatment, else 0",       "example": 0,         "required": True},
            {"name": "outcome",  "dtype": "float",  "description": "Outcome metric value",              "example": 105.3,     "required": True},
        ],
        "notes": "Ensure balanced panel: every unit should appear in every period.",
        "form_fields": {
            "num_treatment_units": {"label": "# Treatment units",  "type": "int",   "default": 5,    "min": 1},
            "num_control_units":   {"label": "# Control units",    "type": "int",   "default": 10,   "min": 1},
            "num_pre_periods":     {"label": "# Pre periods",      "type": "int",   "default": 12,   "min": 1},
            "num_post_periods":    {"label": "# Post periods",     "type": "int",   "default": 8,    "min": 1},
            "baseline_value":      {"label": "Baseline metric mean","type": "float", "default": 100.0},
            "baseline_std":        {"label": "Baseline metric std", "type": "float", "default": 30.0, "min": 0.01},
            "lift_abs":            {"label": "Expected lift (abs)", "type": "float", "default": 5.0},
        },
    },
    "geo_lift": {
        "description": (
            "Market-level weekly data for a Geo Lift test. Each row is one geo × one week."
        ),
        "columns": [
            {"name": "geo_id",    "dtype": "string", "description": "Geography / DMA identifier",      "example": "geo_001",  "required": True},
            {"name": "group",     "dtype": "string", "description": "treatment or control",             "example": "control",  "required": True},
            {"name": "week",      "dtype": "int",    "description": "Week number (1-based)",            "example": 1,          "required": True},
            {"name": "is_post",   "dtype": "int",    "description": "1 if post-treatment, else 0",      "example": 0,          "required": True},
            {"name": "kpi_value", "dtype": "float",  "description": "KPI value (e.g. sales, signups)", "example": 4850.0,     "required": True},
        ],
        "notes": "Include sufficient pre-period weeks (≥ 8) for the model to learn geo-level patterns.",
        "form_fields": {
            "num_treatment_geos": {"label": "# Treatment geos",   "type": "int",   "default": 5,    "min": 1},
            "num_control_geos":   {"label": "# Control geos",     "type": "int",   "default": 15,   "min": 1},
            "num_pre_periods":    {"label": "# Pre weeks",         "type": "int",   "default": 12,   "min": 1},
            "num_post_periods":   {"label": "# Post weeks",        "type": "int",   "default": 6,    "min": 1},
            "baseline_value":     {"label": "Baseline KPI/week",   "type": "float", "default": 5000.0},
            "baseline_std":       {"label": "Baseline std",        "type": "float", "default": 1500.0, "min": 0.01},
            "lift_abs":           {"label": "Expected lift (abs)", "type": "float", "default": 250.0},
        },
    },
    "synthetic_control": {
        "description": (
            "Time-series data for Synthetic Control. One treated unit + J donor units."
        ),
        "columns": [
            {"name": "unit_id",    "dtype": "string", "description": "Unit identifier (treatment or donor_NNN)","example": "treatment","required": True},
            {"name": "is_treated", "dtype": "int",    "description": "1 for treated unit, 0 for donors",        "example": 1,          "required": True},
            {"name": "period",     "dtype": "int",    "description": "Time period index (1-based)",              "example": 1,          "required": True},
            {"name": "is_post",    "dtype": "int",    "description": "1 if post-treatment, else 0",              "example": 0,          "required": True},
            {"name": "outcome",    "dtype": "float",  "description": "Outcome metric value",                     "example": 98.7,       "required": True},
        ],
        "notes": "Long pre-periods (≥ 20) improve the quality of the synthetic counterfactual.",
        "form_fields": {
            "num_donor_units":  {"label": "# Donor units",       "type": "int",   "default": 15,   "min": 2},
            "num_pre_periods":  {"label": "# Pre periods",       "type": "int",   "default": 52,   "min": 5},
            "num_post_periods": {"label": "# Post periods",      "type": "int",   "default": 12,   "min": 1},
            "baseline_value":   {"label": "Baseline metric mean","type": "float", "default": 100.0},
            "baseline_std":     {"label": "Baseline metric std", "type": "float", "default": 20.0, "min": 0.01},
            "lift_abs":         {"label": "Expected lift (abs)", "type": "float", "default": 10.0},
        },
    },
    "matched_market": {
        "description": (
            "Paired market data for a Matched Market test. Each pair has one treatment and one control market."
        ),
        "columns": [
            {"name": "pair_id",   "dtype": "int",    "description": "Matched pair identifier",          "example": 1,             "required": True},
            {"name": "market_id", "dtype": "string", "description": "Market identifier",                "example": "pair_1_T",    "required": True},
            {"name": "group",     "dtype": "string", "description": "treatment or control",             "example": "treatment",   "required": True},
            {"name": "week",      "dtype": "int",    "description": "Week number (1-based)",            "example": 1,             "required": True},
            {"name": "is_post",   "dtype": "int",    "description": "1 if post-treatment, else 0",      "example": 0,             "required": True},
            {"name": "kpi_value", "dtype": "float",  "description": "KPI value for that market/week",   "example": 5200.0,        "required": True},
        ],
        "notes": "Markets within each pair should be as similar as possible in pre-period KPI trends.",
        "form_fields": {
            "num_pairs":        {"label": "# Market pairs",      "type": "int",   "default": 6,    "min": 2},
            "num_pre_periods":  {"label": "# Pre weeks",         "type": "int",   "default": 8,    "min": 1},
            "num_post_periods": {"label": "# Post weeks",        "type": "int",   "default": 4,    "min": 1},
            "baseline_value":   {"label": "Baseline KPI/week",   "type": "float", "default": 5000.0},
            "baseline_std":     {"label": "Baseline std",        "type": "float", "default": 1500.0, "min": 0.01},
            "lift_abs":         {"label": "Expected lift (abs)", "type": "float", "default": 250.0},
        },
    },
    "ddml": {
        "description": (
            "Observational data with covariates for Double/Debiased ML. "
            "Each row is one observational unit."
        ),
        "columns": [
            {"name": "user_id",   "dtype": "int",   "description": "Unique user identifier",                 "example": 1,      "required": True},
            {"name": "x_1…x_K",   "dtype": "float", "description": "Covariate columns (confounders)",        "example": 0.34,   "required": True},
            {"name": "treatment", "dtype": "int",   "description": "Binary treatment indicator (0/1)",        "example": 1,      "required": True},
            {"name": "outcome",   "dtype": "float", "description": "Continuous outcome metric",               "example": 105.2,  "required": True},
        ],
        "notes": "Include as many pre-treatment covariates as possible to satisfy the unconfoundedness assumption.",
        "form_fields": {
            "n_obs":         {"label": "# Observations",  "type": "int",   "default": 5000, "min": 100},
            "n_covariates":  {"label": "# Covariates",    "type": "int",   "default": 10,   "min": 1, "max": 50},
            "baseline_value":{"label": "Baseline mean",   "type": "float", "default": 100.0},
            "baseline_std":  {"label": "Baseline std",    "type": "float", "default": 30.0, "min": 0.01},
            "lift_abs":      {"label": "Expected lift",   "type": "float", "default": 5.0},
        },
    },
}


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
