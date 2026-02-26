"""
Streamlit UI for the Measurement Design Agent.
"""
from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

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

SETUP_TOPIC_LABELS = {
    "baseline_metrics":   "📈 Baseline Metrics",
    "expected_effect":    "🎯 Expected Effect",
    "statistical_design": "⚙️ Statistical Design",
    "method_specific":    "🔧 Method Details",
}

ALL_SETUP_TOPICS = list(SETUP_TOPIC_LABELS.keys())

# Method keys → display names
METHOD_NAMES = {
    "ab_test": "A/B Test",
    "did": "Difference-in-Differences",
    "ddml": "Double/Debiased ML",
    "geo_lift": "Geo Lift Test",
    "synthetic_control": "Synthetic Control",
    "matched_market": "Matched Market Test",
}


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Measurement Design Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Init ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "phase" not in st.session_state:
    st.session_state.phase = None
if "done" not in st.session_state:
    st.session_state.done = False
if "covered_topics" not in st.session_state:
    st.session_state.covered_topics = []
if "report" not in st.session_state:
    st.session_state.report = None

# Setup workflow state
if "setup_active" not in st.session_state:
    st.session_state.setup_active = False
if "setup_phase" not in st.session_state:
    st.session_state.setup_phase = None
if "setup_done" not in st.session_state:
    st.session_state.setup_done = False
if "setup_topics_covered" not in st.session_state:
    st.session_state.setup_topics_covered = []
if "setup_results" not in st.session_state:
    st.session_state.setup_results = None
if "chosen_method" not in st.session_state:
    st.session_state.chosen_method = None
if "red_flags" not in st.session_state:
    st.session_state.red_flags = []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs):
    url = f"{API_BASE}{path}"
    response = httpx.request(method, url, timeout=300, **kwargs)
    response.raise_for_status()
    return response.json()


def _start_session():
    data = _api("POST", "/sessions")
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


def _send_message(user_text: str):
    sid = st.session_state.session_id
    st.session_state.messages.append({"role": "user", "content": user_text})
    data = _api("POST", f"/sessions/{sid}/turn", json={"message": user_text})
    st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
    st.session_state.phase = data["phase"]
    st.session_state.done = data["done"]
    st.session_state.covered_topics = data["covered_topics"]

    if data["done"]:
        _fetch_report()


def _fetch_report():
    sid = st.session_state.session_id
    st.session_state.report = _api("GET", f"/sessions/{sid}/report")


def _start_setup(method_key: str):
    sid = st.session_state.session_id
    data = _api("POST", f"/sessions/{sid}/setup", json={"method_key": method_key})
    st.session_state.setup_active = True
    st.session_state.chosen_method = method_key
    st.session_state.setup_phase = data["setup_phase"]
    st.session_state.setup_done = data["done"]
    st.session_state.setup_topics_covered = data["setup_topics_covered"]
    st.session_state.red_flags = data.get("red_flags", [])
    st.session_state.messages.append({"role": "assistant", "content": data["reply"]})


def _send_setup_message(user_text: str):
    sid = st.session_state.session_id
    st.session_state.messages.append({"role": "user", "content": user_text})
    data = _api("POST", f"/sessions/{sid}/setup/turn", json={"message": user_text})
    st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
    st.session_state.setup_phase = data["setup_phase"]
    st.session_state.setup_done = data["done"]
    st.session_state.setup_topics_covered = data["setup_topics_covered"]
    st.session_state.red_flags = data.get("red_flags", [])

    if data["done"]:
        _fetch_setup_results()


def _fetch_setup_results():
    sid = st.session_state.session_id
    st.session_state.setup_results = _api("GET", f"/sessions/{sid}/setup/results")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Measurement Design Agent")
    st.caption(
        "This tool helps you design a rigorous experiment to measure "
        "the effectiveness of your ad campaign — no statistics background needed."
    )
    st.divider()

    if st.button("🆕 Start New Session", use_container_width=True, type="primary"):
        try:
            _start_session()
            st.rerun()
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")

    st.divider()

    if not st.session_state.setup_active:
        # Show elicitation progress
        st.subheader("Elicitation Progress")
        covered = st.session_state.covered_topics or []
        for topic in ALL_TOPICS:
            label = TOPIC_LABELS.get(topic, topic)
            if topic in covered:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"⬜ {label}")
    else:
        # Show setup progress
        st.subheader("Setup Progress")
        method_name = METHOD_NAMES.get(
            st.session_state.chosen_method, st.session_state.chosen_method
        )
        st.caption(f"Method: **{method_name}**")
        st.divider()

        setup_covered = st.session_state.setup_topics_covered or []
        for topic in ALL_SETUP_TOPICS:
            label = SETUP_TOPIC_LABELS.get(topic, topic)
            if topic in setup_covered:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"⬜ {label}")

        # Show computation phases
        phase = st.session_state.setup_phase or ""
        computation_phases = [
            ("power_analysis", "📊 Power Analysis"),
            ("mde_simulation", "🎯 MDE Simulation"),
            ("synthetic_gen", "🧪 Synthetic Data"),
            ("validation", "✅ Code Validation"),
            ("setup_output", "📋 Setup Report"),
        ]
        phase_order = [p[0] for p in computation_phases]
        current_idx = phase_order.index(phase) if phase in phase_order else -1
        done_idx = len(phase_order) if st.session_state.setup_done else current_idx

        st.divider()
        st.caption("Computation Pipeline")
        for i, (p_key, p_label) in enumerate(computation_phases):
            if i < done_idx:
                st.markdown(f"✅ {p_label}")
            elif i == current_idx and not st.session_state.setup_done:
                st.markdown(f"⏳ {p_label}")
            else:
                st.markdown(f"⬜ {p_label}")

        # Show red flags in sidebar
        rf = st.session_state.red_flags
        if rf:
            st.divider()
            n_crit = sum(1 for f in rf if f.get("severity") == "critical")
            n_warn = sum(1 for f in rf if f.get("severity") == "warning")
            header_parts = []
            if n_crit:
                header_parts.append(f"🚨 {n_crit} critical")
            if n_warn:
                header_parts.append(f"⚠️ {n_warn} warning(s)")
            st.caption(f"Flags: {' · '.join(header_parts)}")
            for f in rf:
                icon = "🚨" if f.get("severity") == "critical" else "⚠️"
                st.markdown(f"{icon} {f.get('title', 'Flag')}")

    if st.session_state.phase or st.session_state.setup_phase:
        st.divider()
        phase_display = st.session_state.setup_phase or st.session_state.phase
        st.caption(f"Phase: `{phase_display}`")

    st.divider()
    st.caption("Powered by Claude · LangGraph · PyMC")

# ── Main chat area ─────────────────────────────────────────────────────────────
st.header("Ad Campaign Measurement Design Assistant")

if not st.session_state.session_id:
    st.info(
        "👈 Click **Start New Session** in the sidebar to begin. "
        "I'll ask you a few simple questions about your campaign and recommend "
        "the best measurement approach."
    )
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Determine which chat input to show
    if st.session_state.setup_active and not st.session_state.setup_done:
        # Setup workflow chat
        if user_input := st.chat_input("Your answer…"):
            try:
                _send_setup_message(user_input)
                st.rerun()
            except Exception as e:
                st.error(f"Error communicating with backend: {e}")

    elif st.session_state.done and not st.session_state.setup_active:
        # Elicitation done, show method selection
        st.success("✅ Your measurement design is complete! Choose a method below to set up the experiment.")

    elif not st.session_state.done:
        # Elicitation chat
        if user_input := st.chat_input("Your answer…"):
            try:
                _send_message(user_input)
                st.rerun()
            except Exception as e:
                st.error(f"Error communicating with backend: {e}")

    if st.session_state.setup_done:
        st.success(
            "🎉 Experiment setup complete! See the Setup Results tab below."
        )


# ── Results panel (shown when elicitation is done) ────────────────────────────
if st.session_state.done and st.session_state.report:
    report = st.session_state.report
    st.divider()
    st.subheader("📦 Your Measurement Design Artefacts")

    tab_report, tab_spec_json, tab_spec_yaml, tab_scaffold, tab_ranking = st.tabs(
        ["📄 Report", "🗂 JSON Spec", "📋 YAML Spec", "🐍 Code Scaffold", "🏆 Rankings"]
    )

    with tab_report:
        st.download_button(
            "⬇ Download Markdown",
            data=report.get("markdown", ""),
            file_name="measurement_design_report.md",
            mime="text/markdown",
        )
        st.markdown(report.get("markdown", ""), unsafe_allow_html=False)

    with tab_spec_json:
        json_str = json.dumps(report.get("json_spec", {}), indent=2)
        st.download_button(
            "⬇ Download JSON",
            data=json_str,
            file_name="measurement_design_spec.json",
            mime="application/json",
        )
        st.code(json_str, language="json")

    with tab_spec_yaml:
        yaml_str = report.get("yaml_spec", "")
        st.download_button(
            "⬇ Download YAML",
            data=yaml_str,
            file_name="measurement_design_spec.yaml",
            mime="text/yaml",
        )
        st.code(yaml_str, language="yaml")

    with tab_scaffold:
        scaffold_str = report.get("scaffold", "")
        st.download_button(
            "⬇ Download Python Scaffold (.py)",
            data=scaffold_str,
            file_name="measurement_scaffold.py",
            mime="text/x-python",
        )
        st.code(scaffold_str, language="python")

    with tab_ranking:
        st.subheader("Method Rankings")
        ranked = report.get("ranked_methods", [])
        for item in ranked:
            score = item.get("score", 0)
            bar_len = int(score / 20)
            bar = "█" * bar_len + "░" * (5 - bar_len)
            st.metric(
                label=f"#{item['rank']} {item.get('key', '').replace('_', ' ').title()}",
                value=f"{score}/100",
                delta=None,
            )
        if ranked:
            st.bar_chart(
                data={item.get("key", str(item["rank"])): item.get("score", 0) for item in ranked},
            )


# ── Method Selection & Setup ──────────────────────────────────────────────────
if st.session_state.done and not st.session_state.setup_active and st.session_state.report:
    st.divider()
    st.subheader("🚀 Set Up Your Experiment")
    st.markdown(
        "Select one of the recommended methods below to proceed with experiment setup. "
        "I'll help you calculate sample sizes, find the minimum detectable effect, "
        "and generate synthetic data to test everything before launch."
    )

    ranked = st.session_state.report.get("ranked_methods", [])
    cols = st.columns(min(len(ranked), 3))

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
                    _start_setup(method_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error starting setup: {e}")


# ── Setup Results Panel ──────────────────────────────────────────────────────
if st.session_state.setup_done and st.session_state.setup_results:
    results = st.session_state.setup_results
    st.divider()
    st.subheader("🔬 Experiment Setup Results")

    tab_setup_report, tab_power, tab_mde, tab_synth_data, tab_validation, tab_flags = st.tabs(
        ["📋 Setup Report", "📊 Power Analysis", "🎯 MDE Results",
         "🧪 Synthetic Data", "✅ Validation", "⚠️ Flags"]
    )

    with tab_setup_report:
        report_md = results.get("setup_report_markdown", "")
        st.download_button(
            "⬇ Download Setup Report",
            data=report_md,
            file_name="experiment_setup_report.md",
            mime="text/markdown",
        )
        st.markdown(report_md, unsafe_allow_html=False)

    with tab_power:
        power = results.get("power_results", {})
        st.markdown("### Power Analysis Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Required Sample Size", f"{power.get('required_sample_size', 'N/A'):,}"
                       if isinstance(power.get('required_sample_size'), (int, float))
                       else "N/A")
        with col2:
            achieved = power.get("achieved_power", 0)
            st.metric("Achieved Power", f"{achieved:.1%}" if achieved else "N/A")
        with col3:
            st.metric("Effect Size", f"{power.get('effect_size_used', 0):.4f}")

        st.markdown(f"**Notes:** {power.get('notes', '')}")

        # Power curve chart
        curve_json = results.get("power_curve_json", "")
        if curve_json:
            try:
                curve_data = json.loads(curve_json)
                if curve_data:
                    import pandas as pd
                    df_curve = pd.DataFrame(curve_data)
                    st.line_chart(df_curve.set_index("n")["power"])
                    st.caption("Power curve: statistical power vs sample size (per group)")
            except (json.JSONDecodeError, Exception):
                pass

    with tab_mde:
        mde = results.get("mde_results", {})
        st.markdown("### Minimum Detectable Effect")
        col1, col2 = st.columns(2)
        with col1:
            mde_abs = mde.get("mde_absolute")
            st.metric("MDE (absolute)", f"{mde_abs:.4f}" if mde_abs else "N/A")
        with col2:
            mde_rel = mde.get("mde_relative_pct")
            st.metric("MDE (relative)", f"{mde_rel:.1f}%" if mde_rel else "N/A")

        st.markdown(
            f"- **Simulations**: {mde.get('n_simulations', 0):,}\n"
            f"- **Target power**: {mde.get('target_power', 0.8):.0%}\n"
            f"- **α**: {mde.get('alpha', 0.05)}"
        )
        st.markdown(f"**Notes:** {mde.get('notes', '')}")

    with tab_synth_data:
        csv_data = results.get("synthetic_data_csv", "")
        if csv_data:
            st.download_button(
                "⬇ Download Synthetic Data (CSV)",
                data=csv_data,
                file_name="synthetic_experiment_data.csv",
                mime="text/csv",
            )

            # Show preview
            try:
                import pandas as pd
                import io
                df_preview = pd.read_csv(io.StringIO(csv_data))
                st.markdown(f"**{len(df_preview):,} rows × {len(df_preview.columns)} columns**")
                st.dataframe(df_preview.head(100), use_container_width=True)

                st.markdown("#### Summary Statistics")
                st.dataframe(df_preview.describe(), use_container_width=True)
            except Exception:
                st.code(csv_data[:5000], language="csv")
        else:
            st.info("No synthetic data was generated.")

    with tab_validation:
        val = results.get("validation_results", {})
        if val.get("success"):
            st.success("✅ Validation passed!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Effect", f"{val.get('true_effect', 0):.4f}")
            with col2:
                st.metric("Estimated Effect", f"{val.get('estimated_effect', 0):.4f}")
            with col3:
                p = val.get("p_value")
                st.metric("p-value", f"{p:.4f}" if p is not None else "N/A")

            ci_lo = val.get("ci_lower")
            ci_hi = val.get("ci_upper")
            if ci_lo is not None and ci_hi is not None:
                st.markdown(f"**95% CI:** [{ci_lo:.4f}, {ci_hi:.4f}]")
            st.markdown(f"**Summary:** {val.get('summary', '')}")
        elif val:
            st.warning(f"⚠️ {val.get('error_message', 'Validation did not complete.')}")
            if val.get("summary"):
                st.markdown(val["summary"])
        else:
            st.info("No validation results available.")

    with tab_flags:
        rf = results.get("red_flags", [])
        if rf:
            n_crit = sum(1 for f in rf if f.get("severity") == "critical")
            n_warn = sum(1 for f in rf if f.get("severity") == "warning")
            if n_crit:
                st.error(f"🚨 {n_crit} critical issue(s) detected")
            if n_warn:
                st.warning(f"⚠️ {n_warn} warning(s) detected")
            for f in rf:
                icon = "🚨" if f.get("severity") == "critical" else "⚠️"
                with st.expander(f"{icon} {f.get('title', 'Flag')}", expanded=True):
                    st.markdown(f.get("detail", ""))
                    st.info(f"💡 **Suggestion:** {f.get('suggestion', '')}")
        else:
            st.success("✅ No major feasibility concerns detected!")
