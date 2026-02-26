"""
Page 2 — Design Report

Shows the elicitation report, JSON/YAML specs, code scaffold, and
a Plotly-powered method-ranking chart.
"""
from __future__ import annotations

import json

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Design Report · Measurement Design Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from shared import (
    METHOD_NAMES,
    init_session_state,
    render_sidebar,
    require_report,
    require_session,
)

init_session_state()
render_sidebar()

# ── Page header ─────────────────────────────────────────────────────────────────
st.header("📄 Design Report & Method Rankings")

if not require_session():
    st.stop()
if not require_report():
    st.stop()

report = st.session_state.report

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_report, tab_json, tab_yaml, tab_scaffold, tab_ranking = st.tabs(
    ["📄 Report", "🗂 JSON Spec", "📋 YAML Spec", "🐍 Code Scaffold", "🏆 Rankings"]
)

# ── Report tab ──────────────────────────────────────────────────────────────────
with tab_report:
    st.download_button(
        "⬇ Download Markdown",
        data=report.get("markdown", ""),
        file_name="measurement_design_report.md",
        mime="text/markdown",
    )
    st.markdown(report.get("markdown", ""), unsafe_allow_html=False)

# ── JSON Spec tab ───────────────────────────────────────────────────────────────
with tab_json:
    json_str = json.dumps(report.get("json_spec", {}), indent=2)
    st.download_button(
        "⬇ Download JSON",
        data=json_str,
        file_name="measurement_design_spec.json",
        mime="application/json",
    )
    st.code(json_str, language="json")

# ── YAML Spec tab ───────────────────────────────────────────────────────────────
with tab_yaml:
    yaml_str = report.get("yaml_spec", "")
    st.download_button(
        "⬇ Download YAML",
        data=yaml_str,
        file_name="measurement_design_spec.yaml",
        mime="text/yaml",
    )
    st.code(yaml_str, language="yaml")

# ── Scaffold tab ────────────────────────────────────────────────────────────────
with tab_scaffold:
    scaffold_str = report.get("scaffold", "")
    st.download_button(
        "⬇ Download Python Scaffold (.py)",
        data=scaffold_str,
        file_name="measurement_scaffold.py",
        mime="text/x-python",
    )
    st.code(scaffold_str, language="python")

# ── Rankings tab — Plotly horizontal bar chart ──────────────────────────────────
with tab_ranking:
    st.subheader("Method Rankings")
    ranked = report.get("ranked_methods", [])

    if ranked:
        # Build data in display order (top rank at top of chart)
        ranked_sorted = sorted(ranked, key=lambda x: x.get("rank", 99), reverse=True)

        names = [
            f"#{r.get('rank', '?')}  {METHOD_NAMES.get(r.get('key', ''), r.get('key', ''))}"
            for r in ranked_sorted
        ]
        scores = [r.get("score", 0) for r in ranked_sorted]

        # Colour scale: green > 70, amber 40-70, red < 40
        colours = []
        for s in scores:
            if s >= 70:
                colours.append("#2ecc71")
            elif s >= 40:
                colours.append("#f39c12")
            else:
                colours.append("#e74c3c")

        fig = go.Figure(go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colours,
            text=[f"{s}/100" for s in scores],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x}/100<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Fit Score",
            xaxis_range=[0, 105],
            yaxis_title="",
            height=max(250, 60 * len(ranked)),
            margin=dict(l=10, r=10, t=30, b=30),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics summary
        st.divider()
        metric_cols = st.columns(min(len(ranked), 4))
        for i, item in enumerate(sorted(ranked, key=lambda x: x.get("rank", 99))):
            with metric_cols[i % len(metric_cols)]:
                st.metric(
                    label=f"#{item['rank']} {METHOD_NAMES.get(item.get('key',''), item.get('key',''))}",
                    value=f"{item.get('score', 0)}/100",
                )
    else:
        st.info("No ranking data available.")
