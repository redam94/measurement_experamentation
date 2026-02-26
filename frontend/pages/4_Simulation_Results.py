"""
Page 4 — Simulation Results

Interactive Plotly charts for:
  1. Power curve (power vs. sample size)
  2. MDE curve (empirical power vs. effect size)
  3. Synthetic-data distributions (treatment vs. control)
  4. Time-series plots (pre/post for panel methods)
  5. Validation (estimated vs. true effect with CI error bars)
  6. Sensitivity analysis (power heatmap across alpha × effect size)
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Simulation Results · Measurement Design Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from shared import (
    METHOD_NAMES,
    PANEL_METHODS,
    init_session_state,
    render_sidebar,
    require_session,
    require_setup_done,
)

init_session_state()
render_sidebar()

# ── Page header ─────────────────────────────────────────────────────────────────
st.header("📊 Simulation Results")

if not require_session():
    st.stop()
if not require_setup_done():
    st.stop()

results = st.session_state.setup_results
method_key = st.session_state.chosen_method or ""
method_name = METHOD_NAMES.get(method_key, method_key.replace("_", " ").title())

st.caption(f"Method: **{method_name}**")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: parse the synthetic CSV once
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _parse_csv(csv_text: str) -> pd.DataFrame | None:
    if not csv_text:
        return None
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None


csv_data = results.get("synthetic_data_csv", "")
df_synth = _parse_csv(csv_data)


# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════

(
    tab_power,
    tab_mde,
    tab_dist,
    tab_ts,
    tab_val,
    tab_sens,
    tab_report,
    tab_flags,
) = st.tabs([
    "📈 Power Curve",
    "🎯 MDE Curve",
    "📊 Distributions",
    "📉 Time Series",
    "✅ Validation",
    "🌡️ Sensitivity",
    "📋 Setup Report",
    "⚠️ Flags",
])


# ── Tab 1: Power Curve ─────────────────────────────────────────────────────────
with tab_power:
    power = results.get("power_results", {})

    # Summary metrics
    st.markdown("### Power Analysis Summary")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        rss = power.get("required_sample_size")
        st.metric("Required Sample Size",
                   f"{rss:,}" if isinstance(rss, (int, float)) else "N/A")
    with mc2:
        achieved = power.get("achieved_power")
        st.metric("Achieved Power",
                   f"{achieved:.1%}" if achieved else "N/A")
    with mc3:
        st.metric("Effect Size Used",
                   f"{power.get('effect_size_used', 0):.4f}")

    st.markdown(f"**Notes:** {power.get('notes', '')}")
    st.divider()

    # Power curve chart
    curve_json = results.get("power_curve_json", "")
    if curve_json:
        try:
            curve_data = json.loads(curve_json)
            if curve_data:
                df_curve = pd.DataFrame(curve_data)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_curve["n"],
                    y=df_curve["power"],
                    mode="lines+markers",
                    name="Power",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=4),
                    hovertemplate="n = %{x:,}<br>Power = %{y:.3f}<extra></extra>",
                ))

                # Target power line
                target = power.get("achieved_power") or 0.80
                fig.add_hline(
                    y=0.80,
                    line_dash="dash",
                    line_color="#e74c3c",
                    annotation_text="Target 80%",
                    annotation_position="top right",
                )

                # Required N annotation
                if isinstance(rss, (int, float)):
                    fig.add_vline(
                        x=rss,
                        line_dash="dot",
                        line_color="#2ecc71",
                        annotation_text=f"n = {rss:,}",
                        annotation_position="top left",
                    )

                fig.update_layout(
                    title="Statistical Power vs. Sample Size (per group)",
                    xaxis_title="Sample Size (n)",
                    yaxis_title="Power",
                    yaxis_range=[0, 1.05],
                    template="plotly_white",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download button for curve data
                st.download_button(
                    "⬇ Download Power Curve Data (CSV)",
                    data=df_curve.to_csv(index=False),
                    file_name="power_curve.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.warning(f"Could not render power curve: {exc}")
    else:
        st.info("No power curve data available.")


# ── Tab 2: MDE Curve ───────────────────────────────────────────────────────────
with tab_mde:
    mde = results.get("mde_results", {})

    # Summary metrics
    st.markdown("### Minimum Detectable Effect")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        mde_abs = mde.get("mde_absolute")
        st.metric("MDE (absolute)", f"{mde_abs:.4f}" if mde_abs else "N/A")
    with mc2:
        mde_rel = mde.get("mde_relative_pct")
        st.metric("MDE (relative)", f"{mde_rel:.1f}%" if mde_rel else "N/A")
    with mc3:
        st.metric("Simulations", f"{mde.get('n_simulations', 0):,}")

    st.markdown(
        f"- **Target power**: {mde.get('target_power', 0.8):.0%}\n"
        f"- **α**: {mde.get('alpha', 0.05)}\n"
        f"- **Notes**: {mde.get('notes', '')}"
    )
    st.divider()

    # MDE curve from the detail endpoint
    mde_detail = st.session_state.mde_detail
    power_by_effect = (mde_detail or {}).get("power_by_effect", [])

    if power_by_effect:
        df_mde = pd.DataFrame(power_by_effect)

        fig = go.Figure()

        # Primary line: power vs absolute effect
        fig.add_trace(go.Scatter(
            x=df_mde["effect_abs"],
            y=df_mde["power"],
            mode="lines+markers",
            name="Empirical Power",
            line=dict(color="#9b59b6", width=2),
            marker=dict(size=5),
            hovertemplate=(
                "Effect (abs): %{x:.4f}<br>"
                "Effect (rel): %{customdata:.1f}%<br>"
                "Power: %{y:.3f}<extra></extra>"
            ),
            customdata=df_mde["effect_rel_pct"],
        ))

        # Target power line
        target_power = mde.get("target_power", 0.80)
        fig.add_hline(
            y=target_power,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Target {target_power:.0%}",
            annotation_position="top right",
        )

        # MDE vertical line
        if mde_abs:
            fig.add_vline(
                x=mde_abs,
                line_dash="dot",
                line_color="#2ecc71",
                annotation_text=f"MDE = {mde_abs:.4f}",
                annotation_position="top left",
            )

        fig.update_layout(
            title="Empirical Power vs. Effect Size (Monte Carlo MDE)",
            xaxis_title="Effect Size (absolute)",
            yaxis_title="Empirical Power",
            yaxis_range=[0, 1.05],
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇ Download MDE Curve Data (CSV)",
            data=df_mde.to_csv(index=False),
            file_name="mde_curve.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "MDE curve data not available. "
            "The power-by-effect detail may not have been fetched."
        )


# ── Tab 3: Distributions ──────────────────────────────────────────────────────
with tab_dist:
    st.markdown("### Synthetic Data Distributions")

    if df_synth is not None and not df_synth.empty:
        # Identify the outcome column and group column
        outcome_col = None
        group_col = None

        for c in ["converted", "outcome", "kpi_value"]:
            if c in df_synth.columns:
                outcome_col = c
                break
        for c in ["group", "is_treated"]:
            if c in df_synth.columns:
                group_col = c
                break

        if outcome_col and group_col:
            # If is_treated is int, map to labels
            plot_df = df_synth.copy()
            if group_col == "is_treated":
                plot_df["group_label"] = plot_df[group_col].map(
                    {1: "Treatment", 0: "Control / Donor"}
                )
                group_col_plot = "group_label"
            else:
                plot_df["group_label"] = plot_df[group_col].str.capitalize()
                group_col_plot = "group_label"

            # For panel data, optionally filter to post-period only
            if "is_post" in plot_df.columns:
                filter_opt = st.radio(
                    "Period filter",
                    ["Post-treatment only", "All periods", "Pre-treatment only"],
                    horizontal=True,
                    key="dist_period_filter",
                )
                if filter_opt == "Post-treatment only":
                    plot_df = plot_df[plot_df["is_post"] == 1]
                elif filter_opt == "Pre-treatment only":
                    plot_df = plot_df[plot_df["is_post"] == 0]

            # Histogram
            fig_hist = px.histogram(
                plot_df,
                x=outcome_col,
                color=group_col_plot,
                barmode="overlay",
                opacity=0.6,
                marginal="box",
                title=f"Distribution of `{outcome_col}` by Group",
                color_discrete_map={"Treatment": "#e74c3c", "Control": "#3498db",
                                     "Control / Donor": "#3498db"},
                template="plotly_white",
                height=500,
            )
            fig_hist.update_layout(
                xaxis_title=outcome_col.replace("_", " ").title(),
                yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Violin plot
            fig_violin = px.violin(
                plot_df,
                x=group_col_plot,
                y=outcome_col,
                color=group_col_plot,
                box=True,
                points="outliers",
                title=f"Violin Plot — `{outcome_col}` by Group",
                color_discrete_map={"Treatment": "#e74c3c", "Control": "#3498db",
                                     "Control / Donor": "#3498db"},
                template="plotly_white",
                height=450,
            )
            st.plotly_chart(fig_violin, use_container_width=True)

            # Summary stats table
            st.markdown("#### Summary Statistics")
            st.dataframe(
                plot_df.groupby(group_col_plot)[outcome_col]
                .describe()
                .round(4),
                use_container_width=True,
            )
        else:
            st.warning("Could not identify outcome and group columns in synthetic data.")
    else:
        st.info("No synthetic data available.")


# ── Tab 4: Time Series ────────────────────────────────────────────────────────
with tab_ts:
    st.markdown("### Time Series — Treatment vs. Control")

    if method_key not in PANEL_METHODS:
        st.info(
            f"Time-series plots are not applicable for **{method_name}**. "
            "This visualisation is designed for panel / geo-based methods."
        )
    elif df_synth is not None and not df_synth.empty:
        # Identify columns
        time_col = None
        for c in ["period", "week"]:
            if c in df_synth.columns:
                time_col = c
                break

        outcome_col = None
        for c in ["outcome", "kpi_value"]:
            if c in df_synth.columns:
                outcome_col = c
                break

        group_col = None
        for c in ["group", "is_treated"]:
            if c in df_synth.columns:
                group_col = c
                break

        if time_col and outcome_col and group_col:
            ts_df = df_synth.copy()
            if group_col == "is_treated":
                ts_df["group_label"] = ts_df[group_col].map(
                    {1: "Treatment", 0: "Control / Donor"}
                )
            else:
                ts_df["group_label"] = ts_df[group_col].str.capitalize()

            # Average outcome per group per period
            agg = (
                ts_df.groupby(["group_label", time_col])[outcome_col]
                .mean()
                .reset_index()
            )

            fig_ts = px.line(
                agg,
                x=time_col,
                y=outcome_col,
                color="group_label",
                markers=True,
                title=f"Mean `{outcome_col}` Over Time by Group",
                color_discrete_map={"Treatment": "#e74c3c", "Control": "#3498db",
                                     "Control / Donor": "#3498db"},
                template="plotly_white",
                height=500,
            )

            # Treatment onset vertical line
            if "is_post" in ts_df.columns:
                pre_periods = ts_df.loc[ts_df["is_post"] == 0, time_col]
                if not pre_periods.empty:
                    onset = pre_periods.max() + 0.5
                    fig_ts.add_vline(
                        x=onset,
                        line_dash="dash",
                        line_color="#7f8c8d",
                        annotation_text="Treatment onset",
                        annotation_position="top left",
                    )
                    # Shading regions
                    x_min = int(ts_df[time_col].min())
                    x_max = int(ts_df[time_col].max())
                    fig_ts.add_vrect(
                        x0=x_min - 0.5, x1=onset,
                        fillcolor="#3498db", opacity=0.05,
                        annotation_text="Pre", annotation_position="top left",
                        line_width=0,
                    )
                    fig_ts.add_vrect(
                        x0=onset, x1=x_max + 0.5,
                        fillcolor="#e74c3c", opacity=0.05,
                        annotation_text="Post", annotation_position="top right",
                        line_width=0,
                    )

            fig_ts.update_layout(
                xaxis_title=time_col.replace("_", " ").title(),
                yaxis_title=f"Mean {outcome_col.replace('_', ' ').title()}",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Optional: individual units
            with st.expander("Show individual unit traces"):
                unit_col = None
                for c in ["unit_id", "geo_id", "market_id"]:
                    if c in ts_df.columns:
                        unit_col = c
                        break
                if unit_col:
                    fig_units = px.line(
                        ts_df,
                        x=time_col,
                        y=outcome_col,
                        color=unit_col,
                        line_group=unit_col,
                        title=f"Individual Unit Traces — `{outcome_col}`",
                        template="plotly_white",
                        height=600,
                    )
                    fig_units.update_traces(opacity=0.5)
                    st.plotly_chart(fig_units, use_container_width=True)
        else:
            st.warning("Could not identify required columns for time series plot.")
    else:
        st.info("No synthetic data available.")


# ── Tab 5: Validation ─────────────────────────────────────────────────────────
with tab_val:
    st.markdown("### Validation — True vs. Estimated Effect")
    val = results.get("validation_results", {})

    if val.get("success"):
        true_eff = val.get("true_effect", 0)
        est_eff = val.get("estimated_effect", 0)
        ci_lo = val.get("ci_lower")
        ci_hi = val.get("ci_upper")
        p_value = val.get("p_value")

        # Metric cards
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("True Effect", f"{true_eff:.4f}")
        with mc2:
            st.metric("Estimated Effect", f"{est_eff:.4f}")
        with mc3:
            st.metric("p-value", f"{p_value:.4f}" if p_value is not None else "N/A")

        if ci_lo is not None and ci_hi is not None:
            st.markdown(f"**95% CI:** [{ci_lo:.4f}, {ci_hi:.4f}]")

        st.divider()

        # Bar chart with error bars
        fig_val = go.Figure()

        # True effect bar
        fig_val.add_trace(go.Bar(
            x=["True Effect"],
            y=[true_eff],
            name="True Effect",
            marker_color="#2ecc71",
            width=0.35,
        ))

        # Estimated effect bar with CI error bars
        error_y_dict = {}
        if ci_lo is not None and ci_hi is not None:
            error_y_dict = dict(
                type="data",
                symmetric=False,
                array=[ci_hi - est_eff],
                arrayminus=[est_eff - ci_lo],
                visible=True,
                color="#2c3e50",
                thickness=2,
                width=8,
            )

        sig_color = "#e74c3c" if (p_value is not None and p_value < 0.05) else "#f39c12"
        fig_val.add_trace(go.Bar(
            x=["Estimated Effect"],
            y=[est_eff],
            name="Estimated Effect",
            marker_color=sig_color,
            width=0.35,
            error_y=error_y_dict if error_y_dict else None,
        ))

        # Reference line at true effect
        fig_val.add_hline(
            y=true_eff,
            line_dash="dash",
            line_color="#2ecc71",
            opacity=0.5,
        )

        significance_text = (
            f"p = {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'} at α=0.05)"
            if p_value is not None
            else ""
        )
        fig_val.update_layout(
            title=f"Validation: True vs. Estimated Effect  {significance_text}",
            yaxis_title="Effect Size",
            template="plotly_white",
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig_val, use_container_width=True)

        st.markdown(f"**Summary:** {val.get('summary', '')}")

    elif val:
        st.warning(f"⚠️ {val.get('error_message', 'Validation did not complete.')}")
        if val.get("summary"):
            st.markdown(val["summary"])
    else:
        st.info("No validation results available.")


# ── Tab 6: Sensitivity Analysis ───────────────────────────────────────────────
with tab_sens:
    st.markdown("### Sensitivity Analysis — Power Heatmap")
    st.caption(
        "How does statistical power change as we vary the significance level (α) "
        "and the effect size?"
    )

    sensitivity = st.session_state.sensitivity_data

    if sensitivity and sensitivity.get("grid"):
        grid = sensitivity["grid"]
        df_grid = pd.DataFrame(grid)

        # Pivot to matrix form
        pivot = df_grid.pivot_table(
            index="alpha",
            columns="effect_rel_pct",
            values="power",
            aggfunc="first",
        )
        pivot = pivot.sort_index(ascending=True)

        # Heatmap
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{c:.1f}%" for c in pivot.columns],
            y=[f"{r}" for r in pivot.index],
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar_title="Power",
            hovertemplate=(
                "Effect: %{x}<br>"
                "α: %{y}<br>"
                "Power: %{z:.3f}<extra></extra>"
            ),
        ))

        # Mark the user's operating point
        user_alpha = results.get("power_results", {}).get("effect_size_used")
        fig_heat.update_layout(
            title="Power Across Significance Level × Effect Size",
            xaxis_title="Relative Effect Size",
            yaxis_title="Significance Level (α)",
            template="plotly_white",
            height=500,
        )

        # Add contour line at 80% power
        fig_contour = go.Figure(data=go.Contour(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdYlGn",
            contours=dict(
                start=0,
                end=1,
                size=0.1,
                showlabels=True,
                labelfont=dict(size=10),
            ),
            colorbar_title="Power",
            hovertemplate=(
                "Effect: %{x:.1f}%<br>"
                "α: %{y}<br>"
                "Power: %{z:.3f}<extra></extra>"
            ),
        ))
        fig_contour.update_layout(
            title="Power Contour Plot — α × Effect Size",
            xaxis_title="Relative Effect Size (%)",
            yaxis_title="Significance Level (α)",
            template="plotly_white",
            height=500,
        )

        # Let user pick which view
        view = st.radio(
            "View",
            ["Heatmap", "Contour"],
            horizontal=True,
            key="sens_view",
        )
        if view == "Heatmap":
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.plotly_chart(fig_contour, use_container_width=True)

        st.download_button(
            "⬇ Download Sensitivity Data (CSV)",
            data=df_grid.to_csv(index=False),
            file_name="sensitivity_analysis.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Sensitivity analysis data not available. "
            "This requires a completed setup session."
        )


# ── Tab 7: Setup Report ──────────────────────────────────────────────────────
with tab_report:
    report_md = results.get("setup_report_markdown", "")
    st.download_button(
        "⬇ Download Setup Report",
        data=report_md,
        file_name="experiment_setup_report.md",
        mime="text/markdown",
    )
    st.markdown(report_md, unsafe_allow_html=False)


# ── Tab 8: Flags ─────────────────────────────────────────────────────────────
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
