"""
Page 5 — Data Templates

Schema documentation + interactive form to generate downloadable CSV
templates for each measurement method.
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Data Templates · Measurement Design Agent",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

from shared import (
    METHOD_NAMES,
    METHOD_SCHEMAS,
    api,
    init_session_state,
    render_sidebar,
)

init_session_state()
render_sidebar()

# ── Page header ─────────────────────────────────────────────────────────────────
st.header("📝 Data Templates")
st.markdown(
    "Download CSV templates and review column schemas for each measurement "
    "method. Use these to prepare your real data before running an experiment."
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Schema Documentation
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Column Schemas by Method")

for method_key, schema in METHOD_SCHEMAS.items():
    method_name = METHOD_NAMES.get(method_key, method_key.replace("_", " ").title())
    with st.expander(f"**{method_name}** — {schema['description'][:80]}…", expanded=False):
        st.markdown(schema["description"])

        # Column table
        col_data = []
        for col in schema["columns"]:
            col_data.append({
                "Column": f"`{col['name']}`",
                "Type": col["dtype"],
                "Required": "✅" if col["required"] else "Optional",
                "Description": col["description"],
                "Example": col["example"],
            })
        st.table(pd.DataFrame(col_data))

        if schema.get("notes"):
            st.info(f"💡 {schema['notes']}")

        # One-click blank-template download from backend
        try:
            tpl = api("GET", f"/methods/{method_key}/template")
            csv_text = tpl.get("csv_template", "")
            if csv_text:
                st.download_button(
                    f"⬇ Download Blank Template — {method_name}",
                    data=csv_text,
                    file_name=f"template_{method_key}.csv",
                    mime="text/csv",
                    key=f"dl_blank_{method_key}",
                )
        except Exception:
            st.caption("(Backend unavailable — start the API to download templates.)")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Interactive Template Generator
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🛠️ Interactive Template Generator")
st.markdown(
    "Select a method and fill in your parameters to generate a realistic "
    "example CSV with synthetic data matching your scenario."
)

selected_method = st.selectbox(
    "Choose a method",
    options=list(METHOD_SCHEMAS.keys()),
    format_func=lambda k: METHOD_NAMES.get(k, k.replace("_", " ").title()),
)

schema = METHOD_SCHEMAS[selected_method]
form_fields = schema.get("form_fields", {})

if not form_fields:
    st.info("No interactive form available for this method.")
    st.stop()

st.markdown(f"**{METHOD_NAMES.get(selected_method, selected_method)}**")
st.caption(schema["description"])

# Build form
with st.form(key="template_form"):
    st.markdown("#### Parameters")

    user_params: dict = {}
    cols = st.columns(min(len(form_fields), 3))
    for i, (field_key, field_cfg) in enumerate(form_fields.items()):
        with cols[i % len(cols)]:
            label = field_cfg["label"]
            ftype = field_cfg["type"]
            default = field_cfg.get("default", 0)

            if ftype == "int":
                user_params[field_key] = st.number_input(
                    label,
                    min_value=field_cfg.get("min", 1),
                    max_value=field_cfg.get("max", 1_000_000),
                    value=default,
                    step=1,
                    key=f"tpl_{selected_method}_{field_key}",
                )
            elif ftype == "float":
                user_params[field_key] = st.number_input(
                    label,
                    min_value=field_cfg.get("min", -1e9),
                    max_value=field_cfg.get("max", 1e9),
                    value=float(default),
                    step=0.001,
                    format="%.4f",
                    key=f"tpl_{selected_method}_{field_key}",
                )

    n_rows = st.slider(
        "Number of example rows",
        min_value=5,
        max_value=500,
        value=50,
        step=5,
        key="tpl_n_rows",
    )

    submitted = st.form_submit_button("Generate Template", type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions for local template generation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_generation_params(method: str, params: dict, n_rows: int) -> dict:
    """Map user form inputs to synthetic generator kwargs."""
    p: dict = {"seed": 42}

    if method == "ab_test":
        p["baseline_rate"] = params.get("baseline_rate", 0.05)
        p["lift_abs"] = params.get("lift_abs", 0.005)
        p["n_per_group"] = max(n_rows // 2, 3)

    elif method == "did":
        p["baseline_metric_value"] = params.get("baseline_value", 100.0)
        p["baseline_metric_std"] = params.get("baseline_std", 30.0)
        p["lift_abs"] = params.get("lift_abs", 5.0)
        p["num_treatment_units"] = params.get("num_treatment_units", 5)
        p["num_control_units"] = params.get("num_control_units", 10)
        p["num_pre_periods"] = params.get("num_pre_periods", 12)
        p["num_post_periods"] = params.get("num_post_periods", 8)

    elif method == "geo_lift":
        p["baseline_metric_value"] = params.get("baseline_value", 5000.0)
        p["baseline_metric_std"] = params.get("baseline_std", 1500.0)
        p["lift_abs"] = params.get("lift_abs", 250.0)
        p["num_treatment_geos"] = params.get("num_treatment_geos", 5)
        p["num_control_geos"] = params.get("num_control_geos", 15)
        p["num_pre_periods"] = params.get("num_pre_periods", 12)
        p["num_post_periods"] = params.get("num_post_periods", 6)

    elif method == "synthetic_control":
        p["baseline_metric_value"] = params.get("baseline_value", 100.0)
        p["baseline_metric_std"] = params.get("baseline_std", 20.0)
        p["lift_abs"] = params.get("lift_abs", 10.0)
        p["num_donor_units"] = params.get("num_donor_units", 15)
        p["num_pre_periods"] = params.get("num_pre_periods", 52)
        p["num_post_periods"] = params.get("num_post_periods", 12)

    elif method == "matched_market":
        p["baseline_metric_value"] = params.get("baseline_value", 5000.0)
        p["baseline_metric_std"] = params.get("baseline_std", 1500.0)
        p["lift_abs"] = params.get("lift_abs", 250.0)
        p["num_pairs"] = params.get("num_pairs", 6)
        p["num_pre_periods"] = params.get("num_pre_periods", 8)
        p["num_post_periods"] = params.get("num_post_periods", 4)

    elif method == "ddml":
        p["baseline_metric_value"] = params.get("baseline_value", 100.0)
        p["baseline_metric_std"] = params.get("baseline_std", 30.0)
        p["lift_abs"] = params.get("lift_abs", 5.0)
        p["n_obs"] = n_rows
        p["n_covariates"] = params.get("n_covariates", 10)

    return p


def _generate_locally(method: str, params: dict) -> str | None:
    """Generate synthetic data using the core measurement_design generators."""
    try:
        from measurement_design.simulation.synthetic import (
            synthetic_ab_test_proportions,
            synthetic_did,
            synthetic_geo_lift,
            synthetic_synthetic_control,
            synthetic_matched_market,
            synthetic_ddml,
        )

        if method == "ab_test":
            result = synthetic_ab_test_proportions(**params)
        elif method == "did":
            result = synthetic_did(**params)
        elif method == "geo_lift":
            result = synthetic_geo_lift(**params)
        elif method == "synthetic_control":
            result = synthetic_synthetic_control(**params)
        elif method == "matched_market":
            result = synthetic_matched_market(**params)
        elif method == "ddml":
            result = synthetic_ddml(**params)
        else:
            return None

        return result.get("csv_string")
    except Exception as e:
        st.warning(f"Local generation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Execute generation when form is submitted
# ═══════════════════════════════════════════════════════════════════════════════

if submitted:
    with st.spinner("Generating synthetic template data…"):
        try:
            gen_params = _build_generation_params(selected_method, user_params, n_rows)
            csv_text = _generate_locally(selected_method, gen_params)

            if csv_text:
                df_preview = pd.read_csv(io.StringIO(csv_text))
                st.success(f"Generated **{len(df_preview):,} rows × {len(df_preview.columns)} columns**")

                st.download_button(
                    "⬇ Download Generated Template (CSV)",
                    data=csv_text,
                    file_name=f"template_{selected_method}_custom.csv",
                    mime="text/csv",
                    key="dl_generated",
                )

                st.dataframe(df_preview.head(100), use_container_width=True)

                st.markdown("#### Summary Statistics")
                st.dataframe(df_preview.describe().round(4), use_container_width=True)
            else:
                st.error("Could not generate template data.")
        except Exception as e:
            st.error(f"Error generating template: {e}")
