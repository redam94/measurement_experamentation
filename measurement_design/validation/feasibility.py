"""
Domain validation: feasibility checks, red-flag detection, parameter sufficiency,
default imputation, and simplified statistical validation of synthetic data.

All functions are pure domain logic — no LLM, no LangGraph, no UI framework deps.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from measurement_design.knowledge.red_flags import RED_FLAG_CATALOG
from measurement_design.knowledge.assumptions import METHOD_ASSUMPTIONS
from measurement_design.simulation.power import compute_power


# ── Unit labels ──────────────────────────────────────────────────────────────

def unit_label_for_method(method_key: str) -> str:
    """Human-readable unit label for sample-size messages."""
    if method_key in ("geo_lift", "matched_market", "synthetic_control"):
        return "markets"
    if method_key == "did":
        return "units (markets x time periods)"
    return "users per group"


# ── Sufficiency checks ──────────────────────────────────────────────────────

def check_params_sufficient(
    topic: str,
    params: dict,
    method_key: str,
) -> tuple[bool, str | None]:
    """
    Check whether extraction got the critical values for the given topic.

    Returns (is_sufficient, missing_description_or_None).
    """
    if topic == "baseline_metrics":
        has_rate = params.get("baseline_rate") is not None
        has_value = params.get("baseline_metric_value") is not None
        has_std = params.get("baseline_metric_std") is not None

        if has_rate:
            # For proportions, std is derived — always sufficient
            return True, None
        if has_value and has_std:
            return True, None
        if has_value and not has_std:
            return False, "missing_std"
        if not has_value and not has_rate:
            if has_std:
                return False, "missing_baseline"
            return False, "missing_both"

    if topic == "expected_effect":
        has_pct = params.get("expected_lift_pct") is not None
        has_abs = params.get("expected_lift_abs") is not None
        if has_pct or has_abs:
            return True, None
        return False, "missing_effect"

    # Other topics: always sufficient (have good defaults)
    return True, None


# ── Red-flag detection ──────────────────────────────────────────────────────

def detect_red_flags(
    method_key: str,
    params: dict,
    facts: dict,
) -> list[dict]:
    """
    Scan current params/facts for feasibility concerns.

    Returns a list of RedFlag dicts (may be empty).
    """
    flags: list[dict] = []

    # ---- Coefficient of variation ----
    baseline_val = params.get("baseline_metric_value")
    baseline_std = params.get("baseline_metric_std")
    lift_pct = params.get("expected_lift_pct")
    lift_abs = params.get("expected_lift_abs")

    if baseline_val and baseline_std and baseline_val > 0:
        cv = baseline_std / baseline_val
        lift_desc = f"{lift_pct:.0%} relative" if lift_pct else (
            f"{lift_abs} absolute" if lift_abs else "unknown"
        )
        if cv > 1.0:
            flags.append({
                "severity": "critical",
                "flag": "high_cv",
                "title": RED_FLAG_CATALOG["high_cv"]["title"],
                "detail": RED_FLAG_CATALOG["high_cv"]["detail"].format(
                    cv=cv, lift=lift_desc
                ),
                "suggestion": RED_FLAG_CATALOG["high_cv"]["suggestion"],
            })
        elif cv > 0.5:
            flags.append({
                "severity": "warning",
                "flag": "high_cv",
                "title": RED_FLAG_CATALOG["high_cv"]["title"],
                "detail": RED_FLAG_CATALOG["high_cv"]["detail"].format(
                    cv=cv, lift=lift_desc
                ),
                "suggestion": RED_FLAG_CATALOG["high_cv"]["suggestion"],
            })

    # ---- Too few geo units ----
    if method_key in ("geo_lift", "matched_market", "did"):
        n_treat = params.get("num_treatment_units") or 0
        n_ctrl = params.get("num_control_units") or 0
        n_total = n_treat + n_ctrl
        if 0 < n_total < 10:
            flags.append({
                "severity": "critical",
                "flag": "too_few_geos",
                "title": RED_FLAG_CATALOG["too_few_geos"]["title"],
                "detail": RED_FLAG_CATALOG["too_few_geos"]["detail"].format(
                    n_total=n_total, n_treat=n_treat, n_ctrl=n_ctrl
                ),
                "suggestion": RED_FLAG_CATALOG["too_few_geos"]["suggestion"],
            })
        elif 0 < n_total < 20:
            flags.append({
                "severity": "warning",
                "flag": "too_few_geos",
                "title": RED_FLAG_CATALOG["too_few_geos"]["title"],
                "detail": RED_FLAG_CATALOG["too_few_geos"]["detail"].format(
                    n_total=n_total, n_treat=n_treat, n_ctrl=n_ctrl
                ),
                "suggestion": RED_FLAG_CATALOG["too_few_geos"]["suggestion"],
            })

    # ---- Short pre-period ----
    if method_key in ("did", "synthetic_control", "matched_market"):
        n_pre = params.get("num_pre_periods") or 0
        if 0 < n_pre < 4:
            flags.append({
                "severity": "critical",
                "flag": "short_pre_period",
                "title": RED_FLAG_CATALOG["short_pre_period"]["title"],
                "detail": RED_FLAG_CATALOG["short_pre_period"]["detail"].format(
                    n_pre=n_pre
                ),
                "suggestion": RED_FLAG_CATALOG["short_pre_period"]["suggestion"],
            })
        elif 0 < n_pre < 8:
            flags.append({
                "severity": "warning",
                "flag": "short_pre_period",
                "title": RED_FLAG_CATALOG["short_pre_period"]["title"],
                "detail": RED_FLAG_CATALOG["short_pre_period"]["detail"].format(
                    n_pre=n_pre
                ),
                "suggestion": RED_FLAG_CATALOG["short_pre_period"]["suggestion"],
            })

    # ---- Tiny expected effect ----
    if lift_pct is not None and lift_pct < 0.03:
        u_label = unit_label_for_method(method_key)
        flags.append({
            "severity": "warning",
            "flag": "tiny_effect",
            "title": RED_FLAG_CATALOG["tiny_effect"]["title"],
            "detail": (
                f"The expected lift of {lift_pct:.1%} is quite small relative to "
                f"the baseline variability. You may need a very large sample of "
                f"{u_label} to detect this reliably."
            ),
            "suggestion": RED_FLAG_CATALOG["tiny_effect"]["suggestion"],
        })

    # ---- High ICC ----
    icc = params.get("icc")
    if icc is not None and icc > 0.2 and method_key in ("did", "geo_lift", "matched_market"):
        flags.append({
            "severity": "warning",
            "flag": "high_icc",
            "title": RED_FLAG_CATALOG["high_icc"]["title"],
            "detail": RED_FLAG_CATALOG["high_icc"]["detail"].format(icc=icc),
            "suggestion": RED_FLAG_CATALOG["high_icc"]["suggestion"],
        })

    # ---- Few donors for synthetic control ----
    if method_key == "synthetic_control":
        n_donors = params.get("num_control_units") or 0
        if 0 < n_donors < 5:
            flags.append({
                "severity": "critical",
                "flag": "few_donors_sc",
                "title": RED_FLAG_CATALOG["few_donors_sc"]["title"],
                "detail": RED_FLAG_CATALOG["few_donors_sc"]["detail"].format(
                    n_donors=n_donors
                ),
                "suggestion": RED_FLAG_CATALOG["few_donors_sc"]["suggestion"],
            })
        elif 0 < n_donors < 10:
            flags.append({
                "severity": "warning",
                "flag": "few_donors_sc",
                "title": RED_FLAG_CATALOG["few_donors_sc"]["title"],
                "detail": RED_FLAG_CATALOG["few_donors_sc"]["detail"].format(
                    n_donors=n_donors
                ),
                "suggestion": RED_FLAG_CATALOG["few_donors_sc"]["suggestion"],
            })

    # ---- Extreme treatment/control imbalance ----
    n_treat = params.get("num_treatment_units") or 0
    n_ctrl = params.get("num_control_units") or 0
    if n_treat > 0 and n_ctrl > 0:
        ratio = max(n_treat, n_ctrl) / min(n_treat, n_ctrl)
        if ratio > 5:
            flags.append({
                "severity": "warning",
                "flag": "extreme_imbalance",
                "title": RED_FLAG_CATALOG["extreme_imbalance"]["title"],
                "detail": RED_FLAG_CATALOG["extreme_imbalance"]["detail"].format(
                    ratio=ratio
                ),
                "suggestion": RED_FLAG_CATALOG["extreme_imbalance"]["suggestion"],
            })

    return flags


# ── Interim power check ─────────────────────────────────────────────────────

def run_interim_power(
    method_key: str,
    params: dict,
    facts: dict,
) -> dict | None:
    """
    Run a quick power calculation if enough params exist (baseline + effect).

    Returns a power-results dict or None if we can't compute yet.
    Never raises — catches all exceptions so the flow is uninterrupted.
    """
    try:
        # Need at least a baseline and an effect
        has_baseline = (
            params.get("baseline_rate") is not None
            or params.get("baseline_metric_value") is not None
        )
        has_effect = (
            params.get("expected_lift_pct") is not None
            or params.get("expected_lift_abs") is not None
        )
        if not (has_baseline and has_effect):
            return None

        # Fill in temporary defaults for missing values so compute_power works
        tmp_params = dict(params)
        tmp_params.setdefault("alpha", 0.05)
        tmp_params.setdefault("power_target", 0.80)
        tmp_params.setdefault("one_sided", False)
        tmp_params.setdefault("n_simulations", 200)  # fast for interim
        tmp_params.setdefault("random_seed", 42)
        tmp_params.setdefault("icc", 0.05)

        if tmp_params.get("baseline_metric_value") and not tmp_params.get("baseline_metric_std"):
            tmp_params["baseline_metric_std"] = tmp_params["baseline_metric_value"] * 0.3

        # Method-specific defaults for interim calculation
        if method_key in ("did", "geo_lift", "matched_market"):
            tmp_params.setdefault("num_treatment_units", 10)
            tmp_params.setdefault("num_control_units", 10)
            tmp_params.setdefault("num_pre_periods", 8)
            tmp_params.setdefault("num_post_periods", 4)
            tmp_params.setdefault("cluster_size", 100)
        elif method_key == "synthetic_control":
            tmp_params.setdefault("num_treatment_units", 1)
            tmp_params.setdefault("num_control_units", 15)
            tmp_params.setdefault("num_pre_periods", 26)
            tmp_params.setdefault("num_post_periods", 8)
        elif method_key == "ddml":
            tmp_params.setdefault("num_treatment_units", 5000)
            tmp_params.setdefault("num_control_units", 5000)

        result = compute_power(method_key, tmp_params, facts)
        return result
    except Exception:
        return None


# ── Default application ─────────────────────────────────────────────────────

def apply_defaults(params: dict, facts: dict) -> None:
    """Fill in sensible defaults for missing parameters. Modifies params in place."""
    if not params.get("baseline_rate") and not params.get("baseline_metric_value"):
        # Try to infer from facts
        kpi = facts.get("kpi", "").lower()
        if "rate" in kpi or "conversion" in kpi or "ctr" in kpi:
            params.setdefault("baseline_rate", 0.03)
        else:
            params.setdefault("baseline_metric_value", 100.0)
            params.setdefault("baseline_metric_std", 30.0)

    if params.get("baseline_metric_value") and not params.get("baseline_metric_std"):
        params["baseline_metric_std"] = params["baseline_metric_value"] * 0.3

    if not params.get("expected_lift_pct") and not params.get("expected_lift_abs"):
        params["expected_lift_pct"] = 0.10  # default 10% relative

    params.setdefault("alpha", 0.05)
    params.setdefault("power_target", 0.80)
    params.setdefault("one_sided", False)
    params.setdefault("n_simulations", 1000)
    params.setdefault("random_seed", 42)
    params.setdefault("icc", 0.05)


# ── Assumptions summary ─────────────────────────────────────────────────────

def build_assumptions_summary(method_key: str) -> str:
    """Build a concise assumptions summary for the review message."""
    method_info = METHOD_ASSUMPTIONS.get(method_key)
    if not method_info:
        return "No specific assumptions documented."
    parts = []
    for a in method_info["assumptions"]:
        parts.append(f"- **{a['name']}**: {a['plain_language']}")
    return "\n".join(parts)


# ── Design problem identification ────────────────────────────────────────────

def identify_design_problems(
    power_results: dict,
    mde_results: dict,
    red_flags: list,
    params: dict,
    method_key: str,
) -> str:
    """Summarise key design problems for the redesign elicitation prompt."""
    problems = []
    achieved_power = power_results.get("achieved_power", 0) or 0
    if achieved_power < 0.80:
        problems.append(
            f"Power is only {achieved_power:.0%} (target is 80%). The test is likely "
            f"to miss a real effect."
        )
    mde_rel = mde_results.get("mde_relative_pct")
    lift_pct = params.get("expected_lift_pct")
    if mde_rel and lift_pct and mde_rel > lift_pct * 100:
        problems.append(
            f"MDE ({mde_rel:.1f}%) is larger than the expected effect "
            f"({lift_pct * 100:.1f}%). The test cannot reliably detect the expected lift."
        )
    for rf in red_flags:
        if rf.get("severity") == "critical":
            problems.append(f"{rf.get('title', 'Issue')}: {rf.get('detail', '')}")
    if not problems:
        problems.append("No critical issues, but review the results to confirm they meet your needs.")
    return "\n".join(f"- {p}" for p in problems)


# ── Synthetic data validation ────────────────────────────────────────────────

def run_validation(method_key: str, synth_data: dict) -> dict:
    """
    Run a quick statistical test on synthetic data to validate the approach.
    This is a simplified version — not the full PyMC scaffold.
    """
    csv_str = synth_data.get("csv_string", "")
    true_effect = synth_data.get("true_effect", 0)

    if not csv_str:
        return {
            "success": False,
            "error_message": "No synthetic data available for validation.",
        }

    try:
        df = pd.read_csv(io.StringIO(csv_str))
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Could not parse synthetic data: {e}",
        }

    try:
        if method_key == "ab_test":
            return _validate_ab_test(df, true_effect)
        elif method_key == "did":
            return _validate_did(df, true_effect)
        elif method_key in ("geo_lift", "matched_market"):
            return _validate_geo(df, true_effect)
        elif method_key == "synthetic_control":
            return _validate_synth_ctrl(df, true_effect)
        elif method_key == "ddml":
            return _validate_ddml(df, true_effect)
        else:
            return _validate_ab_test(df, true_effect)
    except Exception as e:
        return {
            "success": False,
            "estimated_effect": None,
            "true_effect": true_effect,
            "error_message": f"Validation analysis raised an error: {e}",
            "summary": "The synthetic data was generated but validation did not complete.",
        }


def _validate_ab_test(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate A/B test data with a simple two-sample test."""
    if "converted" in df.columns:
        ctrl = df[df["group"] == "control"]["converted"]
        treat = df[df["group"] == "treatment"]["converted"]
    elif "outcome" in df.columns:
        ctrl = df[df["group"] == "control"]["outcome"]
        treat = df[df["group"] == "treatment"]["outcome"]
    else:
        return {"success": False, "error_message": "Unexpected column structure."}

    diff = treat.mean() - ctrl.mean()
    t_stat, p_val = scipy_stats.ttest_ind(treat, ctrl)

    # Bootstrap CI
    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(n_boot):
        c_samp = ctrl.sample(n=len(ctrl), replace=True, random_state=rng.integers(1e9))
        t_samp = treat.sample(n=len(treat), replace=True, random_state=rng.integers(1e9))
        boot_diffs.append(t_samp.mean() - c_samp.mean())
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "success": True,
        "estimated_effect": round(float(diff), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_val), 6),
        "summary": (
            f"Two-sample test: estimated effect = {diff:.4f} "
            f"(true = {true_effect:.4f}), p = {p_val:.4f}. "
            f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
    }


def _validate_did(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate DiD data with a simple DiD estimator."""
    pre_treat = df[(df["group"] == "treatment") & (df["is_post"] == 0)]["outcome"].mean()
    post_treat = df[(df["group"] == "treatment") & (df["is_post"] == 1)]["outcome"].mean()
    pre_ctrl = df[(df["group"] == "control") & (df["is_post"] == 0)]["outcome"].mean()
    post_ctrl = df[(df["group"] == "control") & (df["is_post"] == 1)]["outcome"].mean()

    did_est = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)

    # Simple SE from unit-level DiD
    units = df["unit_id"].unique()
    unit_dids = []
    for u in units:
        ud = df[df["unit_id"] == u]
        pre = ud[ud["is_post"] == 0]["outcome"].mean()
        post = ud[ud["is_post"] == 1]["outcome"].mean()
        group = ud["group"].iloc[0]
        unit_dids.append({"unit": u, "group": group, "diff": post - pre})

    ud_df = pd.DataFrame(unit_dids)
    treat_diffs = ud_df[ud_df["group"] == "treatment"]["diff"]
    ctrl_diffs = ud_df[ud_df["group"] == "control"]["diff"]
    t_stat, p_val = scipy_stats.ttest_ind(treat_diffs, ctrl_diffs)

    se = np.sqrt(treat_diffs.var() / len(treat_diffs) + ctrl_diffs.var() / len(ctrl_diffs))
    ci_lower = did_est - 1.96 * se
    ci_upper = did_est + 1.96 * se

    return {
        "success": True,
        "estimated_effect": round(float(did_est), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_val), 6),
        "summary": (
            f"DiD estimator: {did_est:.4f} (true = {true_effect:.4f}), "
            f"p = {p_val:.4f}. 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
    }


def _validate_geo(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate geo/market data."""
    outcome_col = "kpi_value" if "kpi_value" in df.columns else "outcome"

    # Post-period comparison
    post = df[df["is_post"] == 1]
    treat_markets = post[post["group"] == "treatment"].groupby(
        post[post["group"] == "treatment"].columns[0]
    )[outcome_col].mean()
    ctrl_markets = post[post["group"] == "control"].groupby(
        post[post["group"] == "control"].columns[0]
    )[outcome_col].mean()

    diff = treat_markets.mean() - ctrl_markets.mean()
    t_stat, p_val = scipy_stats.ttest_ind(treat_markets, ctrl_markets)

    se = np.sqrt(treat_markets.var() / len(treat_markets) + ctrl_markets.var() / len(ctrl_markets))
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    return {
        "success": True,
        "estimated_effect": round(float(diff), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_val), 6),
        "summary": (
            f"Geo market comparison: {diff:.4f} (true = {true_effect:.4f}), "
            f"p = {p_val:.4f}. 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
    }


def _validate_synth_ctrl(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate synthetic control data with simple mean comparison."""
    post = df[df["is_post"] == 1]
    treat_mean = post[post["is_treated"] == 1]["outcome"].mean()
    donor_mean = post[post["is_treated"] == 0].groupby("unit_id")["outcome"].mean().mean()

    gap = treat_mean - donor_mean

    return {
        "success": True,
        "estimated_effect": round(float(gap), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": None,
        "ci_upper": None,
        "p_value": None,
        "summary": (
            f"Avg post-period gap: {gap:.4f} (true = {true_effect:.4f}). "
            f"Formal inference requires permutation test — see the full scaffold."
        ),
    }


def _validate_ddml(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate DDML data with OLS partialling out."""
    feature_cols = [c for c in df.columns if c.startswith("x_")]
    X = df[feature_cols].values
    D = df["treatment"].values
    Y = df["outcome"].values

    # Partial out X via OLS
    XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
    beta_d = np.linalg.solve(XtX, X.T @ D)
    D_resid = D - X @ beta_d
    beta_y = np.linalg.solve(XtX, X.T @ Y)
    Y_resid = Y - X @ beta_y

    denom = D_resid @ D_resid
    theta = (D_resid @ Y_resid) / denom
    e = Y_resid - theta * D_resid
    V = np.sum((D_resid ** 2) * (e ** 2)) / (denom ** 2)
    se = np.sqrt(V)

    z = theta / se
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se

    return {
        "success": True,
        "estimated_effect": round(float(theta), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_val), 6),
        "summary": (
            f"Partialling-out estimator: theta = {theta:.4f} (true = {true_effect:.4f}), "
            f"p = {p_val:.4f}. 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
    }
