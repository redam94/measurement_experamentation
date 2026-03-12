"""
Synthetic data generation for each measurement method.

Generates realistic fake datasets that match the user's scenario,
with a known ground-truth treatment effect injected.  These datasets
let the user validate their analysis pipeline before launch.
"""
from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import pandas as pd


# ── A/B Test (proportions) ──────────────────────────────────────────────────

def synthetic_ab_test_proportions(
    baseline_rate: float,
    lift_abs: float,
    n_per_group: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate user-level conversion data for an A/B test."""
    rng = np.random.default_rng(seed)

    n_total = n_per_group * 2
    group = np.array(["control"] * n_per_group + ["treatment"] * n_per_group)
    user_id = np.arange(1, n_total + 1)

    p_ctrl = baseline_rate
    p_treat = min(max(baseline_rate + lift_abs, 0.001), 0.999)

    converted = np.concatenate([
        rng.binomial(1, p_ctrl, n_per_group),
        rng.binomial(1, p_treat, n_per_group),
    ])

    df = pd.DataFrame({
        "user_id": user_id,
        "group": group,
        "converted": converted,
    })

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    return {
        "csv_string": csv_buf.getvalue(),
        "n_rows": n_total,
        "columns": list(df.columns),
        "true_effect": round(lift_abs, 6),
        "description": (
            f"A/B test synthetic data: {n_per_group} users per group, "
            f"baseline rate={p_ctrl:.4f}, treatment rate={p_treat:.4f}, "
            f"true lift={lift_abs:.4f}."
        ),
    }


# ── A/B Test (continuous) ───────────────────────────────────────────────────

def synthetic_ab_test_continuous(
    baseline_mean: float,
    baseline_std: float,
    lift_abs: float,
    n_per_group: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate user-level continuous outcome data for an A/B test."""
    rng = np.random.default_rng(seed)

    n_total = n_per_group * 2
    group = np.array(["control"] * n_per_group + ["treatment"] * n_per_group)
    user_id = np.arange(1, n_total + 1)

    outcome = np.concatenate([
        rng.normal(baseline_mean, baseline_std, n_per_group),
        rng.normal(baseline_mean + lift_abs, baseline_std, n_per_group),
    ])

    df = pd.DataFrame({
        "user_id": user_id,
        "group": group,
        "outcome": np.round(outcome, 4),
    })

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    return {
        "csv_string": csv_buf.getvalue(),
        "n_rows": n_total,
        "columns": list(df.columns),
        "true_effect": round(lift_abs, 6),
        "description": (
            f"A/B test synthetic data (continuous): {n_per_group} users per group, "
            f"baseline μ={baseline_mean:.2f}, σ={baseline_std:.2f}, "
            f"true lift={lift_abs:.4f}."
        ),
    }


# ── Difference-in-Differences ───────────────────────────────────────────────

def synthetic_did(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_abs: float,
    num_treatment_units: int,
    num_control_units: int,
    num_pre_periods: int,
    num_post_periods: int,
    icc: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate panel data for a DiD design."""
    rng = np.random.default_rng(seed)

    N_t = num_treatment_units
    N_c = num_control_units
    N = N_t + N_c
    T_pre = num_pre_periods
    T_post = num_post_periods
    T = T_pre + T_post
    sigma = baseline_metric_std

    sigma_u = sigma * math.sqrt(icc)
    sigma_e = sigma * math.sqrt(1 - icc)

    unit_effects = rng.normal(0, sigma_u, N)
    time_effects = rng.normal(0, sigma_e * 0.2, T)
    # Add a slight upward trend
    trend = np.linspace(0, sigma * 0.1, T)

    rows = []
    for i in range(N):
        is_treat = i < N_t
        unit_name = f"{'T' if is_treat else 'C'}_{i + 1}"
        for t in range(T):
            is_post = t >= T_pre
            effect = lift_abs if (is_treat and is_post) else 0.0
            y = (
                baseline_metric_value
                + unit_effects[i]
                + time_effects[t]
                + trend[t]
                + effect
                + rng.normal(0, sigma_e)
            )
            rows.append({
                "unit_id": unit_name,
                "group": "treatment" if is_treat else "control",
                "period": t + 1,
                "is_post": int(is_post),
                "outcome": round(y, 4),
            })

    df = pd.DataFrame(rows)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    return {
        "csv_string": csv_buf.getvalue(),
        "n_rows": len(df),
        "columns": list(df.columns),
        "true_effect": round(lift_abs, 6),
        "description": (
            f"DiD panel data: {N_t} treatment × {N_c} control units, "
            f"{T_pre} pre + {T_post} post periods, "
            f"baseline={baseline_metric_value:.2f}, true effect={lift_abs:.4f}."
        ),
    }


# ── Geo Lift ────────────────────────────────────────────────────────────────

def synthetic_geo_lift(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_abs: float,
    num_treatment_geos: int,
    num_control_geos: int,
    num_pre_periods: int,
    num_post_periods: int,
    icc: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate market-level weekly data for a geo lift test."""
    rng = np.random.default_rng(seed)

    N_t = num_treatment_geos
    N_c = num_control_geos
    N = N_t + N_c
    T_pre = num_pre_periods
    T_post = num_post_periods
    T = T_pre + T_post
    sigma = baseline_metric_std

    sigma_m = sigma * math.sqrt(icc)
    sigma_e = sigma * math.sqrt(1 - icc)

    # Market baselines (some variation in market size)
    market_scales = rng.uniform(0.5, 2.0, N)
    market_effects = rng.normal(0, sigma_m, N)

    # Shared seasonality
    weekly_seasonal = sigma * 0.15 * np.sin(np.linspace(0, 4 * math.pi, T))

    rows = []
    for i in range(N):
        is_treat = i < N_t
        geo_name = f"geo_{i + 1:03d}"
        for t in range(T):
            is_post = t >= T_pre
            effect = lift_abs * market_scales[i] if (is_treat and is_post) else 0.0
            y = (
                baseline_metric_value * market_scales[i]
                + market_effects[i]
                + weekly_seasonal[t]
                + effect
                + rng.normal(0, sigma_e)
            )
            rows.append({
                "geo_id": geo_name,
                "group": "treatment" if is_treat else "control",
                "week": t + 1,
                "is_post": int(is_post),
                "kpi_value": round(max(y, 0), 4),
            })

    df = pd.DataFrame(rows)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    return {
        "csv_string": csv_buf.getvalue(),
        "n_rows": len(df),
        "columns": list(df.columns),
        "true_effect": round(lift_abs, 6),
        "description": (
            f"Geo lift synthetic data: {N_t} treatment × {N_c} control geos, "
            f"{T_pre} pre + {T_post} post weeks, "
            f"baseline={baseline_metric_value:.2f}/week, true effect={lift_abs:.4f}."
        ),
    }


# ── Synthetic Control ───────────────────────────────────────────────────────

def synthetic_synthetic_control(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_abs: float,
    num_donor_units: int,
    num_pre_periods: int,
    num_post_periods: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate time-series data for synthetic control (1 treated + J donors)."""
    rng = np.random.default_rng(seed)

    J = num_donor_units
    T_pre = num_pre_periods
    T_post = num_post_periods
    T = T_pre + T_post
    sigma = baseline_metric_std

    # Common factor model
    n_factors = 3
    factors = rng.normal(0, 1, (n_factors, T))
    loadings = rng.uniform(0.3, 1.5, (J + 1, n_factors))

    rows = []
    for j in range(J + 1):
        is_treated = j == 0
        unit_name = "treatment" if is_treated else f"donor_{j:03d}"
        for t in range(T):
            is_post = t >= T_pre
            common = sigma * 0.5 * (loadings[j] @ factors[:, t])
            effect = lift_abs if (is_treated and is_post) else 0.0
            y = (
                baseline_metric_value
                + common
                + effect
                + rng.normal(0, sigma * 0.2)
            )
            rows.append({
                "unit_id": unit_name,
                "is_treated": int(is_treated),
                "period": t + 1,
                "is_post": int(is_post),
                "outcome": round(y, 4),
            })

    df = pd.DataFrame(rows)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    return {
        "csv_string": csv_buf.getvalue(),
        "n_rows": len(df),
        "columns": list(df.columns),
        "true_effect": round(lift_abs, 6),
        "description": (
            f"Synthetic control data: 1 treated unit + {J} donors, "
            f"{T_pre} pre + {T_post} post periods, "
            f"baseline={baseline_metric_value:.2f}, true effect={lift_abs:.4f}."
        ),
    }


# ── Matched Market ──────────────────────────────────────────────────────────

def synthetic_matched_market(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_abs: float,
    num_pairs: int,
    num_pre_periods: int,
    num_post_periods: int,
    icc: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate paired market data for a matched market test."""
    rng = np.random.default_rng(seed)

    T_pre = num_pre_periods
    T_post = num_post_periods
    T = T_pre + T_post
    sigma = baseline_metric_std

    sigma_pair = sigma * math.sqrt(icc)
    sigma_e = sigma * math.sqrt(1 - icc)

    rows = []
    for p in range(num_pairs):
        pair_effect = rng.normal(0, sigma_pair)
        pair_scale = rng.uniform(0.7, 1.3)

        for role in ["treatment", "control"]:
            market_name = f"pair_{p + 1}_{role[0].upper()}"
            for t in range(T):
                is_post = t >= T_pre
                effect = lift_abs * pair_scale if (role == "treatment" and is_post) else 0.0
                y = (
                    baseline_metric_value * pair_scale
                    + pair_effect
                    + effect
                    + rng.normal(0, sigma_e)
                )
                rows.append({
                    "pair_id": p + 1,
                    "market_id": market_name,
                    "group": role,
                    "week": t + 1,
                    "is_post": int(is_post),
                    "kpi_value": round(max(y, 0), 4),
                })

    df = pd.DataFrame(rows)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    return {
        "csv_string": csv_buf.getvalue(),
        "n_rows": len(df),
        "columns": list(df.columns),
        "true_effect": round(lift_abs, 6),
        "description": (
            f"Matched market data: {num_pairs} pairs, "
            f"{T_pre} pre + {T_post} post weeks, "
            f"baseline={baseline_metric_value:.2f}/week, true effect={lift_abs:.4f}."
        ),
    }


# ── DDML ────────────────────────────────────────────────────────────────────

def synthetic_ddml(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_abs: float,
    n_obs: int,
    n_covariates: int = 10,
    r2_treatment: float = 0.30,
    r2_outcome: float = 0.40,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate observational data with confounders for DDML."""
    rng = np.random.default_rng(seed)

    sigma = baseline_metric_std

    # Generate covariates
    X = rng.normal(0, 1, (n_obs, n_covariates))
    feature_names = [f"x_{i + 1}" for i in range(n_covariates)]

    # Treatment depends on covariates (confounding)
    beta_d = rng.normal(0, 1, n_covariates)
    beta_d *= math.sqrt(r2_treatment / max(np.var(X @ beta_d), 1e-12))
    logit = X @ beta_d
    prob = 1 / (1 + np.exp(-np.clip(logit, -10, 10)))
    D = rng.binomial(1, prob)

    # Outcome depends on X and D
    beta_y = rng.normal(0, 1, n_covariates)
    beta_y *= sigma * math.sqrt(r2_outcome / max(np.var(X @ beta_y), 1e-12))
    Y = (
        baseline_metric_value
        + X @ beta_y
        + lift_abs * D
        + rng.normal(0, sigma * math.sqrt(max(1 - r2_outcome, 0.01)), n_obs)
    )

    # Build DataFrame
    data = {"user_id": np.arange(1, n_obs + 1)}
    for i, fname in enumerate(feature_names):
        data[fname] = np.round(X[:, i], 4)
    data["treatment"] = D
    data["outcome"] = np.round(Y, 4)

    df = pd.DataFrame(data)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    return {
        "csv_string": csv_buf.getvalue(),
        "n_rows": n_obs,
        "columns": list(df.columns),
        "true_effect": round(lift_abs, 6),
        "description": (
            f"DDML synthetic data: {n_obs:,} obs, {n_covariates} covariates, "
            f"R²(D|X)={r2_treatment:.2f}, R²(Y|X)={r2_outcome:.2f}, "
            f"true effect={lift_abs:.4f}."
        ),
    }


# ── Dispatcher ───────────────────────────────────────────────────────────────

def generate_synthetic_data(
    method_key: str,
    setup_params: dict,
    elicited_facts: dict,
    power_results: dict | None = None,
    mde_results: dict | None = None,
) -> dict[str, Any]:
    """
    Dispatch to the appropriate synthetic data generator.
    Uses the MDE or expected lift as the injected true effect.
    """
    seed = setup_params.get("random_seed", 42)

    baseline_rate = setup_params.get("baseline_rate")
    baseline_val = setup_params.get("baseline_metric_value", 100.0)
    baseline_std = setup_params.get("baseline_metric_std", baseline_val * 0.3)
    lift_pct = setup_params.get("expected_lift_pct")
    lift_abs = setup_params.get("expected_lift_abs")

    n_treat = setup_params.get("num_treatment_units")
    n_ctrl = setup_params.get("num_control_units")
    n_pre = setup_params.get("num_pre_periods")
    n_post = setup_params.get("num_post_periods")

    # Determine effect size: prefer MDE, then user-specified lift
    if mde_results and mde_results.get("mde_absolute") is not None:
        effect_abs = mde_results["mde_absolute"]
    elif lift_abs is not None:
        effect_abs = lift_abs
    elif lift_pct is not None and baseline_rate is not None:
        effect_abs = baseline_rate * lift_pct
    elif lift_pct is not None:
        effect_abs = baseline_val * lift_pct
    else:
        effect_abs = baseline_val * 0.10 if baseline_rate is None else baseline_rate * 0.10

    n_per_group = (power_results or {}).get("required_sample_size") or 1000

    if method_key == "ab_test":
        if baseline_rate is not None and baseline_rate > 0:
            return synthetic_ab_test_proportions(
                baseline_rate=baseline_rate,
                lift_abs=effect_abs,
                n_per_group=n_per_group,
                seed=seed,
            )
        else:
            return synthetic_ab_test_continuous(
                baseline_mean=baseline_val,
                baseline_std=baseline_std,
                lift_abs=effect_abs,
                n_per_group=n_per_group,
                seed=seed,
            )

    elif method_key == "did":
        return synthetic_did(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_abs=effect_abs,
            num_treatment_units=n_treat or 5,
            num_control_units=n_ctrl or 10,
            num_pre_periods=n_pre or int(elicited_facts.get("pre_period_weeks", 12) or 12),
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 8) or 8),
            seed=seed,
        )

    elif method_key == "geo_lift":
        return synthetic_geo_lift(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_abs=effect_abs,
            num_treatment_geos=n_treat or int(elicited_facts.get("num_markets", 8) or 8) // 3,
            num_control_geos=n_ctrl or int(elicited_facts.get("num_markets", 25) or 25) * 2 // 3,
            num_pre_periods=n_pre or int(elicited_facts.get("pre_period_weeks", 12) or 12),
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 6) or 6),
            seed=seed,
        )

    elif method_key == "synthetic_control":
        return synthetic_synthetic_control(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_abs=effect_abs,
            num_donor_units=n_ctrl or int(elicited_facts.get("num_markets", 20) or 20) - 1,
            num_pre_periods=n_pre or int(elicited_facts.get("pre_period_weeks", 52) or 52),
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 12) or 12),
            seed=seed,
        )

    elif method_key == "matched_market":
        num_pairs = (n_treat or 6)
        return synthetic_matched_market(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_abs=effect_abs,
            num_pairs=num_pairs,
            num_pre_periods=n_pre or int(elicited_facts.get("pre_period_weeks", 8) or 8),
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 4) or 4),
            seed=seed,
        )

    elif method_key == "ddml":
        ss_map = {"small (<10k)": 5000, "medium (10k-1M)": 50000, "large (>1M)": 200000}
        n_obs = ss_map.get(
            elicited_facts.get("sample_size_estimate", "medium (10k-1M)"),
            50000,
        )
        return synthetic_ddml(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_abs=effect_abs,
            n_obs=min(n_obs, 100000),  # cap for CSV size
            seed=seed,
        )

    else:
        # Fallback
        return synthetic_ab_test_continuous(
            baseline_mean=baseline_val,
            baseline_std=baseline_std,
            lift_abs=effect_abs,
            n_per_group=n_per_group,
            seed=seed,
        )
