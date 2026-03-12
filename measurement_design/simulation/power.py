"""
Power analysis and sample-size calculations for each measurement method.

Each function returns a PowerResults-compatible dict with:
  - required_sample_size
  - achieved_power
  - power_curve (list of {n, power} dicts)
  - effect_size_used
  - notes
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


# ── Generic helpers ──────────────────────────────────────────────────────────

def _z(alpha: float, one_sided: bool = False) -> float:
    """z critical value for given alpha."""
    if one_sided:
        return stats.norm.ppf(1 - alpha)
    return stats.norm.ppf(1 - alpha / 2)


def _power_curve(
    n_range: list[int],
    power_fn,
) -> list[dict[str, float]]:
    """Compute power for a range of sample sizes."""
    return [{"n": int(n), "power": round(float(power_fn(n)), 4)} for n in n_range]


def _default_n_range(n_required: int) -> list[int]:
    """Generate a sensible range of sample sizes around the required n."""
    low = max(10, n_required // 5)
    high = n_required * 3
    step = max(1, (high - low) // 50)
    return list(range(low, high + 1, step))


# ── A/B Test (two-sample proportions) ───────────────────────────────────────

def power_ab_test(
    baseline_rate: float,
    lift_pct: float | None = None,
    lift_abs: float | None = None,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
    ratio: float = 1.0,
) -> dict[str, Any]:
    """
    Two-sample z-test for proportions.
    ratio = n_control / n_treatment (1.0 = equal allocation).
    """
    p1 = baseline_rate
    if lift_abs is not None:
        p2 = p1 + lift_abs
    elif lift_pct is not None:
        p2 = p1 * (1 + lift_pct)
    else:
        # Default: 10% relative lift
        lift_pct = 0.10
        p2 = p1 * (1 + lift_pct)

    delta = abs(p2 - p1)
    if delta < 1e-12:
        return {
            "required_sample_size": None,
            "achieved_power": 0.0,
            "power_curve": [],
            "effect_size_used": 0.0,
            "notes": "Effect size is zero — cannot compute power.",
        }

    # Pooled variance under alternative
    za = _z(alpha, one_sided)
    zb = stats.norm.ppf(power_target)

    # Per-group sample size (Fleiss formula)
    p_bar = (p1 + ratio * p2) / (1 + ratio)
    n_treat = (
        (za * math.sqrt((1 + 1 / ratio) * p_bar * (1 - p_bar))
         + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2
    ) / (delta ** 2)
    n_treat = int(math.ceil(n_treat))

    # Power function for arbitrary n
    def _power_at_n(n: int) -> float:
        se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / (n * ratio))
        z_stat = delta / se
        if one_sided:
            return float(stats.norm.cdf(z_stat - za))
        return float(stats.norm.cdf(z_stat - za) + stats.norm.cdf(-z_stat - za))

    n_range = _default_n_range(n_treat)
    curve = _power_curve(n_range, _power_at_n)

    return {
        "required_sample_size": n_treat,
        "achieved_power": round(_power_at_n(n_treat), 4),
        "power_curve": curve,
        "effect_size_used": round(delta, 6),
        "notes": (
            f"Per-group size for {power_target:.0%} power at α={alpha}. "
            f"Baseline={p1:.4f}, expected={p2:.4f}, Δ={delta:.4f}. "
            f"Total sample = {n_treat * (1 + int(ratio))} (treatment + control)."
        ),
    }


# ── A/B Test (two-sample means / t-test) ────────────────────────────────────

def power_ab_test_continuous(
    baseline_mean: float,
    baseline_std: float,
    lift_pct: float | None = None,
    lift_abs: float | None = None,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
) -> dict[str, Any]:
    """Two-sample t-test for means."""
    mu1 = baseline_mean
    sigma = baseline_std

    if lift_abs is not None:
        delta = lift_abs
    elif lift_pct is not None:
        delta = mu1 * lift_pct
    else:
        delta = mu1 * 0.10

    if abs(delta) < 1e-12 or sigma < 1e-12:
        return {
            "required_sample_size": None,
            "achieved_power": 0.0,
            "power_curve": [],
            "effect_size_used": 0.0,
            "notes": "Effect size or std is zero — cannot compute power.",
        }

    d = abs(delta) / sigma  # Cohen's d
    za = _z(alpha, one_sided)
    zb = stats.norm.ppf(power_target)
    n_per_group = int(math.ceil(2 * ((za + zb) / d) ** 2))

    def _power_at_n(n: int) -> float:
        se = sigma * math.sqrt(2 / n)
        z_stat = abs(delta) / se
        if one_sided:
            return float(stats.norm.cdf(z_stat - za))
        return float(stats.norm.cdf(z_stat - za) + stats.norm.cdf(-z_stat - za))

    n_range = _default_n_range(n_per_group)
    curve = _power_curve(n_range, _power_at_n)

    return {
        "required_sample_size": n_per_group,
        "achieved_power": round(_power_at_n(n_per_group), 4),
        "power_curve": curve,
        "effect_size_used": round(abs(delta), 6),
        "notes": (
            f"Per-group size for {power_target:.0%} power at α={alpha}. "
            f"Cohen's d={d:.3f}. Total sample = {n_per_group * 2}."
        ),
    }


# ── Difference-in-Differences ───────────────────────────────────────────────

def power_did(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_pct: float | None = None,
    lift_abs: float | None = None,
    num_treatment_units: int = 5,
    num_control_units: int = 10,
    num_pre_periods: int = 12,
    num_post_periods: int = 8,
    icc: float = 0.05,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
) -> dict[str, Any]:
    """
    Power for a DiD design with clustered units and multiple time periods.
    Uses design effect for clustering.
    """
    mu = baseline_metric_value
    sigma = baseline_metric_std

    if lift_abs is not None:
        delta = lift_abs
    elif lift_pct is not None:
        delta = mu * lift_pct
    else:
        delta = mu * 0.10

    N_treat = num_treatment_units
    N_ctrl = num_control_units
    T_pre = num_pre_periods
    T_post = num_post_periods

    # Effective sample: units * post-periods
    n_eff_treat = N_treat * T_post
    n_eff_ctrl = N_ctrl * T_post

    # Design effect for clustering (Kish)
    m_bar = T_post  # observations per cluster
    deff = 1 + (m_bar - 1) * icc
    n_eff_treat_adj = n_eff_treat / deff
    n_eff_ctrl_adj = n_eff_ctrl / deff

    za = _z(alpha, one_sided)

    # SE of the DiD estimator (simplified)
    se_did = sigma * math.sqrt(1 / n_eff_treat_adj + 1 / n_eff_ctrl_adj)

    # Power at these specific group sizes
    z_stat = abs(delta) / se_did
    if one_sided:
        achieved_power = float(stats.norm.cdf(z_stat - za))
    else:
        achieved_power = float(stats.norm.cdf(z_stat - za) + stats.norm.cdf(-z_stat - za))

    # Required total treatment units for target power
    zb = stats.norm.ppf(power_target)
    ratio = N_ctrl / max(N_treat, 1)

    def _power_at_n_treat(n_t: int) -> float:
        n_c = max(1, int(n_t * ratio))
        ne_t = n_t * T_post / deff
        ne_c = n_c * T_post / deff
        se = sigma * math.sqrt(1 / max(ne_t, 1) + 1 / max(ne_c, 1))
        z = abs(delta) / max(se, 1e-12)
        if one_sided:
            return float(stats.norm.cdf(z - za))
        return float(stats.norm.cdf(z - za) + stats.norm.cdf(-z - za))

    # Solve for required treatment units
    required_n = N_treat
    for candidate in range(1, 500):
        if _power_at_n_treat(candidate) >= power_target:
            required_n = candidate
            break

    n_range = list(range(1, max(required_n * 3, 30)))
    curve = [{"n": n, "power": round(_power_at_n_treat(n), 4)} for n in n_range]

    return {
        "required_sample_size": required_n,
        "achieved_power": round(achieved_power, 4),
        "power_curve": curve,
        "effect_size_used": round(abs(delta), 6),
        "notes": (
            f"DiD with {N_treat} treatment × {N_ctrl} control units, "
            f"{T_pre} pre + {T_post} post periods. ICC={icc:.3f}, DEFF={deff:.2f}. "
            f"Achieved power = {achieved_power:.2%}. "
            f"Min treatment units for {power_target:.0%} power ≈ {required_n}."
        ),
    }


# ── Geo Lift / Matched Market (paired / clustered market designs) ────────────

def power_geo_market(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_pct: float | None = None,
    lift_abs: float | None = None,
    num_treatment_units: int = 8,
    num_control_units: int = 25,
    num_post_periods: int = 6,
    icc: float = 0.10,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
) -> dict[str, Any]:
    """
    Power calculation for geo-based designs (geo lift, matched market).
    Accounts for market-level clustering.
    """
    mu = baseline_metric_value
    sigma = baseline_metric_std

    if lift_abs is not None:
        delta = lift_abs
    elif lift_pct is not None:
        delta = mu * lift_pct
    else:
        delta = mu * 0.10

    N_treat = num_treatment_units
    N_ctrl = num_control_units
    T_post = num_post_periods

    deff = 1 + (T_post - 1) * icc

    za = _z(alpha, one_sided)
    zb = stats.norm.ppf(power_target)

    def _power_at_n_treat(n_t: int) -> float:
        n_c = max(1, int(n_t * N_ctrl / max(N_treat, 1)))
        ne_t = n_t * T_post / deff
        ne_c = n_c * T_post / deff
        se = sigma * math.sqrt(1 / max(ne_t, 1) + 1 / max(ne_c, 1))
        z = abs(delta) / max(se, 1e-12)
        if one_sided:
            return float(stats.norm.cdf(z - za))
        return float(stats.norm.cdf(z - za) + stats.norm.cdf(-z - za))

    # Current power
    achieved_power = _power_at_n_treat(N_treat)

    # Solve for required
    required_n = N_treat
    for candidate in range(1, 500):
        if _power_at_n_treat(candidate) >= power_target:
            required_n = candidate
            break

    n_range = list(range(1, max(required_n * 3, 40)))
    curve = [{"n": n, "power": round(_power_at_n_treat(n), 4)} for n in n_range]

    return {
        "required_sample_size": required_n,
        "achieved_power": round(achieved_power, 4),
        "power_curve": curve,
        "effect_size_used": round(abs(delta), 6),
        "notes": (
            f"Geo design with {N_treat} treatment × {N_ctrl} control markets, "
            f"{T_post} post periods. ICC={icc:.3f}, DEFF={deff:.2f}. "
            f"Min treatment markets for {power_target:.0%} power ≈ {required_n}."
        ),
    }


# ── Synthetic Control ───────────────────────────────────────────────────────

def power_synthetic_control(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_pct: float | None = None,
    lift_abs: float | None = None,
    num_donor_units: int = 20,
    num_pre_periods: int = 52,
    num_post_periods: int = 12,
    fit_quality: float = 0.90,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
) -> dict[str, Any]:
    """
    Approximate power for synthetic control via permutation-based inference.
    fit_quality ∈ [0, 1] measures how well donors approximate the treatment unit.
    """
    mu = baseline_metric_value
    sigma = baseline_metric_std

    if lift_abs is not None:
        delta = lift_abs
    elif lift_pct is not None:
        delta = mu * lift_pct
    else:
        delta = mu * 0.10

    T_post = num_post_periods
    J = num_donor_units  # number of donors (for permutation p-value)

    # Residual std after synthetic control fit
    sigma_resid = sigma * math.sqrt(1 - fit_quality ** 2)

    # Test statistic: average post-period gap / se
    se_gap = sigma_resid / math.sqrt(max(T_post, 1))
    z_stat = abs(delta) / max(se_gap, 1e-12)

    za = _z(alpha, one_sided)

    if one_sided:
        achieved_power = float(stats.norm.cdf(z_stat - za))
    else:
        achieved_power = float(stats.norm.cdf(z_stat - za) + stats.norm.cdf(-z_stat - za))

    # Power as a function of post-periods
    def _power_at_t(t: int) -> float:
        se = sigma_resid / math.sqrt(max(t, 1))
        z = abs(delta) / max(se, 1e-12)
        if one_sided:
            return float(stats.norm.cdf(z - za))
        return float(stats.norm.cdf(z - za) + stats.norm.cdf(-z - za))

    # Required post periods
    required_t = T_post
    for t_cand in range(1, 200):
        if _power_at_t(t_cand) >= power_target:
            required_t = t_cand
            break

    t_range = list(range(1, max(required_t * 3, 30)))
    curve = [{"n": t, "power": round(_power_at_t(t), 4)} for t in t_range]

    return {
        "required_sample_size": required_t,  # post-periods needed
        "achieved_power": round(achieved_power, 4),
        "power_curve": curve,
        "effect_size_used": round(abs(delta), 6),
        "notes": (
            f"Synthetic control with {J} donors, {num_pre_periods} pre + {T_post} post periods. "
            f"Fit quality={fit_quality:.2f}, residual σ={sigma_resid:.2f}. "
            f"Min post-periods for {power_target:.0%} power ≈ {required_t}. "
            f"Note: exact power depends on permutation distribution (J+1={J+1} units)."
        ),
    }


# ── DDML (Double/Debiased ML) ───────────────────────────────────────────────

def power_ddml(
    baseline_metric_value: float,
    baseline_metric_std: float,
    lift_pct: float | None = None,
    lift_abs: float | None = None,
    n_obs: int = 50000,
    r2_treatment: float = 0.30,
    r2_outcome: float = 0.40,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
) -> dict[str, Any]:
    """
    Power for DDML: normal approximation based on partially-linear model.
    R² values represent how well covariates predict treatment and outcome.
    """
    sigma = baseline_metric_std
    mu = baseline_metric_value

    if lift_abs is not None:
        delta = lift_abs
    elif lift_pct is not None:
        delta = mu * lift_pct
    else:
        delta = mu * 0.10

    # Residual variance after partialling out covariates
    sigma_resid = sigma * math.sqrt(1 - r2_outcome)
    # Variance of residualised treatment
    # For binary treatment with ~50% treated, var(D)≈0.25
    var_d_resid = 0.25 * (1 - r2_treatment)

    za = _z(alpha, one_sided)
    zb = stats.norm.ppf(power_target)

    def _se_at_n(n: int) -> float:
        return sigma_resid / math.sqrt(max(n * var_d_resid, 1))

    def _power_at_n(n: int) -> float:
        se = _se_at_n(n)
        z = abs(delta) / max(se, 1e-12)
        if one_sided:
            return float(stats.norm.cdf(z - za))
        return float(stats.norm.cdf(z - za) + stats.norm.cdf(-z - za))

    # Required n
    n_required = int(math.ceil(
        (sigma_resid ** 2 / (var_d_resid * delta ** 2)) * (za + zb) ** 2
    )) if abs(delta) > 1e-12 else None

    if n_required is None:
        return {
            "required_sample_size": None,
            "achieved_power": 0.0,
            "power_curve": [],
            "effect_size_used": 0.0,
            "notes": "Effect size is zero — cannot compute power.",
        }

    n_range = _default_n_range(n_required)
    curve = _power_curve(n_range, _power_at_n)

    return {
        "required_sample_size": n_required,
        "achieved_power": round(_power_at_n(n_obs), 4),
        "power_curve": curve,
        "effect_size_used": round(abs(delta), 6),
        "notes": (
            f"DDML with n={n_obs:,}, R²(outcome)={r2_outcome:.2f}, R²(treatment)={r2_treatment:.2f}. "
            f"Residual σ={sigma_resid:.2f}. Required n for {power_target:.0%} power ≈ {n_required:,}. "
            f"Power at current n={n_obs:,} is {_power_at_n(n_obs):.2%}."
        ),
    }


# ── Dispatcher ───────────────────────────────────────────────────────────────

def compute_power(
    method_key: str,
    setup_params: dict,
    elicited_facts: dict,
) -> dict[str, Any]:
    """
    Dispatch to the appropriate power calculation based on the chosen method.
    Returns a PowerResults-compatible dict.
    """
    # Extract common parameters with defaults
    alpha = setup_params.get("alpha", 0.05)
    power_target = setup_params.get("power_target", 0.80)
    one_sided = setup_params.get("one_sided", False)

    baseline_rate = setup_params.get("baseline_rate")
    baseline_val = setup_params.get("baseline_metric_value", 100.0)
    baseline_std = setup_params.get("baseline_metric_std", baseline_val * 0.3)
    lift_pct = setup_params.get("expected_lift_pct")
    lift_abs = setup_params.get("expected_lift_abs")

    n_treat = setup_params.get("num_treatment_units")
    n_ctrl = setup_params.get("num_control_units")
    n_pre = setup_params.get("num_pre_periods")
    n_post = setup_params.get("num_post_periods")
    icc = setup_params.get("icc", 0.05)

    if method_key == "ab_test":
        if baseline_rate is not None and baseline_rate > 0:
            return power_ab_test(
                baseline_rate=baseline_rate,
                lift_pct=lift_pct,
                lift_abs=lift_abs,
                alpha=alpha,
                power_target=power_target,
                one_sided=one_sided,
            )
        else:
            return power_ab_test_continuous(
                baseline_mean=baseline_val,
                baseline_std=baseline_std,
                lift_pct=lift_pct,
                lift_abs=lift_abs,
                alpha=alpha,
                power_target=power_target,
                one_sided=one_sided,
            )

    elif method_key == "did":
        return power_did(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_pct=lift_pct,
            lift_abs=lift_abs,
            num_treatment_units=n_treat or 5,
            num_control_units=n_ctrl or 10,
            num_pre_periods=n_pre or int(elicited_facts.get("pre_period_weeks", 12) or 12),
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 8) or 8),
            icc=icc,
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
        )

    elif method_key in ("geo_lift", "matched_market"):
        return power_geo_market(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_pct=lift_pct,
            lift_abs=lift_abs,
            num_treatment_units=n_treat or int(elicited_facts.get("num_markets", 8) or 8) // 3,
            num_control_units=n_ctrl or int(elicited_facts.get("num_markets", 25) or 25) * 2 // 3,
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 6) or 6),
            icc=icc,
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
        )

    elif method_key == "synthetic_control":
        return power_synthetic_control(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_pct=lift_pct,
            lift_abs=lift_abs,
            num_donor_units=n_ctrl or int(elicited_facts.get("num_markets", 20) or 20) - 1,
            num_pre_periods=n_pre or int(elicited_facts.get("pre_period_weeks", 52) or 52),
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 12) or 12),
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
        )

    elif method_key == "ddml":
        # Map sample_size_estimate to a number
        ss_map = {"small (<10k)": 5000, "medium (10k-1M)": 100000, "large (>1M)": 1000000}
        n_obs = ss_map.get(
            elicited_facts.get("sample_size_estimate", "medium (10k-1M)"),
            50000,
        )
        return power_ddml(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            lift_pct=lift_pct,
            lift_abs=lift_abs,
            n_obs=n_obs,
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
        )

    else:
        # Fallback: treat as a continuous two-sample test
        return power_ab_test_continuous(
            baseline_mean=baseline_val,
            baseline_std=baseline_std,
            lift_pct=lift_pct,
            lift_abs=lift_abs,
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
        )
