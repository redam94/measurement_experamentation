"""
Monte Carlo simulation for Minimum Detectable Effect (MDE).

For each candidate effect size we:
  1. Simulate N_SIM experiments under that effect size.
  2. Apply the appropriate test statistic.
  3. Compute rejection rate → empirical power.
  4. The MDE is the smallest effect where power ≥ target.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


# ── Generic MDE search ──────────────────────────────────────────────────────

def _search_mde(
    simulate_and_test_fn,
    baseline_value: float,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
    seed: int = 42,
    max_relative_effect: float = 0.50,
    n_grid: int = 25,
) -> dict[str, Any]:
    """
    Generic MDE search using a grid of relative effect sizes.

    simulate_and_test_fn(effect_abs, rng) -> p_value
        Must return a p-value for a single simulated experiment.
    """
    rng = np.random.default_rng(seed)

    # Build grid of candidate relative effects (e.g. 1% to 50%)
    effects_rel = np.linspace(0.005, max_relative_effect, n_grid)
    effects_abs = effects_rel * abs(baseline_value) if baseline_value != 0 else effects_rel

    power_by_effect: list[dict[str, float]] = []
    mde_abs: float | None = None
    mde_rel: float | None = None

    for i, (eff_rel, eff_abs) in enumerate(zip(effects_rel, effects_abs)):
        rejections = 0
        for _ in range(n_simulations):
            p_val = simulate_and_test_fn(eff_abs, rng)
            if one_sided:
                if p_val < alpha:
                    rejections += 1
            else:
                if p_val < alpha:
                    rejections += 1

        empirical_power = rejections / n_simulations
        power_by_effect.append({
            "effect_abs": round(float(eff_abs), 6),
            "effect_rel_pct": round(float(eff_rel * 100), 2),
            "power": round(empirical_power, 4),
        })

        if mde_abs is None and empirical_power >= power_target:
            mde_abs = float(eff_abs)
            mde_rel = float(eff_rel)

    return {
        "mde_absolute": mde_abs,
        "mde_relative_pct": round(mde_rel * 100, 2) if mde_rel else None,
        "power_by_effect": power_by_effect,
        "n_simulations": n_simulations,
        "alpha": alpha,
        "target_power": power_target,
    }


# ── A/B Test MDE (proportions) ──────────────────────────────────────────────

def mde_ab_test_proportions(
    baseline_rate: float,
    n_per_group: int,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo MDE for a two-sample proportions test."""

    def _sim(effect_abs: float, rng: np.random.Generator) -> float:
        p_ctrl = baseline_rate
        p_treat = baseline_rate + effect_abs
        p_treat = min(max(p_treat, 0.001), 0.999)

        x_ctrl = rng.binomial(n_per_group, p_ctrl)
        x_treat = rng.binomial(n_per_group, p_treat)

        p_hat_c = x_ctrl / n_per_group
        p_hat_t = x_treat / n_per_group
        p_hat_pool = (x_ctrl + x_treat) / (2 * n_per_group)

        se = math.sqrt(max(p_hat_pool * (1 - p_hat_pool) * 2 / n_per_group, 1e-15))
        z = (p_hat_t - p_hat_c) / se

        if one_sided:
            return float(1 - stats.norm.cdf(z))
        return float(2 * (1 - stats.norm.cdf(abs(z))))

    result = _search_mde(
        _sim,
        baseline_value=baseline_rate,
        n_simulations=n_simulations,
        alpha=alpha,
        power_target=power_target,
        one_sided=one_sided,
        seed=seed,
        max_relative_effect=1.0,  # up to 100% relative lift for small rates
    )
    result["notes"] = (
        f"Monte Carlo MDE for two-sample proportions (n={n_per_group}/group, "
        f"baseline={baseline_rate:.4f}, {n_simulations} simulations)."
    )
    return result


# ── A/B Test MDE (continuous) ────────────────────────────────────────────────

def mde_ab_test_continuous(
    baseline_mean: float,
    baseline_std: float,
    n_per_group: int,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo MDE for a two-sample t-test."""

    def _sim(effect_abs: float, rng: np.random.Generator) -> float:
        ctrl = rng.normal(baseline_mean, baseline_std, n_per_group)
        treat = rng.normal(baseline_mean + effect_abs, baseline_std, n_per_group)
        t_stat, p_val = stats.ttest_ind(treat, ctrl, alternative="greater" if one_sided else "two-sided")
        return float(p_val)

    result = _search_mde(
        _sim,
        baseline_value=baseline_mean,
        n_simulations=n_simulations,
        alpha=alpha,
        power_target=power_target,
        one_sided=one_sided,
        seed=seed,
    )
    result["notes"] = (
        f"Monte Carlo MDE for two-sample t-test (n={n_per_group}/group, "
        f"mean={baseline_mean:.2f}, σ={baseline_std:.2f}, {n_simulations} sims)."
    )
    return result


# ── DiD MDE ─────────────────────────────────────────────────────────────────

def mde_did(
    baseline_metric_value: float,
    baseline_metric_std: float,
    num_treatment_units: int,
    num_control_units: int,
    num_pre_periods: int,
    num_post_periods: int,
    icc: float = 0.05,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo MDE for a DiD design with panel data."""

    N_t = num_treatment_units
    N_c = num_control_units
    T_pre = num_pre_periods
    T_post = num_post_periods
    T = T_pre + T_post
    sigma = baseline_metric_std

    # Decompose variance: between-unit σ_u and within-unit σ_e
    sigma_u = sigma * math.sqrt(icc)
    sigma_e = sigma * math.sqrt(1 - icc)

    def _sim(effect_abs: float, rng: np.random.Generator) -> float:
        N = N_t + N_c
        # Unit random effects
        unit_effects = rng.normal(0, sigma_u, N)
        # Time effects (shared)
        time_effects = rng.normal(0, sigma_e * 0.3, T)

        # Build panel: y_{it} = mu + unit_i + time_t + treat_effect + noise
        y = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                y[i, t] = (
                    baseline_metric_value
                    + unit_effects[i]
                    + time_effects[t]
                    + rng.normal(0, sigma_e)
                )
                # Add treatment effect in post-period for treatment units
                if i < N_t and t >= T_pre:
                    y[i, t] += effect_abs

        # DiD estimator
        pre_treat = y[:N_t, :T_pre].mean()
        post_treat = y[:N_t, T_pre:].mean()
        pre_ctrl = y[N_t:, :T_pre].mean()
        post_ctrl = y[N_t:, T_pre:].mean()
        did_hat = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)

        # Clustered SE (unit-level clustering)
        unit_dids = np.zeros(N)
        for i in range(N):
            pre_i = y[i, :T_pre].mean()
            post_i = y[i, T_pre:].mean()
            unit_dids[i] = post_i - pre_i

        treat_diffs = unit_dids[:N_t]
        ctrl_diffs = unit_dids[N_t:]
        se = math.sqrt(
            np.var(treat_diffs, ddof=1) / N_t + np.var(ctrl_diffs, ddof=1) / N_c
        )

        if se < 1e-12:
            return 1.0
        t_stat = did_hat / se
        df = N_t + N_c - 2
        if one_sided:
            return float(1 - stats.t.cdf(t_stat, df))
        return float(2 * (1 - stats.t.cdf(abs(t_stat), df)))

    result = _search_mde(
        _sim,
        baseline_value=baseline_metric_value,
        n_simulations=n_simulations,
        alpha=alpha,
        power_target=power_target,
        one_sided=one_sided,
        seed=seed,
    )
    result["notes"] = (
        f"Monte Carlo MDE for DiD ({N_t} treat × {N_c} ctrl units, "
        f"{T_pre} pre + {T_post} post periods, ICC={icc:.3f}, {n_simulations} sims)."
    )
    return result


# ── Geo Lift / Matched Market MDE ───────────────────────────────────────────

def mde_geo_market(
    baseline_metric_value: float,
    baseline_metric_std: float,
    num_treatment_units: int,
    num_control_units: int,
    num_post_periods: int,
    icc: float = 0.10,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo MDE for geo-based market experiment."""

    N_t = num_treatment_units
    N_c = num_control_units
    T = num_post_periods
    sigma = baseline_metric_std

    sigma_m = sigma * math.sqrt(icc)
    sigma_e = sigma * math.sqrt(1 - icc)

    def _sim(effect_abs: float, rng: np.random.Generator) -> float:
        # Market random effects
        market_eff_t = rng.normal(0, sigma_m, N_t)
        market_eff_c = rng.normal(0, sigma_m, N_c)

        # Generate post-period data
        y_treat = np.zeros((N_t, T))
        y_ctrl = np.zeros((N_c, T))

        for i in range(N_t):
            for t in range(T):
                y_treat[i, t] = (
                    baseline_metric_value + market_eff_t[i]
                    + effect_abs + rng.normal(0, sigma_e)
                )
        for i in range(N_c):
            for t in range(T):
                y_ctrl[i, t] = (
                    baseline_metric_value + market_eff_c[i]
                    + rng.normal(0, sigma_e)
                )

        # Market-level means over post-period
        mean_t = y_treat.mean(axis=1)
        mean_c = y_ctrl.mean(axis=1)

        diff = mean_t.mean() - mean_c.mean()
        se = math.sqrt(np.var(mean_t, ddof=1) / N_t + np.var(mean_c, ddof=1) / N_c)

        if se < 1e-12:
            return 1.0
        t_stat = diff / se
        df = N_t + N_c - 2
        if one_sided:
            return float(1 - stats.t.cdf(t_stat, df))
        return float(2 * (1 - stats.t.cdf(abs(t_stat), df)))

    result = _search_mde(
        _sim,
        baseline_value=baseline_metric_value,
        n_simulations=n_simulations,
        alpha=alpha,
        power_target=power_target,
        one_sided=one_sided,
        seed=seed,
    )
    result["notes"] = (
        f"Monte Carlo MDE for geo/market design ({N_t} treat × {N_c} ctrl markets, "
        f"{T} post periods, ICC={icc:.3f}, {n_simulations} sims)."
    )
    return result


# ── Synthetic Control MDE ───────────────────────────────────────────────────

def mde_synthetic_control(
    baseline_metric_value: float,
    baseline_metric_std: float,
    num_donor_units: int,
    num_pre_periods: int,
    num_post_periods: int,
    n_simulations: int = 500,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Monte Carlo MDE for synthetic control via permutation test.
    Simplified: uses regression-based SC weights and permutation p-value.
    """
    J = num_donor_units
    T_pre = num_pre_periods
    T_post = num_post_periods
    T = T_pre + T_post
    sigma = baseline_metric_std

    def _sim(effect_abs: float, rng: np.random.Generator) -> float:
        # Generate J+1 correlated time series (treatment + J donors)
        # Common factor model: y_jt = mu + f_t * lambda_j + e_jt
        factors = rng.normal(0, sigma * 0.5, T)
        loadings = rng.uniform(0.5, 1.5, J + 1)

        data = np.zeros((J + 1, T))
        for j in range(J + 1):
            for t in range(T):
                data[j, t] = (
                    baseline_metric_value
                    + factors[t] * loadings[j]
                    + rng.normal(0, sigma * 0.3)
                )

        # Add treatment effect to unit 0 in post-period
        data[0, T_pre:] += effect_abs

        # Fit SC weights using pre-period OLS (donors predicting treatment)
        Y_pre_treat = data[0, :T_pre]
        X_pre_donors = data[1:, :T_pre].T  # (T_pre, J)

        # Ridge regression for stability
        lam = 0.01 * np.eye(J)
        XtX = X_pre_donors.T @ X_pre_donors + lam
        XtY = X_pre_donors.T @ Y_pre_treat
        try:
            weights = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            return 1.0

        # Normalise weights to sum to 1
        weights = np.maximum(weights, 0)
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones(J) / J

        # Compute post-period gap
        synthetic_post = (data[1:, T_pre:].T @ weights)
        gap = data[0, T_pre:] - synthetic_post
        avg_gap_treat = gap.mean()

        # Permutation: compute gap for each donor (placebo test)
        placebo_gaps = []
        for j in range(1, J + 1):
            # Fit SC for donor j using remaining donors
            remaining = [k for k in range(J + 1) if k != j]
            Y_pj = data[j, :T_pre]
            X_pj = data[remaining][:, :T_pre].T
            Jr = len(remaining)
            lam_j = 0.01 * np.eye(Jr)
            XtX_j = X_pj.T @ X_pj + lam_j
            XtY_j = X_pj.T @ Y_pj
            try:
                w_j = np.linalg.solve(XtX_j, XtY_j)
            except np.linalg.LinAlgError:
                continue
            w_j = np.maximum(w_j, 0)
            ws = w_j.sum()
            if ws > 0:
                w_j /= ws
            else:
                w_j = np.ones(Jr) / Jr
            synth_j = (data[remaining][:, T_pre:].T @ w_j)
            gap_j = data[j, T_pre:] - synth_j
            placebo_gaps.append(gap_j.mean())

        if not placebo_gaps:
            return 1.0

        # Permutation p-value
        all_gaps = [avg_gap_treat] + placebo_gaps
        rank = sum(1 for g in all_gaps if abs(g) >= abs(avg_gap_treat))
        p_val = rank / len(all_gaps)
        return float(p_val)

    # Use fewer simulations for SC (it's expensive)
    result = _search_mde(
        _sim,
        baseline_value=baseline_metric_value,
        n_simulations=min(n_simulations, 500),
        alpha=alpha,
        power_target=power_target,
        one_sided=one_sided,
        seed=seed,
        n_grid=15,
    )
    result["notes"] = (
        f"Monte Carlo MDE for synthetic control ({J} donors, "
        f"{T_pre} pre + {T_post} post periods, {min(n_simulations, 500)} sims). "
        f"Uses OLS-based SC weights with permutation p-values."
    )
    return result


# ── DDML MDE ────────────────────────────────────────────────────────────────

def mde_ddml(
    baseline_metric_value: float,
    baseline_metric_std: float,
    n_obs: int,
    r2_treatment: float = 0.30,
    r2_outcome: float = 0.40,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    power_target: float = 0.80,
    one_sided: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo MDE for DDML (partially-linear model)."""

    sigma = baseline_metric_std

    def _sim(effect_abs: float, rng: np.random.Generator) -> float:
        # Generate X, D, Y from a partially linear model
        n_covariates = 5
        X = rng.normal(0, 1, (n_obs, n_covariates))

        # Treatment propensity depends on X
        beta_d = rng.normal(0, 1, n_covariates) * math.sqrt(r2_treatment / n_covariates)
        logit = X @ beta_d
        prob = 1 / (1 + np.exp(-logit))
        D = rng.binomial(1, prob)

        # Outcome depends on X and D
        beta_y = rng.normal(0, 1, n_covariates) * math.sqrt(r2_outcome / n_covariates)
        Y = (
            baseline_metric_value
            + X @ (beta_y * sigma)
            + effect_abs * D
            + rng.normal(0, sigma * math.sqrt(1 - r2_outcome), n_obs)
        )

        # Simple "partialling out" via OLS (approximation of DML)
        # Residualise D on X
        XtX = X.T @ X
        try:
            beta_dx = np.linalg.solve(XtX, X.T @ D)
            D_resid = D - X @ beta_dx
            beta_yx = np.linalg.solve(XtX, X.T @ Y)
            Y_resid = Y - X @ beta_yx
        except np.linalg.LinAlgError:
            return 1.0

        # Estimate treatment effect from residuals
        denom = D_resid @ D_resid
        if denom < 1e-12:
            return 1.0
        theta_hat = (D_resid @ Y_resid) / denom

        # SE (heteroskedasticity-robust)
        e = Y_resid - theta_hat * D_resid
        V = np.sum((D_resid ** 2) * (e ** 2)) / (denom ** 2)
        se = math.sqrt(max(V, 1e-15))

        t_stat = theta_hat / se
        if one_sided:
            return float(1 - stats.norm.cdf(t_stat))
        return float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    result = _search_mde(
        _sim,
        baseline_value=baseline_metric_value,
        n_simulations=n_simulations,
        alpha=alpha,
        power_target=power_target,
        one_sided=one_sided,
        seed=seed,
    )
    result["notes"] = (
        f"Monte Carlo MDE for DDML (n={n_obs:,}, R²(Y|X)={r2_outcome:.2f}, "
        f"R²(D|X)={r2_treatment:.2f}, {n_simulations} sims)."
    )
    return result


# ── Dispatcher ───────────────────────────────────────────────────────────────

def compute_mde(
    method_key: str,
    setup_params: dict,
    elicited_facts: dict,
    power_results: dict | None = None,
) -> dict[str, Any]:
    """
    Dispatch to the appropriate MDE simulation based on the chosen method.
    Uses power_results to determine sample sizes where applicable.
    """
    alpha = setup_params.get("alpha", 0.05)
    power_target = setup_params.get("power_target", 0.80)
    one_sided = setup_params.get("one_sided", False)
    n_sims = setup_params.get("n_simulations", 1000)
    seed = setup_params.get("random_seed", 42)

    baseline_rate = setup_params.get("baseline_rate")
    baseline_val = setup_params.get("baseline_metric_value", 100.0)
    baseline_std = setup_params.get("baseline_metric_std", baseline_val * 0.3)

    n_treat = setup_params.get("num_treatment_units")
    n_ctrl = setup_params.get("num_control_units")
    n_pre = setup_params.get("num_pre_periods")
    n_post = setup_params.get("num_post_periods")
    icc = setup_params.get("icc", 0.05)

    # Get sample size from power results if available
    n_per_group = (power_results or {}).get("required_sample_size") or 1000

    if method_key == "ab_test":
        if baseline_rate is not None and baseline_rate > 0:
            return mde_ab_test_proportions(
                baseline_rate=baseline_rate,
                n_per_group=n_per_group,
                n_simulations=n_sims,
                alpha=alpha,
                power_target=power_target,
                one_sided=one_sided,
                seed=seed,
            )
        else:
            return mde_ab_test_continuous(
                baseline_mean=baseline_val,
                baseline_std=baseline_std,
                n_per_group=n_per_group,
                n_simulations=n_sims,
                alpha=alpha,
                power_target=power_target,
                one_sided=one_sided,
                seed=seed,
            )

    elif method_key == "did":
        return mde_did(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            num_treatment_units=n_treat or 5,
            num_control_units=n_ctrl or 10,
            num_pre_periods=n_pre or int(elicited_facts.get("pre_period_weeks", 12) or 12),
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 8) or 8),
            icc=icc,
            n_simulations=n_sims,
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
            seed=seed,
        )

    elif method_key in ("geo_lift", "matched_market"):
        return mde_geo_market(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            num_treatment_units=n_treat or int(elicited_facts.get("num_markets", 8) or 8) // 3,
            num_control_units=n_ctrl or int(elicited_facts.get("num_markets", 25) or 25) * 2 // 3,
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 6) or 6),
            icc=icc,
            n_simulations=n_sims,
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
            seed=seed,
        )

    elif method_key == "synthetic_control":
        return mde_synthetic_control(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            num_donor_units=n_ctrl or int(elicited_facts.get("num_markets", 20) or 20) - 1,
            num_pre_periods=n_pre or int(elicited_facts.get("pre_period_weeks", 52) or 52),
            num_post_periods=n_post or int(elicited_facts.get("test_duration_weeks", 12) or 12),
            n_simulations=min(n_sims, 500),
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
            seed=seed,
        )

    elif method_key == "ddml":
        ss_map = {"small (<10k)": 5000, "medium (10k-1M)": 100000, "large (>1M)": 1000000}
        n_obs = ss_map.get(
            elicited_facts.get("sample_size_estimate", "medium (10k-1M)"),
            50000,
        )
        return mde_ddml(
            baseline_metric_value=baseline_val,
            baseline_metric_std=baseline_std,
            n_obs=n_obs,
            n_simulations=n_sims,
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
            seed=seed,
        )

    else:
        # Fallback
        return mde_ab_test_continuous(
            baseline_mean=baseline_val,
            baseline_std=baseline_std,
            n_per_group=n_per_group,
            n_simulations=n_sims,
            alpha=alpha,
            power_target=power_target,
            one_sided=one_sided,
            seed=seed,
        )
