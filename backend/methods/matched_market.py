"""
Matched Market Testing method module.
"""
from __future__ import annotations

from .base import ExperimentMethod, DesignSpec, clamp


class MatchedMarketTest(ExperimentMethod):
    key = "matched_market"
    name = "Matched Market Testing"
    short_description = (
        "Identify pairs of highly similar markets. Run the campaign in one market of each pair "
        "and hold out the other. Compare outcomes across matched pairs."
    )

    def score(self, facts: dict) -> float:
        score = 45.0

        # Needs geo holdout
        if facts.get("geo_holdout_feasible"):
            score += 20
        else:
            score -= 20

        # Needs multiple markets to form pairs
        num_markets = facts.get("num_markets") or 0
        if num_markets >= 10:
            score += 15
        elif num_markets >= 6:
            score += 8
        elif num_markets >= 4:
            score += 0
        else:
            score -= 15

        # Some pre-period history helps with matching
        pre_weeks = facts.get("pre_period_weeks") or 0
        if pre_weeks >= 4:
            score += 10
        else:
            score -= 5

        # Covariates improve matching quality
        if facts.get("has_rich_covariates"):
            score += 5

        # Works whether brand-controlled or platform
        if facts.get("campaign_type") in ("brand_controlled", "platform_only"):
            score += 5

        return clamp(score)

    def generate_spec(self, facts: dict, explanation: str = "") -> DesignSpec:
        num_markets = facts.get("num_markets") or 10
        pre_weeks = facts.get("pre_period_weeks") or 4
        duration = facts.get("test_duration_weeks") or 4
        n_pairs = num_markets // 2

        return DesignSpec(
            method_key=self.key,
            method_name=self.name,
            score=self.score(facts),
            explanation=explanation,
            primary_objective=facts.get("primary_objective", ""),
            kpi=facts.get("kpi", ""),
            treatment_assignment=(
                f"Pair {num_markets} markets into {n_pairs} matched pairs based on "
                "historical KPI trend and market descriptors. Randomly assign one market "
                "per pair to treatment (receives campaign) and one to control (holdout)."
            ),
            control_definition=(
                "Within each matched pair, the untreated market serves as the control. "
                "Analysis averages treatment effects across all pairs."
            ),
            randomization_unit="geo/market",
            pre_period_weeks=pre_weeks,
            test_duration_weeks=duration,
            num_units=f"{num_markets} markets in {n_pairs} pairs",
            statistical_approach=(
                "Bayesian paired comparison: per-pair treatment effects averaged "
                "with hierarchical partial pooling across pairs."
            ),
            primary_model=(
                "Hierarchical Bayesian model: θ_pair ~ Normal(δ, σ_between); "
                "Y_post − Y_pre ~ Normal(θ_pair × treated, σ_within)."
            ),
            minimum_detectable_effect=f"~10–15% relative lift with {n_pairs} pairs.",
            implementation_steps=[
                f"Compile market descriptors and {pre_weeks} weeks of pre-period KPI.",
                "Run propensity score or Euclidean distance matching to form market pairs.",
                "Verify match quality: pre-period correlation within pairs should be ≥0.9.",
                "Randomly assign one market per pair to treatment.",
                f"Run campaign in treatment markets for {duration} weeks.",
                "Compute per-pair diff-in-means and aggregate with Bayesian hierarchical model.",
            ],
            data_requirements=[
                f"Pre-period KPI per market: ≥{pre_weeks} weeks.",
                f"Post-period KPI per market: {duration} weeks.",
                "Market descriptors for matching: size, category mix, regional indicators.",
            ],
            assumptions=[
                "Matched pairs are exchangeable absent the treatment.",
                "No cross-market spillover within pairs (geographical separation required).",
                "Matching quality is high enough that unobserved confounders are balanced.",
            ],
            caveats=[
                "With few pairs (<5), statistical power may be low.",
                "Poor market matching biases the estimate — validate match quality rigorously.",
            ],
            pros=[
                "Combines the simplicity of A/B testing with geo-level execution.",
                "Matching reduces variance compared to unmatched geo designs.",
                "Easy to explain: 'we tested in matched cities' resonates with stakeholders.",
            ],
            cons=[
                "Requires sufficient pairs — hard to implement with very few markets.",
                "Matching is an art; mismatches inflate variance or bias.",
            ],
        )

    def generate_scaffold(self, facts: dict) -> str:
        kpi = facts.get("kpi", "conversions")
        pre = facts.get("pre_period_weeks") or 4
        dur = facts.get("test_duration_weeks") or 4
        return f'''"""
Matched Market Test — Bayesian Hierarchical Paired Design Scaffold
KPI: {kpi} | Pre: {pre}w | Test: {dur}w
Generated by the Measurement Design Agent
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ── 1. Load data ───────────────────────────────────────────────────────────────
# Expected columns: ["pair_id", "market_id", "treated", "pre_kpi", "post_kpi"]
# Each pair_id has exactly 2 rows: one treated, one control.
df = pd.read_csv("your_matched_market_data.csv")

pairs, pair_idx = np.unique(df["pair_id"], return_inverse=True)
n_pairs = len(pairs)

# Difference-in-means per pair
df_sorted = df.sort_values(["pair_id", "treated"])
treat_post = df_sorted.loc[df_sorted["treated"] == 1, "post_kpi"].values
ctrl_post  = df_sorted.loc[df_sorted["treated"] == 0, "post_kpi"].values
treat_pre  = df_sorted.loc[df_sorted["treated"] == 1, "pre_kpi"].values
ctrl_pre   = df_sorted.loc[df_sorted["treated"] == 0, "pre_kpi"].values

# Period-over-period lift per pair (DiD per pair)
pair_delta = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)

# ── 2. Hierarchical Bayesian model ─────────────────────────────────────────────
with pm.Model() as mmt_model:
    # Hyper-parameters (pooled treatment effect)
    delta_mu    = pm.Normal("delta_mu",    mu=0,  sigma=10)
    delta_sigma = pm.HalfNormal("delta_sigma", sigma=5)

    # Per-pair treatment effects
    theta_pair  = pm.Normal("theta_pair", mu=delta_mu, sigma=delta_sigma, shape=n_pairs)

    # Observation noise
    sigma_obs   = pm.HalfNormal("sigma_obs", sigma=5)

    # Likelihood: observed per-pair delta
    obs = pm.Normal("obs", mu=theta_pair, sigma=sigma_obs, observed=pair_delta)

    idata = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

# ── 3. Results ─────────────────────────────────────────────────────────────────
summary = az.summary(idata, var_names=["delta_mu", "delta_sigma"])
print(summary)

prob_pos = float((idata.posterior["delta_mu"] > 0).mean())
d_med    = float(idata.posterior["delta_mu"].median())
print(f"\\nPooled treatment effect: {{d_med:.4f}}")
print(f"P(treatment > control):  {{prob_pos:.1%}}")

az.plot_posterior(idata, var_names=["delta_mu"], ref_val=0)
plt.title("Posterior of Matched Market pooled treatment effect")
plt.tight_layout()
plt.savefig("matched_market_posterior.png", dpi=150)
plt.show()
'''
