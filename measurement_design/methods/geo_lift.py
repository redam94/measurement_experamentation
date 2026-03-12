"""
Geo Lift Test method module.
"""
from __future__ import annotations

from .base import ExperimentMethod, DesignSpec, clamp


class GeoLiftTest(ExperimentMethod):
    key = "geo_lift"
    name = "Geo Lift Test"
    short_description = (
        "Run the campaign in a randomly selected set of geographic markets and withhold it "
        "from others. Measure the incremental lift in treated markets vs. held-out markets."
    )

    def score(self, facts: dict) -> float:
        score = 45.0

        # Core: geo holdout must be feasible
        if facts.get("geo_holdout_feasible"):
            score += 25
        else:
            score -= 20

        # More markets = better power
        num_markets = facts.get("num_markets") or 0
        if num_markets >= 20:
            score += 15
        elif num_markets >= 10:
            score += 10
        elif num_markets >= 5:
            score += 0
        else:
            score -= 10

        # Works best when platform controls geo delivery
        if facts.get("campaign_type") in ("platform_only", "brand_controlled"):
            score += 5

        # Pre-period helps with market matching
        pre_weeks = facts.get("pre_period_weeks") or 0
        if pre_weeks >= 4:
            score += 5
        else:
            score -= 5

        # Scale doesn't matter as much (aggregated to geo level)
        if facts.get("randomization_unit") in ("geo", "market"):
            score += 10

        return clamp(score)

    def generate_spec(self, facts: dict, explanation: str = "") -> DesignSpec:
        num_markets = facts.get("num_markets") or 20
        pre_weeks = facts.get("pre_period_weeks") or 4
        duration = facts.get("test_duration_weeks") or 4

        return DesignSpec(
            method_key=self.key,
            method_name=self.name,
            score=self.score(facts),
            explanation=explanation,
            primary_objective=facts.get("primary_objective", ""),
            kpi=facts.get("kpi", ""),
            treatment_assignment=(
                f"Randomly assign ~50% of available markets ({num_markets // 2} markets) to receive "
                "the campaign; hold out the remainder as control."
            ),
            control_definition="Holdout markets: identical geo units where the ad does not run.",
            randomization_unit="geo/market",
            pre_period_weeks=pre_weeks,
            test_duration_weeks=duration,
            num_units=f"{num_markets} markets",
            statistical_approach=(
                "Bayesian geo-level hierarchical model; or GeoLift open-source framework "
                "(Meta) using time-series synthetic control at the market level."
            ),
            primary_model=(
                "Hierarchical Normal model per market with pooled treatment effect δ. "
                "Alternatively: GeoLift's Bayesian structural time-series model."
            ),
            minimum_detectable_effect=f"~8–15% relative lift with {num_markets} markets.",
            implementation_steps=[
                f"Identify {num_markets} distinct, independently targetable markets.",
                f"Collect {pre_weeks}+ weeks of pre-period KPI data per market.",
                "Balance treatment/control markets by size and trend (use GeoLift power analysis).",
                "Block ads in control markets at platform level for full test duration.",
                f"Run test for {duration} weeks; collect daily/weekly KPI per market.",
                "Run hierarchical Bayesian model; report posterior lift and 90% CI.",
            ],
            data_requirements=[
                "Weekly or daily KPI aggregated by market/geo.",
                "Market descriptors: size (population, revenue baseline), region.",
                f"At least {pre_weeks} weeks of clean pre-campaign history.",
            ],
            assumptions=[
                "Geographic targeting is enforced — no ad spills across market boundaries.",
                "Markets are roughly homogeneous in consumer behavior (or differences are controlled).",
                "No cross-market contamination (e.g., consumers traveling between markets).",
            ],
            caveats=[
                "Low number of markets (<10) severely limits statistical power.",
                "Geographic spillover (ads seen in holdout markets) will bias estimates downward.",
            ],
            pros=[
                "Works even when user-level holdout is impossible.",
                "Clean causal interpretation if geo holdout is enforced.",
                "Easily understood by media buyers and planners.",
            ],
            cons=[
                "Requires sufficient market count for power.",
                "Geographic aggregation loses individual-level precision.",
            ],
        )

    def generate_scaffold(self, facts: dict) -> str:
        kpi = facts.get("kpi", "conversions")
        pre = facts.get("pre_period_weeks") or 4
        dur = facts.get("test_duration_weeks") or 4
        return f'''"""
Geo Lift Test — Bayesian Hierarchical Scaffold
KPI: {kpi} | Pre-period: {pre}w | Test: {dur}w
Generated by the Measurement Design Agent
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ── 1. Load geo-level data ─────────────────────────────────────────────────────
# Expected columns: ["market_id", "week", "kpi", "treated", "is_post"]
# treated=1 → ad ran in this market  |  is_post=1 → post campaign launch
df = pd.read_csv("your_geo_data.csv")
df["treated_post"] = df["treated"] * df["is_post"]

markets, mkt_idx = np.unique(df["market_id"], return_inverse=True)
n_markets = len(markets)
y = df["{kpi}"].values.astype(float)

# ── 2. Hierarchical Bayesian geo lift model ────────────────────────────────────
with pm.Model() as geo_model:
    # Market-level baselines (partial pooling)
    mu_alpha       = pm.Normal("mu_alpha", mu=0, sigma=10)
    sigma_alpha    = pm.HalfNormal("sigma_alpha", sigma=5)
    alpha_market   = pm.Normal("alpha_market", mu=mu_alpha, sigma=sigma_alpha, shape=n_markets)

    # Week trend (shared across markets for simplicity — extend to per-market if needed)
    sigma_obs      = pm.HalfNormal("sigma_obs", sigma=5)

    # Shared treatment effect (lift)
    delta          = pm.Normal("delta", mu=0, sigma=10)

    # Expected KPI
    mu = alpha_market[mkt_idx] + delta * df["treated_post"].values

    # Likelihood
    obs = pm.Normal("obs", mu=mu, sigma=sigma_obs, observed=y)

    idata = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

# ── 3. Results ─────────────────────────────────────────────────────────────────
summary = az.summary(idata, var_names=["delta"])
print(summary)

prob_lift = float((idata.posterior["delta"] > 0).mean())
print(f"\\nP(geo lift > 0) = {{prob_lift:.1%}}")

az.plot_posterior(idata, var_names=["delta"], ref_val=0)
plt.title("Posterior of Geo Lift treatment effect (δ)")
plt.tight_layout()
plt.savefig("geo_lift_posterior.png", dpi=150)
plt.show()
'''
