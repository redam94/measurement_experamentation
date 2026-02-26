"""
Synthetic Control method module.
"""
from __future__ import annotations

from .base import ExperimentMethod, DesignSpec, clamp


class SyntheticControl(ExperimentMethod):
    key = "synthetic_control"
    name = "Synthetic Control"
    short_description = (
        "Construct a weighted combination of un-treated comparison units that closely "
        "tracks the treated unit(s) before the campaign. The post-campaign gap between "
        "the actual and the synthetic control measures campaign impact."
    )

    def score(self, facts: dict) -> float:
        score = 40.0

        # Needs long pre-period — that's the whole point
        pre_weeks = facts.get("pre_period_weeks") or 0
        if pre_weeks >= 20:
            score += 25
        elif pre_weeks >= 12:
            score += 12
        elif pre_weeks >= 6:
            score += 0
        else:
            score -= 20

        # Works best when FEW units are treated (1–3 treated markets)
        num_markets = facts.get("num_markets") or 0
        if num_markets >= 15:
            score += 15       # many donor markets available
        elif num_markets >= 8:
            score += 8
        else:
            score -= 10

        # Geo/market level is the natural unit
        if facts.get("randomization_unit") in ("geo", "market"):
            score += 10

        # Not ideal if RCT is possible
        if facts.get("can_run_rct"):
            score -= 10

        # Campaign type: works best when brand-controlled (can ensure clean donor pool)
        if facts.get("campaign_type") == "brand_controlled":
            score += 5

        return clamp(score)

    def generate_spec(self, facts: dict, explanation: str = "") -> DesignSpec:
        pre_weeks = facts.get("pre_period_weeks") or 20
        duration = facts.get("test_duration_weeks") or 4
        num_markets = facts.get("num_markets") or 15

        return DesignSpec(
            method_key=self.key,
            method_name=self.name,
            score=self.score(facts),
            explanation=explanation,
            primary_objective=facts.get("primary_objective", ""),
            kpi=facts.get("kpi", ""),
            treatment_assignment=(
                "One or a small number of markets receive the campaign (treated unit). "
                f"The remaining {num_markets - 2}+ markets form the donor pool."
            ),
            control_definition=(
                "Synthetic control: a data-driven weighted average of donor markets that "
                "best tracks the treated market's pre-campaign trajectory."
            ),
            randomization_unit="geo/market",
            pre_period_weeks=pre_weeks,
            test_duration_weeks=duration,
            num_units=f"{num_markets} markets (1–3 treated, rest donor pool)",
            statistical_approach=(
                "Constrained optimization to find donor weights W* minimising "
                "||X_treated − X_donors × W||². "
                "Inference via permutation (placebo) tests."
            ),
            primary_model=(
                "Abadie-Diamond-Hainmueller synthetic control. "
                "Bayesian variant (BSCM) available for probabilistic inference."
            ),
            minimum_detectable_effect=(
                f"Depends on pre-period fit quality; typically good with ≥20 pre-period observations."
            ),
            implementation_steps=[
                f"Designate 1–3 treated markets and {num_markets - 3}+ donor markets.",
                f"Collect {pre_weeks} weeks of weekly KPI data for ALL markets.",
                "Optimise donor weights to minimise pre-period RMSPE between treated and synthetic.",
                "Validate: pre-period fit should be tight (RMSPE < 5% of mean outcome).",
                f"Launch campaign in treated markets for {duration} weeks.",
                "Post-period gap = incremental effect. Run placebo tests on all donor markets for p-values.",
            ],
            data_requirements=[
                f"Weekly KPI time series per market, covering ≥{pre_weeks} pre-period weeks.",
                "Market-level predictor variables (optional: improves donor weight quality).",
                "Clean donor pool: donor markets must NOT have received the campaign.",
            ],
            assumptions=[
                "No interference between treated and donor markets.",
                "Synthetic control can interpolate (not extrapolate) treated unit behaviour.",
                "Donor pool is large enough to construct a good match.",
            ],
            caveats=[
                "Requires long pre-period — unreliable with <8 pre-period observations.",
                "If many markets are treated, DiD or Geo Lift is more appropriate.",
                "Inference relies on permutation tests — valid only with many donor units.",
            ],
            pros=[
                "Excellent transparency: synthetic counterfactual is directly visualisable.",
                "No functional form assumption on the treatment effect.",
                "Handles a single treated market where DiD may fail.",
            ],
            cons=[
                "Very data-hungry in the time dimension.",
                "Construction of good synthetic control depends on donor pool quality.",
            ],
        )

    def generate_scaffold(self, facts: dict) -> str:
        kpi = facts.get("kpi", "conversions")
        pre = facts.get("pre_period_weeks") or 20
        return f'''"""
Synthetic Control — Bayesian Structural Time Series Scaffold
KPI: {kpi} | Pre-period: {pre} weeks
Generated by the Measurement Design Agent

Uses the BSTS (Bayesian Structural Time Series) approach via PyMC.
For the classic Abadie SCM, see the 'pysyncon' or 'SyntheticControlMethods' packages.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ── 1. Load data ───────────────────────────────────────────────────────────────
# Expected:
#   treated_series.csv  → (T,) weekly {kpi} for the treated market
#   donor_matrix.csv    → (T, J) weekly {kpi} for J donor markets
treated = pd.read_csv("treated_series.csv", index_col=0).squeeze().values
donors  = pd.read_csv("donor_matrix.csv",   index_col=0).values  # shape (T, J)

T_pre  = {pre}          # pre-period observations
T_post = len(treated) - T_pre

y_pre_treated = treated[:T_pre]
y_pre_donors  = donors[:T_pre, :]
y_post_treated = treated[T_pre:]
y_post_donors  = donors[T_pre:, :]

J = donors.shape[1]  # number of donor markets

# ── 2. Bayesian Synthetic Control ─────────────────────────────────────────────
with pm.Model() as sc_model:
    # Donor weights (Dirichlet prior → sum to 1, non-negative)
    w = pm.Dirichlet("weights", a=np.ones(J))

    # Synthetic control in pre-period
    sc_pre = pm.Deterministic("sc_pre", y_pre_donors @ w)

    # Noise in pre-period
    sigma = pm.HalfNormal("sigma", sigma=1)
    obs_pre = pm.Normal("obs_pre", mu=sc_pre, sigma=sigma, observed=y_pre_treated)

    # Sample posterior weights
    idata = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

# ── 3. Post-period counterfactual & gap ───────────────────────────────────────
w_samples = idata.posterior["weights"].values  # (chains, draws, J)
w_flat    = w_samples.reshape(-1, J)           # (samples, J)

# Counterfactual trajectory (what would have happened without the campaign)
cf = y_post_donors @ w_flat.T  # shape (T_post, samples)

# Gap = actual − counterfactual
gap = y_post_treated[:, None] - cf  # shape (T_post, samples)

mean_gap  = gap.mean(axis=1)
ci_lo     = np.percentile(gap, 5,  axis=1)
ci_hi     = np.percentile(gap, 95, axis=1)
total_lift = gap.sum(axis=0)

print("\\n── Synthetic Control Results ─────────────────────────────────────────────")
print(f"  Mean total lift over {{T_post}} post periods: {{total_lift.mean():.2f}}")
print(f"  90% CI: [{{np.percentile(total_lift, 5):.2f}}, {{np.percentile(total_lift, 95):.2f}}]")
print(f"  P(total lift > 0): {{(total_lift > 0).mean():.1%}}")

# ── 4. Plot ────────────────────────────────────────────────────────────────────
t_all = np.arange(len(treated))
t_pre  = t_all[:T_pre]
t_post = t_all[T_pre:]
sc_pre_mean = (y_pre_donors @ w_flat.T).mean(axis=1)
cf_mean = cf.mean(axis=1)

plt.figure(figsize=(12, 5))
plt.plot(t_all, treated,  "b-",  label="Treated market (actual)")
plt.plot(t_pre, sc_pre_mean, "r--", label="Synthetic control (pre-fit)")
plt.plot(t_post, cf_mean,    "r--")
plt.fill_between(t_post, ci_lo, ci_hi, alpha=0.2, color="red", label="90% CI")
plt.axvline(T_pre - 0.5, color="k", linestyle=":", label="Campaign launch")
plt.legend()
plt.title("Synthetic Control: Actual vs. Counterfactual")
plt.xlabel("Week")
plt.ylabel("{kpi}")
plt.tight_layout()
plt.savefig("synthetic_control.png", dpi=150)
plt.show()
'''
