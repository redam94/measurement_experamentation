"""
Double Debiased Machine Learning (DDML) method module.
"""
from __future__ import annotations

from .base import ExperimentMethod, DesignSpec, clamp


class DoubleDML(ExperimentMethod):
    key = "ddml"
    name = "Double Debiased Machine Learning (DDML)"
    short_description = (
        "Use machine learning to residualise out confounding variables from both the treatment "
        "and the outcome, then estimate the causal effect on the residuals. Ideal for "
        "observational data with rich feature sets."
    )

    def score(self, facts: dict) -> float:
        score = 40.0  # start lower — requires rich covariates

        # Core requirement: rich covariates
        if facts.get("has_rich_covariates"):
            score += 30
        else:
            score -= 20

        # Best in observational / platform-controlled settings
        if facts.get("campaign_type") in ("observational", "platform_only"):
            score += 20
        elif facts.get("can_run_rct"):
            score -= 15  # A/B is simpler if RCT is possible

        # Benefits from larger data
        scale = facts.get("sample_size_estimate", "unknown")
        if "large" in scale:
            score += 10
        elif "medium" in scale:
            score += 5
        elif "small" in scale:
            score -= 10  # ML doesn't work well on tiny datasets

        # Some historical data helps
        if facts.get("has_historical_data"):
            score += 5

        return clamp(score)

    def generate_spec(self, facts: dict, explanation: str = "") -> DesignSpec:
        return DesignSpec(
            method_key=self.key,
            method_name=self.name,
            score=self.score(facts),
            explanation=explanation,
            primary_objective=facts.get("primary_objective", ""),
            kpi=facts.get("kpi", ""),
            treatment_assignment=(
                "Treatment is ad exposure (observed, not randomized). "
                "Use propensity model to estimate likelihood of exposure given covariates."
            ),
            control_definition=(
                "Unexposed units (users/markets that did not receive the ad during the window), "
                "after statistical adjustment for all observed confounders."
            ),
            randomization_unit=facts.get("randomization_unit", "user"),
            pre_period_weeks=facts.get("pre_period_weeks"),
            test_duration_weeks=facts.get("test_duration_weeks"),
            num_units=facts.get("sample_size_estimate", "unknown"),
            statistical_approach=(
                "Neyman-orthogonal score / Frisch-Waugh-Lovell theorem: "
                "residualise D and Y on X using cross-fitted ML models, "
                "then regress Ỹ on D̃."
            ),
            primary_model=(
                "Stage 1: gradient boosted trees (XGBoost) for E[D|X] and E[Y|X]. "
                "Stage 2: linear regression on residuals for θ (ATE)."
            ),
            minimum_detectable_effect="Depends on covariate predictive power; typically better than naive OLS.",
            implementation_steps=[
                "Assemble covariate matrix X (user/market features).",
                "Label treatment indicator D (1 = ad-exposed, 0 = unexposed).",
                "Use K-fold cross-fitting: for each fold, train ML models on out-of-fold data.",
                "Compute residuals: Ỹ = Y − Ê[Y|X], D̃ = D − Ê[D|X].",
                "Regress Ỹ on D̃ to obtain causal estimate θ and confidence interval.",
                "Perform sensitivity analysis (Rosenbaum bounds) for unmeasured confounding.",
            ],
            data_requirements=[
                f"Rich covariate matrix: {facts.get('covariate_description', 'user/market features')}.",
                f"Binary or continuous treatment indicator (ad exposure for {facts.get('kpi', 'KPI')}).",
                "Outcome variable (KPI) per observation.",
                "Large enough sample for ML cross-fitting (≥10k observations recommended).",
            ],
            assumptions=[
                "Unconfoundedness: no unmeasured variables drive both ad exposure and outcomes.",
                "Overlap: every unit has non-zero probability of exposure and non-exposure.",
                "Stable Unit Treatment Value Assumption (SUTVA).",
            ],
            caveats=[
                "Cannot handle unmeasured confounders — this is the core limitation.",
                "Requires large datasets for ML models to be reliable.",
                "Interpret θ as Average Treatment Effect on the Treated (ATT), not pure RCT-level ATE.",
            ],
            pros=[
                "Exploits rich observational data without a holdout group.",
                "ML models flexibly account for non-linear confounders.",
                "Neyman-orthogonality makes the final estimate robust to first-stage model misspecification.",
            ],
            cons=[
                "Assumptions are fundamentally untestable — requires domain knowledge.",
                "Computationally intensive for very large datasets.",
            ],
        )

    def generate_scaffold(self, facts: dict) -> str:
        kpi = facts.get("kpi", "outcome")
        covs = facts.get("covariate_description", "user features")
        return f'''"""
Double Debiased Machine Learning — Scaffold
KPI: {kpi} | Covariates: {covs}
Generated by the Measurement Design Agent

Note: This uses a pure-Python implementation of the DML estimator with scikit-learn.
For production, consider the EconML or doubleml Python packages.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# ── 1. Load data ───────────────────────────────────────────────────────────────
# Expected columns: covariate columns + "treatment" (0/1) + "{kpi}"
df = pd.read_csv("your_observational_data.csv")

# Define feature / treatment / outcome
covariate_cols = [c for c in df.columns if c not in ["treatment", "{kpi}"]]
X = df[covariate_cols].values
D = df["treatment"].values          # ad exposure indicator
Y = df["{kpi}"].values.astype(float)

# ── 2. Cross-fitted DML ────────────────────────────────────────────────────────
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

Y_resid = np.zeros_like(Y)
D_resid = np.zeros_like(D, dtype=float)

scaler = StandardScaler()

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]
    D_tr, D_te = D[train_idx], D[test_idx]

    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Stage 1a: predict outcome from covariates
    m_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    m_model.fit(X_tr_s, Y_tr)
    Y_resid[test_idx] = Y_te - m_model.predict(X_te_s)

    # Stage 1b: predict treatment from covariates (propensity model)
    e_model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    e_model.fit(X_tr_s, D_tr)
    D_resid[test_idx] = D_te - e_model.predict_proba(X_te_s)[:, 1]

    print(f"  Fold {{fold+1}}/{{K}} complete")

# ── 3. Stage 2 regression on residuals ────────────────────────────────────────
# θ̂ = (D̃ᵀỸ) / (D̃ᵀD̃)
theta_hat = np.dot(D_resid, Y_resid) / np.dot(D_resid, D_resid)

# Heteroskedasticity-robust standard error
n        = len(Y)
psi      = D_resid * (Y_resid - theta_hat * D_resid)
var_hat  = np.mean(psi**2) / (np.mean(D_resid**2)**2) / n
se_hat   = np.sqrt(var_hat)

t_stat   = theta_hat / se_hat
p_value  = 2 * stats.norm.sf(abs(t_stat))
ci_lo    = theta_hat - 1.96 * se_hat
ci_hi    = theta_hat + 1.96 * se_hat

print("\\n── DML Results ──────────────────────────────────────────────────────────")
print(f"  ATE estimate:  {{theta_hat:.4f}}")
print(f"  Std error:     {{se_hat:.4f}}")
print(f"  95% CI:        [{{ci_lo:.4f}}, {{ci_hi:.4f}}]")
print(f"  p-value:       {{p_value:.4f}}")
direction = "increases" if theta_hat > 0 else "decreases"
print(f"  Interpretation: On average, ad exposure " + direction)
print(f"    '{kpi}' by {{abs(theta_hat):.4f}} units (95% CI: {{ci_lo:.4f}} to {{ci_hi:.4f}}).")
'''
