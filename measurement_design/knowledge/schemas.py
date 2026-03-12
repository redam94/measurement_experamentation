"""
Domain knowledge: method schemas, display names, topic labels, and related constants.

These are structural definitions about the measurement domain, independent of
any UI framework or LLM. Both frontend and backend can import from here.
"""
from __future__ import annotations

from typing import Any

# ── Topic labels (elicitation) ──────────────────────────────────────────────────
TOPIC_LABELS = {
    "objective":         "Objective",
    "randomization":     "Randomisation",
    "data_history":      "Data History",
    "geo_structure":     "Geography",
    "treatment_control": "Treatment Control",
    "covariates":        "Covariates",
    "scale":             "Scale & Duration",
}

ALL_TOPICS = list(TOPIC_LABELS.keys())

# ── Topic labels (setup) ───────────────────────────────────────────────────────
SETUP_TOPIC_LABELS = {
    "baseline_metrics":   "Baseline Metrics",
    "expected_effect":    "Expected Effect",
    "statistical_design": "Statistical Design",
    "method_specific":    "Method Details",
}

ALL_SETUP_TOPICS = list(SETUP_TOPIC_LABELS.keys())

# ── Method key → display name ──────────────────────────────────────────────────
METHOD_NAMES = {
    "ab_test":           "A/B Test",
    "did":               "Difference-in-Differences",
    "ddml":              "Double/Debiased ML",
    "geo_lift":          "Geo Lift Test",
    "synthetic_control": "Synthetic Control",
    "matched_market":    "Matched Market Test",
}

# Methods that have panel / time-series structure
PANEL_METHODS = {"did", "geo_lift", "synthetic_control", "matched_market"}

# ── Computation phases (setup pipeline) ─────────────────────────────────────────
COMPUTATION_PHASES = [
    ("power_analysis",  "Power Analysis"),
    ("mde_simulation",  "MDE Simulation"),
    ("synthetic_gen",   "Synthetic Data"),
    ("validation",      "Code Validation"),
    ("review_results",  "Results Review"),
    ("setup_output",    "Setup Report"),
]

# ── Data-template schemas per method ────────────────────────────────────────────
METHOD_SCHEMAS: dict[str, dict[str, Any]] = {
    "ab_test": {
        "description": (
            "User-level data for a randomised A/B test. Each row represents one "
            "user assigned to either `control` or `treatment`."
        ),
        "columns": [
            {"name": "user_id",   "dtype": "int",    "description": "Unique user identifier",         "example": 1,          "required": True},
            {"name": "group",     "dtype": "string", "description": "Assignment group",                "example": "control",  "required": True},
            {"name": "converted", "dtype": "int",    "description": "Binary conversion flag (0/1)",    "example": 0,          "required": False},
            {"name": "outcome",   "dtype": "float",  "description": "Continuous outcome metric",       "example": 42.5,       "required": False},
        ],
        "notes": "Include *either* `converted` (proportions test) or `outcome` (continuous test), not both.",
        "form_fields": {
            "n_per_group":    {"label": "Users per group",    "type": "int",   "default": 1000, "min": 10},
            "baseline_rate":  {"label": "Baseline conv rate", "type": "float", "default": 0.05, "min": 0.001, "max": 0.999},
            "lift_abs":       {"label": "Expected lift (abs)","type": "float", "default": 0.005},
        },
    },
    "did": {
        "description": (
            "Panel data for Difference-in-Differences. Each row is one unit x one time period."
        ),
        "columns": [
            {"name": "unit_id",  "dtype": "string", "description": "Unit / market identifier",          "example": "T_1",     "required": True},
            {"name": "group",    "dtype": "string", "description": "treatment or control",              "example": "treatment","required": True},
            {"name": "period",   "dtype": "int",    "description": "Time period index (1-based)",       "example": 1,         "required": True},
            {"name": "is_post",  "dtype": "int",    "description": "1 if post-treatment, else 0",       "example": 0,         "required": True},
            {"name": "outcome",  "dtype": "float",  "description": "Outcome metric value",              "example": 105.3,     "required": True},
        ],
        "notes": "Ensure balanced panel: every unit should appear in every period.",
        "form_fields": {
            "num_treatment_units": {"label": "# Treatment units",  "type": "int",   "default": 5,    "min": 1},
            "num_control_units":   {"label": "# Control units",    "type": "int",   "default": 10,   "min": 1},
            "num_pre_periods":     {"label": "# Pre periods",      "type": "int",   "default": 12,   "min": 1},
            "num_post_periods":    {"label": "# Post periods",     "type": "int",   "default": 8,    "min": 1},
            "baseline_value":      {"label": "Baseline metric mean","type": "float", "default": 100.0},
            "baseline_std":        {"label": "Baseline metric std", "type": "float", "default": 30.0, "min": 0.01},
            "lift_abs":            {"label": "Expected lift (abs)", "type": "float", "default": 5.0},
        },
    },
    "geo_lift": {
        "description": (
            "Market-level weekly data for a Geo Lift test. Each row is one geo x one week."
        ),
        "columns": [
            {"name": "geo_id",    "dtype": "string", "description": "Geography / DMA identifier",      "example": "geo_001",  "required": True},
            {"name": "group",     "dtype": "string", "description": "treatment or control",             "example": "control",  "required": True},
            {"name": "week",      "dtype": "int",    "description": "Week number (1-based)",            "example": 1,          "required": True},
            {"name": "is_post",   "dtype": "int",    "description": "1 if post-treatment, else 0",      "example": 0,          "required": True},
            {"name": "kpi_value", "dtype": "float",  "description": "KPI value (e.g. sales, signups)", "example": 4850.0,     "required": True},
        ],
        "notes": "Include sufficient pre-period weeks (>= 8) for the model to learn geo-level patterns.",
        "form_fields": {
            "num_treatment_geos": {"label": "# Treatment geos",   "type": "int",   "default": 5,    "min": 1},
            "num_control_geos":   {"label": "# Control geos",     "type": "int",   "default": 15,   "min": 1},
            "num_pre_periods":    {"label": "# Pre weeks",         "type": "int",   "default": 12,   "min": 1},
            "num_post_periods":   {"label": "# Post weeks",        "type": "int",   "default": 6,    "min": 1},
            "baseline_value":     {"label": "Baseline KPI/week",   "type": "float", "default": 5000.0},
            "baseline_std":       {"label": "Baseline std",        "type": "float", "default": 1500.0, "min": 0.01},
            "lift_abs":           {"label": "Expected lift (abs)", "type": "float", "default": 250.0},
        },
    },
    "synthetic_control": {
        "description": (
            "Time-series data for Synthetic Control. One treated unit + J donor units."
        ),
        "columns": [
            {"name": "unit_id",    "dtype": "string", "description": "Unit identifier (treatment or donor_NNN)","example": "treatment","required": True},
            {"name": "is_treated", "dtype": "int",    "description": "1 for treated unit, 0 for donors",        "example": 1,          "required": True},
            {"name": "period",     "dtype": "int",    "description": "Time period index (1-based)",              "example": 1,          "required": True},
            {"name": "is_post",    "dtype": "int",    "description": "1 if post-treatment, else 0",              "example": 0,          "required": True},
            {"name": "outcome",    "dtype": "float",  "description": "Outcome metric value",                     "example": 98.7,       "required": True},
        ],
        "notes": "Long pre-periods (>= 20) improve the quality of the synthetic counterfactual.",
        "form_fields": {
            "num_donor_units":  {"label": "# Donor units",       "type": "int",   "default": 15,   "min": 2},
            "num_pre_periods":  {"label": "# Pre periods",       "type": "int",   "default": 52,   "min": 5},
            "num_post_periods": {"label": "# Post periods",      "type": "int",   "default": 12,   "min": 1},
            "baseline_value":   {"label": "Baseline metric mean","type": "float", "default": 100.0},
            "baseline_std":     {"label": "Baseline metric std", "type": "float", "default": 20.0, "min": 0.01},
            "lift_abs":         {"label": "Expected lift (abs)", "type": "float", "default": 10.0},
        },
    },
    "matched_market": {
        "description": (
            "Paired market data for a Matched Market test. Each pair has one treatment and one control market."
        ),
        "columns": [
            {"name": "pair_id",   "dtype": "int",    "description": "Matched pair identifier",          "example": 1,             "required": True},
            {"name": "market_id", "dtype": "string", "description": "Market identifier",                "example": "pair_1_T",    "required": True},
            {"name": "group",     "dtype": "string", "description": "treatment or control",             "example": "treatment",   "required": True},
            {"name": "week",      "dtype": "int",    "description": "Week number (1-based)",            "example": 1,             "required": True},
            {"name": "is_post",   "dtype": "int",    "description": "1 if post-treatment, else 0",      "example": 0,             "required": True},
            {"name": "kpi_value", "dtype": "float",  "description": "KPI value for that market/week",   "example": 5200.0,        "required": True},
        ],
        "notes": "Markets within each pair should be as similar as possible in pre-period KPI trends.",
        "form_fields": {
            "num_pairs":        {"label": "# Market pairs",      "type": "int",   "default": 6,    "min": 2},
            "num_pre_periods":  {"label": "# Pre weeks",         "type": "int",   "default": 8,    "min": 1},
            "num_post_periods": {"label": "# Post weeks",        "type": "int",   "default": 4,    "min": 1},
            "baseline_value":   {"label": "Baseline KPI/week",   "type": "float", "default": 5000.0},
            "baseline_std":     {"label": "Baseline std",        "type": "float", "default": 1500.0, "min": 0.01},
            "lift_abs":         {"label": "Expected lift (abs)", "type": "float", "default": 250.0},
        },
    },
    "ddml": {
        "description": (
            "Observational data with covariates for Double/Debiased ML. "
            "Each row is one observational unit."
        ),
        "columns": [
            {"name": "user_id",   "dtype": "int",   "description": "Unique user identifier",                 "example": 1,      "required": True},
            {"name": "x_1...x_K", "dtype": "float", "description": "Covariate columns (confounders)",        "example": 0.34,   "required": True},
            {"name": "treatment", "dtype": "int",   "description": "Binary treatment indicator (0/1)",        "example": 1,      "required": True},
            {"name": "outcome",   "dtype": "float", "description": "Continuous outcome metric",               "example": 105.2,  "required": True},
        ],
        "notes": "Include as many pre-treatment covariates as possible to satisfy the unconfoundedness assumption.",
        "form_fields": {
            "n_obs":         {"label": "# Observations",  "type": "int",   "default": 5000, "min": 100},
            "n_covariates":  {"label": "# Covariates",    "type": "int",   "default": 10,   "min": 1, "max": 50},
            "baseline_value":{"label": "Baseline mean",   "type": "float", "default": 100.0},
            "baseline_std":  {"label": "Baseline std",    "type": "float", "default": 30.0, "min": 0.01},
            "lift_abs":      {"label": "Expected lift",   "type": "float", "default": 5.0},
        },
    },
}
