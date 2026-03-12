"""
Domain knowledge: elicitation and setup topic metadata.

Defines the structural metadata for each elicitation topic (what facts it produces,
when to follow up, why it matters) and setup topic (expected output fields).

The actual LLM prompt templates (question text, extraction prompts) stay in
backend/prompts/. This module provides the domain-level topic structure.
"""
from __future__ import annotations

# ── Elicitation topic metadata ───────────────────────────────────────────────

ELICITATION_TOPIC_META: dict[str, dict] = {
    "objective": {
        "output_fields": {
            "primary_objective": {
                "type": "enum",
                "values": ["conversion", "awareness", "retention", "engagement", "other"],
                "description": "High-level campaign objective category",
            },
            "kpi": {
                "type": "string",
                "description": "Specific metric name (e.g. 'purchases', 'brand recall score')",
            },
        },
        "followup_triggers": ["kpi is 'unknown'", "primary_objective is 'other'"],
        "why_it_matters": (
            "Knowing the specific metric helps us choose the right statistical method "
            "and design an experiment that can actually detect a meaningful change."
        ),
    },
    "randomization": {
        "output_fields": {
            "randomization_unit": {
                "type": "enum",
                "values": ["user", "device", "geo", "market", "unknown"],
                "description": "The level at which randomization occurs",
            },
            "can_run_rct": {
                "type": "bool",
                "description": "Whether the brand/team can control exposure",
            },
        },
        "followup_triggers": ["randomization_unit is 'unknown'"],
        "why_it_matters": (
            "Whether you can create a fair comparison group (people who don't see the ad) "
            "determines which measurement methods are available. If you can, we have "
            "more powerful options. If not, we still have great approaches — they just "
            "work a bit differently."
        ),
    },
    "data_history": {
        "output_fields": {
            "pre_period_weeks": {
                "type": "int_or_null",
                "description": "Weeks of pre-campaign historical data",
            },
            "has_historical_data": {
                "type": "bool",
                "description": "Whether any meaningful history exists",
            },
        },
        "followup_triggers": ["pre_period_weeks is null and has_historical_data is true"],
        "why_it_matters": (
            "Historical data lets us establish what 'normal' looks like before the "
            "campaign starts. The more history we have, the more confident we can be "
            "that any change we see was actually caused by the campaign."
        ),
    },
    "geo_structure": {
        "output_fields": {
            "num_markets": {
                "type": "int_or_null",
                "description": "Number of distinct geographic markets",
            },
            "geo_holdout_feasible": {
                "type": "bool",
                "description": "Whether some geos can be kept ad-free as a control",
            },
        },
        "followup_triggers": ["num_markets is null"],
        "why_it_matters": (
            "The number of separate locations determines whether we can use powerful "
            "geographic comparison methods. Even 10–15 locations can be enough if "
            "some can serve as a comparison group."
        ),
    },
    "treatment_control": {
        "output_fields": {
            "campaign_type": {
                "type": "enum",
                "values": ["brand_controlled", "platform_only", "mixed", "observational"],
                "description": "Who controls campaign targeting and timing",
            },
            "control_group_exists": {
                "type": "bool",
                "description": "Whether a formal no-ad control group is planned",
            },
        },
        "followup_triggers": [],
        "why_it_matters": (
            "How much control you have affects which testing methods are feasible. "
            "More control means more options, but even with less control we can still "
            "design a solid measurement approach."
        ),
    },
    "covariates": {
        "output_fields": {
            "has_rich_covariates": {
                "type": "bool",
                "description": "Whether meaningful user/market features are available",
            },
            "covariate_description": {
                "type": "string",
                "description": "Short description of available covariates",
            },
        },
        "followup_triggers": [],
        "why_it_matters": (
            "Background data helps us reduce 'noise' in the results — like accounting "
            "for the fact that some customers spend more than others regardless of "
            "the campaign. The more we know, the smaller the experiment can be."
        ),
    },
    "scale": {
        "output_fields": {
            "sample_size_estimate": {
                "type": "enum",
                "values": ["small (<10k)", "medium (10k-1M)", "large (>1M)", "unknown"],
                "description": "Approximate reach of the campaign",
            },
            "test_duration_weeks": {
                "type": "int_or_null",
                "description": "Planned campaign duration in weeks",
            },
        },
        "followup_triggers": ["sample_size_estimate is 'unknown'"],
        "why_it_matters": (
            "The number of people and the duration together determine how much data "
            "we'll have to work with. More data means we can detect smaller effects "
            "and be more confident in the results."
        ),
    },
}

ALL_ELICITATION_TOPICS = list(ELICITATION_TOPIC_META.keys())


# ── Setup topic metadata ────────────────────────────────────────────────────

SETUP_TOPIC_META: dict[str, dict] = {
    "baseline_metrics": {
        "output_fields": {
            "baseline_rate": {
                "type": "float_or_null",
                "description": "Baseline rate/proportion (0–1), if applicable",
            },
            "baseline_metric_value": {
                "type": "float_or_null",
                "description": "Average metric value",
            },
            "baseline_metric_std": {
                "type": "float_or_null",
                "description": "Standard deviation / variability of the metric",
            },
        },
        "critical_fields": ["baseline_metric_std"],
        "why_it_matters": (
            "The baseline value and its variability determine how many observations "
            "you need to detect a given effect size."
        ),
    },
    "expected_effect": {
        "output_fields": {
            "expected_lift_pct": {
                "type": "float_or_null",
                "description": "Relative lift as decimal (0.10 = 10%)",
            },
            "expected_lift_abs": {
                "type": "float_or_null",
                "description": "Absolute change in the metric",
            },
        },
        "critical_fields": [],
        "why_it_matters": (
            "The expected effect size directly determines the minimum sample size "
            "needed and whether the test is feasible."
        ),
    },
    "statistical_design": {
        "output_fields": {
            "alpha": {
                "type": "float",
                "description": "Significance level (default 0.05)",
                "default": 0.05,
            },
            "power_target": {
                "type": "float",
                "description": "Desired statistical power (default 0.80)",
                "default": 0.80,
            },
            "one_sided": {
                "type": "bool",
                "description": "Whether to use a one-sided test (default False)",
                "default": False,
            },
        },
        "critical_fields": [],
        "why_it_matters": (
            "These parameters control the trade-off between sensitivity and "
            "false-positive risk."
        ),
    },
    "method_specific": {
        "output_fields": {
            "num_treatment_units": {"type": "int_or_null", "description": "Number of treatment units"},
            "num_control_units": {"type": "int_or_null", "description": "Number of control units"},
            "num_pre_periods": {"type": "int_or_null", "description": "Number of pre-campaign periods"},
            "num_post_periods": {"type": "int_or_null", "description": "Number of post-campaign periods"},
            "cluster_size": {"type": "int_or_null", "description": "Average units per cluster"},
            "icc": {"type": "float_or_null", "description": "Intra-cluster correlation"},
        },
        "critical_fields": [],
        "why_it_matters": (
            "Method-specific parameters determine the exact design of the experiment."
        ),
    },
}

ALL_SETUP_TOPIC_KEYS = list(SETUP_TOPIC_META.keys())
