"""
Domain knowledge: red flag catalog for experiment design feasibility checks.

Each entry defines a design concern with its title, detail template, and suggestion.
Detail and suggestion strings may contain format placeholders (e.g. {cv:.0%}, {n_total})
that the caller fills in with situation-specific values.
"""
from __future__ import annotations

RED_FLAG_CATALOG: dict[str, dict] = {
    "high_cv": {
        "title": "High variability relative to effect size",
        "detail": (
            "Your KPI's coefficient of variation is {cv:.0%}. With an expected lift "
            "of {lift}, the signal-to-noise ratio is low, meaning you'll need a very "
            "large sample to detect the effect reliably."
        ),
        "suggestion": (
            "Consider: (a) using a variance-reduction technique like CUPED/regression "
            "adjustment, (b) targeting a larger effect or a less noisy KPI, or "
            "(c) extending the test duration to accumulate more data."
        ),
    },
    "too_few_geos": {
        "title": "Too few geographic units",
        "detail": (
            "You have {n_total} total markets ({n_treat} treatment + {n_ctrl} control). "
            "Geo-based methods typically need at least 15–20 total markets for reliable "
            "inference, and ideally 30+."
        ),
        "suggestion": (
            "Consider: (a) adding more markets to both groups, (b) switching to a "
            "user-level A/B test if randomisation is feasible, or (c) using synthetic "
            "control with a single treatment region and many donors."
        ),
    },
    "short_pre_period": {
        "title": "Pre-period may be too short",
        "detail": (
            "You have {n_pre} pre-campaign periods. Methods like DiD and synthetic "
            "control rely on the pre-period to establish a baseline trend — fewer than "
            "8 periods makes parallel-trends assumptions harder to verify."
        ),
        "suggestion": (
            "If possible, extend the pre-period to at least 8–12 weeks. Alternatively, "
            "if you have daily data, consider using daily granularity for more observations."
        ),
    },
    "tiny_effect": {
        "title": "Very small expected effect",
        "detail": (
            "The expected lift of {lift} is quite small relative to the baseline "
            "variability. You would need approximately {required_n:,} {unit_label} "
            "to detect this reliably."
        ),
        "suggestion": (
            "Consider: (a) whether a larger effect is plausible (even 2× the lift "
            "cuts required sample size by ~75%), (b) running the test longer, or "
            "(c) accepting lower power (e.g., 70%) if this is an exploratory test."
        ),
    },
    "high_icc": {
        "title": "High intra-cluster correlation",
        "detail": (
            "The ICC of {icc:.2f} means observations within each cluster are quite "
            "similar, reducing the effective sample size substantially."
        ),
        "suggestion": (
            "Consider randomising at a finer level (e.g., user-level instead of "
            "market-level), increasing the number of clusters, or using covariates "
            "to absorb within-cluster correlation."
        ),
    },
    "few_donors_sc": {
        "title": "Too few donor units for synthetic control",
        "detail": (
            "You have only {n_donors} donor units. Synthetic control works best with "
            "a rich donor pool (at least 10, ideally 20+) so the algorithm can find "
            "a good weighted combination to match the treatment unit."
        ),
        "suggestion": (
            "Consider: (a) adding more potential donor regions, (b) using sub-regions "
            "or DMAs instead of states, or (c) switching to DiD if you can designate "
            "a proper control group."
        ),
    },
    "extreme_imbalance": {
        "title": "Large treatment/control imbalance",
        "detail": (
            "Your ratio of treatment to control units is {ratio:.1f}:1. Very "
            "unbalanced designs lose statistical efficiency."
        ),
        "suggestion": (
            "Aim for a ratio between 1:1 and 3:1. If you can't change group sizes, "
            "consider stratified randomisation or matching to improve balance."
        ),
    },
    "insufficient_sample": {
        "title": "Available sample may be too small",
        "detail": (
            "Based on your inputs, you'd need ~{required_n:,} {unit_label} but you "
            "indicated having roughly {available_n:,}. This means you're likely "
            "under-powered for the target effect size."
        ),
        "suggestion": (
            "Options: (a) run the test longer to accumulate more observations, "
            "(b) increase the treatment proportion, (c) accept a larger minimum "
            "detectable effect, or (d) reduce noise with covariates (CUPED)."
        ),
    },
}
