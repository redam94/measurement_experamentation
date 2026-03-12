"""
Domain type definitions for measurement experiment design.

These are pure data types with no framework dependencies. LangGraph-specific
state types (AgentState, SetupState) live in backend/graph/ and compose these.
"""
from __future__ import annotations

from typing import Literal
from typing_extensions import TypedDict


# ── Elicited facts structure ──────────────────────────────────────────────────

class ElicitedFacts(TypedDict, total=False):
    """Structured knowledge extracted from the conversation."""

    # 1. Objective
    primary_objective: str          # e.g. "conversion", "awareness", "retention"
    kpi: str                        # e.g. "revenue", "installs", "ROAS"

    # 2. Randomization unit
    randomization_unit: str         # "user", "device", "geo", "market", "unknown"
    can_run_rct: bool               # True if brand controls assignment

    # 3. Data history
    pre_period_weeks: int | None    # weeks of historical data available
    has_historical_data: bool

    # 4. Geographic structure
    num_markets: int | None         # number of distinct geos/markets
    geo_holdout_feasible: bool

    # 5. Treatment / control
    campaign_type: str              # "brand_controlled", "platform_only", "observational"
    control_group_exists: bool

    # 6. Covariate richness
    has_rich_covariates: bool       # demographic, behavioral, contextual features
    covariate_description: str

    # 7. Scale / budget
    sample_size_estimate: str       # "small (<10k)", "medium (10k-1M)", "large (>1M)"
    test_duration_weeks: int | None

    # Extra free-text context captured by LLM extraction
    additional_context: str


# ── Phase literals ────────────────────────────────────────────────────────────

Phase = Literal["elicit", "clarify", "score", "recommend", "output", "done"]

ELICITATION_TOPICS = [
    "objective",
    "randomization",
    "data_history",
    "geo_structure",
    "treatment_control",
    "covariates",
    "scale",
]


# ── Setup parameters gathered from user ──────────────────────────────────────

class SetupParams(TypedDict, total=False):
    """Parameters the user provides (or confirms defaults for) during setup."""

    # Baseline metrics
    baseline_rate: float | None          # e.g. 0.03 for 3% conversion rate
    baseline_metric_value: float | None  # e.g. mean weekly sales = 5000
    baseline_metric_std: float | None    # standard deviation of baseline metric

    # Expected / desired effect
    expected_lift_pct: float | None      # e.g. 0.10 for 10% relative lift
    expected_lift_abs: float | None      # absolute lift (e.g. +0.003 in conv rate)

    # Statistical design
    alpha: float             # significance level, default 0.05
    power_target: float      # desired power, default 0.80
    one_sided: bool          # one-sided test? default False

    # Method-specific
    num_treatment_units: int | None      # markets, users, etc.
    num_control_units: int | None
    num_pre_periods: int | None
    num_post_periods: int | None
    cluster_size: int | None             # units per cluster (for clustered designs)
    icc: float | None                    # intra-cluster correlation

    # Simulation control
    n_simulations: int       # Monte Carlo replications, default 1000
    random_seed: int         # reproducibility, default 42


# ── Computed results ─────────────────────────────────────────────────────────

class PowerResults(TypedDict, total=False):
    """Output of the power / sample-size calculation."""
    required_sample_size: int | None
    achieved_power: float | None
    power_curve: list[dict[str, float]]   # [{n: ..., power: ...}, ...]
    effect_size_used: float | None
    notes: str


class MDEResults(TypedDict, total=False):
    """Output of the Monte Carlo MDE simulation."""
    mde_absolute: float | None
    mde_relative_pct: float | None
    power_by_effect: list[dict[str, float]]  # [{effect: ..., power: ...}, ...]
    n_simulations: int
    alpha: float
    target_power: float
    notes: str


class SyntheticDataResult(TypedDict, total=False):
    """Output of synthetic data generation."""
    csv_string: str                # synthetic data as CSV
    n_rows: int
    columns: list[str]
    true_effect: float             # ground-truth effect injected
    description: str               # human-readable summary


class ValidationResult(TypedDict, total=False):
    """Output of running the scaffold code against synthetic data."""
    success: bool
    estimated_effect: float | None
    true_effect: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    summary: str
    error_message: str


# ── Setup phases and topics ──────────────────────────────────────────────────

SetupPhase = Literal[
    "",
    "setup_welcome",
    "setup_elicit",
    "power_analysis",
    "mde_simulation",
    "synthetic_gen",
    "validation",
    "review_results",
    "redesign_elicit",
    "setup_output",
    "setup_done",
]

SETUP_TOPICS = [
    "baseline_metrics",
    "expected_effect",
    "statistical_design",
    "method_specific",
]

# Maximum follow-up rounds per topic before applying defaults and moving on
MAX_FOLLOWUP_ROUNDS = 2


# ── Red flag severity levels ─────────────────────────────────────────────────

RedFlagSeverity = Literal["warning", "critical"]


class RedFlag(TypedDict, total=False):
    """A feasibility concern detected from the user's inputs."""
    severity: RedFlagSeverity
    flag: str          # short machine-readable tag  (e.g. "high_cv")
    title: str         # one-line human-readable heading
    detail: str        # explanation of the problem
    suggestion: str    # actionable recommendation
