"""
LangGraph state schema for the measurement design agent.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


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


# ── Main agent state ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Complete mutable state carried through the LangGraph graph."""

    session_id: str

    # Full conversation history (append-only via add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Structured facts extracted so far
    elicited_facts: ElicitedFacts

    # Which elicitation topics have been covered
    covered_topics: list[str]

    # Current phase of the agent lifecycle
    phase: Phase

    # Method scores: method_key → score (0–100)
    scores: dict[str, float]

    # Ordered list of method keys after scoring
    ranked_methods: list[str]

    # Number of clarification rounds already run
    clarify_rounds: int

    # Final output artifacts (populated in output phase)
    report_markdown: str
    spec_json: dict[str, Any]
    spec_yaml: str
    scaffold_code: str

    # Pending question to surface to the user (set by question_node)
    pending_question: str

    # Flag set by the streaming API to detect when the graph has finished
    done: bool
