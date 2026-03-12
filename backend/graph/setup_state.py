"""
LangGraph state schema for the experiment setup workflow.

Domain types (SetupParams, PowerResults, etc.) live in the measurement_design
core library. This module defines the LangGraph-specific SetupState that
composes those domain types with LangGraph annotations.
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Re-export domain types for backward compatibility
from measurement_design.types import (
    SetupParams,
    SetupPhase,
    PowerResults,
    MDEResults,
    SyntheticDataResult,
    ValidationResult,
    RedFlag,
    RedFlagSeverity,
    SETUP_TOPICS,
    MAX_FOLLOWUP_ROUNDS,
)


# ── Main setup workflow state ────────────────────────────────────────────────

class SetupState(TypedDict):
    """Complete mutable state for the setup workflow."""

    session_id: str

    # Conversation (append-only)
    messages: Annotated[list[BaseMessage], add_messages]

    # Carried over from the elicitation workflow
    chosen_method_key: str
    chosen_method_name: str
    elicited_facts: dict[str, Any]
    scores: dict[str, float]
    ranked_methods: list[str]

    # Setup elicitation
    setup_params: SetupParams
    setup_topics_covered: list[str]

    # Workflow control
    setup_phase: SetupPhase
    pending_question: str

    # Adaptive elicitation — follow-ups & feasibility
    followup_round: int              # sub-round counter within current topic
    feasibility_checked: bool        # True after interim power + red-flag check
    interim_power_result: dict       # cached mid-conversation power estimate
    red_flags: list[RedFlag]         # accumulated feasibility concerns

    # Computed results
    power_results: PowerResults
    mde_results: MDEResults
    synthetic_data: SyntheticDataResult
    validation_results: ValidationResult

    # Final artifacts
    setup_report_markdown: str
    power_curve_json: str           # JSON-serialised chart data
    synthetic_data_csv: str         # downloadable CSV

    # Done flag
    done: bool
