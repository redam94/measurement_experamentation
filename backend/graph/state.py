"""
LangGraph state schema for the measurement design agent.

Domain types (ElicitedFacts, Phase, etc.) live in the measurement_design
core library. This module defines the LangGraph-specific AgentState that
composes those domain types with LangGraph annotations.
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Re-export domain types for backward compatibility
from measurement_design.types import (
    ElicitedFacts,
    Phase,
    ELICITATION_TOPICS,
)


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

    # Number of follow-up rounds on the current topic (reset when topic changes)
    followup_round: int

    # Flag set by the streaming API to detect when the graph has finished
    done: bool
