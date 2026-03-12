"""
Conditional edge routing functions for the LangGraph graph.

Delegates to measurement_design.workflow.transitions for the actual
state machine logic. Every graph invocation does ONE conversational
turn then reaches END.
"""
from __future__ import annotations

from .state import AgentState
from measurement_design.workflow.transitions import (
    next_elicitation_step,
    after_question_step,
)


def route_entry(state: AgentState) -> str:
    """Dispatch to the correct node based on the current phase."""
    step = next_elicitation_step(state.get("phase", ""))
    return f"{step}_node"


def route_after_question(state: AgentState) -> str:
    """
    After question_node processes a user reply:
    - If all topics are now covered -> 'score_node'
    - Otherwise -> '__end__'  (wait for the next user message)
    """
    step = after_question_step(state.get("phase", ""))
    return f"{step}_node" if step else "__end__"
