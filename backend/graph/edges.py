"""
Conditional edge routing functions for the LangGraph graph.

Every graph invocation does ONE conversational turn then reaches END.
"""
from __future__ import annotations

from .state import AgentState, ELICITATION_TOPICS


def route_entry(state: AgentState) -> str:
    """
    Dispatch to the correct node based on the current phase.
    Called by the entry_router passthrough node.
    """
    phase = state.get("phase", "")
    if not phase:
        return "welcome_node"
    if phase == "elicit":
        return "question_node"
    if phase == "score":
        return "score_node"
    if phase == "recommend":
        return "recommend_node"
    if phase == "output":
        return "output_node"
    # fallback: keep eliciting
    return "question_node"


def route_after_question(state: AgentState) -> str:
    """
    After question_node processes a user reply:
    - If all topics are now covered -> 'score_node'  (run the rest of the pipeline)
    - Otherwise -> '__end__'  (wait for the next user message)
    """
    if state.get("phase") == "score":
        return "score_node"
    return "__end__"
