"""
Conditional edge routing for the experiment setup workflow.

Delegates to measurement_design.workflow.transitions for the actual
state machine logic. Same single-turn-per-invocation pattern as the
elicitation graph.
"""
from __future__ import annotations

from .setup_state import SetupState
from measurement_design.workflow.transitions import (
    next_setup_step,
    after_setup_question_step,
    after_review_step,
)


def route_setup_entry(state: SetupState) -> str:
    """Dispatch to the correct node based on setup_phase."""
    step = next_setup_step(state.get("setup_phase", ""))
    return f"{step}_node"


def route_after_setup_question(state: SetupState) -> str:
    """
    After setup_question_node:
    - If all setup topics covered -> 'power_analysis_node'
    - Otherwise -> '__end__' (wait for user)
    """
    step = after_setup_question_step(state.get("setup_phase", ""))
    return f"{step}_node" if step else "__end__"


def route_after_review(state: SetupState) -> str:
    """
    After review_results_node:
    - If user accepts -> 'setup_output_node'
    - If user wants to re-run -> 'power_analysis_node'
    - Otherwise -> '__end__' (wait for user input)
    """
    step = after_review_step(state.get("setup_phase", ""))
    return f"{step}_node" if step else "__end__"
