"""
Conditional edge routing for the experiment setup workflow.
Same single-turn-per-invocation pattern as the elicitation graph.
"""
from __future__ import annotations

from .setup_state import SetupState, SETUP_TOPICS


def route_setup_entry(state: SetupState) -> str:
    """
    Dispatch to the correct node based on setup_phase.
    Called by the setup_entry_router passthrough node.
    """
    phase = state.get("setup_phase", "")

    if not phase or phase == "setup_welcome":
        return "setup_welcome_node"
    if phase == "setup_elicit":
        return "setup_question_node"
    if phase == "power_analysis":
        return "power_analysis_node"
    if phase == "mde_simulation":
        return "mde_simulation_node"
    if phase == "synthetic_gen":
        return "synthetic_data_node"
    if phase == "validation":
        return "validation_node"
    if phase == "review_results":
        return "review_results_node"
    if phase == "redesign_elicit":
        return "redesign_question_node"
    if phase == "setup_output":
        return "setup_output_node"

    # Fallback
    return "setup_question_node"


def route_after_setup_question(state: SetupState) -> str:
    """
    After setup_question_node:
    - If all setup topics covered → 'power_analysis_node' (run computation pipeline)
    - Otherwise → '__end__' (wait for user)
    """
    if state.get("setup_phase") == "power_analysis":
        return "power_analysis_node"
    return "__end__"


def route_after_review(state: SetupState) -> str:
    """
    After review_results_node:
    - If user accepts → 'setup_output_node' (generate final report)
    - If user wants to re-run → 'power_analysis_node' (re-run computation)
    - Otherwise → '__end__' (wait for user input)
    """
    phase = state.get("setup_phase", "")
    if phase == "setup_output":
        return "setup_output_node"
    if phase == "power_analysis":
        return "power_analysis_node"
    return "__end__"
