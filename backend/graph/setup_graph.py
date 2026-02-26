"""
LangGraph graph assembly for the experiment setup workflow.

Same entry_router pattern as the elicitation graph:
each invocation does EXACTLY ONE turn, then reaches END.

Welcome invocation (phase=''):       router → setup_welcome_node → END
Elicitation turns (phase='setup_elicit'): router → setup_question_node → END
Final elicitation (all topics done):      router → setup_question_node → power → mde → synthetic → validation → review → END
Re-entry for computation phases:          router → {phase}_node → ... → END
Review results (phase='review_results'):  router → review_results_node → output → END  (if accepted)
                                          router → review_results_node → END           (if needs redesign)
Redesign elicit (phase='redesign_elicit'):router → redesign_question_node → power → ... → review → END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from .setup_state import SetupState
from .setup_nodes import (
    setup_welcome_node,
    setup_question_node,
    power_analysis_node,
    mde_simulation_node,
    synthetic_data_node,
    validation_node,
    review_results_node,
    redesign_question_node,
    setup_output_node,
)
from .setup_edges import route_setup_entry, route_after_setup_question, route_after_review


def build_setup_graph():
    builder = StateGraph(SetupState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("setup_entry_router", lambda state: state)  # passthrough
    builder.add_node("setup_welcome_node",      setup_welcome_node)
    builder.add_node("setup_question_node",     setup_question_node)
    builder.add_node("power_analysis_node",     power_analysis_node)
    builder.add_node("mde_simulation_node",     mde_simulation_node)
    builder.add_node("synthetic_data_node",     synthetic_data_node)
    builder.add_node("validation_node",         validation_node)
    builder.add_node("review_results_node",     review_results_node)
    builder.add_node("redesign_question_node",  redesign_question_node)
    builder.add_node("setup_output_node",       setup_output_node)

    # ── Entry ─────────────────────────────────────────────────────────────
    builder.set_entry_point("setup_entry_router")

    builder.add_conditional_edges(
        "setup_entry_router",
        route_setup_entry,
        {
            "setup_welcome_node":      "setup_welcome_node",
            "setup_question_node":     "setup_question_node",
            "power_analysis_node":     "power_analysis_node",
            "mde_simulation_node":     "mde_simulation_node",
            "synthetic_data_node":     "synthetic_data_node",
            "validation_node":         "validation_node",
            "review_results_node":     "review_results_node",
            "redesign_question_node":  "redesign_question_node",
            "setup_output_node":       "setup_output_node",
        },
    )

    # welcome → END  (return greeting, wait for user)
    builder.add_edge("setup_welcome_node", END)

    # question → END  (still eliciting) or → power_analysis_node
    builder.add_conditional_edges(
        "setup_question_node",
        route_after_setup_question,
        {
            "__end__":              END,
            "power_analysis_node":  "power_analysis_node",
        },
    )

    # Computation pipeline: power → mde → synthetic → validation → review
    builder.add_edge("power_analysis_node",  "mde_simulation_node")
    builder.add_edge("mde_simulation_node",  "synthetic_data_node")
    builder.add_edge("synthetic_data_node",  "validation_node")
    builder.add_edge("validation_node",      "review_results_node")

    # review_results → END (wait for user)
    #                → setup_output_node (user accepts)
    #                → power_analysis_node (user wants to re-run with changes)
    builder.add_conditional_edges(
        "review_results_node",
        route_after_review,
        {
            "__end__":              END,
            "setup_output_node":    "setup_output_node",
            "power_analysis_node":  "power_analysis_node",
        },
    )

    # redesign_question → power_analysis (always re-runs pipeline)
    builder.add_edge("redesign_question_node", "power_analysis_node")

    # setup_output → END (done)
    builder.add_edge("setup_output_node", END)

    return builder.compile()


# Module-level compiled graph
setup_graph = build_setup_graph()
