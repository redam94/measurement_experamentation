"""LangGraph graph assembly and compilation.

Each graph invocation performs EXACTLY ONE conversational turn, then reaches END.

First invocation  (phase=''):       entry_router → welcome_node → END
Elicitation turns (phase='elicit'):  entry_router → question_node → END
Final elicitation (all topics done): entry_router → question_node → score_node → recommend_node → output_node → END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    welcome_node,
    question_node,
    score_node,
    recommend_node,
    output_node,
)
from .edges import route_entry, route_after_question


def build_graph():
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("entry_router",   lambda state: state)  # passthrough
    builder.add_node("welcome_node",   welcome_node)
    builder.add_node("question_node",  question_node)
    builder.add_node("score_node",     score_node)
    builder.add_node("recommend_node", recommend_node)
    builder.add_node("output_node",    output_node)

    # ── Entry ─────────────────────────────────────────────────────────────
    builder.set_entry_point("entry_router")

    # Router dispatches to the correct node based on current phase.
    builder.add_conditional_edges(
        "entry_router",
        route_entry,
        {
            "welcome_node":  "welcome_node",
            "question_node": "question_node",
            "score_node":    "score_node",
            "recommend_node": "recommend_node",
            "output_node":   "output_node",
        },
    )

    # welcome → END  (return greeting, wait for user)
    builder.add_edge("welcome_node", END)

    # question → END  (still eliciting) or → score_node (all topics done)
    builder.add_conditional_edges(
        "question_node",
        route_after_question,
        {
            "__end__":    END,
            "score_node": "score_node",
        },
    )

    # score → recommend → output → END  (runs to completion in one shot)
    builder.add_edge("score_node",     "recommend_node")
    builder.add_edge("recommend_node", "output_node")
    builder.add_edge("output_node",    END)

    return builder.compile()


# Module-level compiled graph (imported by the API)
graph = build_graph()
