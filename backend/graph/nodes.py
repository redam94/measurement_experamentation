"""
LangGraph node implementations for the measurement design agent.

Thin wrappers around measurement_design.workflow.ElicitationWorkflow.
Each node reads LangGraph state, converts to plain types, calls the
domain workflow, and converts the result back to LangGraph state updates.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .state import AgentState
from measurement_design.workflow import ElicitationWorkflow
from ..adapters.llm_adapter import AnthropicLLMService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_on_token(config: RunnableConfig | None = None):
    """Extract on_token callback from LangGraph config (bridges asyncio.Queue)."""
    if config is None:
        return None
    token_queue = (config.get("configurable") or {}).get("token_queue")
    return token_queue.put if token_queue else None


def _conversation_from_messages(messages: list, limit: int = 6) -> list[dict[str, str]]:
    """Convert LangChain BaseMessage list to plain conversation dicts."""
    return [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in messages[-limit:]
        if isinstance(m, (HumanMessage, AIMessage))
    ]


def _last_human_text(messages: list) -> str | None:
    """Extract the content of the last HumanMessage."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return None


def _make_workflow() -> ElicitationWorkflow:
    """Create an ElicitationWorkflow with the default LLM adapter."""
    return ElicitationWorkflow(AnthropicLLMService())


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def welcome_node(state: AgentState) -> dict:
    """Emit the welcome message and seed the state."""
    wf = _make_workflow()
    result = wf.get_welcome()

    return {
        "messages": [AIMessage(content=result["response"])],
        "phase": result["phase"],
        "covered_topics": result["covered_topics"],
        "elicited_facts": result["facts"],
        "scores": result["scores"],
        "ranked_methods": result["ranked_methods"],
        "clarify_rounds": 0,
        "followup_round": result["followup_round"],
        "report_markdown": "",
        "spec_json": {},
        "spec_yaml": "",
        "scaffold_code": "",
        "done": result["done"],
        "pending_question": result["response"],
    }


async def question_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Process the user's latest reply: extract facts, check sufficiency,
    ask follow-up or next question.
    """
    wf = _make_workflow()
    on_token = _get_on_token(config)
    messages = list(state.get("messages", []))

    result = await wf.handle_question_turn(
        user_reply=_last_human_text(messages),
        facts=dict(state.get("elicited_facts") or {}),
        covered_topics=list(state.get("covered_topics", [])),
        followup_round=state.get("followup_round", 0),
        conversation=_conversation_from_messages(messages),
        on_token=on_token,
    )

    update: dict[str, Any] = {
        "elicited_facts": result["facts"],
        "covered_topics": result["covered_topics"],
        "phase": result["phase"],
        "followup_round": result["followup_round"],
    }

    if result.get("response"):
        update["messages"] = [AIMessage(content=result["response"])]
        update["pending_question"] = result["response"]
    else:
        update["messages"] = []

    return update


async def score_node(state: AgentState, config: RunnableConfig) -> dict:
    """Score all six methods and transition to recommend phase."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = await wf.score_and_rank(
        facts=dict(state.get("elicited_facts") or {}),
        on_token=on_token,
    )

    return {
        "scores": result["scores"],
        "ranked_methods": result["ranked_methods"],
        "phase": result["phase"],
        "messages": [AIMessage(content=result["response"])],
    }


async def recommend_node(state: AgentState, config: RunnableConfig) -> dict:
    """Build ranked summary message from scores and prepare for output."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = await wf.build_recommendations(
        facts=dict(state.get("elicited_facts") or {}),
        scores=dict(state.get("scores") or {}),
        on_token=on_token,
    )

    return {
        "phase": result["phase"],
        "messages": [AIMessage(content=result["response"])],
    }


async def output_node(state: AgentState, config: RunnableConfig) -> dict:
    """Generate all three output artifacts and mark the session done."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = await wf.generate_outputs(
        facts=dict(state.get("elicited_facts") or {}),
        scores=dict(state.get("scores") or {}),
        on_token=on_token,
    )

    return {
        "report_markdown": result["report_markdown"],
        "spec_json": result["spec_json"],
        "spec_yaml": result["spec_yaml"],
        "scaffold_code": result["scaffold_code"],
        "phase": result["phase"],
        "done": result["done"],
        "messages": [AIMessage(content=result["response"])],
    }
