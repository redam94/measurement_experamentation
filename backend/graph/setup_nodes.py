"""
LangGraph node implementations for the experiment setup workflow.

Thin wrappers around measurement_design.workflow.SetupWorkflow.
Each node reads LangGraph state, converts to plain types, calls the
domain workflow, and converts the result back to LangGraph state updates.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .setup_state import SetupState
from measurement_design.workflow import SetupWorkflow
from ..adapters.llm_adapter import AnthropicLLMService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_on_token(config: RunnableConfig | None = None):
    """Extract on_token callback from LangGraph config."""
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


def _make_workflow() -> SetupWorkflow:
    """Create a SetupWorkflow with the default LLM adapter."""
    return SetupWorkflow(AnthropicLLMService())


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def setup_welcome_node(state: SetupState, config: RunnableConfig) -> dict:
    """Emit the setup welcome message with the first question."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = await wf.get_welcome(
        method_key=state.get("chosen_method_key", ""),
        method_name=state.get("chosen_method_name", "your chosen method"),
        on_token=on_token,
    )

    return {
        "messages": [AIMessage(content=result["response"])],
        "setup_phase": result["setup_phase"],
        "setup_topics_covered": result["setup_topics_covered"],
        "setup_params": result["setup_params"],
        "power_results": {},
        "mde_results": {},
        "synthetic_data": {},
        "validation_results": {},
        "setup_report_markdown": "",
        "power_curve_json": "",
        "synthetic_data_csv": "",
        "done": result["done"],
        "pending_question": result["response"],
        "followup_round": result["followup_round"],
        "feasibility_checked": result["feasibility_checked"],
        "interim_power_result": result["interim_power_result"],
        "red_flags": result["red_flags"],
    }


async def setup_question_node(state: SetupState, config: RunnableConfig) -> dict:
    """Process user reply for the current setup topic, extract params, ask next."""
    wf = _make_workflow()
    on_token = _get_on_token(config)
    messages = list(state.get("messages", []))
    interim_power = dict(state.get("interim_power_result") or {}) or None

    result = await wf.handle_question_turn(
        user_reply=_last_human_text(messages),
        params=dict(state.get("setup_params") or {}),
        covered=list(state.get("setup_topics_covered", [])),
        method_key=state.get("chosen_method_key", "ab_test"),
        facts=dict(state.get("elicited_facts") or {}),
        followup_round=state.get("followup_round", 0),
        feasibility_checked=state.get("feasibility_checked", False),
        red_flags=list(state.get("red_flags") or []),
        interim_power=interim_power,
        conversation=_conversation_from_messages(messages),
        on_token=on_token,
    )

    update: dict[str, Any] = {
        "setup_params": result["setup_params"],
        "setup_topics_covered": result["setup_topics_covered"],
        "setup_phase": result["setup_phase"],
        "red_flags": result["red_flags"],
    }

    if result.get("response"):
        update["messages"] = [AIMessage(content=result["response"])]
        update["pending_question"] = result["response"]
    else:
        update["messages"] = []

    if "followup_round" in result:
        update["followup_round"] = result["followup_round"]
    if "feasibility_checked" in result:
        update["feasibility_checked"] = result["feasibility_checked"]
    if "interim_power_result" in result:
        update["interim_power_result"] = result["interim_power_result"]

    return update


async def power_analysis_node(state: SetupState, config: RunnableConfig) -> dict:
    """Run power / sample-size calculations."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = wf.run_power_analysis(
        method_key=state.get("chosen_method_key", "ab_test"),
        params=dict(state.get("setup_params") or {}),
        facts=dict(state.get("elicited_facts") or {}),
        on_token=on_token,
    )

    msg = result["response"]
    if on_token:
        await on_token(msg + "\n\n")

    return {
        "power_results": result["power_results"],
        "setup_phase": result["setup_phase"],
        "messages": [AIMessage(content=msg)],
        "power_curve_json": result["power_curve_json"],
    }


async def mde_simulation_node(state: SetupState, config: RunnableConfig) -> dict:
    """Run Monte Carlo MDE simulation."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = wf.run_mde_simulation(
        method_key=state.get("chosen_method_key", "ab_test"),
        params=dict(state.get("setup_params") or {}),
        facts=dict(state.get("elicited_facts") or {}),
        power_results=dict(state.get("power_results") or {}),
    )

    msg = result["response"]
    if on_token:
        await on_token(msg + "\n\n")

    return {
        "mde_results": result["mde_results"],
        "setup_phase": result["setup_phase"],
        "messages": [AIMessage(content=msg)],
    }


async def synthetic_data_node(state: SetupState, config: RunnableConfig) -> dict:
    """Generate synthetic data matching the user's scenario."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = wf.generate_synthetic(
        method_key=state.get("chosen_method_key", "ab_test"),
        params=dict(state.get("setup_params") or {}),
        facts=dict(state.get("elicited_facts") or {}),
        power_results=dict(state.get("power_results") or {}),
        mde_results=dict(state.get("mde_results") or {}),
    )

    msg = result["response"]
    if on_token:
        await on_token(msg + "\n\n")

    return {
        "synthetic_data": result["synthetic_data"],
        "synthetic_data_csv": result["synthetic_data_csv"],
        "setup_phase": result["setup_phase"],
        "messages": [AIMessage(content=msg)],
    }


async def validation_node(state: SetupState, config: RunnableConfig) -> dict:
    """Validate the analysis scaffold against synthetic data."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = wf.run_validation(
        method_key=state.get("chosen_method_key", "ab_test"),
        synth=dict(state.get("synthetic_data") or {}),
    )

    msg = result["response"]
    if on_token:
        await on_token(msg + "\n\n")

    return {
        "validation_results": result["validation_results"],
        "setup_phase": result["setup_phase"],
        "messages": [AIMessage(content=msg)],
    }


async def review_results_node(state: SetupState, config: RunnableConfig) -> dict:
    """
    Present results to user and ask if they want to proceed or adjust.
    Two modes: first entry (after validation) or re-entry (user responded).
    """
    wf = _make_workflow()
    on_token = _get_on_token(config)
    messages = list(state.get("messages", []))

    # Determine if this is first entry or re-entry
    last_msg = messages[-1] if messages else None
    is_reentry = isinstance(last_msg, HumanMessage)

    if not is_reentry:
        # First entry: present results and ask
        result = await wf.review_results_first_entry(
            method_key=state.get("chosen_method_key", "ab_test"),
            method_name=state.get("chosen_method_name", ""),
            params=dict(state.get("setup_params") or {}),
            power_results=dict(state.get("power_results") or {}),
            mde_results=dict(state.get("mde_results") or {}),
            red_flags=list(state.get("red_flags") or []),
            on_token=on_token,
        )
    else:
        # Re-entry: analyze user response
        result = await wf.review_results_reentry(
            user_text=last_msg.content,
            params=dict(state.get("setup_params") or {}),
            power_results=dict(state.get("power_results") or {}),
            mde_results=dict(state.get("mde_results") or {}),
            red_flags=list(state.get("red_flags") or []),
            method_key=state.get("chosen_method_key", "ab_test"),
            on_token=on_token,
        )

    update: dict[str, Any] = {
        "setup_phase": result["setup_phase"],
        "messages": [AIMessage(content=result["response"])],
    }

    if "done" in result:
        update["done"] = result["done"]
    if "setup_params" in result:
        update["setup_params"] = result["setup_params"]

    return update


async def setup_output_node(state: SetupState, config: RunnableConfig) -> dict:
    """Generate the final setup report and mark the workflow done."""
    wf = _make_workflow()
    on_token = _get_on_token(config)

    result = await wf.generate_report(
        method_key=state.get("chosen_method_key", "ab_test"),
        method_name=state.get("chosen_method_name", ""),
        facts=dict(state.get("elicited_facts") or {}),
        params=dict(state.get("setup_params") or {}),
        power_results=dict(state.get("power_results") or {}),
        mde_results=dict(state.get("mde_results") or {}),
        synth=dict(state.get("synthetic_data") or {}),
        validation=dict(state.get("validation_results") or {}),
        red_flags=list(state.get("red_flags") or []),
        on_token=on_token,
    )

    return {
        "setup_report_markdown": result["setup_report_markdown"],
        "setup_phase": result["setup_phase"],
        "done": result["done"],
        "messages": [AIMessage(content=result["response"])],
        "red_flags": result["red_flags"],
    }


async def redesign_question_node(state: SetupState, config: RunnableConfig) -> dict:
    """Process the user's specific design change request and trigger re-computation."""
    wf = _make_workflow()
    on_token = _get_on_token(config)
    messages = list(state.get("messages", []))

    user_reply = _last_human_text(messages) or ""

    result = await wf.handle_redesign(
        user_reply=user_reply,
        params=dict(state.get("setup_params") or {}),
        on_token=on_token,
    )

    return {
        "setup_params": result["setup_params"],
        "setup_phase": result["setup_phase"],
        "messages": [AIMessage(content=result["response"])],
    }
