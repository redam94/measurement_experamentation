"""
LangGraph node implementations for the measurement design agent.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .state import AgentState, ELICITATION_TOPICS
from ..prompts.system import SYSTEM_PROMPT, WELCOME_MESSAGE
from ..prompts.questions import TOPIC_INDEX, EXTRACTION_SYSTEM
from ..scoring.scorer import score_methods, rank_methods, generate_explanations, build_ranked_report_data
from ..output.report import generate_report
from ..output.spec import generate_spec_json, generate_spec_yaml
from ..output.scaffold import generate_combined_scaffold


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_FOLLOWUP_ROUNDS = 2  # max follow-up attempts per topic before accepting unknowns


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm(model: str = "claude-opus-4-5") -> ChatAnthropic:
    return ChatAnthropic(model=model, temperature=0.2, max_tokens=2048)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


async def _extract_facts(
    user_reply: str,
    topic_meta: dict,
    llm: ChatAnthropic,
) -> dict:
    """Call LLM to extract structured facts from the user's reply for a given topic."""
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM),
        HumanMessage(
            content=(
                f"Extraction instruction:\n{topic_meta['extraction_prompt']}\n\n"
                f"User reply:\n{user_reply}"
            )
        ),
    ]
    response = await llm.ainvoke(messages)
    raw = _strip_json_fence(response.content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _check_extraction_sufficient(topic: str, extracted: dict, facts: dict) -> tuple[bool, str | None]:
    """
    Check whether extraction got the critical values for this topic.

    Returns (is_sufficient, missing_description_or_None).
    """
    if topic == "objective":
        kpi = facts.get("kpi") or extracted.get("kpi")
        obj = facts.get("primary_objective") or extracted.get("primary_objective")
        if kpi in (None, "unknown") and obj in (None, "other", "unknown"):
            return False, "the specific metric (KPI) they want to move"
        if kpi in (None, "unknown"):
            return False, "the specific metric (KPI) they want to move"
        return True, None

    if topic == "randomization":
        unit = facts.get("randomization_unit") or extracted.get("randomization_unit")
        if unit in (None, "unknown"):
            return False, "whether they can control who sees the ad"
        return True, None

    if topic == "data_history":
        weeks = facts.get("pre_period_weeks") or extracted.get("pre_period_weeks")
        has_data = facts.get("has_historical_data")
        if has_data is None:
            has_data = extracted.get("has_historical_data")
        if weeks is None and has_data:
            return False, "roughly how many weeks/months of historical data they have"
        return True, None

    if topic == "geo_structure":
        markets = facts.get("num_markets") or extracted.get("num_markets")
        if markets is None:
            return False, "how many markets/locations are involved"
        return True, None

    if topic == "scale":
        size = facts.get("sample_size_estimate") or extracted.get("sample_size_estimate")
        if size in (None, "unknown"):
            return False, "a rough sense of audience size"
        return True, None

    # treatment_control and covariates: always sufficient (no critical unknowns)
    return True, None


async def _ask_question(
    topic_meta: dict,
    conversation_history: list,
    facts_so_far: dict,
    llm: ChatAnthropic,
    *,
    is_followup: bool = False,
    missing_info: str | None = None,
) -> str:
    """
    Ask the next elicitation question, rephrased naturally given conversation history.

    When is_followup=True, the question is a gentle follow-up because the initial
    answer was incomplete. The missing_info describes what we still need.
    """
    context = "\n".join(
        [
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in conversation_history[-6:]  # last 3 turns
        ]
    )

    # Build adaptive instruction parts
    instruction_parts = []

    if is_followup and missing_info:
        # Use the topic's dedicated followup_question if available
        followup_q = topic_meta.get("followup_question")
        if followup_q:
            instruction_parts.append(
                f"This is a FOLLOW-UP. The user already answered, but we still need to "
                f"understand: {missing_info}.\n"
                f"Use this follow-up question as inspiration (rephrase naturally): "
                f"{followup_q}\n"
                f"Acknowledge what they already shared, then ask specifically for the "
                f"missing information. Offer concrete examples or options to choose from."
            )
        else:
            instruction_parts.append(
                f"This is a FOLLOW-UP. The user already answered, but we still need to "
                f"understand: {missing_info}.\n"
                f"Acknowledge what they already shared, then ask specifically for the "
                f"missing information. Offer concrete examples or options to choose from."
            )
    else:
        base_question = topic_meta["question"]
        instruction_parts.append(
            f"Base question: {base_question}"
        )

    # Add examples and clarification hints for the LLM to use
    examples = topic_meta.get("examples", [])
    hints = topic_meta.get("clarification_hints", [])
    why = topic_meta.get("why_it_matters", "")

    if examples:
        instruction_parts.append(
            f"Example answers the user might give (use these to frame your question "
            f"if helpful): {'; '.join(examples[:3])}"
        )
    if hints and is_followup:
        instruction_parts.append(
            f"If the user seems confused, try one of these angles: "
            f"{'; '.join(hints[:2])}"
        )
    if why:
        instruction_parts.append(
            f"Why this matters (use sparingly — only if the user asks why): {why}"
        )

    instruction_parts.append(
        "Rephrase naturally to fit the conversation flow. Keep it in plain English, "
        "ONE to THREE sentences at most. Use a friendly, encouraging tone. "
        "If offering examples, use bullet points. "
        "Do NOT answer the question — only ask it."
    )

    prompt = (
        f"The next topic to ask about is: {topic_meta['topic']}\n\n"
        f"Facts gathered so far: {json.dumps(facts_so_far, default=str)}\n\n"
        f"Recent conversation:\n{context}\n\n"
        + "\n\n".join(instruction_parts)
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = await llm.ainvoke(messages)
    return response.content.strip()


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def welcome_node(state: AgentState) -> dict:
    """Emit the welcome message and seed the state."""
    return {
        "messages": [AIMessage(content=WELCOME_MESSAGE)],
        "phase": "elicit",
        "covered_topics": [],
        "elicited_facts": {},
        "scores": {},
        "ranked_methods": [],
        "clarify_rounds": 0,
        "followup_round": 0,
        "report_markdown": "",
        "spec_json": {},
        "spec_yaml": "",
        "scaffold_code": "",
        "done": False,
        "pending_question": WELCOME_MESSAGE,
    }


async def question_node(state: AgentState) -> dict:
    """
    Process the user's latest reply:
      1. Extract structured facts for the current topic.
      2. Check if the extraction got all critical information.
      3. If critical info is missing and we haven't exceeded follow-up limit,
         ask a follow-up question (topic stays uncovered).
      4. If sufficient or max follow-ups reached, mark topic covered.
      5. If topics remain, ask the next question.
      6. If all topics done, run a completeness check and set phase='score'.
    """
    llm = _make_llm()
    covered = list(state.get("covered_topics", []))
    facts = dict(state.get("elicited_facts") or {})
    messages = list(state.get("messages", []))
    followup_round = state.get("followup_round", 0)

    # The first uncovered topic is the one the user is answering right now.
    remaining = [t for t in ELICITATION_TOPICS if t not in covered]

    if not remaining:
        # Edge case: all topics already covered → score
        return {
            "elicited_facts": facts,
            "covered_topics": covered,
            "phase": "score",
            "messages": [],
            "followup_round": 0,
        }

    current_topic = remaining[0]
    current_meta = TOPIC_INDEX[current_topic]

    # Extract facts from the latest human message
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    extracted = {}
    if last_human:
        extracted = await _extract_facts(last_human.content, current_meta, llm)
        # Merge into facts (skip null/None values)
        for k, v in extracted.items():
            if v is not None:
                facts[k] = v

    # Check if we got the critical information for this topic
    is_sufficient, missing_desc = _check_extraction_sufficient(
        current_topic, extracted, facts
    )

    if not is_sufficient and followup_round < MAX_FOLLOWUP_ROUNDS:
        # Ask a follow-up for the SAME topic (don't mark as covered)
        question_text = await _ask_question(
            current_meta, messages, facts, llm,
            is_followup=True,
            missing_info=missing_desc,
        )
        return {
            "elicited_facts": facts,
            "covered_topics": covered,  # NOT marked covered yet
            "messages": [AIMessage(content=question_text)],
            "pending_question": question_text,
            "phase": "elicit",
            "followup_round": followup_round + 1,
        }

    # Topic is complete (sufficient or max follow-ups reached)
    covered_now = covered + [current_topic]
    still_remaining = [t for t in ELICITATION_TOPICS if t not in covered_now]

    if still_remaining:
        # Ask the next question and reset follow-up counter
        next_meta = TOPIC_INDEX[still_remaining[0]]
        question_text = await _ask_question(next_meta, messages, facts, llm)

        return {
            "elicited_facts": facts,
            "covered_topics": covered_now,
            "messages": [AIMessage(content=question_text)],
            "pending_question": question_text,
            "phase": "elicit",
            "followup_round": 0,
        }

    # All topics covered → transition to scoring pipeline
    return {
        "elicited_facts": facts,
        "covered_topics": covered_now,
        "phase": "score",
        "messages": [],
        "followup_round": 0,
    }

async def score_node(state: AgentState) -> dict:
    """Score all six methods and transition to recommend phase."""
    llm = _make_llm()
    facts = dict(state.get("elicited_facts") or {})
    scores = score_methods(facts)
    ranked = rank_methods(scores)

    # Generate LLM explanations
    explanations = await generate_explanations(facts, scores, llm)

    # Notify user that scoring is happening
    scoring_msg = (
        "I now have everything I need. Let me score the six measurement approaches "
        "against your situation…"
    )

    return {
        "scores": scores,
        "ranked_methods": ranked,
        "phase": "recommend",
        "messages": [AIMessage(content=scoring_msg)],
    }


async def recommend_node(state: AgentState) -> dict:
    """Build ranked summary message from scores and prepare for output."""
    llm = _make_llm()
    facts = dict(state.get("elicited_facts") or {})
    scores = dict(state.get("scores") or {})
    ranked = list(state.get("ranked_methods") or [])

    # Re-generate explanations (or use a fresh call)
    explanations = await generate_explanations(facts, scores, llm)
    ranked_data = build_ranked_report_data(facts, scores, explanations)

    # Build a short chatbot summary message
    summary_lines = [
        "Here are my recommendations, ranked for your situation:\n",
    ]
    for item in ranked_data:
        bar = "█" * int(item["score"] / 20)  # 0–5 blocks
        summary_lines.append(
            f"**#{item['rank']} {item['name']}** — {item['score']}/100 {bar}\n"
            f"> {item['explanation']}\n"
        )
    summary_lines.append(
        "\nI'm now generating your full design report, structured spec, and analysis code. "
        "This will just take a moment…"
    )

    return {
        "phase": "output",
        "messages": [AIMessage(content="\n".join(summary_lines))],
    }


async def output_node(state: AgentState) -> dict:
    """Generate all three output artifacts and mark the session done."""
    llm = _make_llm()
    facts = dict(state.get("elicited_facts") or {})
    scores = dict(state.get("scores") or {})

    explanations = await generate_explanations(facts, scores, llm)
    ranked_data = build_ranked_report_data(facts, scores, explanations)

    report_md = generate_report(facts, ranked_data)
    spec_json = generate_spec_json(facts, ranked_data)
    spec_yaml = generate_spec_yaml(spec_json)
    scaffold = generate_combined_scaffold(ranked_data)

    done_msg = (
        "✅ **Your measurement design is ready!**\n\n"
        "You'll find three downloadable artefacts in the panel on the right:\n"
        "- 📄 **Full Markdown Report** — executive summary, method comparison, implementation checklist\n"
        "- 🗂 **Design Spec** (JSON + YAML) — machine-readable design for your analytics team\n"
        "- 🐍 **PyMC Scaffold** — working Bayesian analysis code for the top-ranked method\n\n"
        "If you'd like to revisit any assumptions or explore a different scenario, "
        "just start a new session."
    )

    return {
        "report_markdown": report_md,
        "spec_json": spec_json,
        "spec_yaml": spec_yaml,
        "scaffold_code": scaffold,
        "phase": "done",
        "done": True,
        "messages": [AIMessage(content=done_msg)],
    }
