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


async def _ask_question(
    topic_meta: dict,
    conversation_history: list,
    facts_so_far: dict,
    llm: ChatAnthropic,
) -> str:
    """Ask the next elicitation question, rephrased naturally given conversation history."""
    base_question = topic_meta["question"]
    context = "\n".join(
        [
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in conversation_history[-6:]  # last 3 turns
        ]
    )
    prompt = (
        f"The next topic to ask about is: {topic_meta['topic']}\n\n"
        f"Base question: {base_question}\n\n"
        f"Recent conversation:\n{context}\n\n"
        "Rephrase the base question naturally to fit the conversation. "
        "Keep it in plain English, ONE sentence or two at most. "
        "Do NOT answer the question — only ask it."
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
      2. Mark that topic as covered.
      3. If topics remain, ask the next question and return (→ END).
      4. If all topics are done, set phase='score' (→ score_node).

    This node is called once per user turn, never self-loops.
    """
    llm = _make_llm()
    covered = list(state.get("covered_topics", []))
    facts = dict(state.get("elicited_facts") or {})
    messages = list(state.get("messages", []))

    # The first uncovered topic is the one the user is answering right now.
    remaining = [t for t in ELICITATION_TOPICS if t not in covered]

    if not remaining:
        # Edge case: all topics already covered → score
        return {
            "elicited_facts": facts,
            "covered_topics": covered,
            "phase": "score",
            "messages": [],
        }

    current_topic = remaining[0]
    current_meta = TOPIC_INDEX[current_topic]

    # Extract facts from the latest human message
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human:
        extracted = await _extract_facts(last_human.content, current_meta, llm)
        facts.update(extracted)

    # Mark this topic as covered
    covered_now = covered + [current_topic]
    still_remaining = [t for t in ELICITATION_TOPICS if t not in covered_now]

    if still_remaining:
        # Ask the next question and return → END (wait for user)
        next_meta = TOPIC_INDEX[still_remaining[0]]
        question_text = await _ask_question(next_meta, messages, facts, llm)

        return {
            "elicited_facts": facts,
            "covered_topics": covered_now,
            "messages": [AIMessage(content=question_text)],
            "pending_question": question_text,
            "phase": "elicit",
        }

    # All topics covered → transition to scoring pipeline
    return {
        "elicited_facts": facts,
        "covered_topics": covered_now,
        "phase": "score",
        "messages": [],
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
