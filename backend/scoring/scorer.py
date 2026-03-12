"""
Scoring engine: score all methods against elicited facts,
then use Claude to generate prose explanations.

Pure scoring functions are delegated to measurement_design.scoring.
This module adds the LLM-dependent generate_explanations() function.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from measurement_design.methods import ALL_METHODS, METHOD_MAP
from measurement_design.scoring import score_methods, rank_methods
from ..prompts.questions import SCORING_EXPLANATION_PROMPT


async def generate_explanations(
    facts: dict,
    scores: dict[str, float],
    llm: ChatAnthropic,
) -> dict[str, str]:
    """Use Claude to generate plain-English explanations for each method score."""
    prompt = SCORING_EXPLANATION_PROMPT.format(
        facts_json=json.dumps(facts, indent=2, default=str),
        scores_json=json.dumps(scores, indent=2),
    )
    messages = [
        SystemMessage(content=(
            "You are an expert marketing measurement scientist. "
            "Return ONLY valid JSON — a single object mapping method keys to explanation strings. "
            "No markdown fences, no preamble."
        )),
        HumanMessage(content=prompt),
    ]
    response = await llm.ainvoke(messages)
    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback to empty explanations
        return {m.key: "" for m in ALL_METHODS}


# Re-export build_ranked_report_data from core for backward compatibility
from measurement_design.scoring import build_ranked_report_data  # noqa: F811, E402
