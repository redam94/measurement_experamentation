"""
Scoring engine: score all methods against elicited facts,
then use Claude to generate prose explanations.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..methods import ALL_METHODS, METHOD_MAP
from ..prompts.questions import SCORING_EXPLANATION_PROMPT


def score_methods(facts: dict) -> dict[str, float]:
    """Run all method score() functions and return a score map."""
    return {m.key: m.score(facts) for m in ALL_METHODS}


def rank_methods(scores: dict[str, float]) -> list[str]:
    """Return method keys sorted by score descending."""
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


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


def build_ranked_report_data(
    facts: dict,
    scores: dict[str, float],
    explanations: dict[str, str],
) -> list[dict[str, Any]]:
    """Build a list of method data dicts ordered by score (descending)."""
    ranked_keys = rank_methods(scores)
    result = []
    for rank, key in enumerate(ranked_keys, start=1):
        method = METHOD_MAP[key]
        spec = method.generate_spec(facts, explanation=explanations.get(key, ""))
        scaffold = method.generate_scaffold(facts)
        result.append(
            {
                "rank": rank,
                "key": key,
                "name": method.name,
                "score": round(scores[key], 1),
                "short_description": method.short_description,
                "explanation": explanations.get(key, ""),
                "spec": spec,
                "scaffold": scaffold,
            }
        )
    return result
