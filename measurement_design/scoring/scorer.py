"""
Scoring engine: score all methods against elicited facts and rank them.

Pure domain logic — no LLM dependencies. The LLM-based explanation
generation lives in the backend adapter layer.
"""
from __future__ import annotations

from typing import Any

from measurement_design.methods import ALL_METHODS, METHOD_MAP


def score_methods(facts: dict) -> dict[str, float]:
    """Run all method score() functions and return a score map."""
    return {m.key: m.score(facts) for m in ALL_METHODS}


def rank_methods(scores: dict[str, float]) -> list[str]:
    """Return method keys sorted by score descending."""
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


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
