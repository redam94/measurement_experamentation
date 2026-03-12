"""
JSON + YAML design spec generator.
"""
from __future__ import annotations

import json
from typing import Any

import yaml


def generate_spec_json(
    facts: dict,
    ranked_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a fully serialisable design specification dict."""
    spec_dict: dict[str, Any] = {
        "version": "1.0",
        "elicited_facts": facts,
        "ranked_methods": [],
    }
    for item in ranked_data:
        spec = item["spec"]
        spec_dict["ranked_methods"].append(
            {
                "rank": item["rank"],
                "key": item["key"],
                "name": item["name"],
                "score": item["score"],
                "short_description": item["short_description"],
                "explanation": item["explanation"],
                "design": spec.to_dict() if spec else {},
            }
        )
    return spec_dict


def generate_spec_yaml(spec_json: dict[str, Any]) -> str:
    """Serialise the spec dict to a YAML string."""
    return yaml.dump(spec_json, allow_unicode=True, sort_keys=False, width=100)
