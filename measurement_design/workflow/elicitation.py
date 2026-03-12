"""
Domain workflow service for the elicitation conversation.

Encapsulates all business logic for the elicitation phase:
fact extraction, sufficiency checks, question generation,
scoring, recommendations, and output artifact generation.

Uses the LLMService port for all LLM interactions — no
framework dependencies (no langchain, langgraph, etc.).
"""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..ports import LLMService
from ..types import ELICITATION_TOPICS
from ..prompts.system import SYSTEM_PROMPT, WELCOME_MESSAGE
from ..prompts.questions import TOPIC_INDEX, EXTRACTION_SYSTEM, SCORING_EXPLANATION_PROMPT
from ..scoring import score_methods, rank_methods, build_ranked_report_data
from ..output.report import generate_report
from ..output.spec import generate_spec_json, generate_spec_yaml
from ..output.scaffold import generate_combined_scaffold
from ..methods import ALL_METHODS


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_FOLLOWUP_ROUNDS = 2  # max follow-up attempts per topic


# ── Private helpers ──────────────────────────────────────────────────────────

def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _check_extraction_sufficient(
    topic: str, extracted: dict, facts: dict,
) -> tuple[bool, str | None]:
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

    # treatment_control and covariates: always sufficient
    return True, None


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    """Convert conversation history dicts to a context string."""
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in conversation[-6:]
    )


# ── Workflow class ───────────────────────────────────────────────────────────

class ElicitationWorkflow:
    """Domain service for the elicitation conversation workflow."""

    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    # ── Welcome ──────────────────────────────────────────────────────────

    def get_welcome(self) -> dict[str, Any]:
        """Return welcome state + message."""
        return {
            "response": WELCOME_MESSAGE,
            "phase": "elicit",
            "facts": {},
            "covered_topics": [],
            "scores": {},
            "ranked_methods": [],
            "followup_round": 0,
            "done": False,
        }

    # ── Question turn ────────────────────────────────────────────────────

    async def handle_question_turn(
        self,
        user_reply: str | None,
        facts: dict[str, Any],
        covered_topics: list[str],
        followup_round: int,
        conversation: list[dict[str, str]],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """
        Process user reply → extract facts → check sufficiency →
        ask follow-up or next question → return state updates.
        """
        remaining = [t for t in ELICITATION_TOPICS if t not in covered_topics]

        if not remaining:
            return {
                "response": None,
                "phase": "score",
                "facts": facts,
                "covered_topics": covered_topics,
                "followup_round": 0,
            }

        current_topic = remaining[0]
        current_meta = TOPIC_INDEX[current_topic]

        # Extract facts from user reply
        extracted: dict = {}
        if user_reply:
            extracted = await self._extract_facts(user_reply, current_meta)
            for k, v in extracted.items():
                if v is not None:
                    facts[k] = v

        # Check sufficiency
        is_sufficient, missing_desc = _check_extraction_sufficient(
            current_topic, extracted, facts,
        )

        if not is_sufficient and followup_round < MAX_FOLLOWUP_ROUNDS:
            question_text = await self._ask_question(
                current_meta, conversation, facts,
                is_followup=True, missing_info=missing_desc,
                on_token=on_token,
            )
            return {
                "response": question_text,
                "phase": "elicit",
                "facts": facts,
                "covered_topics": covered_topics,
                "followup_round": followup_round + 1,
            }

        # Topic complete
        covered_now = covered_topics + [current_topic]
        still_remaining = [t for t in ELICITATION_TOPICS if t not in covered_now]

        if still_remaining:
            next_meta = TOPIC_INDEX[still_remaining[0]]
            question_text = await self._ask_question(
                next_meta, conversation, facts,
                on_token=on_token,
            )
            return {
                "response": question_text,
                "phase": "elicit",
                "facts": facts,
                "covered_topics": covered_now,
                "followup_round": 0,
            }

        # All topics covered → transition to scoring
        return {
            "response": None,
            "phase": "score",
            "facts": facts,
            "covered_topics": covered_now,
            "followup_round": 0,
        }

    # ── Score & rank ─────────────────────────────────────────────────────

    async def score_and_rank(
        self,
        facts: dict[str, Any],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Score all methods, generate explanations."""
        scores = score_methods(facts)
        ranked = rank_methods(scores)
        explanations = await self._generate_explanations(facts, scores)

        scoring_msg = (
            "I now have everything I need. Let me score the six measurement approaches "
            "against your situation\u2026"
        )
        if on_token:
            await on_token(scoring_msg)

        return {
            "response": scoring_msg,
            "phase": "recommend",
            "scores": scores,
            "ranked_methods": ranked,
            "explanations": explanations,
        }

    # ── Recommendations ──────────────────────────────────────────────────

    async def build_recommendations(
        self,
        facts: dict[str, Any],
        scores: dict[str, float],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Build ranked summary with progress bars."""
        explanations = await self._generate_explanations(facts, scores)
        ranked_data = build_ranked_report_data(facts, scores, explanations)

        summary_lines = [
            "Here are my recommendations, ranked for your situation:\n",
        ]
        for item in ranked_data:
            bar = "\u2588" * int(item["score"] / 20)
            summary_lines.append(
                f"**#{item['rank']} {item['name']}** \u2014 {item['score']}/100 {bar}\n"
                f"> {item['explanation']}\n"
            )
        summary_lines.append(
            "\nI'm now generating your full design report, structured spec, and analysis code. "
            "This will just take a moment\u2026"
        )

        summary_text = "\n".join(summary_lines)
        if on_token:
            await on_token(summary_text)

        return {
            "response": summary_text,
            "phase": "output",
        }

    # ── Output artifacts ─────────────────────────────────────────────────

    async def generate_outputs(
        self,
        facts: dict[str, Any],
        scores: dict[str, float],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Generate report, spec, scaffold and return all artifacts."""
        explanations = await self._generate_explanations(facts, scores)
        ranked_data = build_ranked_report_data(facts, scores, explanations)

        report_md = generate_report(facts, ranked_data)
        spec_json = generate_spec_json(facts, ranked_data)
        spec_yaml = generate_spec_yaml(spec_json)
        scaffold = generate_combined_scaffold(ranked_data)

        done_msg = (
            "\u2705 **Your measurement design is ready!**\n\n"
            "You'll find three downloadable artefacts in the panel on the right:\n"
            "- \U0001f4c4 **Full Markdown Report** \u2014 executive summary, method comparison, implementation checklist\n"
            "- \U0001f5c2 **Design Spec** (JSON + YAML) \u2014 machine-readable design for your analytics team\n"
            "- \U0001f40d **PyMC Scaffold** \u2014 working Bayesian analysis code for the top-ranked method\n\n"
            "If you'd like to revisit any assumptions or explore a different scenario, "
            "just start a new session."
        )

        if on_token:
            await on_token(done_msg)

        return {
            "response": done_msg,
            "phase": "done",
            "done": True,
            "report_markdown": report_md,
            "spec_json": spec_json,
            "spec_yaml": spec_yaml,
            "scaffold_code": scaffold,
        }

    # ── Private LLM helpers ──────────────────────────────────────────────

    async def _extract_facts(self, user_reply: str, topic_meta: dict) -> dict:
        """Call LLM to extract structured facts from the user's reply."""
        user_prompt = (
            f"Extraction instruction:\n{topic_meta['extraction_prompt']}\n\n"
            f"User reply:\n{user_reply}"
        )
        try:
            return await self.llm.generate_json(EXTRACTION_SYSTEM, user_prompt)
        except (json.JSONDecodeError, Exception):
            return {}

    async def _ask_question(
        self,
        topic_meta: dict,
        conversation: list[dict[str, str]],
        facts_so_far: dict,
        *,
        is_followup: bool = False,
        missing_info: str | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Generate a conversational elicitation question."""
        context = _format_conversation(conversation)

        instruction_parts: list[str] = []

        if is_followup and missing_info:
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
            instruction_parts.append(f"Base question: {base_question}")

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
                f"Why this matters (use sparingly \u2014 only if the user asks why): {why}"
            )

        instruction_parts.append(
            "Rephrase naturally to fit the conversation flow. Keep it in plain English, "
            "ONE to THREE sentences at most. Use a friendly, encouraging tone. "
            "If offering examples, use bullet points. "
            "Do NOT answer the question \u2014 only ask it."
        )

        user_prompt = (
            f"The next topic to ask about is: {topic_meta['topic']}\n\n"
            f"Facts gathered so far: {json.dumps(facts_so_far, default=str)}\n\n"
            f"Recent conversation:\n{context}\n\n"
            + "\n\n".join(instruction_parts)
        )

        if on_token is not None:
            chunks: list[str] = []
            async for token in self.llm.stream_text(SYSTEM_PROMPT, user_prompt):
                await on_token(token)
                chunks.append(token)
            return "".join(chunks).strip()

        return await self.llm.generate_text(SYSTEM_PROMPT, user_prompt)

    async def _generate_explanations(
        self, facts: dict, scores: dict[str, float],
    ) -> dict[str, str]:
        """Use LLM to generate plain-English explanations for each method score."""
        prompt = SCORING_EXPLANATION_PROMPT.format(
            facts_json=json.dumps(facts, indent=2, default=str),
            scores_json=json.dumps(scores, indent=2),
        )
        system = (
            "You are an expert marketing measurement scientist. "
            "Return ONLY valid JSON \u2014 a single object mapping method keys to explanation strings. "
            "No markdown fences, no preamble."
        )
        try:
            return await self.llm.generate_json(system, prompt)
        except (json.JSONDecodeError, Exception):
            return {m.key: "" for m in ALL_METHODS}
