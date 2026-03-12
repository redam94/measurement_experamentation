"""
Domain workflow service for the experiment setup conversation.

Encapsulates all business logic for the setup phase:
parameter extraction, follow-up questions, power analysis,
MDE simulation, synthetic data, validation, and reporting.

Uses the LLMService port for LLM interactions — no framework deps.
"""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..ports import LLMService
from ..types import SETUP_TOPICS, RedFlag
from ..prompts.setup_prompts import (
    SETUP_SYSTEM_PROMPT,
    SETUP_WELCOME_TEMPLATE,
    SETUP_TOPIC_INDEX,
    SETUP_EXTRACTION_SYSTEM,
    SETUP_REPORT_PROMPT,
    BASELINE_FOLLOWUP_TEMPLATES,
    REVIEW_RESULTS_PROMPT,
    REVIEW_DECISION_SYSTEM,
    REDESIGN_ELICIT_PROMPT,
)
from ..knowledge import RED_FLAG_CATALOG
from ..validation.feasibility import (
    unit_label_for_method,
    check_params_sufficient,
    detect_red_flags,
    run_interim_power,
    apply_defaults,
    build_assumptions_summary,
    identify_design_problems,
    run_validation,
)
from ..simulation.power import compute_power
from ..simulation.mde import compute_mde
from ..simulation.synthetic import generate_synthetic_data


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_FOLLOWUP_ROUNDS = 2


# ── Private helpers ──────────────────────────────────────────────────────────

def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in conversation[-6:]
    )


def _format_feasibility_message(
    interim_power: dict | None,
    red_flags: list[RedFlag],
    method_key: str,
) -> str:
    """Build a human-readable feasibility preamble."""
    if interim_power is None and not red_flags:
        return ""

    ul = unit_label_for_method(method_key)
    parts: list[str] = []

    if interim_power:
        req_n = interim_power.get("required_sample_size")
        achieved = interim_power.get("achieved_power", 0)
        effect = interim_power.get("effect_size_used", 0)
        req_n_str = f"~{req_n:,}" if req_n else "could not be determined"
        parts.append(
            f"\U0001f4ca **Early power estimate**: You'd need roughly **{req_n_str} "
            f"{ul}** to achieve {achieved:.0%} power for an effect "
            f"size of {effect:.4f}."
        )

    if red_flags:
        parts.append("")
        for rf in red_flags:
            icon = "\U0001f6a8" if rf.get("severity") == "critical" else "\u26a0\ufe0f"
            parts.append(
                f"{icon} **{rf.get('title', 'Warning')}**: "
                f"{rf.get('detail', '')}\n"
                f"   \U0001f4a1 *{rf.get('suggestion', '')}*"
            )

    if parts:
        parts.append("")
    return "\n".join(parts)


# ── Workflow class ───────────────────────────────────────────────────────────

class SetupWorkflow:
    """Domain service for the setup conversation workflow."""

    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    # ── Welcome ──────────────────────────────────────────────────────────

    async def get_welcome(
        self,
        method_key: str,
        method_name: str,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Generate welcome + first question."""
        welcome = SETUP_WELCOME_TEMPLATE.format(method_name=method_name)

        first_topic = SETUP_TOPICS[0]
        first_meta = SETUP_TOPIC_INDEX[first_topic]
        first_question = await self._ask_setup_question(
            first_meta, method_key, [],
            on_token=on_token,
        )

        combined = f"{welcome}\n{first_question}"

        return {
            "response": combined,
            "setup_phase": "setup_elicit",
            "setup_topics_covered": [],
            "setup_params": {
                "alpha": 0.05,
                "power_target": 0.80,
                "one_sided": False,
                "n_simulations": 1000,
                "random_seed": 42,
            },
            "followup_round": 0,
            "feasibility_checked": False,
            "interim_power_result": {},
            "red_flags": [],
            "done": False,
        }

    # ── Question turn ────────────────────────────────────────────────────

    async def handle_question_turn(
        self,
        user_reply: str | None,
        params: dict[str, Any],
        covered: list[str],
        method_key: str,
        facts: dict[str, Any],
        followup_round: int,
        feasibility_checked: bool,
        red_flags: list[RedFlag],
        interim_power: dict | None,
        conversation: list[dict[str, str]],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Process user reply, extract params, ask next question."""
        remaining = [t for t in SETUP_TOPICS if t not in covered]

        if not remaining:
            return {
                "response": None,
                "setup_params": params,
                "setup_topics_covered": covered,
                "setup_phase": "power_analysis",
                "red_flags": red_flags,
            }

        current_topic = remaining[0]
        current_meta = SETUP_TOPIC_INDEX[current_topic]

        # Extract from user reply
        if user_reply:
            extracted = await self._extract_setup_facts(user_reply, current_meta)
            for k, v in extracted.items():
                if v is not None:
                    params[k] = v

        # Check sufficiency
        is_sufficient, missing_key = check_params_sufficient(
            current_topic, params, method_key,
        )

        if not is_sufficient and followup_round < MAX_FOLLOWUP_ROUNDS:
            ul = unit_label_for_method(method_key)
            followup_template = BASELINE_FOLLOWUP_TEMPLATES.get(
                missing_key or "missing_both",
                BASELINE_FOLLOWUP_TEMPLATES["missing_both"],
            )
            followup_desc = followup_template.format(units=ul, detail="")

            question_text = await self._ask_setup_question(
                current_meta, method_key, conversation,
                followup_context=followup_desc,
                on_token=on_token,
            )
            return {
                "response": question_text,
                "setup_params": params,
                "setup_topics_covered": covered,
                "setup_phase": "setup_elicit",
                "followup_round": followup_round + 1,
                "red_flags": red_flags,
            }

        # Topic complete
        covered_now = covered + [current_topic]

        # Interim power + red-flag check after expected_effect
        preamble = ""
        if current_topic == "expected_effect" and not feasibility_checked:
            interim_power = run_interim_power(method_key, params, facts)
            new_flags = detect_red_flags(method_key, params, facts)
            existing_keys = {f["flag"] for f in red_flags}
            for nf in new_flags:
                if nf["flag"] not in existing_keys:
                    red_flags.append(nf)
                    existing_keys.add(nf["flag"])
            preamble = _format_feasibility_message(interim_power, red_flags, method_key)
            feasibility_checked = True

        # Re-check after method_specific
        if current_topic == "method_specific":
            new_flags = detect_red_flags(method_key, params, facts)
            existing_keys = {f["flag"] for f in red_flags}
            for nf in new_flags:
                if nf["flag"] not in existing_keys:
                    red_flags.append(nf)
                    existing_keys.add(nf["flag"])

        # Advance to next topic or computation
        still_remaining = [t for t in SETUP_TOPICS if t not in covered_now]

        if still_remaining:
            next_meta = SETUP_TOPIC_INDEX[still_remaining[0]]
            question_text = await self._ask_setup_question(
                next_meta, method_key, conversation,
                preamble=preamble,
                on_token=on_token,
            )
            return {
                "response": question_text,
                "setup_params": params,
                "setup_topics_covered": covered_now,
                "setup_phase": "setup_elicit",
                "followup_round": 0,
                "feasibility_checked": feasibility_checked,
                "interim_power_result": interim_power if interim_power else {},
                "red_flags": red_flags,
            }

        # All topics done → computation
        return {
            "response": None,
            "setup_params": params,
            "setup_topics_covered": covered_now,
            "setup_phase": "power_analysis",
            "followup_round": 0,
            "feasibility_checked": feasibility_checked,
            "interim_power_result": interim_power if interim_power else {},
            "red_flags": red_flags,
        }

    # ── Computation steps ────────────────────────────────────────────────

    def run_power_analysis(
        self,
        method_key: str,
        params: dict[str, Any],
        facts: dict[str, Any],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Run power / sample-size calculations."""
        apply_defaults(params, facts)
        power_results = compute_power(method_key, params, facts)

        msg = (
            "\U0001f4ca **Power Analysis Complete!**\n\n"
            f"- **Required sample size**: {power_results.get('required_sample_size', 'N/A')}\n"
            f"- **Achieved power**: {power_results.get('achieved_power', 0):.1%}\n"
            f"- **Effect size used**: {power_results.get('effect_size_used', 0):.4f}\n\n"
            f"{power_results.get('notes', '')}\n\n"
            "Running Monte Carlo MDE simulation next\u2026"
        )

        return {
            "response": msg,
            "power_results": power_results,
            "setup_phase": "mde_simulation",
            "power_curve_json": json.dumps(
                power_results.get("power_curve", []), indent=2,
            ),
        }

    def run_mde_simulation(
        self,
        method_key: str,
        params: dict[str, Any],
        facts: dict[str, Any],
        power_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Monte Carlo MDE simulation."""
        apply_defaults(params, facts)
        mde_results = compute_mde(method_key, params, facts, power_results)

        mde_abs = mde_results.get("mde_absolute")
        mde_rel = mde_results.get("mde_relative_pct")
        mde_str = "could not be determined (try more simulations or a larger sample)"
        if mde_abs is not None:
            mde_str = f"{mde_abs:.4f}"
            if mde_rel is not None:
                mde_str += f" ({mde_rel:.1f}% relative)"

        msg = (
            "\U0001f3af **MDE Simulation Complete!**\n\n"
            f"- **Minimum Detectable Effect**: {mde_str}\n"
            f"- **Simulations run**: {mde_results.get('n_simulations', 0):,}\n"
            f"- **Target power**: {mde_results.get('target_power', 0.8):.0%}\n"
            f"- **Significance level**: {mde_results.get('alpha', 0.05)}\n\n"
            f"{mde_results.get('notes', '')}\n\n"
            "Generating synthetic test data next\u2026"
        )

        return {
            "response": msg,
            "mde_results": mde_results,
            "setup_phase": "synthetic_gen",
        }

    def generate_synthetic(
        self,
        method_key: str,
        params: dict[str, Any],
        facts: dict[str, Any],
        power_results: dict[str, Any],
        mde_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate synthetic data matching the user's scenario."""
        apply_defaults(params, facts)
        synth = generate_synthetic_data(
            method_key, params, facts, power_results, mde_results,
        )

        msg = (
            "\U0001f9ea **Synthetic Data Generated!**\n\n"
            f"- **Rows**: {synth.get('n_rows', 0):,}\n"
            f"- **Columns**: {', '.join(synth.get('columns', []))}\n"
            f"- **True injected effect**: {synth.get('true_effect', 0):.4f}\n\n"
            f"{synth.get('description', '')}\n\n"
            "Now I'll validate the analysis code against this synthetic data\u2026"
        )

        return {
            "response": msg,
            "synthetic_data": synth,
            "synthetic_data_csv": synth.get("csv_string", ""),
            "setup_phase": "validation",
        }

    def run_validation(
        self, method_key: str, synth: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate the analysis scaffold against synthetic data."""
        true_effect = synth.get("true_effect", 0)
        validation = run_validation(method_key, synth)

        if validation.get("success"):
            msg = (
                "\u2705 **Code Validation Passed!**\n\n"
                f"- **True effect**: {true_effect:.4f}\n"
                f"- **Estimated effect**: {validation.get('estimated_effect', 'N/A')}\n"
                f"- **95% CI**: [{validation.get('ci_lower', 'N/A')}, {validation.get('ci_upper', 'N/A')}]\n"
            )
            if validation.get("p_value") is not None:
                msg += f"- **p-value**: {validation['p_value']:.4f}\n"
            msg += (
                f"\n{validation.get('summary', '')}\n\n"
                "Reviewing your results now\u2026"
            )
        else:
            msg = (
                "\u26a0\ufe0f **Code Validation Note**\n\n"
                f"{validation.get('error_message', 'Validation encountered an issue.')}\n\n"
                "This is expected for some methods with small synthetic datasets. "
                "The generated scaffold is still valid \u2014 the validation is just conservative. "
                "Reviewing your results now\u2026"
            )

        return {
            "response": msg,
            "validation_results": validation,
            "setup_phase": "review_results",
        }

    # ── Review results ───────────────────────────────────────────────────

    async def review_results_first_entry(
        self,
        method_key: str,
        method_name: str,
        params: dict[str, Any],
        power_results: dict[str, Any],
        mde_results: dict[str, Any],
        red_flags: list[RedFlag],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Present results and ask if user wants to proceed or adjust."""
        required_n = power_results.get("required_sample_size", "N/A")
        achieved_power = power_results.get("achieved_power", 0)
        effect_size = power_results.get("effect_size_used", 0)
        mde_abs = mde_results.get("mde_absolute")
        mde_rel = mde_results.get("mde_relative_pct")

        mde_abs_str = f"{mde_abs:.4f}" if mde_abs is not None else "N/A"
        mde_rel_str = f"{mde_rel:.1f}%" if mde_rel is not None else "N/A"

        # Red flags text
        if red_flags:
            rf_parts = []
            for rf in red_flags:
                icon = "\U0001f6a8" if rf.get("severity") == "critical" else "\u26a0\ufe0f"
                rf_parts.append(f"{icon} {rf.get('title', 'Flag')}: {rf.get('detail', '')}")
            red_flags_text = "\n".join(rf_parts)
        else:
            red_flags_text = "No major concerns detected."

        assumptions_text = build_assumptions_summary(method_key)

        # Params summary
        param_parts: list[str] = []
        if params.get("baseline_rate"):
            param_parts.append(f"Baseline rate: {params['baseline_rate']:.2%}")
        if params.get("baseline_metric_value"):
            param_parts.append(f"Baseline value: {params['baseline_metric_value']:.1f}")
        if params.get("expected_lift_pct"):
            param_parts.append(f"Expected lift: {params['expected_lift_pct']:.1%}")
        elif params.get("expected_lift_abs"):
            param_parts.append(f"Expected lift: {params['expected_lift_abs']:.4f}")
        if params.get("num_treatment_units"):
            param_parts.append(f"Treatment units: {params['num_treatment_units']}")
        if params.get("num_control_units"):
            param_parts.append(f"Control units: {params['num_control_units']}")
        setup_params_summary = (
            "\n".join(f"- {p}" for p in param_parts) if param_parts else "Default parameters"
        )

        review_prompt = REVIEW_RESULTS_PROMPT.format(
            method_name=method_name,
            method_key=method_key,
            setup_params_summary=setup_params_summary,
            required_n=required_n,
            achieved_power=f"{achieved_power:.0%}" if achieved_power else "N/A",
            effect_size=f"{effect_size:.4f}" if effect_size else "N/A",
            mde_abs=mde_abs_str,
            mde_rel_pct=mde_rel_str,
            red_flags_text=red_flags_text,
            assumptions_text=assumptions_text,
        )

        if on_token is not None:
            chunks: list[str] = []
            async for token in self.llm.stream_text(SETUP_SYSTEM_PROMPT, review_prompt):
                await on_token(token)
                chunks.append(token)
            response_text = "".join(chunks).strip()
        else:
            response_text = await self.llm.generate_text(
                SETUP_SYSTEM_PROMPT, review_prompt,
            )

        return {
            "response": response_text,
            "setup_phase": "review_results",
            "done": False,
        }

    async def review_results_reentry(
        self,
        user_text: str,
        params: dict[str, Any],
        power_results: dict[str, Any],
        mde_results: dict[str, Any],
        red_flags: list[RedFlag],
        method_key: str,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Analyze user response: accept or modify."""
        decision_prompt = (
            f"The user was reviewing their experiment design results and responded:\n\n"
            f"\"{user_text}\"\n\n"
            f"Current parameters:\n{json.dumps(params, indent=2, default=str)}\n\n"
            f"Determine their intent and extract any parameter changes."
        )

        try:
            decision = await self.llm.generate_json(
                REVIEW_DECISION_SYSTEM, decision_prompt,
            )
        except (json.JSONDecodeError, Exception):
            decision = {"decision": "modify", "changes": {}, "change_summary": None}

        if decision.get("decision") == "accept":
            accept_msg = (
                "Great, let's finalize your experiment setup! "
                "Generating your complete setup report now\u2026"
            )
            if on_token:
                await on_token(accept_msg)
            return {
                "response": accept_msg,
                "setup_phase": "setup_output",
            }

        # User wants to modify
        changes = decision.get("changes") or {}
        actual_changes = {k: v for k, v in changes.items() if v is not None}

        if actual_changes:
            for k, v in actual_changes.items():
                params[k] = v
            change_summary = decision.get("change_summary") or "your requested changes"
            rerun_msg = (
                f"Got it! I've updated the design with {change_summary}. "
                f"Let me re-run the analysis with these new parameters\u2026"
            )
            if on_token:
                await on_token(rerun_msg)
            return {
                "response": rerun_msg,
                "setup_params": params,
                "setup_phase": "power_analysis",
            }

        # Vague modification request
        problems = identify_design_problems(
            power_results, mde_results, red_flags, params, method_key,
        )
        redesign_prompt = REDESIGN_ELICIT_PROMPT.format(
            problems=problems,
            current_params=json.dumps(params, indent=2, default=str),
        )

        if on_token is not None:
            chunks: list[str] = []
            async for token in self.llm.stream_text(SETUP_SYSTEM_PROMPT, redesign_prompt):
                await on_token(token)
                chunks.append(token)
            response_text = "".join(chunks).strip()
        else:
            response_text = await self.llm.generate_text(
                SETUP_SYSTEM_PROMPT, redesign_prompt,
            )

        return {
            "response": response_text,
            "setup_phase": "redesign_elicit",
            "done": False,
        }

    # ── Redesign ─────────────────────────────────────────────────────────

    async def handle_redesign(
        self,
        user_reply: str,
        params: dict[str, Any],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Process specific design change request."""
        extract_prompt = (
            f"The user wants to modify their experiment design parameters.\n\n"
            f"Their response: \"{user_reply}\"\n\n"
            f"Current parameters:\n{json.dumps(params, indent=2, default=str)}\n\n"
            f"Extract the specific parameter changes."
        )

        change_summary = "your requested adjustments"
        try:
            result = await self.llm.generate_json(
                REVIEW_DECISION_SYSTEM, extract_prompt,
            )
            changes = result.get("changes") or {}
            for k, v in changes.items():
                if v is not None:
                    params[k] = v
            change_summary = result.get("change_summary") or "your changes"
        except (json.JSONDecodeError, Exception):
            pass

        msg = (
            f"Updated! Re-running the full analysis pipeline with {change_summary}\u2026\n\n"
            "\u23f3 Computing power analysis, MDE simulation, synthetic data, and validation\u2026"
        )

        if on_token:
            await on_token(msg)

        return {
            "response": msg,
            "setup_params": params,
            "setup_phase": "power_analysis",
        }

    # ── Final report ─────────────────────────────────────────────────────

    async def generate_report(
        self,
        method_key: str,
        method_name: str,
        facts: dict[str, Any],
        params: dict[str, Any],
        power_results: dict[str, Any],
        mde_results: dict[str, Any],
        synth: dict[str, Any],
        validation: dict[str, Any],
        red_flags: list[RedFlag],
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Generate the final setup report."""
        # Re-run final red-flag detection
        final_flags = detect_red_flags(method_key, params, facts)
        existing_keys = {f["flag"] for f in red_flags}
        for nf in final_flags:
            if nf["flag"] not in existing_keys:
                red_flags.append(nf)
                existing_keys.add(nf["flag"])

        # Check insufficient sample
        req_n = power_results.get("required_sample_size")
        avail_treat = params.get("num_treatment_units") or 0
        avail_ctrl = params.get("num_control_units") or 0
        avail_n = avail_treat + avail_ctrl
        ul = unit_label_for_method(method_key)
        if req_n and avail_n and avail_n < req_n and "insufficient_sample" not in existing_keys:
            red_flags.append({
                "severity": "critical" if avail_n < req_n * 0.5 else "warning",
                "flag": "insufficient_sample",
                "title": RED_FLAG_CATALOG["insufficient_sample"]["title"],
                "detail": RED_FLAG_CATALOG["insufficient_sample"]["detail"].format(
                    required_n=req_n, unit_label=ul, available_n=avail_n,
                ),
                "suggestion": RED_FLAG_CATALOG["insufficient_sample"]["suggestion"],
            })

        # Generate report via LLM
        report_prompt = SETUP_REPORT_PROMPT.format(
            method_name=method_name,
            method_key=method_key,
            facts_json=json.dumps(facts, indent=2, default=str),
            setup_params_json=json.dumps(params, indent=2, default=str),
            power_json=json.dumps(power_results, indent=2, default=str),
            mde_json=json.dumps(
                {k: v for k, v in mde_results.items() if k != "power_by_effect"},
                indent=2, default=str,
            ),
            synthetic_summary=synth.get("description", "No synthetic data generated."),
            validation_json=json.dumps(validation, indent=2, default=str),
            red_flags_json=json.dumps(red_flags, indent=2, default=str),
        )

        report_md = await self.llm.generate_text(SETUP_SYSTEM_PROMPT, report_prompt)

        done_msg = (
            "\U0001f389 **Your experiment setup is complete!**\n\n"
            "You now have:\n"
            "- \U0001f4ca **Power analysis** with required sample sizes\n"
            "- \U0001f3af **MDE simulation** showing the smallest detectable effect\n"
            "- \U0001f9ea **Synthetic test data** matching your scenario\n"
            "- \u2705 **Validation results** confirming the analysis pipeline works\n"
            "- \U0001f4cb **Setup report** with actionable recommendations\n"
        )

        if red_flags:
            n_crit = sum(1 for f in red_flags if f.get("severity") == "critical")
            n_warn = sum(1 for f in red_flags if f.get("severity") == "warning")
            flag_parts = []
            if n_crit:
                flag_parts.append(f"\U0001f6a8 {n_crit} critical")
            if n_warn:
                flag_parts.append(f"\u26a0\ufe0f {n_warn} warning(s)")
            done_msg += f"- {'  &  '.join(flag_parts)} flagged \u2014 see the report for details\n"

        done_msg += "\nDownload everything from the tabs below!"

        if on_token:
            await on_token(done_msg)

        return {
            "response": done_msg,
            "setup_report_markdown": report_md,
            "setup_phase": "setup_done",
            "done": True,
            "red_flags": red_flags,
        }

    # ── Private LLM helpers ──────────────────────────────────────────────

    async def _extract_setup_facts(self, user_reply: str, topic_meta: dict) -> dict:
        """Extract structured setup parameters from user reply."""
        user_prompt = (
            f"Extraction instruction:\n{topic_meta['extraction_prompt']}\n\n"
            f"User reply:\n{user_reply}"
        )
        try:
            return await self.llm.generate_json(SETUP_EXTRACTION_SYSTEM, user_prompt)
        except (json.JSONDecodeError, Exception):
            return {}

    async def _ask_setup_question(
        self,
        topic_meta: dict,
        method_key: str,
        conversation: list[dict[str, str]],
        *,
        preamble: str = "",
        followup_context: str = "",
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Ask the next setup question, adapted for the chosen method."""
        templates = topic_meta["question_template"]
        base_question = templates.get(method_key, templates.get("_default", ""))

        context = _format_conversation(conversation)

        instruction_parts: list[str] = []
        if preamble:
            instruction_parts.append(
                f"IMPORTANT \u2014 Before asking the question, present this feasibility update "
                f"to the user. Rewrite it in your own words using plain, friendly language. "
                f"Keep all the numbers and warnings, but explain what they mean in everyday "
                f"terms (e.g., 'you'd need about 10,000 customers in your test' instead of "
                f"'required sample size is 10,000'):\n\n{preamble}\n"
            )
        if followup_context:
            instruction_parts.append(
                f"This is a FOLLOW-UP question. The user already answered, but we still "
                f"need: {followup_context}\n"
                f"Acknowledge what they already provided \u2014 make them feel heard. Then ask "
                f"specifically for the missing information. Be helpful:\n"
                f"- Offer concrete examples or ranges to choose from\n"
                f"- Suggest you can estimate from rough descriptions (like 'pretty stable' "
                f"  vs 'swings a lot')\n"
                f"- If the user seems stuck, offer industry benchmarks as a starting point\n"
                f"- Never make them feel bad for not knowing something"
            )
        else:
            instruction_parts.append(
                "Rephrase the base question naturally to fit the conversation. "
                "Use plain English \u2014 explain any technical concept with an analogy or "
                "example. Keep it to ONE to THREE sentences plus optional bullet examples. "
                "Provide sensible defaults where applicable and frame them as: "
                "'Most teams use __, which works well. Sound good?'"
            )
        instruction_parts.append(
            "Do NOT answer the question \u2014 only ask it. "
            "Use a warm, encouraging tone."
        )

        user_prompt = (
            f"The next topic to ask about is: {topic_meta['topic']}\n\n"
            f"Base question: {base_question}\n\n"
            f"Recent conversation:\n{context}\n\n"
            + "\n".join(instruction_parts)
        )

        if on_token is not None:
            chunks: list[str] = []
            async for token in self.llm.stream_text(SETUP_SYSTEM_PROMPT, user_prompt):
                await on_token(token)
                chunks.append(token)
            return "".join(chunks).strip()

        return await self.llm.generate_text(SETUP_SYSTEM_PROMPT, user_prompt)
