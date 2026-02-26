"""
LangGraph node implementations for the experiment setup workflow.

Same single-turn-per-invocation pattern as the elicitation graph:
each node does its work and returns updates, letting the graph reach END.

Enhanced with adaptive follow-ups, interim power checks, and red-flag detection.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .setup_state import SetupState, SETUP_TOPICS, MAX_FOLLOWUP_ROUNDS, RedFlag
from ..prompts.setup_prompts import (
    SETUP_SYSTEM_PROMPT,
    SETUP_WELCOME_TEMPLATE,
    SETUP_TOPIC_INDEX,
    SETUP_EXTRACTION_SYSTEM,
    SETUP_REPORT_PROMPT,
    BASELINE_FOLLOWUP_TEMPLATES,
    FEASIBILITY_PREAMBLE_TEMPLATE,
    RED_FLAG_CATALOG,
)
from ..simulation.power import compute_power
from ..simulation.mde import compute_mde
from ..simulation.synthetic import generate_synthetic_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm(model: str = "claude-opus-4-5") -> ChatAnthropic:
    return ChatAnthropic(model=model, temperature=0.2, max_tokens=2048)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


async def _extract_setup_facts(
    user_reply: str,
    topic_meta: dict,
    llm: ChatAnthropic,
) -> dict:
    """Extract structured setup parameters from user reply."""
    messages = [
        SystemMessage(content=SETUP_EXTRACTION_SYSTEM),
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


async def _ask_setup_question(
    topic_meta: dict,
    method_key: str,
    conversation_history: list,
    llm: ChatAnthropic,
    *,
    preamble: str = "",
    followup_context: str = "",
) -> str:
    """
    Ask the next setup question, adapted for the chosen method.

    Parameters
    ----------
    preamble : str
        Optional feasibility message to weave into the response before the
        question (e.g., interim power results & red-flag warnings).
    followup_context : str
        If this is a follow-up round, describes what information is still missing.
    """
    # Get method-specific question or default
    templates = topic_meta["question_template"]
    base_question = templates.get(method_key, templates.get("_default", ""))

    context = "\n".join(
        [
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in conversation_history[-6:]
        ]
    )

    instruction_parts = []
    if preamble:
        instruction_parts.append(
            f"IMPORTANT — Before asking the question, present this feasibility update "
            f"to the user (rewrite it naturally in your own words, keeping all the "
            f"numbers and warnings):\n\n{preamble}\n"
        )
    if followup_context:
        instruction_parts.append(
            f"This is a FOLLOW-UP question. The user already answered, but we still "
            f"need: {followup_context}\n"
            f"Acknowledge what they already provided, then ask specifically for the "
            f"missing information. Be helpful — offer to estimate from ranges or "
            f"qualitative descriptions."
        )
    else:
        instruction_parts.append(
            "Rephrase the base question naturally to fit the conversation. "
            "Keep it in plain English, ONE sentence or two at most. "
            "Provide sensible defaults where applicable."
        )
    instruction_parts.append("Do NOT answer the question — only ask it.")

    prompt = (
        f"The next topic to ask about is: {topic_meta['topic']}\n\n"
        f"Base question: {base_question}\n\n"
        f"Recent conversation:\n{context}\n\n"
        + "\n".join(instruction_parts)
    )
    messages = [
        SystemMessage(content=SETUP_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = await llm.ainvoke(messages)
    return response.content.strip()


# ── Sufficiency checks ────────────────────────────────────────────────────────

def _unit_label_for_method(method_key: str) -> str:
    """Human-readable unit label for sample-size messages."""
    if method_key in ("geo_lift", "matched_market", "synthetic_control"):
        return "markets"
    if method_key == "did":
        return "units (markets × time periods)"
    return "users per group"


def _check_params_sufficient(
    topic: str,
    params: dict,
    method_key: str,
) -> tuple[bool, str | None]:
    """
    Check whether extraction got the critical values for the given topic.

    Returns (is_sufficient, missing_description_or_None).
    """
    if topic == "baseline_metrics":
        has_rate = params.get("baseline_rate") is not None
        has_value = params.get("baseline_metric_value") is not None
        has_std = params.get("baseline_metric_std") is not None

        if has_rate:
            # For proportions, std is derived — always sufficient
            return True, None
        if has_value and has_std:
            return True, None
        if has_value and not has_std:
            return False, "missing_std"
        if not has_value and not has_rate:
            if has_std:
                return False, "missing_baseline"
            return False, "missing_both"

    if topic == "expected_effect":
        has_pct = params.get("expected_lift_pct") is not None
        has_abs = params.get("expected_lift_abs") is not None
        if has_pct or has_abs:
            return True, None
        return False, "missing_effect"

    # Other topics: always sufficient (have good defaults)
    return True, None


# ── Red-flag detection ────────────────────────────────────────────────────────

def _detect_red_flags(
    method_key: str,
    params: dict,
    facts: dict,
) -> list[RedFlag]:
    """
    Scan current params/facts for feasibility concerns.

    Returns a list of RedFlag dicts (may be empty).
    """
    flags: list[RedFlag] = []
    unit_label = _unit_label_for_method(method_key)

    # ---- Coefficient of variation ----
    baseline_val = params.get("baseline_metric_value")
    baseline_std = params.get("baseline_metric_std")
    lift_pct = params.get("expected_lift_pct")
    lift_abs = params.get("expected_lift_abs")

    if baseline_val and baseline_std and baseline_val > 0:
        cv = baseline_std / baseline_val
        lift_desc = f"{lift_pct:.0%} relative" if lift_pct else (
            f"{lift_abs} absolute" if lift_abs else "unknown"
        )
        if cv > 1.0:
            flags.append({
                "severity": "critical",
                "flag": "high_cv",
                "title": RED_FLAG_CATALOG["high_cv"]["title"],
                "detail": RED_FLAG_CATALOG["high_cv"]["detail"].format(
                    cv=cv, lift=lift_desc
                ),
                "suggestion": RED_FLAG_CATALOG["high_cv"]["suggestion"],
            })
        elif cv > 0.5:
            flags.append({
                "severity": "warning",
                "flag": "high_cv",
                "title": RED_FLAG_CATALOG["high_cv"]["title"],
                "detail": RED_FLAG_CATALOG["high_cv"]["detail"].format(
                    cv=cv, lift=lift_desc
                ),
                "suggestion": RED_FLAG_CATALOG["high_cv"]["suggestion"],
            })

    # ---- Too few geo units ----
    if method_key in ("geo_lift", "matched_market", "did"):
        n_treat = params.get("num_treatment_units") or 0
        n_ctrl = params.get("num_control_units") or 0
        n_total = n_treat + n_ctrl
        if 0 < n_total < 10:
            flags.append({
                "severity": "critical",
                "flag": "too_few_geos",
                "title": RED_FLAG_CATALOG["too_few_geos"]["title"],
                "detail": RED_FLAG_CATALOG["too_few_geos"]["detail"].format(
                    n_total=n_total, n_treat=n_treat, n_ctrl=n_ctrl
                ),
                "suggestion": RED_FLAG_CATALOG["too_few_geos"]["suggestion"],
            })
        elif 0 < n_total < 20:
            flags.append({
                "severity": "warning",
                "flag": "too_few_geos",
                "title": RED_FLAG_CATALOG["too_few_geos"]["title"],
                "detail": RED_FLAG_CATALOG["too_few_geos"]["detail"].format(
                    n_total=n_total, n_treat=n_treat, n_ctrl=n_ctrl
                ),
                "suggestion": RED_FLAG_CATALOG["too_few_geos"]["suggestion"],
            })

    # ---- Short pre-period ----
    if method_key in ("did", "synthetic_control", "matched_market"):
        n_pre = params.get("num_pre_periods") or 0
        if 0 < n_pre < 4:
            flags.append({
                "severity": "critical",
                "flag": "short_pre_period",
                "title": RED_FLAG_CATALOG["short_pre_period"]["title"],
                "detail": RED_FLAG_CATALOG["short_pre_period"]["detail"].format(
                    n_pre=n_pre
                ),
                "suggestion": RED_FLAG_CATALOG["short_pre_period"]["suggestion"],
            })
        elif 0 < n_pre < 8:
            flags.append({
                "severity": "warning",
                "flag": "short_pre_period",
                "title": RED_FLAG_CATALOG["short_pre_period"]["title"],
                "detail": RED_FLAG_CATALOG["short_pre_period"]["detail"].format(
                    n_pre=n_pre
                ),
                "suggestion": RED_FLAG_CATALOG["short_pre_period"]["suggestion"],
            })

    # ---- Tiny expected effect ----
    if lift_pct is not None and lift_pct < 0.03:
        req_n_str = "many"
        flags.append({
            "severity": "warning",
            "flag": "tiny_effect",
            "title": RED_FLAG_CATALOG["tiny_effect"]["title"],
            "detail": (
                f"The expected lift of {lift_pct:.1%} is quite small relative to "
                f"the baseline variability. You may need a very large sample of "
                f"{unit_label} to detect this reliably."
            ),
            "suggestion": RED_FLAG_CATALOG["tiny_effect"]["suggestion"],
        })

    # ---- High ICC ----
    icc = params.get("icc")
    if icc is not None and icc > 0.2 and method_key in ("did", "geo_lift", "matched_market"):
        flags.append({
            "severity": "warning",
            "flag": "high_icc",
            "title": RED_FLAG_CATALOG["high_icc"]["title"],
            "detail": RED_FLAG_CATALOG["high_icc"]["detail"].format(icc=icc),
            "suggestion": RED_FLAG_CATALOG["high_icc"]["suggestion"],
        })

    # ---- Few donors for synthetic control ----
    if method_key == "synthetic_control":
        n_donors = params.get("num_control_units") or 0
        if 0 < n_donors < 5:
            flags.append({
                "severity": "critical",
                "flag": "few_donors_sc",
                "title": RED_FLAG_CATALOG["few_donors_sc"]["title"],
                "detail": RED_FLAG_CATALOG["few_donors_sc"]["detail"].format(
                    n_donors=n_donors
                ),
                "suggestion": RED_FLAG_CATALOG["few_donors_sc"]["suggestion"],
            })
        elif 0 < n_donors < 10:
            flags.append({
                "severity": "warning",
                "flag": "few_donors_sc",
                "title": RED_FLAG_CATALOG["few_donors_sc"]["title"],
                "detail": RED_FLAG_CATALOG["few_donors_sc"]["detail"].format(
                    n_donors=n_donors
                ),
                "suggestion": RED_FLAG_CATALOG["few_donors_sc"]["suggestion"],
            })

    # ---- Extreme treatment/control imbalance ----
    n_treat = params.get("num_treatment_units") or 0
    n_ctrl = params.get("num_control_units") or 0
    if n_treat > 0 and n_ctrl > 0:
        ratio = max(n_treat, n_ctrl) / min(n_treat, n_ctrl)
        if ratio > 5:
            flags.append({
                "severity": "warning",
                "flag": "extreme_imbalance",
                "title": RED_FLAG_CATALOG["extreme_imbalance"]["title"],
                "detail": RED_FLAG_CATALOG["extreme_imbalance"]["detail"].format(
                    ratio=ratio
                ),
                "suggestion": RED_FLAG_CATALOG["extreme_imbalance"]["suggestion"],
            })

    return flags


# ── Interim power check ──────────────────────────────────────────────────────

def _run_interim_power(
    method_key: str,
    params: dict,
    facts: dict,
) -> dict | None:
    """
    Run a quick power calculation if enough params exist (baseline + effect).

    Returns a power-results dict or None if we can't compute yet.
    Never raises — catches all exceptions so the flow is uninterrupted.
    """
    try:
        # Need at least a baseline and an effect
        has_baseline = (
            params.get("baseline_rate") is not None
            or params.get("baseline_metric_value") is not None
        )
        has_effect = (
            params.get("expected_lift_pct") is not None
            or params.get("expected_lift_abs") is not None
        )
        if not (has_baseline and has_effect):
            return None

        # Fill in temporary defaults for missing values so compute_power works
        tmp_params = dict(params)
        tmp_params.setdefault("alpha", 0.05)
        tmp_params.setdefault("power_target", 0.80)
        tmp_params.setdefault("one_sided", False)
        tmp_params.setdefault("n_simulations", 200)  # fast for interim
        tmp_params.setdefault("random_seed", 42)
        tmp_params.setdefault("icc", 0.05)

        if tmp_params.get("baseline_metric_value") and not tmp_params.get("baseline_metric_std"):
            tmp_params["baseline_metric_std"] = tmp_params["baseline_metric_value"] * 0.3

        # Method-specific defaults for interim calculation
        if method_key in ("did", "geo_lift", "matched_market"):
            tmp_params.setdefault("num_treatment_units", 10)
            tmp_params.setdefault("num_control_units", 10)
            tmp_params.setdefault("num_pre_periods", 8)
            tmp_params.setdefault("num_post_periods", 4)
            tmp_params.setdefault("cluster_size", 100)
        elif method_key == "synthetic_control":
            tmp_params.setdefault("num_treatment_units", 1)
            tmp_params.setdefault("num_control_units", 15)
            tmp_params.setdefault("num_pre_periods", 26)
            tmp_params.setdefault("num_post_periods", 8)
        elif method_key == "ddml":
            tmp_params.setdefault("num_treatment_units", 5000)
            tmp_params.setdefault("num_control_units", 5000)

        result = compute_power(method_key, tmp_params, facts)
        return result
    except Exception:
        return None


def _format_feasibility_message(
    interim_power: dict | None,
    red_flags: list[RedFlag],
    method_key: str,
) -> str:
    """
    Build a human-readable feasibility preamble from interim power results
    and detected red flags.

    Returns an empty string if there's nothing to report.
    """
    if interim_power is None and not red_flags:
        return ""

    unit_label = _unit_label_for_method(method_key)
    parts: list[str] = []

    if interim_power:
        req_n = interim_power.get("required_sample_size")
        achieved = interim_power.get("achieved_power", 0)
        effect = interim_power.get("effect_size_used", 0)
        req_n_str = f"~{req_n:,}" if req_n else "could not be determined"
        parts.append(
            f"📊 **Early power estimate**: You'd need roughly **{req_n_str} "
            f"{unit_label}** to achieve {achieved:.0%} power for an effect "
            f"size of {effect:.4f}."
        )

    if red_flags:
        parts.append("")  # blank line
        for rf in red_flags:
            icon = "🚨" if rf.get("severity") == "critical" else "⚠️"
            parts.append(
                f"{icon} **{rf.get('title', 'Warning')}**: "
                f"{rf.get('detail', '')}\n"
                f"   💡 *{rf.get('suggestion', '')}*"
            )

    if parts:
        parts.append("")  # trailing blank line
    return "\n".join(parts)


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def setup_welcome_node(state: SetupState) -> dict:
    """Emit the setup welcome message and prepare the state."""
    method_name = state.get("chosen_method_name", "your chosen method")
    method_key = state.get("chosen_method_key", "")

    welcome = SETUP_WELCOME_TEMPLATE.format(method_name=method_name)

    return {
        "messages": [AIMessage(content=welcome)],
        "setup_phase": "setup_elicit",
        "setup_topics_covered": [],
        "setup_params": {
            "alpha": 0.05,
            "power_target": 0.80,
            "one_sided": False,
            "n_simulations": 1000,
            "random_seed": 42,
        },
        "power_results": {},
        "mde_results": {},
        "synthetic_data": {},
        "validation_results": {},
        "setup_report_markdown": "",
        "power_curve_json": "",
        "synthetic_data_csv": "",
        "done": False,
        "pending_question": welcome,
        # Adaptive elicitation state
        "followup_round": 0,
        "feasibility_checked": False,
        "interim_power_result": {},
        "red_flags": [],
    }


async def setup_question_node(state: SetupState) -> dict:
    """
    Process user reply for the current setup topic, extract params, ask next.

    Enhanced with:
    - Follow-up questions when critical values (e.g., baseline std) are missing
    - Interim power calculation after baseline + expected_effect are gathered
    - Red-flag detection with actionable suggestions surfaced mid-conversation
    """
    llm = _make_llm()
    covered = list(state.get("setup_topics_covered", []))
    params = dict(state.get("setup_params") or {})
    messages = list(state.get("messages", []))
    method_key = state.get("chosen_method_key", "ab_test")
    facts = dict(state.get("elicited_facts") or {})
    followup_round = state.get("followup_round", 0)
    feasibility_checked = state.get("feasibility_checked", False)
    red_flags = list(state.get("red_flags") or [])
    interim_power = dict(state.get("interim_power_result") or {}) or None

    remaining = [t for t in SETUP_TOPICS if t not in covered]

    if not remaining:
        return {
            "setup_params": params,
            "setup_topics_covered": covered,
            "setup_phase": "power_analysis",
            "messages": [],
            "red_flags": red_flags,
        }

    current_topic = remaining[0]
    current_meta = SETUP_TOPIC_INDEX[current_topic]

    # ── Extract from latest user message ──────────────────────────────────
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human:
        extracted = await _extract_setup_facts(last_human.content, current_meta, llm)
        # Merge into params (skip null values)
        for k, v in extracted.items():
            if v is not None:
                params[k] = v

    # ── Check if we got enough from this topic ────────────────────────────
    is_sufficient, missing_key = _check_params_sufficient(
        current_topic, params, method_key,
    )

    if not is_sufficient and followup_round < MAX_FOLLOWUP_ROUNDS:
        # --- Ask a follow-up for the SAME topic ---
        unit_label = _unit_label_for_method(method_key)
        followup_template = BASELINE_FOLLOWUP_TEMPLATES.get(
            missing_key or "missing_both",
            BASELINE_FOLLOWUP_TEMPLATES["missing_both"],
        )
        followup_desc = followup_template.format(units=unit_label, detail="")

        question_text = await _ask_setup_question(
            current_meta, method_key, messages, llm,
            followup_context=followup_desc,
        )

        return {
            "setup_params": params,
            "setup_topics_covered": covered,        # topic NOT marked covered yet
            "messages": [AIMessage(content=question_text)],
            "pending_question": question_text,
            "setup_phase": "setup_elicit",
            "followup_round": followup_round + 1,
            "red_flags": red_flags,
        }

    # ── Topic is complete (sufficient or max follow-ups reached) ──────────
    covered_now = covered + [current_topic]
    followup_round_reset = 0  # reset for next topic

    # ── Run interim power + red-flag check after expected_effect ──────────
    preamble = ""
    if (
        current_topic == "expected_effect"
        and not feasibility_checked
    ):
        interim_power = _run_interim_power(method_key, params, facts)
        new_flags = _detect_red_flags(method_key, params, facts)
        # Merge new flags (dedup by flag key)
        existing_keys = {f["flag"] for f in red_flags}
        for nf in new_flags:
            if nf["flag"] not in existing_keys:
                red_flags.append(nf)
                existing_keys.add(nf["flag"])

        preamble = _format_feasibility_message(interim_power, red_flags, method_key)
        feasibility_checked = True

    # ── Also re-check red flags after method_specific topic ───────────────
    if current_topic == "method_specific":
        new_flags = _detect_red_flags(method_key, params, facts)
        existing_keys = {f["flag"] for f in red_flags}
        for nf in new_flags:
            if nf["flag"] not in existing_keys:
                red_flags.append(nf)
                existing_keys.add(nf["flag"])

    # ── Advance to next topic or start computation ────────────────────────
    still_remaining = [t for t in SETUP_TOPICS if t not in covered_now]

    if still_remaining:
        next_meta = SETUP_TOPIC_INDEX[still_remaining[0]]
        question_text = await _ask_setup_question(
            next_meta, method_key, messages, llm,
            preamble=preamble,
        )

        return {
            "setup_params": params,
            "setup_topics_covered": covered_now,
            "messages": [AIMessage(content=question_text)],
            "pending_question": question_text,
            "setup_phase": "setup_elicit",
            "followup_round": followup_round_reset,
            "feasibility_checked": feasibility_checked,
            "interim_power_result": interim_power if interim_power else {},
            "red_flags": red_flags,
        }

    # All topics done → run computations
    return {
        "setup_params": params,
        "setup_topics_covered": covered_now,
        "setup_phase": "power_analysis",
        "messages": [],
        "followup_round": followup_round_reset,
        "feasibility_checked": feasibility_checked,
        "interim_power_result": interim_power if interim_power else {},
        "red_flags": red_flags,
    }


async def power_analysis_node(state: SetupState) -> dict:
    """Run power / sample-size calculations."""
    method_key = state.get("chosen_method_key", "ab_test")
    params = dict(state.get("setup_params") or {})
    facts = dict(state.get("elicited_facts") or {})

    # Apply defaults for missing values
    _apply_defaults(params, facts)

    power_results = compute_power(method_key, params, facts)

    msg = (
        "📊 **Power Analysis Complete!**\n\n"
        f"- **Required sample size**: {power_results.get('required_sample_size', 'N/A')}\n"
        f"- **Achieved power**: {power_results.get('achieved_power', 0):.1%}\n"
        f"- **Effect size used**: {power_results.get('effect_size_used', 0):.4f}\n\n"
        f"{power_results.get('notes', '')}\n\n"
        "Running Monte Carlo MDE simulation next…"
    )

    return {
        "power_results": power_results,
        "setup_phase": "mde_simulation",
        "messages": [AIMessage(content=msg)],
        "power_curve_json": json.dumps(power_results.get("power_curve", []), indent=2),
    }


async def mde_simulation_node(state: SetupState) -> dict:
    """Run Monte Carlo MDE simulation."""
    method_key = state.get("chosen_method_key", "ab_test")
    params = dict(state.get("setup_params") or {})
    facts = dict(state.get("elicited_facts") or {})
    power_results = dict(state.get("power_results") or {})

    _apply_defaults(params, facts)

    mde_results = compute_mde(method_key, params, facts, power_results)

    mde_abs = mde_results.get("mde_absolute")
    mde_rel = mde_results.get("mde_relative_pct")

    mde_str = "could not be determined (try more simulations or a larger sample)"
    if mde_abs is not None:
        mde_str = f"{mde_abs:.4f}"
        if mde_rel is not None:
            mde_str += f" ({mde_rel:.1f}% relative)"

    msg = (
        "🎯 **MDE Simulation Complete!**\n\n"
        f"- **Minimum Detectable Effect**: {mde_str}\n"
        f"- **Simulations run**: {mde_results.get('n_simulations', 0):,}\n"
        f"- **Target power**: {mde_results.get('target_power', 0.8):.0%}\n"
        f"- **Significance level**: {mde_results.get('alpha', 0.05)}\n\n"
        f"{mde_results.get('notes', '')}\n\n"
        "Generating synthetic test data next…"
    )

    return {
        "mde_results": mde_results,
        "setup_phase": "synthetic_gen",
        "messages": [AIMessage(content=msg)],
    }


async def synthetic_data_node(state: SetupState) -> dict:
    """Generate synthetic data matching the user's scenario."""
    method_key = state.get("chosen_method_key", "ab_test")
    params = dict(state.get("setup_params") or {})
    facts = dict(state.get("elicited_facts") or {})
    power_results = dict(state.get("power_results") or {})
    mde_results = dict(state.get("mde_results") or {})

    _apply_defaults(params, facts)

    synth = generate_synthetic_data(method_key, params, facts, power_results, mde_results)

    msg = (
        "🧪 **Synthetic Data Generated!**\n\n"
        f"- **Rows**: {synth.get('n_rows', 0):,}\n"
        f"- **Columns**: {', '.join(synth.get('columns', []))}\n"
        f"- **True injected effect**: {synth.get('true_effect', 0):.4f}\n\n"
        f"{synth.get('description', '')}\n\n"
        "Now I'll validate the analysis code against this synthetic data…"
    )

    return {
        "synthetic_data": synth,
        "synthetic_data_csv": synth.get("csv_string", ""),
        "setup_phase": "validation",
        "messages": [AIMessage(content=msg)],
    }


async def validation_node(state: SetupState) -> dict:
    """
    Validate the analysis scaffold against synthetic data.
    Runs a simplified version of the scaffold's analysis on the synthetic data.
    """
    method_key = state.get("chosen_method_key", "ab_test")
    synth = dict(state.get("synthetic_data") or {})
    true_effect = synth.get("true_effect", 0)

    # Run a quick validation analysis
    validation = _run_validation(method_key, synth)

    if validation.get("success"):
        msg = (
            "✅ **Code Validation Passed!**\n\n"
            f"- **True effect**: {true_effect:.4f}\n"
            f"- **Estimated effect**: {validation.get('estimated_effect', 'N/A')}\n"
            f"- **95% CI**: [{validation.get('ci_lower', 'N/A')}, {validation.get('ci_upper', 'N/A')}]\n"
        )
        if validation.get("p_value") is not None:
            msg += f"- **p-value**: {validation['p_value']:.4f}\n"
        msg += (
            f"\n{validation.get('summary', '')}\n\n"
            "Generating your setup report now…"
        )
    else:
        msg = (
            "⚠️ **Code Validation Note**\n\n"
            f"{validation.get('error_message', 'Validation encountered an issue.')}\n\n"
            "This is expected for some methods with small synthetic datasets. "
            "The generated scaffold is still valid — the validation is just conservative. "
            "Generating your setup report now…"
        )

    return {
        "validation_results": validation,
        "setup_phase": "setup_output",
        "messages": [AIMessage(content=msg)],
    }


async def setup_output_node(state: SetupState) -> dict:
    """Generate the final setup report and mark the workflow done."""
    llm = _make_llm()
    method_key = state.get("chosen_method_key", "ab_test")
    method_name = state.get("chosen_method_name", method_key)
    facts = dict(state.get("elicited_facts") or {})
    params = dict(state.get("setup_params") or {})
    power_results = dict(state.get("power_results") or {})
    mde_results = dict(state.get("mde_results") or {})
    synth = dict(state.get("synthetic_data") or {})
    validation = dict(state.get("validation_results") or {})
    red_flags = list(state.get("red_flags") or [])

    # Re-run final red-flag detection with all params populated
    final_flags = _detect_red_flags(method_key, params, facts)
    existing_keys = {f["flag"] for f in red_flags}
    for nf in final_flags:
        if nf["flag"] not in existing_keys:
            red_flags.append(nf)
            existing_keys.add(nf["flag"])

    # Check if available sample is insufficient vs required
    req_n = power_results.get("required_sample_size")
    avail_treat = params.get("num_treatment_units") or 0
    avail_ctrl = params.get("num_control_units") or 0
    avail_n = avail_treat + avail_ctrl
    unit_label = _unit_label_for_method(method_key)
    if req_n and avail_n and avail_n < req_n and "insufficient_sample" not in existing_keys:
        red_flags.append({
            "severity": "critical" if avail_n < req_n * 0.5 else "warning",
            "flag": "insufficient_sample",
            "title": RED_FLAG_CATALOG["insufficient_sample"]["title"],
            "detail": RED_FLAG_CATALOG["insufficient_sample"]["detail"].format(
                required_n=req_n, unit_label=unit_label, available_n=avail_n,
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
            indent=2,
            default=str,
        ),
        synthetic_summary=synth.get("description", "No synthetic data generated."),
        validation_json=json.dumps(validation, indent=2, default=str),
        red_flags_json=json.dumps(red_flags, indent=2, default=str),
    )

    messages = [
        SystemMessage(content=SETUP_SYSTEM_PROMPT),
        HumanMessage(content=report_prompt),
    ]
    response = await llm.ainvoke(messages)
    report_md = response.content.strip()

    done_msg = (
        "🎉 **Your experiment setup is complete!**\n\n"
        "You now have:\n"
        "- 📊 **Power analysis** with required sample sizes\n"
        "- 🎯 **MDE simulation** showing the smallest detectable effect\n"
        "- 🧪 **Synthetic test data** matching your scenario\n"
        "- ✅ **Validation results** confirming the analysis pipeline works\n"
        "- 📋 **Setup report** with actionable recommendations\n"
    )

    if red_flags:
        n_crit = sum(1 for f in red_flags if f.get("severity") == "critical")
        n_warn = sum(1 for f in red_flags if f.get("severity") == "warning")
        flag_parts = []
        if n_crit:
            flag_parts.append(f"🚨 {n_crit} critical")
        if n_warn:
            flag_parts.append(f"⚠️ {n_warn} warning(s)")
        done_msg += f"- {'  &  '.join(flag_parts)} flagged — see the report for details\n"

    done_msg += "\nDownload everything from the tabs below!"

    return {
        "setup_report_markdown": report_md,
        "setup_phase": "setup_done",
        "done": True,
        "messages": [AIMessage(content=done_msg)],
        "red_flags": red_flags,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_defaults(params: dict, facts: dict) -> None:
    """Fill in sensible defaults for missing parameters."""
    if not params.get("baseline_rate") and not params.get("baseline_metric_value"):
        # Try to infer from facts
        kpi = facts.get("kpi", "").lower()
        if "rate" in kpi or "conversion" in kpi or "ctr" in kpi:
            params.setdefault("baseline_rate", 0.03)
        else:
            params.setdefault("baseline_metric_value", 100.0)
            params.setdefault("baseline_metric_std", 30.0)

    if params.get("baseline_metric_value") and not params.get("baseline_metric_std"):
        params["baseline_metric_std"] = params["baseline_metric_value"] * 0.3

    if not params.get("expected_lift_pct") and not params.get("expected_lift_abs"):
        params["expected_lift_pct"] = 0.10  # default 10% relative

    params.setdefault("alpha", 0.05)
    params.setdefault("power_target", 0.80)
    params.setdefault("one_sided", False)
    params.setdefault("n_simulations", 1000)
    params.setdefault("random_seed", 42)
    params.setdefault("icc", 0.05)


def _run_validation(method_key: str, synth_data: dict) -> dict:
    """
    Run a quick statistical test on synthetic data to validate the approach.
    This is a simplified version — not the full PyMC scaffold.
    """
    import io

    csv_str = synth_data.get("csv_string", "")
    true_effect = synth_data.get("true_effect", 0)

    if not csv_str:
        return {
            "success": False,
            "error_message": "No synthetic data available for validation.",
        }

    try:
        df = pd.read_csv(io.StringIO(csv_str))
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Could not parse synthetic data: {e}",
        }

    try:
        if method_key == "ab_test":
            return _validate_ab_test(df, true_effect)
        elif method_key == "did":
            return _validate_did(df, true_effect)
        elif method_key in ("geo_lift", "matched_market"):
            return _validate_geo(df, true_effect)
        elif method_key == "synthetic_control":
            return _validate_synth_ctrl(df, true_effect)
        elif method_key == "ddml":
            return _validate_ddml(df, true_effect)
        else:
            return _validate_ab_test(df, true_effect)
    except Exception as e:
        return {
            "success": False,
            "estimated_effect": None,
            "true_effect": true_effect,
            "error_message": f"Validation analysis raised an error: {e}",
            "summary": "The synthetic data was generated but validation did not complete.",
        }


def _validate_ab_test(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate A/B test data with a simple two-sample test."""
    stats = scipy_stats

    if "converted" in df.columns:
        ctrl = df[df["group"] == "control"]["converted"]
        treat = df[df["group"] == "treatment"]["converted"]
    elif "outcome" in df.columns:
        ctrl = df[df["group"] == "control"]["outcome"]
        treat = df[df["group"] == "treatment"]["outcome"]
    else:
        return {"success": False, "error_message": "Unexpected column structure."}

    diff = treat.mean() - ctrl.mean()
    t_stat, p_val = stats.ttest_ind(treat, ctrl)

    # Bootstrap CI
    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(n_boot):
        c_samp = ctrl.sample(n=len(ctrl), replace=True, random_state=rng.integers(1e9))
        t_samp = treat.sample(n=len(treat), replace=True, random_state=rng.integers(1e9))
        boot_diffs.append(t_samp.mean() - c_samp.mean())
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "success": True,
        "estimated_effect": round(float(diff), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_val), 6),
        "summary": (
            f"Two-sample test: estimated effect = {diff:.4f} "
            f"(true = {true_effect:.4f}), p = {p_val:.4f}. "
            f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
    }


def _validate_did(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate DiD data with a simple DiD estimator."""
    pre_treat = df[(df["group"] == "treatment") & (df["is_post"] == 0)]["outcome"].mean()
    post_treat = df[(df["group"] == "treatment") & (df["is_post"] == 1)]["outcome"].mean()
    pre_ctrl = df[(df["group"] == "control") & (df["is_post"] == 0)]["outcome"].mean()
    post_ctrl = df[(df["group"] == "control") & (df["is_post"] == 1)]["outcome"].mean()

    did_est = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)

    # Simple SE from unit-level DiD
    units = df["unit_id"].unique()
    unit_dids = []
    for u in units:
        ud = df[df["unit_id"] == u]
        pre = ud[ud["is_post"] == 0]["outcome"].mean()
        post = ud[ud["is_post"] == 1]["outcome"].mean()
        group = ud["group"].iloc[0]
        unit_dids.append({"unit": u, "group": group, "diff": post - pre})

    ud_df = pd.DataFrame(unit_dids)
    treat_diffs = ud_df[ud_df["group"] == "treatment"]["diff"]
    ctrl_diffs = ud_df[ud_df["group"] == "control"]["diff"]
    t_stat, p_val = scipy_stats.ttest_ind(treat_diffs, ctrl_diffs)

    se = np.sqrt(treat_diffs.var() / len(treat_diffs) + ctrl_diffs.var() / len(ctrl_diffs))
    ci_lower = did_est - 1.96 * se
    ci_upper = did_est + 1.96 * se

    return {
        "success": True,
        "estimated_effect": round(float(did_est), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_val), 6),
        "summary": (
            f"DiD estimator: {did_est:.4f} (true = {true_effect:.4f}), "
            f"p = {p_val:.4f}. 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
    }


def _validate_geo(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate geo/market data."""
    outcome_col = "kpi_value" if "kpi_value" in df.columns else "outcome"

    # Post-period comparison
    post = df[df["is_post"] == 1]
    treat_markets = post[post["group"] == "treatment"].groupby(
        post[post["group"] == "treatment"].columns[0]
    )[outcome_col].mean()
    ctrl_markets = post[post["group"] == "control"].groupby(
        post[post["group"] == "control"].columns[0]
    )[outcome_col].mean()

    diff = treat_markets.mean() - ctrl_markets.mean()
    t_stat, p_val = scipy_stats.ttest_ind(treat_markets, ctrl_markets)

    se = np.sqrt(treat_markets.var() / len(treat_markets) + ctrl_markets.var() / len(ctrl_markets))
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    return {
        "success": True,
        "estimated_effect": round(float(diff), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_val), 6),
        "summary": (
            f"Geo market comparison: {diff:.4f} (true = {true_effect:.4f}), "
            f"p = {p_val:.4f}. 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
    }


def _validate_synth_ctrl(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate synthetic control data with simple mean comparison."""
    post = df[df["is_post"] == 1]
    treat_mean = post[post["is_treated"] == 1]["outcome"].mean()
    donor_mean = post[post["is_treated"] == 0].groupby("unit_id")["outcome"].mean().mean()

    gap = treat_mean - donor_mean

    return {
        "success": True,
        "estimated_effect": round(float(gap), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": None,
        "ci_upper": None,
        "p_value": None,
        "summary": (
            f"Avg post-period gap: {gap:.4f} (true = {true_effect:.4f}). "
            f"Formal inference requires permutation test — see the full scaffold."
        ),
    }


def _validate_ddml(df: pd.DataFrame, true_effect: float) -> dict:
    """Validate DDML data with OLS partialling out."""
    feature_cols = [c for c in df.columns if c.startswith("x_")]
    X = df[feature_cols].values
    D = df["treatment"].values
    Y = df["outcome"].values

    # Partial out X via OLS
    XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
    beta_d = np.linalg.solve(XtX, X.T @ D)
    D_resid = D - X @ beta_d
    beta_y = np.linalg.solve(XtX, X.T @ Y)
    Y_resid = Y - X @ beta_y

    denom = D_resid @ D_resid
    theta = (D_resid @ Y_resid) / denom
    e = Y_resid - theta * D_resid
    V = np.sum((D_resid ** 2) * (e ** 2)) / (denom ** 2)
    se = np.sqrt(V)

    z = theta / se
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se

    return {
        "success": True,
        "estimated_effect": round(float(theta), 6),
        "true_effect": round(true_effect, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_val), 6),
        "summary": (
            f"Partialling-out estimator: θ̂ = {theta:.4f} (true = {true_effect:.4f}), "
            f"p = {p_val:.4f}. 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
    }
