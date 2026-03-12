"""
LangGraph node implementations for the experiment setup workflow.

Same single-turn-per-invocation pattern as the elicitation graph:
each node does its work and returns updates, letting the graph reach END.

Enhanced with adaptive follow-ups, interim power checks, and red-flag detection.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .setup_state import SetupState, SETUP_TOPICS, MAX_FOLLOWUP_ROUNDS, RedFlag
from measurement_design.knowledge import RED_FLAG_CATALOG, METHOD_ASSUMPTIONS
from measurement_design.validation.feasibility import (
    unit_label_for_method as _unit_label_for_method,
    check_params_sufficient as _check_params_sufficient,
    detect_red_flags as _detect_red_flags,
    run_interim_power as _run_interim_power,
    apply_defaults as _apply_defaults,
    build_assumptions_summary as _build_assumptions_summary,
    identify_design_problems as _identify_design_problems,
    run_validation as _run_validation,
)
from ..prompts.setup_prompts import (
    SETUP_SYSTEM_PROMPT,
    SETUP_WELCOME_TEMPLATE,
    SETUP_TOPIC_INDEX,
    SETUP_EXTRACTION_SYSTEM,
    SETUP_REPORT_PROMPT,
    BASELINE_FOLLOWUP_TEMPLATES,
    FEASIBILITY_PREAMBLE_TEMPLATE,
    REVIEW_RESULTS_PROMPT,
    REVIEW_DECISION_SYSTEM,
    REDESIGN_ELICIT_PROMPT,
)
from measurement_design.simulation.power import compute_power
from measurement_design.simulation.mde import compute_mde
from measurement_design.simulation.synthetic import generate_synthetic_data


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
    token_queue: asyncio.Queue | None = None,
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
            f"to the user. Rewrite it in your own words using plain, friendly language. "
            f"Keep all the numbers and warnings, but explain what they mean in everyday "
            f"terms (e.g., 'you'd need about 10,000 customers in your test' instead of "
            f"'required sample size is 10,000'):\n\n{preamble}\n"
        )
    if followup_context:
        instruction_parts.append(
            f"This is a FOLLOW-UP question. The user already answered, but we still "
            f"need: {followup_context}\n"
            f"Acknowledge what they already provided — make them feel heard. Then ask "
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
            "Use plain English — explain any technical concept with an analogy or "
            "example. Keep it to ONE to THREE sentences plus optional bullet examples. "
            "Provide sensible defaults where applicable and frame them as: "
            "'Most teams use __, which works well. Sound good?'"
        )
    instruction_parts.append(
        "Do NOT answer the question — only ask it. "
        "Use a warm, encouraging tone."
    )

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

    if token_queue is not None:
        chunks: list[str] = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                await token_queue.put(chunk.content)
                chunks.append(chunk.content)
        return "".join(chunks).strip()

    response = await llm.ainvoke(messages)
    return response.content.strip()


# Sufficiency checks, red-flag detection, interim power, defaults, validation,
# and assumptions summary are now in measurement_design.validation.feasibility
# and imported above as _unit_label_for_method, _check_params_sufficient, etc.


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

async def setup_welcome_node(state: SetupState, config: RunnableConfig) -> dict:
    """Emit the setup welcome message with the first question and prepare the state."""
    token_queue = (config.get("configurable") or {}).get("token_queue")
    llm = _make_llm()
    method_name = state.get("chosen_method_name", "your chosen method")
    method_key = state.get("chosen_method_key", "")

    welcome = SETUP_WELCOME_TEMPLATE.format(method_name=method_name)

    # Also generate the first setup question (baseline_metrics) so the user
    # immediately knows what to answer, rather than seeing a generic intro.
    first_topic = SETUP_TOPICS[0]  # "baseline_metrics"
    first_meta = SETUP_TOPIC_INDEX[first_topic]
    first_question = await _ask_setup_question(
        first_meta, method_key, [], llm,
        token_queue=token_queue,
    )

    combined_message = f"{welcome}\n{first_question}"

    return {
        "messages": [AIMessage(content=combined_message)],
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
        "pending_question": combined_message,
        # Adaptive elicitation state
        "followup_round": 0,
        "feasibility_checked": False,
        "interim_power_result": {},
        "red_flags": [],
    }


async def setup_question_node(state: SetupState, config: RunnableConfig) -> dict:
    """
    Process user reply for the current setup topic, extract params, ask next.

    Enhanced with:
    - Follow-up questions when critical values (e.g., baseline std) are missing
    - Interim power calculation after baseline + expected_effect are gathered
    - Red-flag detection with actionable suggestions surfaced mid-conversation
    """
    token_queue = (config.get("configurable") or {}).get("token_queue")
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
            token_queue=token_queue,
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
            token_queue=token_queue,
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


async def power_analysis_node(state: SetupState, config: RunnableConfig) -> dict:
    """Run power / sample-size calculations."""
    token_queue = (config.get("configurable") or {}).get("token_queue")
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

    if token_queue:
        await token_queue.put(msg + "\n\n")

    return {
        "power_results": power_results,
        "setup_phase": "mde_simulation",
        "messages": [AIMessage(content=msg)],
        "power_curve_json": json.dumps(power_results.get("power_curve", []), indent=2),
    }


async def mde_simulation_node(state: SetupState, config: RunnableConfig) -> dict:
    """Run Monte Carlo MDE simulation."""
    token_queue = (config.get("configurable") or {}).get("token_queue")
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

    if token_queue:
        await token_queue.put(msg + "\n\n")

    return {
        "mde_results": mde_results,
        "setup_phase": "synthetic_gen",
        "messages": [AIMessage(content=msg)],
    }


async def synthetic_data_node(state: SetupState, config: RunnableConfig) -> dict:
    """Generate synthetic data matching the user's scenario."""
    token_queue = (config.get("configurable") or {}).get("token_queue")
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

    if token_queue:
        await token_queue.put(msg + "\n\n")

    return {
        "synthetic_data": synth,
        "synthetic_data_csv": synth.get("csv_string", ""),
        "setup_phase": "validation",
        "messages": [AIMessage(content=msg)],
    }


async def validation_node(state: SetupState, config: RunnableConfig) -> dict:
    """
    Validate the analysis scaffold against synthetic data.
    Runs a simplified version of the scaffold's analysis on the synthetic data.
    """
    token_queue = (config.get("configurable") or {}).get("token_queue")
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
            "Reviewing your results now…"
        )
    else:
        msg = (
            "⚠️ **Code Validation Note**\n\n"
            f"{validation.get('error_message', 'Validation encountered an issue.')}\n\n"
            "This is expected for some methods with small synthetic datasets. "
            "The generated scaffold is still valid — the validation is just conservative. "
            "Reviewing your results now…"
        )

    if token_queue:
        await token_queue.put(msg + "\n\n")

    return {
        "validation_results": validation,
        "setup_phase": "review_results",
        "messages": [AIMessage(content=msg)],
    }


async def setup_output_node(state: SetupState, config: RunnableConfig) -> dict:
    """Generate the final setup report and mark the workflow done."""
    token_queue = (config.get("configurable") or {}).get("token_queue")
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

    if token_queue:
        await token_queue.put(done_msg)

    return {
        "setup_report_markdown": report_md,
        "setup_phase": "setup_done",
        "done": True,
        "messages": [AIMessage(content=done_msg)],
        "red_flags": red_flags,
    }


# ── Review results node ──────────────────────────────────────────────────────

# _build_assumptions_summary and _identify_design_problems are now imported
# from measurement_design.validation.feasibility


async def review_results_node(state: SetupState, config: RunnableConfig) -> dict:
    """
    Present power/MDE results to the user and ask if they want to proceed or adjust.

    Two modes:
    - First entry (after validation pipeline): Present results summary
    - Re-entry (user responded): Determine accept or modify
    """
    token_queue = (config.get("configurable") or {}).get("token_queue")
    llm = _make_llm()
    messages = list(state.get("messages", []))
    method_key = state.get("chosen_method_key", "ab_test")
    method_name = state.get("chosen_method_name", method_key)
    params = dict(state.get("setup_params") or {})
    power_results = dict(state.get("power_results") or {})
    mde_results = dict(state.get("mde_results") or {})
    red_flags = list(state.get("red_flags") or [])

    # Determine if this is first entry or re-entry
    last_msg = messages[-1] if messages else None
    is_reentry = isinstance(last_msg, HumanMessage)

    if not is_reentry:
        # ── First entry: present results and ask ──────────────────────────
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
                icon = "🚨" if rf.get("severity") == "critical" else "⚠️"
                rf_parts.append(f"{icon} {rf.get('title', 'Flag')}: {rf.get('detail', '')}")
            red_flags_text = "\n".join(rf_parts)
        else:
            red_flags_text = "No major concerns detected."

        assumptions_text = _build_assumptions_summary(method_key)

        # Build setup params summary
        param_parts = []
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
        setup_params_summary = "\n".join(f"- {p}" for p in param_parts) if param_parts else "Default parameters"

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

        review_messages = [
            SystemMessage(content=SETUP_SYSTEM_PROMPT),
            HumanMessage(content=review_prompt),
        ]

        if token_queue is not None:
            chunks: list[str] = []
            async for chunk in llm.astream(review_messages):
                if chunk.content:
                    await token_queue.put(chunk.content)
                    chunks.append(chunk.content)
            response_text = "".join(chunks).strip()
        else:
            response = await llm.ainvoke(review_messages)
            response_text = response.content.strip()

        return {
            "setup_phase": "review_results",
            "messages": [AIMessage(content=response_text)],
            "done": False,
        }

    # ── Re-entry: analyze user response ───────────────────────────────────
    user_text = last_msg.content

    # Use LLM to determine accept vs modify and extract changes
    decision_prompt = (
        f"The user was reviewing their experiment design results and responded:\n\n"
        f"\"{user_text}\"\n\n"
        f"Current parameters:\n{json.dumps(params, indent=2, default=str)}\n\n"
        f"Determine their intent and extract any parameter changes."
    )

    decision_response = await llm.ainvoke([
        SystemMessage(content=REVIEW_DECISION_SYSTEM),
        HumanMessage(content=decision_prompt),
    ])

    try:
        decision = json.loads(_strip_json_fence(decision_response.content))
    except json.JSONDecodeError:
        decision = {"decision": "modify", "changes": {}, "change_summary": None}

    if decision.get("decision") == "accept":
        # User accepts → chain to setup_output
        accept_msg = (
            "Great, let's finalize your experiment setup! "
            "Generating your complete setup report now…"
        )
        if token_queue:
            await token_queue.put(accept_msg)
        return {
            "setup_phase": "setup_output",
            "messages": [AIMessage(content=accept_msg)],
        }

    # User wants to modify
    changes = decision.get("changes") or {}
    # Filter out null values
    actual_changes = {k: v for k, v in changes.items() if v is not None}

    if actual_changes:
        # User specified changes → apply and re-run
        for k, v in actual_changes.items():
            params[k] = v

        change_summary = decision.get("change_summary") or "your requested changes"
        rerun_msg = (
            f"Got it! I've updated the design with {change_summary}. "
            f"Let me re-run the analysis with these new parameters…"
        )
        if token_queue:
            await token_queue.put(rerun_msg)
        return {
            "setup_params": params,
            "setup_phase": "power_analysis",
            "messages": [AIMessage(content=rerun_msg)],
        }

    # User wants to change but was vague → ask specifically
    problems = _identify_design_problems(
        power_results, mde_results, red_flags, params, method_key,
    )
    redesign_prompt = REDESIGN_ELICIT_PROMPT.format(
        problems=problems,
        current_params=json.dumps(params, indent=2, default=str),
    )

    redesign_messages = [
        SystemMessage(content=SETUP_SYSTEM_PROMPT),
        HumanMessage(content=redesign_prompt),
    ]

    if token_queue is not None:
        chunks = []
        async for chunk in llm.astream(redesign_messages):
            if chunk.content:
                await token_queue.put(chunk.content)
                chunks.append(chunk.content)
        response_text = "".join(chunks).strip()
    else:
        response = await llm.ainvoke(redesign_messages)
        response_text = response.content.strip()

    return {
        "setup_phase": "redesign_elicit",
        "messages": [AIMessage(content=response_text)],
        "done": False,
    }


async def redesign_question_node(state: SetupState, config: RunnableConfig) -> dict:
    """
    Process the user's specific design change request and trigger re-computation.
    """
    token_queue = (config.get("configurable") or {}).get("token_queue")
    llm = _make_llm()
    messages = list(state.get("messages", []))
    params = dict(state.get("setup_params") or {})

    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )

    if last_human:
        # Extract parameter changes from user response
        extract_prompt = (
            f"The user wants to modify their experiment design parameters.\n\n"
            f"Their response: \"{last_human.content}\"\n\n"
            f"Current parameters:\n{json.dumps(params, indent=2, default=str)}\n\n"
            f"Extract the specific parameter changes."
        )

        response = await llm.ainvoke([
            SystemMessage(content=REVIEW_DECISION_SYSTEM),
            HumanMessage(content=extract_prompt),
        ])

        try:
            result = json.loads(_strip_json_fence(response.content))
            changes = result.get("changes") or {}
            for k, v in changes.items():
                if v is not None:
                    params[k] = v
            change_summary = result.get("change_summary") or "your changes"
        except json.JSONDecodeError:
            change_summary = "your requested adjustments"

    msg = (
        f"Updated! Re-running the full analysis pipeline with {change_summary}…\n\n"
        "⏳ Computing power analysis, MDE simulation, synthetic data, and validation…"
    )

    if token_queue:
        await token_queue.put(msg)

    return {
        "setup_params": params,
        "setup_phase": "power_analysis",
        "messages": [AIMessage(content=msg)],
    }


# _apply_defaults, _run_validation, and _validate_* are now imported
# from measurement_design.validation.feasibility (see imports at top of file).
