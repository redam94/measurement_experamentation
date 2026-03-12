"""
Phase transition logic for the measurement design workflows.

These pure functions define the state machine semantics — which step comes
next based on the current phase.  They return abstract step names (e.g.
"welcome", "question"), **not** LangGraph node names.  The backend maps
step names to node names (e.g. "welcome" -> "welcome_node").
"""
from __future__ import annotations


# ── Elicitation workflow transitions ────────────────────────────────────────

def next_elicitation_step(phase: str) -> str:
    """Map elicitation phase to the next logical step name."""
    if not phase:
        return "welcome"
    if phase == "elicit":
        return "question"
    if phase == "score":
        return "score"
    if phase == "recommend":
        return "recommend"
    if phase == "output":
        return "output"
    # fallback: keep eliciting
    return "question"


def after_question_step(phase: str) -> str | None:
    """
    After question processing: return next step or None (wait for user).

    Returns "score" if all topics done (phase transitioned to "score"),
    otherwise None meaning the graph should END and wait for the next message.
    """
    if phase == "score":
        return "score"
    return None


# ── Setup workflow transitions ──────────────────────────────────────────────

def next_setup_step(setup_phase: str) -> str:
    """Map setup phase to the next logical step name."""
    if not setup_phase or setup_phase == "setup_welcome":
        return "setup_welcome"
    if setup_phase == "setup_elicit":
        return "setup_question"
    if setup_phase == "power_analysis":
        return "power_analysis"
    if setup_phase == "mde_simulation":
        return "mde_simulation"
    if setup_phase == "synthetic_gen":
        return "synthetic_data"
    if setup_phase == "validation":
        return "validation"
    if setup_phase == "review_results":
        return "review_results"
    if setup_phase == "redesign_elicit":
        return "redesign_question"
    if setup_phase == "setup_output":
        return "setup_output"
    # fallback
    return "setup_question"


def after_setup_question_step(setup_phase: str) -> str | None:
    """
    After setup question processing.

    Returns "power_analysis" if all topics done, else None (wait for user).
    """
    if setup_phase == "power_analysis":
        return "power_analysis"
    return None


def after_review_step(setup_phase: str) -> str | None:
    """
    After review results.

    Returns "setup_output" if user accepts, "power_analysis" if re-running,
    or None (wait for user input).
    """
    if setup_phase == "setup_output":
        return "setup_output"
    if setup_phase == "power_analysis":
        return "power_analysis"
    return None
