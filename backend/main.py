"""
FastAPI application — entry point for the measurement design agent backend.
"""
from __future__ import annotations

import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

load_dotenv()

from .graph.graph import graph
from .graph.state import AgentState
from .graph.setup_graph import setup_graph
from .methods import METHOD_MAP

app = FastAPI(
    title="Measurement Design Agent",
    description=(
        "Agentic framework that elicits experimental design requirements "
        "from non-experts and recommends causal measurement methods for ad campaigns."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store  {session_id: AgentState}
# Replace with Redis / DB for production.
_sessions: dict[str, dict[str, Any]] = {}
_setup_sessions: dict[str, dict[str, Any]] = {}


# ── Request / Response models ──────────────────────────────────────────────────

class CreateSessionResponse(BaseModel):
    session_id: str
    message: str
    phase: str


class TurnRequest(BaseModel):
    message: str


class TurnResponse(BaseModel):
    reply: str
    phase: str
    done: bool
    covered_topics: list[str]


class ReportResponse(BaseModel):
    markdown: str
    json_spec: dict[str, Any]
    yaml_spec: str
    scaffold: str
    ranked_methods: list[dict[str, Any]]


class StartSetupRequest(BaseModel):
    method_key: str


class SetupTurnResponse(BaseModel):
    reply: str
    setup_phase: str
    done: bool
    setup_topics_covered: list[str]
    red_flags: list[dict[str, Any]] = []


class SetupResultsResponse(BaseModel):
    setup_report_markdown: str
    power_results: dict[str, Any]
    mde_results: dict[str, Any]
    synthetic_data_csv: str
    validation_results: dict[str, Any]
    power_curve_json: str
    red_flags: list[dict[str, Any]] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    """
    Start a new elicitation session.
    Returns the session ID and the agent's opening message.
    """
    session_id = str(uuid.uuid4())

    # Seed empty state and run the welcome node
    initial_state: dict[str, Any] = {
        "session_id": session_id,
        "messages": [],
        "elicited_facts": {},
        "covered_topics": [],
        "phase": "",
        "scores": {},
        "ranked_methods": [],
        "clarify_rounds": 0,
        "report_markdown": "",
        "spec_json": {},
        "spec_yaml": "",
        "scaffold_code": "",
        "pending_question": "",
        "done": False,
    }

    result = await graph.ainvoke(initial_state)
    _sessions[session_id] = result

    # Get the agent's first message
    msgs = result.get("messages", [])
    last_msg = msgs[-1].content if msgs else ""

    return CreateSessionResponse(
        session_id=session_id,
        message=last_msg,
        phase=result.get("phase", "elicit"),
    )


@app.post("/sessions/{session_id}/turn", response_model=TurnResponse)
async def handle_turn(session_id: str, body: TurnRequest) -> TurnResponse:
    """
    Submit a user message and get the agent's next response.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = _sessions[session_id]

    if state.get("done"):
        return TurnResponse(
            reply="This session is complete. Please start a new session.",
            phase="done",
            done=True,
            covered_topics=state.get("covered_topics", []),
        )

    # Append user message and re-invoke
    state["messages"] = list(state.get("messages", [])) + [
        HumanMessage(content=body.message)
    ]

    result = await graph.ainvoke(state)
    _sessions[session_id] = result

    msgs = result.get("messages", [])
    # Get the last AI message
    from langchain_core.messages import AIMessage
    last_ai = next(
        (m for m in reversed(msgs) if isinstance(m, AIMessage)),
        None,
    )
    reply = last_ai.content if last_ai else ""

    return TurnResponse(
        reply=reply,
        phase=result.get("phase", "elicit"),
        done=bool(result.get("done", False)),
        covered_topics=result.get("covered_topics", []),
    )


@app.get("/sessions/{session_id}/report", response_model=ReportResponse)
async def get_report(session_id: str) -> ReportResponse:
    """
    Retrieve the generated report, spec, and scaffold for a completed session.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = _sessions[session_id]

    if not state.get("done"):
        raise HTTPException(
            status_code=400,
            detail="Session is not yet complete. Continue the conversation.",
        )

    # Build ranked method summary for the response
    from .scoring.scorer import score_methods, rank_methods, generate_explanations, build_ranked_report_data
    from langchain_anthropic import ChatAnthropic

    facts = state.get("elicited_facts") or {}
    scores = state.get("scores") or {}

    ranked_summary = [
        {
            "rank": i + 1,
            "key": key,
            "score": round(scores.get(key, 0), 1),
        }
        for i, key in enumerate(state.get("ranked_methods", []))
    ]

    return ReportResponse(
        markdown=state.get("report_markdown", ""),
        json_spec=state.get("spec_json") or {},
        yaml_spec=state.get("spec_yaml", ""),
        scaffold=state.get("scaffold_code", ""),
        ranked_methods=ranked_summary,
    )


@app.get("/sessions/{session_id}/status")
async def get_status(session_id: str) -> dict:
    """Lightweight session status check."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = _sessions[session_id]
    return {
        "session_id": session_id,
        "phase": state.get("phase", "unknown"),
        "done": state.get("done", False),
        "covered_topics": state.get("covered_topics", []),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session from memory."""
    _sessions.pop(session_id, None)
    _setup_sessions.pop(session_id, None)
    return {"deleted": session_id}


# ── Setup Workflow Endpoints ──────────────────────────────────────────────────

@app.post("/sessions/{session_id}/setup", response_model=SetupTurnResponse)
async def start_setup(session_id: str, body: StartSetupRequest) -> SetupTurnResponse:
    """
    Start the setup workflow for a completed elicitation session.
    The user picks a method from the ranked list.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    elicitation_state = _sessions[session_id]
    if not elicitation_state.get("done"):
        raise HTTPException(
            status_code=400,
            detail="Elicitation session is not complete. Finish the conversation first.",
        )

    method_key = body.method_key
    method = METHOD_MAP.get(method_key)
    if not method:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method key: {method_key}. "
                   f"Valid keys: {list(METHOD_MAP.keys())}",
        )

    # Seed setup state from elicitation results
    initial_setup_state: dict[str, Any] = {
        "session_id": session_id,
        "messages": [],
        "chosen_method_key": method_key,
        "chosen_method_name": method.name,
        "elicited_facts": dict(elicitation_state.get("elicited_facts") or {}),
        "scores": dict(elicitation_state.get("scores") or {}),
        "ranked_methods": list(elicitation_state.get("ranked_methods") or []),
        "setup_params": {},
        "setup_topics_covered": [],
        "setup_phase": "",
        "pending_question": "",
        "power_results": {},
        "mde_results": {},
        "synthetic_data": {},
        "validation_results": {},
        "setup_report_markdown": "",
        "power_curve_json": "",
        "synthetic_data_csv": "",
        "done": False,
        # Adaptive elicitation state
        "followup_round": 0,
        "feasibility_checked": False,
        "interim_power_result": {},
        "red_flags": [],
    }

    result = await setup_graph.ainvoke(initial_setup_state)
    _setup_sessions[session_id] = result

    msgs = result.get("messages", [])
    last_msg = msgs[-1].content if msgs else ""

    return SetupTurnResponse(
        reply=last_msg,
        setup_phase=result.get("setup_phase", "setup_elicit"),
        done=False,
        setup_topics_covered=result.get("setup_topics_covered", []),
        red_flags=result.get("red_flags", []),
    )


@app.post("/sessions/{session_id}/setup/turn", response_model=SetupTurnResponse)
async def handle_setup_turn(session_id: str, body: TurnRequest) -> SetupTurnResponse:
    """
    Submit a user message in the setup workflow.
    """
    if session_id not in _setup_sessions:
        raise HTTPException(
            status_code=404,
            detail="Setup session not found. Start setup first via POST /sessions/{id}/setup.",
        )

    state = _setup_sessions[session_id]

    if state.get("done"):
        return SetupTurnResponse(
            reply="Setup is complete. See the results via GET /sessions/{id}/setup/results.",
            setup_phase="setup_done",
            done=True,
            setup_topics_covered=state.get("setup_topics_covered", []),
        )

    # Append user message and re-invoke
    state["messages"] = list(state.get("messages", [])) + [
        HumanMessage(content=body.message)
    ]

    result = await setup_graph.ainvoke(state)
    _setup_sessions[session_id] = result

    msgs = result.get("messages", [])
    from langchain_core.messages import AIMessage
    last_ai = next(
        (m for m in reversed(msgs) if isinstance(m, AIMessage)),
        None,
    )
    reply = last_ai.content if last_ai else ""

    return SetupTurnResponse(
        reply=reply,
        setup_phase=result.get("setup_phase", "setup_elicit"),
        done=bool(result.get("done", False)),
        setup_topics_covered=result.get("setup_topics_covered", []),
        red_flags=result.get("red_flags", []),
    )


@app.get("/sessions/{session_id}/setup/results", response_model=SetupResultsResponse)
async def get_setup_results(session_id: str) -> SetupResultsResponse:
    """Retrieve power analysis, MDE, synthetic data, and validation results."""
    if session_id not in _setup_sessions:
        raise HTTPException(
            status_code=404,
            detail="Setup session not found.",
        )

    state = _setup_sessions[session_id]
    if not state.get("done"):
        raise HTTPException(
            status_code=400,
            detail="Setup is not yet complete. Continue the conversation.",
        )

    return SetupResultsResponse(
        setup_report_markdown=state.get("setup_report_markdown", ""),
        power_results=state.get("power_results") or {},
        mde_results={
            k: v
            for k, v in (state.get("mde_results") or {}).items()
            if k != "power_by_effect"   # omit large list from JSON response
        },
        synthetic_data_csv=state.get("synthetic_data_csv", ""),
        validation_results=state.get("validation_results") or {},
        power_curve_json=state.get("power_curve_json", ""),
        red_flags=state.get("red_flags") or [],
    )


@app.get("/sessions/{session_id}/setup/status")
async def get_setup_status(session_id: str) -> dict:
    """Lightweight setup session status."""
    if session_id not in _setup_sessions:
        raise HTTPException(status_code=404, detail="Setup session not found.")
    state = _setup_sessions[session_id]
    return {
        "session_id": session_id,
        "setup_phase": state.get("setup_phase", "unknown"),
        "done": state.get("done", False),
        "setup_topics_covered": state.get("setup_topics_covered", []),
        "chosen_method_key": state.get("chosen_method_key", ""),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
