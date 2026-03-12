"""
FastAPI application — entry point for the measurement design agent backend.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from starlette.responses import StreamingResponse

load_dotenv()

from .graph.graph import graph
from .graph.state import AgentState
from .graph.setup_graph import setup_graph
from measurement_design.methods import METHOD_MAP
from .database import (
    init_db,
    save_session,
    load_session,
    list_sessions as db_list_sessions,
    delete_session as db_delete_session,
    save_setup_session,
    load_setup_session,
)

app = FastAPI(
    title="Measurement Design Agent",
    description=(
        "Agentic framework that elicits experimental design requirements "
        "from non-experts and recommends causal measurement methods for ad campaigns."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


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


class FAQRequest(BaseModel):
    messages: list[dict[str, str]]  # [{"role": "user", "content": "..."}]
    method_key: str | None = None   # optional — focus on a specific method


class FAQResponse(BaseModel):
    reply: str


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


# ── Elicitation Endpoints ─────────────────────────────────────────────────────

@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    """Start a new elicitation session."""
    session_id = str(uuid.uuid4())

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
    save_session(session_id, result)

    msgs = result.get("messages", [])
    last_msg = msgs[-1].content if msgs else ""

    return CreateSessionResponse(
        session_id=session_id,
        message=last_msg,
        phase=result.get("phase", "elicit"),
    )


@app.get("/sessions")
async def list_all_sessions() -> list[dict]:
    """List all persisted sessions (newest first)."""
    return db_list_sessions()


@app.post("/sessions/{session_id}/turn", response_model=TurnResponse)
async def handle_turn(session_id: str, body: TurnRequest) -> TurnResponse:
    """Submit a user message and get the agent's next response."""
    state = load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.get("done"):
        return TurnResponse(
            reply="This session is complete. Please start a new session.",
            phase="done",
            done=True,
            covered_topics=state.get("covered_topics", []),
        )

    state["messages"] = list(state.get("messages", [])) + [
        HumanMessage(content=body.message)
    ]

    result = await graph.ainvoke(state)
    save_session(session_id, result)

    msgs = result.get("messages", [])
    last_ai = next(
        (m for m in reversed(msgs) if isinstance(m, AIMessage)), None,
    )
    reply = last_ai.content if last_ai else ""

    return TurnResponse(
        reply=reply,
        phase=result.get("phase", "elicit"),
        done=bool(result.get("done", False)),
        covered_topics=result.get("covered_topics", []),
    )


@app.post("/sessions/{session_id}/turn/stream")
async def handle_turn_stream(session_id: str, body: TurnRequest):
    """SSE streaming variant of the elicitation turn endpoint."""
    state = load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.get("done"):
        async def _done():
            yield _sse_event({"token": "This session is complete."})
            yield _sse_event({"done": True, "phase": "done", "is_done": True,
                              "covered_topics": state.get("covered_topics", [])})
        return StreamingResponse(_done(), media_type="text/event-stream")

    state["messages"] = list(state.get("messages", [])) + [
        HumanMessage(content=body.message)
    ]

    async def generate():
        yield _sse_event({"status": "thinking"})

        token_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def run_graph():
            try:
                config = {"configurable": {"token_queue": token_queue}}
                result = await graph.ainvoke(state, config=config)
                await token_queue.put(None)  # sentinel to signal completion
                return result
            except Exception:
                await token_queue.put(None)  # unblock reader on error
                raise

        graph_task = asyncio.create_task(run_graph())

        # Stream tokens as they arrive from the graph nodes
        while True:
            item = await token_queue.get()
            if item is None:
                break
            yield _sse_event({"token": item})

        result = await graph_task
        save_session(session_id, result)

        yield _sse_event({
            "done": True,
            "phase": result.get("phase", "elicit"),
            "is_done": bool(result.get("done", False)),
            "covered_topics": result.get("covered_topics", []),
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/sessions/{session_id}/report", response_model=ReportResponse)
async def get_report(session_id: str) -> ReportResponse:
    """Retrieve the generated report for a completed session."""
    state = load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not state.get("done"):
        raise HTTPException(
            status_code=400,
            detail="Session is not yet complete. Continue the conversation.",
        )

    scores = state.get("scores") or {}
    ranked_summary = [
        {"rank": i + 1, "key": key, "score": round(scores.get(key, 0), 1)}
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
    state = load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "phase": state.get("phase", "unknown"),
        "done": state.get("done", False),
        "covered_topics": state.get("covered_topics", []),
    }


@app.get("/sessions/{session_id}/restore")
async def restore_session(session_id: str) -> dict:
    """Return everything the frontend needs to resume a session."""
    state = load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs = state.get("messages", [])
    chat_messages = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant",
         "content": m.content}
        for m in msgs
        if isinstance(m, (HumanMessage, AIMessage))
    ]

    result: dict[str, Any] = {
        "session_id": session_id,
        "phase": state.get("phase", ""),
        "done": state.get("done", False),
        "covered_topics": state.get("covered_topics", []),
        "messages": chat_messages,
    }

    if state.get("done"):
        scores = state.get("scores") or {}
        result["report"] = {
            "markdown": state.get("report_markdown", ""),
            "json_spec": state.get("spec_json") or {},
            "yaml_spec": state.get("spec_yaml", ""),
            "scaffold": state.get("scaffold_code", ""),
            "ranked_methods": [
                {"rank": i + 1, "key": key, "score": round(scores.get(key, 0), 1)}
                for i, key in enumerate(state.get("ranked_methods", []))
            ],
        }

    # Check for setup session too
    setup_state = load_setup_session(session_id)
    if setup_state:
        setup_msgs = setup_state.get("messages", [])
        setup_chat = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant",
             "content": m.content}
            for m in setup_msgs
            if isinstance(m, (HumanMessage, AIMessage))
        ]
        result["setup"] = {
            "active": True,
            "chosen_method_key": setup_state.get("chosen_method_key", ""),
            "setup_phase": setup_state.get("setup_phase", ""),
            "setup_done": bool(setup_state.get("done", False)),
            "setup_topics_covered": setup_state.get("setup_topics_covered", []),
            "red_flags": setup_state.get("red_flags", []),
            "messages": setup_chat,
        }

    return result


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str) -> dict:
    db_delete_session(session_id)
    return {"deleted": session_id}


# ── Setup Workflow Endpoints ──────────────────────────────────────────────────

@app.post("/sessions/{session_id}/setup", response_model=SetupTurnResponse)
async def start_setup(session_id: str, body: StartSetupRequest) -> SetupTurnResponse:
    """Start the setup workflow for a completed elicitation session."""
    elicitation_state = load_session(session_id)
    if elicitation_state is None:
        raise HTTPException(status_code=404, detail="Session not found")

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
        "followup_round": 0,
        "feasibility_checked": False,
        "interim_power_result": {},
        "red_flags": [],
    }

    result = await setup_graph.ainvoke(initial_setup_state)
    save_setup_session(session_id, result)

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
    """Submit a user message in the setup workflow."""
    state = load_setup_session(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Setup session not found. Start setup first via POST /sessions/{id}/setup.",
        )

    if state.get("done"):
        return SetupTurnResponse(
            reply="Setup is complete. See the results via GET /sessions/{id}/setup/results.",
            setup_phase="setup_done",
            done=True,
            setup_topics_covered=state.get("setup_topics_covered", []),
        )

    state["messages"] = list(state.get("messages", [])) + [
        HumanMessage(content=body.message)
    ]

    result = await setup_graph.ainvoke(state)
    save_setup_session(session_id, result)

    msgs = result.get("messages", [])
    last_ai = next(
        (m for m in reversed(msgs) if isinstance(m, AIMessage)), None,
    )
    reply = last_ai.content if last_ai else ""

    return SetupTurnResponse(
        reply=reply,
        setup_phase=result.get("setup_phase", "setup_elicit"),
        done=bool(result.get("done", False)),
        setup_topics_covered=result.get("setup_topics_covered", []),
        red_flags=result.get("red_flags", []),
    )


@app.post("/sessions/{session_id}/setup/turn/stream")
async def handle_setup_turn_stream(session_id: str, body: TurnRequest):
    """SSE streaming variant of the setup turn endpoint."""
    state = load_setup_session(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Setup session not found.",
        )

    if state.get("done"):
        async def _done():
            yield _sse_event({"token": "Setup is complete."})
            yield _sse_event({"done": True, "setup_phase": "setup_done",
                              "setup_done": True,
                              "setup_topics_covered": state.get("setup_topics_covered", []),
                              "red_flags": state.get("red_flags", [])})
        return StreamingResponse(_done(), media_type="text/event-stream")

    state["messages"] = list(state.get("messages", [])) + [
        HumanMessage(content=body.message)
    ]

    async def generate():
        yield _sse_event({"status": "thinking"})

        token_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def run_graph():
            try:
                config = {"configurable": {"token_queue": token_queue}}
                result = await setup_graph.ainvoke(state, config=config)
                await token_queue.put(None)  # sentinel to signal completion
                return result
            except Exception:
                await token_queue.put(None)  # unblock reader on error
                raise

        graph_task = asyncio.create_task(run_graph())

        # Stream tokens as they arrive from the graph nodes
        while True:
            item = await token_queue.get()
            if item is None:
                break
            yield _sse_event({"token": item})

        result = await graph_task
        save_setup_session(session_id, result)

        yield _sse_event({
            "done": True,
            "setup_phase": result.get("setup_phase", "setup_elicit"),
            "setup_done": bool(result.get("done", False)),
            "setup_topics_covered": result.get("setup_topics_covered", []),
            "red_flags": result.get("red_flags", []),
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/sessions/{session_id}/setup/results", response_model=SetupResultsResponse)
async def get_setup_results(session_id: str) -> SetupResultsResponse:
    state = load_setup_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Setup session not found.")

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
            if k != "power_by_effect"
        },
        synthetic_data_csv=state.get("synthetic_data_csv", ""),
        validation_results=state.get("validation_results") or {},
        power_curve_json=state.get("power_curve_json", ""),
        red_flags=state.get("red_flags") or [],
    )


@app.get("/sessions/{session_id}/setup/status")
async def get_setup_status(session_id: str) -> dict:
    state = load_setup_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Setup session not found.")
    return {
        "session_id": session_id,
        "setup_phase": state.get("setup_phase", "unknown"),
        "done": state.get("done", False),
        "setup_topics_covered": state.get("setup_topics_covered", []),
        "chosen_method_key": state.get("chosen_method_key", ""),
    }


@app.get("/sessions/{session_id}/setup/mde-detail")
async def get_mde_detail(session_id: str) -> dict:
    state = load_setup_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Setup session not found.")
    if not state.get("done"):
        raise HTTPException(status_code=400, detail="Setup is not yet complete.")
    return state.get("mde_results") or {}


@app.get("/sessions/{session_id}/setup/sensitivity")
async def get_sensitivity(session_id: str) -> dict:
    """Compute power across a grid of (alpha, effect_size) combinations."""
    state = load_setup_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Setup session not found.")
    if not state.get("done"):
        raise HTTPException(status_code=400, detail="Setup is not yet complete.")

    from measurement_design.simulation import compute_power

    base_params = dict(state.get("setup_params") or {})
    facts = dict(state.get("elicited_facts") or {})
    method_key = state.get("chosen_method_key", "ab_test")

    baseline_rate = base_params.get("baseline_rate")
    baseline_val = base_params.get("baseline_metric_value", 100.0)
    base_value = baseline_rate if (baseline_rate and baseline_rate > 0) else baseline_val

    alphas = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
    effect_pcts = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]

    grid: list[dict] = []
    for a in alphas:
        for ep in effect_pcts:
            params = dict(base_params)
            params["alpha"] = a
            if baseline_rate and baseline_rate > 0:
                params["expected_lift_abs"] = baseline_rate * ep
                params.pop("expected_lift_pct", None)
            else:
                params["expected_lift_abs"] = baseline_val * ep
                params.pop("expected_lift_pct", None)
            try:
                res = compute_power(method_key, params, facts)
                power_val = res.get("achieved_power", 0) or 0
            except Exception:
                power_val = 0.0
            grid.append({
                "alpha": a,
                "effect_rel_pct": round(ep * 100, 1),
                "effect_abs": round(base_value * ep, 6),
                "power": round(power_val, 4),
            })

    return {
        "grid": grid,
        "alphas": alphas,
        "effect_pcts": [round(e * 100, 1) for e in effect_pcts],
        "method_key": method_key,
        "note": "Power computed analytically for each (alpha, effect_size) pair.",
    }


@app.get("/methods/{method_key}/template")
async def get_method_template(method_key: str) -> dict:
    """Return column schema and CSV template for a given method."""
    from measurement_design.simulation.synthetic import (
        synthetic_ab_test_proportions,
        synthetic_ab_test_continuous,
        synthetic_did,
        synthetic_geo_lift,
        synthetic_synthetic_control,
        synthetic_matched_market,
        synthetic_ddml,
    )

    templates = {
        "ab_test": lambda: synthetic_ab_test_proportions(
            baseline_rate=0.05, lift_abs=0.005, n_per_group=3, seed=0,
        ),
        "ab_test_continuous": lambda: synthetic_ab_test_continuous(
            baseline_mean=100.0, baseline_std=30.0, lift_abs=5.0, n_per_group=3, seed=0,
        ),
        "did": lambda: synthetic_did(
            baseline_metric_value=100.0, baseline_metric_std=30.0, lift_abs=5.0,
            num_treatment_units=2, num_control_units=2, num_pre_periods=2, num_post_periods=2,
            seed=0,
        ),
        "geo_lift": lambda: synthetic_geo_lift(
            baseline_metric_value=5000.0, baseline_metric_std=1500.0, lift_abs=250.0,
            num_treatment_geos=2, num_control_geos=2, num_pre_periods=2, num_post_periods=2,
            seed=0,
        ),
        "synthetic_control": lambda: synthetic_synthetic_control(
            baseline_metric_value=100.0, baseline_metric_std=20.0, lift_abs=10.0,
            num_donor_units=3, num_pre_periods=3, num_post_periods=2, seed=0,
        ),
        "matched_market": lambda: synthetic_matched_market(
            baseline_metric_value=5000.0, baseline_metric_std=1500.0, lift_abs=250.0,
            num_pairs=2, num_pre_periods=2, num_post_periods=2, seed=0,
        ),
        "ddml": lambda: synthetic_ddml(
            baseline_metric_value=100.0, baseline_metric_std=30.0, lift_abs=5.0,
            n_obs=5, n_covariates=3, seed=0,
        ),
    }

    if method_key not in templates:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method: {method_key}. Valid: {list(templates.keys())}",
        )

    result = templates[method_key]()
    return {
        "method_key": method_key,
        "columns": result["columns"],
        "csv_template": result["csv_string"],
        "n_rows": result["n_rows"],
        "description": result["description"],
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}


# ── FAQ Chat Endpoints ────────────────────────────────────────────────────────

def _build_faq_messages(body: FAQRequest):
    """Construct the LLM messages list for FAQ chat."""
    from langchain_core.messages import SystemMessage
    from measurement_design.knowledge import METHOD_ASSUMPTIONS
    from measurement_design.prompts.setup_prompts import FAQ_SYSTEM_PROMPT

    system_content = FAQ_SYSTEM_PROMPT
    if body.method_key and body.method_key in METHOD_ASSUMPTIONS:
        method_info = METHOD_ASSUMPTIONS[body.method_key]
        assumptions_text = "\n\n".join(
            f"### {a['name']}\n"
            f"**What it means:** {a['plain_language']}\n"
            f"**Why it matters:** {a['why_it_matters']}\n"
            f"**When it's violated:** {a['when_violated']}\n"
            f"**How to check:** {a['how_to_check']}"
            for a in method_info["assumptions"]
        )
        terms_text = "\n".join(
            f"- **{term}**: {defn}"
            for term, defn in method_info.get("key_terms", {}).items()
        )
        system_content += (
            f"\n\n## Focused Method: {method_info['name']}\n\n"
            f"### Assumptions:\n{assumptions_text}\n\n"
            f"### Key Terms:\n{terms_text}"
        )

    messages = [SystemMessage(content=system_content)]
    for msg in body.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    return messages


@app.post("/faq", response_model=FAQResponse)
async def faq_chat(body: FAQRequest) -> FAQResponse:
    """Non-streaming FAQ chat."""
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-opus-4-5", temperature=0.3, max_tokens=2048)
    messages = _build_faq_messages(body)
    response = await llm.ainvoke(messages)
    return FAQResponse(reply=response.content.strip())


@app.post("/faq/stream")
async def faq_chat_stream(body: FAQRequest):
    """Real token-level streaming FAQ chat via SSE."""
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-opus-4-5", temperature=0.3, max_tokens=2048)
    messages = _build_faq_messages(body)

    async def generate():
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield _sse_event({"token": chunk.content})
        yield _sse_event({"done": True})

    return StreamingResponse(generate(), media_type="text/event-stream")
