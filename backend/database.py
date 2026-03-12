"""
SQLite session persistence for the measurement design agent.

Stores elicitation sessions, setup sessions, and FAQ conversation history
as JSON blobs.  LangChain message objects are serialised to/from plain dicts.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# ── DB location ───────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).resolve().parent.parent / "sessions.db"


# ── JSON helpers ──────────────────────────────────────────────────────────────


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalars / arrays that sneak into state dicts."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    result: list[dict] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append({"type": "human", "content": m.content, "id": m.id})
        elif isinstance(m, AIMessage):
            result.append({"type": "ai", "content": m.content, "id": m.id})
        elif isinstance(m, SystemMessage):
            result.append({"type": "system", "content": m.content, "id": m.id})
        else:
            result.append({"type": type(m).__name__, "content": str(m.content), "id": getattr(m, "id", None)})
    return result


def _deserialize_messages(data: list[dict]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for m in data:
        t = m.get("type", "")
        content = m.get("content", "")
        mid = m.get("id")
        kwargs: dict[str, Any] = {"content": content}
        if mid:
            kwargs["id"] = mid
        if t == "human":
            result.append(HumanMessage(**kwargs))
        elif t == "ai":
            result.append(AIMessage(**kwargs))
        elif t == "system":
            result.append(SystemMessage(**kwargs))
        # skip unknowns silently
    return result


def serialize_state(state: dict) -> str:
    """Convert a LangGraph state dict to a JSON string."""
    s = dict(state)
    if "messages" in s:
        s["messages"] = _serialize_messages(s["messages"])
    return json.dumps(s, cls=_NumpyEncoder, default=str)


def deserialize_state(json_str: str) -> dict:
    """Convert a JSON string back to a LangGraph state dict."""
    s: dict = json.loads(json_str)
    if "messages" in s and isinstance(s["messages"], list):
        s["messages"] = _deserialize_messages(s["messages"])
    return s


# ── Connection helpers ────────────────────────────────────────────────────────


@contextmanager
def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Schema init ───────────────────────────────────────────────────────────────


def init_db() -> None:
    """Create tables if they don't exist.  Safe to call multiple times."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                state_json   TEXT NOT NULL,
                label        TEXT DEFAULT '',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS setup_sessions (
                session_id   TEXT PRIMARY KEY,
                state_json   TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faq_conversations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                messages_json TEXT NOT NULL,
                method_key   TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
        """)


# ── Session CRUD ──────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Elicitation sessions --

def save_session(session_id: str, state: dict) -> None:
    now = _now()
    blob = serialize_state(state)
    phase = state.get("phase", "")
    done = state.get("done", False)
    label = f"{'✅ ' if done else ''}{phase}"
    with _connect() as conn:
        conn.execute(
            """INSERT INTO sessions (session_id, state_json, label, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET state_json=excluded.state_json,
               label=excluded.label, updated_at=excluded.updated_at""",
            (session_id, blob, label, now, now),
        )


def load_session(session_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT state_json FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if row is None:
        return None
    return deserialize_state(row["state_json"])


def list_sessions() -> list[dict]:
    """Return lightweight summaries of all sessions, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT session_id, label, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM setup_sessions WHERE session_id=?", (session_id,))


# -- Setup sessions --

def save_setup_session(session_id: str, state: dict) -> None:
    now = _now()
    blob = serialize_state(state)
    with _connect() as conn:
        conn.execute(
            """INSERT INTO setup_sessions (session_id, state_json, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET state_json=excluded.state_json,
               updated_at=excluded.updated_at""",
            (session_id, blob, now, now),
        )


def load_setup_session(session_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT state_json FROM setup_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if row is None:
        return None
    return deserialize_state(row["state_json"])


# -- FAQ conversations --

def save_faq_conversation(messages: list[dict], method_key: str | None = None) -> int:
    now = _now()
    blob = json.dumps(messages, cls=_NumpyEncoder, default=str)
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO faq_conversations (messages_json, method_key, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            (blob, method_key, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def update_faq_conversation(faq_id: int, messages: list[dict]) -> None:
    now = _now()
    blob = json.dumps(messages, cls=_NumpyEncoder, default=str)
    with _connect() as conn:
        conn.execute(
            "UPDATE faq_conversations SET messages_json=?, updated_at=? WHERE id=?",
            (blob, now, faq_id),
        )
