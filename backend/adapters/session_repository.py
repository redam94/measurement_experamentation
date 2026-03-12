"""
Session repository adapter: implements the SessionRepository port
using the SQLite database module.
"""
from __future__ import annotations

from measurement_design.ports import SessionRepository
from ..database import (
    init_db,
    save_session,
    load_session,
    list_sessions,
    delete_session,
    save_setup_session,
    load_setup_session,
)


class SQLiteSessionRepository:
    """Adapter implementing SessionRepository using SQLite via database.py."""

    def __init__(self) -> None:
        init_db()

    def save(self, session_id: str, state: dict) -> None:
        save_session(session_id, state)

    def load(self, session_id: str) -> dict | None:
        return load_session(session_id)

    def list_all(self) -> list[dict]:
        return list_sessions()

    def delete(self, session_id: str) -> None:
        delete_session(session_id)

    # Setup session convenience methods
    def save_setup(self, session_id: str, state: dict) -> None:
        save_setup_session(session_id, state)

    def load_setup(self, session_id: str) -> dict | None:
        return load_setup_session(session_id)
