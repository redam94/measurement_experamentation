"""
Port interfaces for the measurement design domain.

These are abstract interfaces that the domain defines and infrastructure
(backend) implements. Following Hexagonal Architecture / Ports & Adapters:

- Driven ports (outbound): What the domain needs from the outside world
- Driving ports (inbound): What the domain exposes as use cases
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from .types import PowerResults, MDEResults, SyntheticDataResult


# ── Driven Ports (outbound) ──────────────────────────────────────────────────
# The domain defines these; the backend provides concrete implementations.

class LLMService(Protocol):
    """Port for LLM interactions."""

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate free-form text from the LLM."""
        ...

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Generate structured JSON from the LLM."""
        ...

    def stream_text(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        """Stream text tokens from the LLM as an async iterator."""
        ...


class SessionRepository(Protocol):
    """Port for session persistence."""

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist session state."""
        ...

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load session state by ID. Returns None if not found."""
        ...

    def list_all(self) -> list[dict[str, Any]]:
        """List all sessions (newest first)."""
        ...

    def delete(self, session_id: str) -> None:
        """Delete a session by ID."""
        ...


# ── Driving Ports (inbound) ──────────────────────────────────────────────────
# Use cases the domain exposes. The backend (FastAPI) calls these.

class ScoringService(Protocol):
    """Score and rank methods against elicited facts."""

    def score_all(self, facts: dict[str, Any]) -> dict[str, float]:
        """Score all methods, returning {method_key: score}."""
        ...

    def rank(self, scores: dict[str, float]) -> list[str]:
        """Return method keys sorted by score descending."""
        ...

    def build_report_data(
        self,
        facts: dict[str, Any],
        scores: dict[str, float],
        explanations: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Build ranked method data dicts for reporting."""
        ...


class SimulationService(Protocol):
    """Power analysis, MDE, and synthetic data generation."""

    def compute_power(
        self, method_key: str, params: dict[str, Any],
    ) -> PowerResults:
        """Run power / sample-size analysis for a method."""
        ...

    def compute_mde(
        self, method_key: str, params: dict[str, Any], facts: dict[str, Any],
    ) -> MDEResults:
        """Run Monte Carlo MDE simulation."""
        ...

    def generate_synthetic(
        self, method_key: str, params: dict[str, Any],
    ) -> SyntheticDataResult:
        """Generate synthetic experiment data."""
        ...


class OutputService(Protocol):
    """Report, spec, and scaffold generation."""

    def generate_report(
        self, ranked_data: list[dict[str, Any]], facts: dict[str, Any],
    ) -> str:
        """Generate markdown report."""
        ...

    def generate_spec_json(self, ranked_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate JSON spec."""
        ...

    def generate_spec_yaml(self, ranked_data: list[dict[str, Any]]) -> str:
        """Generate YAML spec."""
        ...

    def generate_scaffold(self, method_key: str, facts: dict[str, Any]) -> str:
        """Generate PyMC code scaffold for a method."""
        ...
